# plotters.py

# Core imports - lightweight
import logging

# Setup logger
logger = logging.getLogger(__name__)


class ROCPlotter:
    def __init__(self, class_names=None, figsize=(10, 5)):
        self.class_names = class_names or ['Class 0 (Bearish)', 'Class 1 (Neutral)', 'Class 2 (Bullish)']
        self.figsize = figsize
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']

    def _calculate_roc_for_class(self, trained_models, X_test, y_test, class_idx):
        """Calculate ROC for one class"""
        # Lazy import sklearn metrics
        from sklearn.metrics import roc_curve, auc
        import numpy as np

        y_binary = (y_test == class_idx).astype(int)
        roc_data = []

        for i, model_info in enumerate(trained_models):
            model_name = model_info['name']
            model = model_info['model']

            try:
                # Get prediction probabilities
                y_pred_proba = model.predict_proba(X_test)[:, class_idx]

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_binary, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                roc_data.append((model_name, fpr, tpr, roc_auc))

            except Exception as e:
                logger.error(f"Model {model_name} ROC error: {e}")
                continue

        return roc_data

    def _plot_single_class_roc(self, ax, roc_data, class_name):
        """Plot ROC for one class"""
        for i, (model_name, fpr, tpr, auc_score) in enumerate(roc_data):
            ax.plot(fpr, tpr, color=self.colors[i % len(self.colors)], lw=2,
                    label=f'{model_name} (AUC = {auc_score:.3f})')

        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5,
                label='Random (AUC = 0.500)')

        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {class_name}')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

    def plot_all_classes(self, trained_models, X_test, y_test):
        """Main function - plot all ROCs"""
        # Lazy import matplotlib
        import matplotlib.pyplot as plt

        logger.info("Generating ROC curves...")

        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        for class_idx in range(len(self.class_names)):
            roc_data = self._calculate_roc_for_class(trained_models, X_test, y_test, class_idx)
            self._plot_single_class_roc(axes[class_idx], roc_data, self.class_names[class_idx])

        plt.tight_layout()
        logger.info("ROC curves generated successfully")
        return fig


class PrecisionRecallPlotter:
    def __init__(self, class_names=None, figsize=(10, 5)):
        self.class_names = class_names or ['Class 0 (Bearish)', 'Class 1 (Neutral)', 'Class 2 (Bullish)']
        self.figsize = figsize
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    def _calculate_pr_for_class(self, models, X_test, y_test, class_idx):
        """Calculate P-R curve for ONE class"""
        # Lazy import sklearn metrics
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import numpy as np

        y_binary = (y_test == class_idx).astype(int)
        pr_data = []

        for i, model_info in enumerate(models):
            model_name = model_info['name']
            model = model_info['model']

            try:
                # Probability for one class
                y_scores = model.predict_proba(X_test)[:, class_idx]

                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(y_binary, y_scores)
                avg_precision = average_precision_score(y_binary, y_scores)

                pr_data.append((model_name, precision, recall, avg_precision))

            except Exception as e:
                logger.error(f"Model {model_name} PR error: {e}")
                continue

        return pr_data, y_binary

    def _plot_single_class_pr(self, ax, pr_data, y_binary, class_name):
        """Plot P-R curve for ONE class"""
        import numpy as np

        # Plot curves for all models
        for i, (model_name, precision, recall, avg_precision) in enumerate(pr_data):
            ax.plot(recall, precision,
                    color=self.colors[i % len(self.colors)],
                    label=f'{model_name} (AP = {avg_precision:.3f})',
                    lw=2)

        # Baseline (chance level)
        no_skill = len(y_binary[y_binary == 1]) / len(y_binary)
        ax.plot([0, 1], [no_skill, no_skill],
                linestyle='--', color='black', alpha=0.5,
                label=f'Chance level ({no_skill:.3f})')

        # Formatting
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'P-R Curve - {class_name}')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

    def plot_all_classes(self, trained_models, X_test, y_test):
        """Generate all Precision-Recall curves"""
        # Lazy import matplotlib
        import matplotlib.pyplot as plt

        logger.info("Generating Precision-Recall curves...")

        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        for class_idx in range(len(self.class_names)):
            pr_data, y_binary = self._calculate_pr_for_class(
                trained_models, X_test, y_test, class_idx
            )

            self._plot_single_class_pr(
                axes[class_idx], pr_data, y_binary, self.class_names[class_idx]
            )

        plt.tight_layout()
        logger.info("Precision-Recall curves generated successfully")
        return fig