# Multi-stage build for better optimization
FROM public.ecr.aws/lambda/python:3.11 AS builder

# Install build dependencies + OpenMP (Python 3.11 still uses yum)
RUN yum update -y && \
    yum install -y gcc gcc-c++ cmake make libgomp && \
    yum clean all

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements and install directly to /opt/python (Lambda layer format)
COPY requirements.txt .
RUN uv pip install -r requirements.txt --target /opt/python

# Final stage
FROM public.ecr.aws/lambda/python:3.11

# Install runtime dependencies
RUN yum update -y && \
    yum install -y libgomp && \
    yum clean all

# Copy installed packages from builder to Lambda's expected location
COPY --from=builder /opt/python /opt/python

# Set PYTHONPATH so Python can find packages
ENV PYTHONPATH="/opt/python:${PYTHONPATH}"

# Set cache directories BEFORE downloading model
ENV TRANSFORMERS_CACHE=/tmp
ENV HF_HOME=/tmp  
ENV SENTENCE_TRANSFORMERS_HOME=/tmp

# Copy your source code
COPY data/ ./data/
COPY deployment/ ./deployment/
COPY models/ ./models/
COPY training/ ./training/
COPY lambda_handler.py ./

CMD ["lambda_handler.lambda_handler"]