#!/bin/bash

# Szybki setup Streamlit production
sudo tee /etc/systemd/system/streamlit-trading.service > /dev/null <<EOF
[Unit]
Description=Streamlit Trading App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/eur_usd_trading_system 
ExecStart=/usr/bin/python3 -m streamlit run app.py --server.port=8081 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Uruchom
sudo systemctl daemon-reload
sudo systemctl enable streamlit-trading.service
sudo systemctl start streamlit-trading.service

# Dodaj cron (health check + restart)
(crontab -l; echo "*/5 * * * * curl -f http://localhost:8081 || sudo systemctl restart streamlit-trading.service") | crontab -
(crontab -l; echo "0 3 * * * sudo systemctl restart streamlit-trading.service") | crontab -

echo "Done! 🚀"