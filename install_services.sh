#!/bin/bash
set -e
sudo cp /home/sidgupta3391/trading-engine/trading-engine-alpha.service \
        /home/sidgupta3391/trading-engine/trading-engine-protection.service \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-engine-protection.service trading-engine-alpha.service
sudo systemctl start trading-engine-protection.service
echo "=== Protection Plane status ==="
systemctl status trading-engine-protection.service --no-pager -l
