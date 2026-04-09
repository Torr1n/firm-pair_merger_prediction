#!/bin/bash
# Bootstrap script for the K_max convergence sweep EC2 instance.
# Runs as root via EC2 user-data on first boot.
#
# This script installs project dependencies and checks out the exact sweep commit.
# Week 1 outputs are pulled manually after SSH using `aws configure --profile torrin`.

set -euo pipefail

REPO_COMMIT="${repo_commit}"

LOG="/home/ubuntu/bootstrap.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== K_max sweep bootstrap started at $(date -u) ==="

apt-get update -qq
apt-get install -y -qq awscli python3.10-venv

cd /home/ubuntu
sudo -u ubuntu git clone https://github.com/Torr1n/firm-pair_merger_prediction.git
cd firm-pair_merger_prediction
sudo -u ubuntu git checkout "$REPO_COMMIT"

sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu bash -c "source venv/bin/activate && pip install --quiet -r requirements.txt"

sudo -u ubuntu mkdir -p output/week2_inputs output/kmax_sweep/status

echo ""
echo "=== K_max sweep bootstrap complete at $(date -u) ==="
echo ""
echo "=== Next steps ==="
echo "1. SSH in: ssh -i ~/.ssh/KEY.pem ubuntu@<public-ip>"
echo "2. aws configure --profile torrin"
echo "3. cd firm-pair_merger_prediction"
echo "4. aws s3 cp s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/patent_vectors_50d.parquet output/week2_inputs/patent_vectors_50d.parquet --profile torrin"
echo "5. aws s3 cp s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/gvkey_map.parquet output/week2_inputs/gvkey_map.parquet --profile torrin"
echo "6. source venv/bin/activate"
echo "7. bash scripts/start_kmax_sweep.sh"
