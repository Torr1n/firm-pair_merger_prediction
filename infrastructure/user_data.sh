#!/bin/bash
# Bootstrap script for the patent vectorization pipeline EC2 instance.
# Runs as root via EC2 user-data on first boot.
#
# This script:
#   1. Installs project Python dependencies
#   2. Pulls data from S3
#   3. Clones the project repo
#
# The pipeline itself is run manually via SSH (not automated) so we can
# monitor progress and catch issues before they burn compute hours.

set -euo pipefail

S3_BUCKET="${s3_bucket}"
S3_PREFIX="${s3_prefix}"

LOG="/home/ubuntu/bootstrap.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Bootstrap started at $(date -u) ==="

# --- System setup ---
apt-get update -qq
apt-get install -y -qq awscli

# --- Project setup ---
cd /home/ubuntu

# Clone the repo
sudo -u ubuntu git clone https://github.com/Torr1n/firm-pair_merger_prediction.git
cd firm-pair_merger_prediction

# Create venv and install dependencies
sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu bash -c "source venv/bin/activate && pip install --quiet -r requirements.txt"

# --- Pull data from S3 ---
sudo -u ubuntu mkdir -p data output/embeddings

echo "Pulling data from s3://$S3_BUCKET/$S3_PREFIX/data/v2/ ..."
sudo -u ubuntu aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/data/v2/" data/ --recursive

# --- Verify ---
echo ""
echo "=== Bootstrap complete at $(date -u) ==="
echo "Files in data/:"
ls -lh data/

echo ""
echo "=== Next steps ==="
echo "1. SSH in:  ssh -i ~/.ssh/KEY.pem ubuntu@<public-ip>"
echo "2. cd firm-pair_merger_prediction"
echo "3. source venv/bin/activate"
echo "4. python scripts/run_full_pipeline.py 2>&1 | tee output/pipeline.log"
echo "5. After completion: aws s3 sync output/ s3://$S3_BUCKET/$S3_PREFIX/output/"
