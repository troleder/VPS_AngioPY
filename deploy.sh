#!/bin/bash
# deploy.sh - Automates Git push, VPS pull, and service restarts for AngioPy

# Exit immediately if any command fails
set -e

# VPS Connection Parameters
VPS_IP="72.61.189.195"
VPS_USER="root"
VPS_PATH="/var/www/analiza-dicom"

# Check if commit message was provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide a commit message."
    echo "Usage: ./deploy.sh \"Your commit message\""
    exit 1
fi

COMMIT_MSG="$1"

echo "=== 1. Staging and committing changes locally ==="
git add .
git commit -m "$COMMIT_MSG" || echo "No changes to commit."

echo "=== 2. Pushing changes to GitHub ==="
git push origin master

echo "=== 3. Pulling changes on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "cd ${VPS_PATH} && git pull origin master"

echo "=== 4. Setting correct file ownership on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "chown -R www-data:www-data ${VPS_PATH} && chmod -R 775 ${VPS_PATH}/local_cache ${VPS_PATH}/ecrf_data ${VPS_PATH}/reports 2>/dev/null || true"

echo "=== 5. Restarting Streamlit services on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "systemctl restart analiza-dicom@8501 analiza-dicom@8502 analiza-dicom@8503 analiza-dicom@8504"

echo "=== 6. Verifying service states ==="
ssh ${VPS_USER}@${VPS_IP} "systemctl is-active analiza-dicom@8501 analiza-dicom@8502 analiza-dicom@8503 analiza-dicom@8504"

echo "🎉 Deployment completed successfully and services restarted!"
