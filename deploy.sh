#!/bin/bash
# deploy.sh - Automates React build, Git push, VPS pull, Nginx configuration, and service restarts for AngioPy

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

echo "=== 1. Building React Admin Panel locally ==="
cd admin-panel
npm run build
cd ..
rm -rf admin-dist
cp -R admin-panel/dist admin-dist

echo "=== 2. Staging and committing changes locally ==="
git add .
git commit -m "$COMMIT_MSG" || echo "No changes to commit."

echo "=== 3. Pushing changes to GitHub ==="
git push origin master

echo "=== 4. Pulling changes on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "cd ${VPS_PATH} && git pull origin master"

echo "=== 5. Installing pip dependencies on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "cd ${VPS_PATH} && .venv/bin/pip install fastapi uvicorn python-multipart"

echo "=== 6. Setting up systemd Admin API service on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "cp ${VPS_PATH}/scratch/analiza-dicom-api.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable analiza-dicom-api"

echo "=== 7. Patches and restarts Nginx config ==="
ssh ${VPS_USER}@${VPS_IP} "python3 ${VPS_PATH}/scratch/patch_nginx.py && systemctl restart nginx"

echo "=== 8. Setting correct file ownership on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "chown -R www-data:www-data ${VPS_PATH} && chmod -R 775 ${VPS_PATH}/local_cache ${VPS_PATH}/ecrf_data ${VPS_PATH}/reports 2>/dev/null || true"

echo "=== 9. Restarting all services on VPS ==="
ssh ${VPS_USER}@${VPS_IP} "systemctl restart analiza-dicom-api analiza-dicom@8501 analiza-dicom@8502 analiza-dicom@8503 analiza-dicom@8504"

echo "=== 10. Verifying service states ==="
ssh ${VPS_USER}@${VPS_IP} "systemctl is-active analiza-dicom-api analiza-dicom@8501 analiza-dicom@8502 analiza-dicom@8503 analiza-dicom@8504"

echo "🎉 Deployment completed successfully and services restarted!"
