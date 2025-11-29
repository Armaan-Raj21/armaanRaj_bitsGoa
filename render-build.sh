#!/usr/bin/env bash
# Exit on error
set -o errexit

STORAGE_DIR=/opt/render/project/src/var
mkdir -p $STORAGE_DIR

# Install Poppler on Linux (Render Server)
apt-get update && apt-get install -y poppler-utils

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
