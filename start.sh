#!/bin/bash

# Install additional system dependencies
apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create model directory
mkdir -p /app/models

# Download the model if it doesn't exist
if [ ! -f /app/models/yolov8n.pt ]; then
    echo "Downloading YOLO model..."
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O /app/models/yolov8n.pt
fi

# Start the application with gunicorn
cd src && gunicorn --bind 0.0.0.0:$PORT main:app
