#!/bin/bash

# Download the model if it doesn't exist
if [ ! -f /app/models/yolov8n.pt ]; then
    echo "Downloading YOLO model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('/app/models/yolov8n.pt')"
fi

# Start the application with gunicorn
cd src && gunicorn --bind 0.0.0.0:$PORT main:app
