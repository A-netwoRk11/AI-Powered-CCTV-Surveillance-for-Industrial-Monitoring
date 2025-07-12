# Use a minimal Python base image
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Second stage
FROM python:3.12-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY templates/ ./templates/
COPY static/ ./static/

# Create necessary directories
RUN mkdir -p models && \
    mkdir -p output/screenshots output/uploads output/videos

# Download model at runtime
ENV MODEL_PATH=/app/models/yolov8n.pt
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('$MODEL_PATH')"

# Expose port
EXPOSE 5000

# Set the entrypoint using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--chdir", "src", "main:app"]
