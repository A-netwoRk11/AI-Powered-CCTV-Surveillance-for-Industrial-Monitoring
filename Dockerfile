# Use a minimal Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy only necessary source files and models
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/
COPY templates/ ./templates/
COPY static/ ./static/

# Expose port (change if your app uses a different port)
EXPOSE 5000

# Set the entrypoint (adjust if your main file is different)
CMD ["python", "src/main.py"]
