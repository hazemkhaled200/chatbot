# Use an official Python base image with CUDA support for GPU, fallback to Ubuntu-based Python for CPU
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

# Set environment variables for non-interactive setup and Python optimizations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/huggingface_cache \
    CUDA_HOME=/usr/local/cuda

# Install Python, system dependencies, and CUDA toolkit for deep learning
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip git curl && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy project files to the container
COPY . .  

# Create and activate a virtual environment, then install dependencies
RUN python3 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Expose the Flask API port (default is 5000, but Railway assigns dynamically)
EXPOSE 5000

# Run Gunicorn with correct port binding for Railway
CMD ["/app/venv/bin/gunicorn", "--workers", "4", "--bind", "0.0.0.0:$PORT", "app:app"]
