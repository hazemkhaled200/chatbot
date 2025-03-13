# Use an official Python base image with CUDA support (for GPU) or a standard Python image (for CPU)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/huggingface_cache \
    CUDA_HOME=/usr/local/cuda

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip git curl && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Create and activate a virtual environment
RUN python3 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Set the entry point
CMD ["/app/venv/bin/python", "app.py"]
