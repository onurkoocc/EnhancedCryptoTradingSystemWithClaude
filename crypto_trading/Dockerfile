# Use TensorFlow GPU image as base
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    lm-sensors \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

# Create necessary directories
RUN mkdir -p data logs results models

# Set environment variables
ENV PYTHONPATH=/app
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command: use startup script
ENTRYPOINT ["./startup.sh"]

# Default arguments: backtest mode
CMD ["--mode", "backtest"]