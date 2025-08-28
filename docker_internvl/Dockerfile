FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for optimal inference
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install gliclass==0.1.11
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create application directory and models directory
WORKDIR /app
RUN mkdir -p /app/models

# Copy pre-downloaded models into the container
COPY /app/models/FLARE-gliclass-small-v1.0 /app/models/FLARE-gliclass-small-v1.0 
COPY /app/models/FLARE-InternVL3-8B-hf /app/models/FLARE-InternVL3-8B-hf
COPY /app/models/InternVL3-8B-hf /app/models/InternVL3-8B-hf

# Copy inference script and related files
COPY inference.py /app/
COPY predict.sh /app/

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Make scripts executable
RUN chmod +x /app/predict.sh

# Set working directory
WORKDIR /app

CMD ["python", "inference.py", "--help"] 