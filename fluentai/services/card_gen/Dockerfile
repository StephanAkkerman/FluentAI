# To build image run: docker build -t card_gen_service:latest -f fluentai/services/card_gen/Dockerfile .
#  To run the container: docker run --gpus all \
#     -v "$(pwd)/logs:/FluentAI/logs" \
#     -v "$(pwd)/models:/FluentAI/models" \
#     -v "$(pwd)/img:/FluentAI/img" \
#     -p 8000:8000 \
#     card_gen_service:latest


# Use the CUDA 12.4 runtime image with cuDNN support
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Install Python 3.10 and other necessary utilities
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

# Set the working directory
WORKDIR /FluentAI

# Copy only the requirements file first to leverage Docker cache
COPY ./requirements/requirements.txt .

# Install pip packages
RUN pip install -e .

# Install PyTorch 2.5.1 compatible with CUDA 12.4.0
RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Ensure the logs directory exists and create the fluentAI.log file
RUN mkdir -p logs && touch logs/fluentAI.log

# Expose the port if your service listens on one
EXPOSE 8000

# Set the entrypoint command
CMD ["python3", "fluentai/services/card_gen/main.py"]
