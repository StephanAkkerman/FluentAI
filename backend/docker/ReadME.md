# 🐳 Docker Setup for FluentAI Backend

This directory contains Docker configuration files for running the FluentAI backend service.

## 📂 Directory Structure
```
FluentAI/
├── config.yaml             # Root-level configuration file
├── backend/
│   ├── docker/             # This directory
│   │   ├── Dockerfile      # Main Dockerfile for Linux NVIDIA
│   │   ├── Dockerfile.mac  # Special Dockerfile for Apple Silicon
│   │   ├── docker-compose.yml 
│   │   └── README.md         
```
## ⚙️ Configuration
The Docker setup uses the project's root-level configuration file at /FluentAI/config.yaml. Make sure this file exists and contains the necessary configuration.

## 🚀 Usage
Run all commands from the docker directory:
```
/FluentAI/backend/docker
```
### Choose the appropriate version for your hardware:
#### 🖥️ For Linux/Windows with NVIDIA GPU:
```
docker-compose up -d fluentai-gpu
```
#### 💻 For Linux/Windows CPU only:
```
docker-compose up -d fluentai-cpu
```
#### 🍎 For Apple Silicon Macs (⚠️ work in progress 🚧):
```
docker-compose up -d fluentai-apple
```

### 🌐 View logs:
```
docker-compose logs -f
```

### 📊 Accessing the API
- CPU version: http://localhost:8000
- GPU version: http://localhost:8001
- Apple Silicon version: http://localhost:8002

The FastAPI documentation is available at /docs (e.g., http://localhost:8000/docs)
or visit: https://fastapi.tiangolo.com/#example-upgrade 

### 🛑 Stopping the Containers
```
docker-compose down
```

### 🔄 Rebuilding after Code Changes
```
docker-compose up -d --build fluentai-[cpu|gpu|apple]
```
### 💾 Data Persistence
The following Docker volumes are used for data persistence:
- models_data: Caches downloaded ML models
- datasets_data: Stores datasets
- local_data: Stores generated files (images, audio, etc.)
- logs_data: Stores application logs

## ⚠️ Troubleshooting
### Missing or incorrect config file
Make sure the config file exists at /FluentAI/config.yaml. The Docker container mounts this file directly into the container.
### Out of memory errors
Increase the memory limit for your Docker container in Docker Desktop settings.
### GPU not detected
For NVIDIA GPU users, ensure NVIDIA Container Toolkit is properly installed:
```
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
### Network issues downloading models
If you experience issues downloading models, check network connectivity and ensure Hugging Face credentials are correctly set up if needed.