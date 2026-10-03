# AltiMap in one image: React viewer + FastAPI + the height models, weights included.
#
#   docker build -t altimap .
#   docker run --gpus all -p 8000:8000 altimap          # then open http://localhost:8000
#
# Weights are baked in (~4.4 GB), so the container starts ready. Only GeoTIFF uploads need the
# internet, for the Copernicus / SRTM DEM tiles. Needs an NVIDIA driver for CUDA 12.8 and the
# NVIDIA container toolkit on the host; without a GPU it runs on the CPU (slowly).

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Same versions as the development environment (SETUP.md §2).
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir h5py==3.16.0 numpy==2.5.2 scipy==1.18.1 rasterio==1.5.1 \
        pillow==12.3.0 fastapi==0.141.1 uvicorn==0.54.0 python-multipart==0.0.32 \
        pystac-client==0.9.0 planetary-computer==1.0.0 huggingface_hub==1.33.0 shapely==2.1.2 \
        requests==2.34.2 trimesh==5.1.0 transformers==5.17.0 safetensors==0.8.0

# Model code and weights (SETUP.md §3): RS3DAda code at the commit we developed against, our
# fine-tuned v1 and v2, Meta CHMv2 (public mirror, DINOv3 licence), and the DINOv2 encoder code
# that torch.hub would otherwise fetch from GitHub on the first upload.
RUN git clone https://github.com/JTRNEO/SynRS3D.git viewer/cache/SynRS3D \
    && git -C viewer/cache/SynRS3D checkout ab5a485 && rm -rf viewer/cache/SynRS3D/.git
RUN python -c "from huggingface_hub import hf_hub_download, snapshot_download; import shutil; \
hf_hub_download('Dilavesh/altimap-height', 'best.pth', local_dir='viewer/cache'); \
shutil.move(hf_hub_download('Dilavesh/altimap-height', 'v2/best.pth', local_dir='/tmp/hf'), \
            'viewer/cache/best_v2.pth'); \
snapshot_download('WEO-SAS/chm-meta-v2', allow_patterns=['*.json', 'model.safetensors', 'LICENSE.md'])" \
    && python -c "import torch; torch.hub.list('facebookresearch/dinov2', trust_repo=True)" \
    && rm -rf /tmp/hf

COPY src/ src/
COPY viewer/ viewer/
COPY --from=web /web/dist frontend/dist
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "viewer.server", "--host", "0.0.0.0"]
