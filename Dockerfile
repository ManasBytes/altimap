# Single DigitalOcean App Platform service: build the Three.js dashboard and
# serve it with the FastAPI reconstruction API on the same origin.
FROM node:22-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    ALTIMAP_WEB_DIR=/app/frontend/dist \
    PORT=8080
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgdal-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-viewer.txt requirements-viewer.txt
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir . -r requirements-viewer.txt
COPY viewer/ viewer/
COPY --from=frontend-build /app/frontend/dist frontend/dist
RUN mkdir -p frontend/dist/data-uploads/classifications \
    frontend/dist/data-uploads/reconstructions \
    frontend/dist/data-uploads/scenes
EXPOSE 8080
CMD ["sh", "-c", "uvicorn viewer.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
