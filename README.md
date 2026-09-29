# AltiMap

Single-view optical remote-sensing imagery to metric elevation models, with an interactive Three.js flythrough viewer.

See `docs/superpowers/specs/` for the design.

For the trained RGB surface model, measured validation results, refined workspace
controls and single-process local launch, see [Surface workspace](docs/surface-workspace.md).
Quick start on the prepared workstation: `bash scripts/run-local.sh` → http://localhost:8080.

## Repository layout

- `viewer/` — FastAPI reconstruction service, direct GeoTIFF/DEM handling,
  class-mask export, Overture building-height refinement, and GLB export.
- `frontend/` — the complete Vite + React + Three.js workspace, including
  its proxy configuration and bundled GAMUS preview assets.
- `src/altimap/` and `tests/` — reusable analysis library and validation.

The frontend and backend deliberately live in this repository. Generated
uploads, model caches, virtual environments, and fetched footprint caches are
excluded because they can be recreated locally and should not be committed.

## Run the integrated app

Use two terminals from the repository root:

```bash
# Terminal 1: the reconstruction API
.venv-da3/bin/python -m viewer.server --host 127.0.0.1 --port 8080

# Terminal 2: the Three.js dashboard
cd frontend
npm ci
npm run dev -- --host 0.0.0.0
```

The local single-process command in `docs/surface-workspace.md` builds the
dashboard first and serves `/api`, `/data-uploads`, and the compiled UI on port
8080. For Vite development, the Vite server proxies `/api` and `/data-uploads`
to this service.

## One-service DigitalOcean deployment

The root `Dockerfile` builds the exact `frontend/` Three.js dashboard and
serves its compiled files from the same FastAPI process. `/api` and generated
`/data-uploads` URLs therefore remain same-origin and require no separate
frontend deployment or CORS rewrite.

Create an App Platform app from the `main` branch and select the repository
Dockerfile, or use `doctl` with the included spec:

```bash
doctl apps create --spec .do/app.yaml
```

The service listens on `0.0.0.0:$PORT` (DigitalOcean supplies the port; the
image defaults to 8080). `apps-s-1vcpu-2gb` is selected because depth and image
classification models need more memory than the smallest container. Generated
uploads are ephemeral on App Platform; use object storage for permanent
exports.

## Upload reconstruction routes

The live viewer exposes `POST /api/reconstruct` and deliberately takes two
different routes:

- **GeoTIFF with CRS + affine transform:** preserves the original spatial grid.
  An embedded DEM/DSM/elevation band is used directly when declared; otherwise
  a reference terrain DEM is read for the GeoTIFF footprint. Where the local
  Overture building cache covers that footprint, real building footprints and
  recorded heights are extruded exactly as in `demdirect`. Semantic masks do
  not change the metric elevation layer: trees, roads, and unmeasured
  buildings remain labels unless their height is present in the GeoTIFF or in
  the georeferenced Overture footprint record. The browser uses a 1025² live
  elevation grid (1024×1024 terrain cells, four times the prior cell count).
  The response includes
  a CRS-preserving class mask, float32 elevation GeoTIFF, and textured GLB. In
  the browser, this opens at true scale (elevation range divided by the mapped
  ground footprint), following the direct `demdirect.html` route.
- **PNG/JPEG, or a TIFF without georeferencing:** runs the image-only relative
  RGB-depth plus seven-class semantic pipeline, returning a relative terrain
  preview and textured GLB. It makes no DEM request and never claims metric
  elevation without a spatial anchor. Its final fused surface is used once by
  the viewer—class scaling is not applied again.
