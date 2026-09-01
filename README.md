# AltiMap

Single-view optical remote-sensing imagery to metric elevation models, with an interactive Three.js flythrough viewer.

See `docs/superpowers/specs/` for the design.

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
.venv-da3/bin/python -m viewer.server --host 127.0.0.1 --port 8000

# Terminal 2: the Three.js dashboard
cd frontend
npm ci
npm run dev -- --host 0.0.0.0
```

The Vite development server proxies `/api` and `/data-uploads` to the local
viewer service, so drag-and-drop PNG/JPEG/GeoTIFF reconstruction works from
the dashboard without a separate URL configuration.

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
