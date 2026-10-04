# AltiMap viewer

React + Three.js frontend for the single-image elevation pipeline. The production build is
served by `viewer.server` alongside the inference API. See [SETUP.md](../SETUP.md) for Python
and checkpoint setup, and [ARCHITECTURE.md](../ARCHITECTURE.md) for the data contracts.

## Run

Requires Node.js 20+ and the inference environment described in SETUP.md.

```bash
npm ci
npm run build
# From the repository root:
.venv-da3/bin/python -m viewer.server
```

Open http://127.0.0.1:8000. For frontend development, `npm run dev` starts Vite;
`vite.config.js` proxies `/api` and `/data-uploads` to the local server on port 8000.

For Vercel deployment with Hugging Face prepared model outputs and a separate GPU VM, see
[DEPLOYMENT.md](../docs/DEPLOYMENT.md) and `.env.example`. `VITE_API_BASE` sets the live API
origin (`disabled` for prepared demos only); `VITE_DEMO_CATALOG` supplies the saved-scene
catalog. Image → Demo scenes works independently of API availability, with original textures,
geometry, elevation grids, validation and exports. Reference scenes are the separate LiDAR
preview gallery. All `VITE_*` configuration is public browser code.

## Workstation

- **Image:** upload PNG/JPG/GeoTIFF, load a prepared demo, or choose a GAMUS reference scene; set GSD, quality,
  reference type, GCPs and base DEM. Raster/GeoJSON downloads appear here.
- **View:** City model / Exact DSM, RGB/Surface/Height/Classes/Slope/Error layers,
  contours and vertical exaggeration.
- **Measure:** point height/slope, A→B profiles, PNG scale reprocessing and waypoint routes.
- **Accuracy:** reference metrics, scatter, per-class errors, DEM agreement and GCP results.
- **Context:** OpenStreetMap bridges, flood defences and critical facilities for GeoTIFFs.

The top bar contains Open image, GLB Export, route playback, fullscreen and theme controls.
Inspector tabs support arrow keys, Home and End. On mobile, the viewport sits above the
scrollable inspector. Fonts are bundled locally.

## Navigation and scene data

Orbit by dragging; fly with WASD/QE. Walk follows the ground using WASD and Shift to run.
In Measure, double-click A and B for a height profile. For a route, enable Set points, click at
least two terrain positions, then Play; markers can be selected and moved with transform arrows.

Uploads show model estimates at 1× vertical scale. SRTM is the default absolute-elevation base;
Copernicus is selectable. PNG/JPG without a known GSD use an explicitly reported experimental
0.33 m/pixel assumption. Select the reference type explicitly: nDSM/AGL is height above ground,
while DSM is absolute elevation. Auto-detect can misclassify references near sea level.

Catalog images in `public/` are GAMUS LiDAR reference previews, not model predictions. They
start at 0.5× exaggeration and are labelled accordingly. The mesh uses 1025² vertices on capable
GPUs or 513² on integrated/software/mobile graphics; `?detail=high|standard` overrides this.

Recording guide: [demo video script](../docs/DEMO_VIDEO_SCRIPT.md).
