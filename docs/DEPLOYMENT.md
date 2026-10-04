# Vercel viewer, Hugging Face demos, GPU VM inference

The viewer can operate independently of the inference server. Public Hugging Face dataset
files hold complete prepared model outputs; Vercel serves the React frontend. The A5000 VM
processes new uploads when online. There is no automatic cloud inference fallback when it is
off: prepared scenes remain usable and new uploads show an availability status.

Published demo: [altimap-demo.vercel.app](https://altimap-demo.vercel.app).
Prepared scenes: [Dilavesh/altimap-demo](https://huggingface.co/datasets/Dilavesh/altimap-demo).
Live uploads are currently disabled pending the VM's HTTPS endpoint.

## Frontend

Import this Git repository into Vercel, select the `updated-dilavesh-new` branch and set Root
Directory to `frontend`. `frontend/vercel.json` selects Vite, `npm run build` and `dist`.
Set these public build-time variables and redeploy:

| Variable | Value |
|---|---|
| `VITE_DEMO_CATALOG` | `https://huggingface.co/datasets/Dilavesh/altimap-demo/resolve/main/index.json` |
| `VITE_API_BASE` | The VM's HTTPS origin, e.g. `https://api.example.org`, without `/api` |

Use `VITE_API_BASE=disabled` until the VM endpoint is configured. A blank value means the
local same-origin server/Vite proxy; it is not the correct setting for a Vercel-only demo.
These variables are bundled into browser JavaScript. HF/Vercel access tokens belong in the
deployment environment, never in `VITE_*` variables. Live download URLs resolve against the
API origin; prepared-scene textures and downloads resolve against the HF result URL.

In **Image → Demo scenes**, choose a prepared model output. View, Walk, Measure, Accuracy,
Context, GLB export and raster downloads use saved results without contacting the VM.
**Reference scenes** remain the separate GAMUS LiDAR preview gallery. Saved results retain
the original processing time and validation, rather than claiming fresh inference. Live API
availability is checked every 30 seconds with a 5-second timeout; Check again retries sooner.
With a configured demo catalog, the first prepared scene opens automatically. Local uploads
without a catalog still start with an empty workspace.

Vercel's [Hobby plan](https://vercel.com/docs/plans/hobby) covers personal, non-commercial
projects within its free limits. See the official [Vite deployment guide](https://vercel.com/docs/frameworks/frontend/vite).

The current site was deployed directly with the Vercel CLI. GitHub automatic deployment is
not connected: Vercel's repository connection returned an access error for the linked account.
To publish updates after authenticating to the same Vercel account:

```bash
cd frontend
npx vercel deploy --prod
```

The project stores the two public variables above for production and preview builds.

## Prepare and publish scenes

A completed `/api/estimate` request now saves `result.json` beside `meta.json` in
`viewer/web/data-uploads/scenes/<id>/`. This contains textures, geometry, terrain, elevation
grids, validation and facility positions. Older `meta.json` files alone cannot restore a
complete scene; re-upload the imagery and optional reference to produce a new result.

```bash
.venv/bin/python -m viewer.demo_bundle \
  viewer/web/data-uploads/scenes/<id>/result.json \
  --out output/hf-demo --label "My demo scene"
```

Repeat for additional scenes, or pass multiple result paths without `--label`. The bundle
contains a catalog, PNG display images, full display state and original raster/GeoJSON
downloads. It does not copy original uploads or references. Generated bundles stay untracked.
Image-derived building geometry remains the restored renderer; map-assisted results are
rejected by the bundler. No raw elevation raster values or validation scores are changed.

After signing in to HF, publish to the public dataset:

```bash
hf upload Dilavesh/altimap-demo output/hf-demo . --repo-type dataset
```

Include a dataset card with imagery, model and OSM attribution, acquisition/evaluation limits
and the scene-specific validation scores. Only publish scenes intended for the public demo. The catalog prioritizes the Sikkim
town/forest imagery and Pittsburgh bridges; Philadelphia City Hall is excluded. Indian
scene heights and the bridges scene have no attached independent height reference.
Maxar RGB previews retain the Open Data Program attribution and CC BY-NC 4.0 terms.
The current model weights already live at [Dilavesh/altimap-height](https://huggingface.co/Dilavesh/altimap-height).
HF public storage is [best-effort](https://huggingface.co/docs/hub/storage-limits); the static
dataset is storage, not a running GPU backend.

## Start the VM API

Install Docker Compose and the NVIDIA Container Toolkit as described in [SETUP.md](../SETUP.md).
The existing Dockerfile includes the models and weights; budget its documented build disk
space. The A5000 has sufficient VRAM for this pipeline, which previously ran on 6 GB.

Point a stable DNS hostname at the VM. Open inbound TCP 80/443 for Caddy HTTPS provisioning,
then run from the repository root:

```bash
ALTIMAP_API_HOST=api.example.org docker compose -f deploy/compose.yml up -d --build
curl https://api.example.org/api/health
docker compose -f deploy/compose.yml exec app nvidia-smi
```

Compose reserves one NVIDIA GPU and persists uploads and HTTPS certificates. The app has no
public container port; Caddy exposes only the viewer's health/inference/progress/download
routes. Caddy uses the supplied hostname for [automatic HTTPS](https://caddyserver.com/docs/automatic-https).
Keep `ALTIMAP_API_HOST` set when running later Compose commands (or use a local env file).

Set Vercel's `VITE_API_BASE` to that HTTPS origin and redeploy. Uploads go directly from the
browser to the VM, with the API's existing CORS support. They do not pass through a Vercel
Function. VM uploads are transient relative to VM availability: their download links work
while the VM is online. Publish selected completed results to HF for lasting demo access.
The first live request loads the models; turning the VM on does not itself run inference.

If the VM has no hostname yet, establish its stable HTTPS endpoint before enabling live
uploads. A temporary tunnel can be used for a presentation, but its URL must be updated
in Vercel whenever it changes.

## Deployment verification (2026-10-05)

- Python regression suite: 141 passed, 1 skipped; frontend helper tests: 4 passed.
- Frontend production build, Compose configuration and Caddy validation passed.
- The public Vercel URL returned HTTP 200 without login. All seven curated HF model-output scenes
  loaded and switched without runtime errors while live processing was disabled; no
  health/inference/progress requests were made.
- Walk/Exit walk, saved reference validation and GLB export worked. The GLB contained
  embedded texture images. A browser-downloaded nDSM GeoTIFF matched the original bytes.
- Sikkim opens by default; Philadelphia is absent from the current public dataset and catalog.
  The Accuracy tab explicitly distinguishes missing independent references from DEM agreement.
  All 33 bundled raster/GeoJSON/sidecar exports matched their original processing outputs.
- Scene switching found and fixed a renderer cleanup error involving grouped facility
  sprites. HF image pixel reads use anonymous CORS to keep measurement canvases readable.
- A fresh upload through the separate local Vite frontend reached the inference API and
  produced working API-origin raster download links.
- VM live inference has not been deployed: its connection details are still needed.
