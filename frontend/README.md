# GAMUS Terrain Studio

The integrated React + Three.js frontend for AltiMap. It uses aligned GAMUS
RGB, height, depth, and semantic-class previews, and supports flythrough
playback, WASD/QE navigation, layer switching, adjustable exaggeration,
scene switching, draggable waypoint routes, height profiles, and export.
The scene browser exposes the bundled GAMUS previews in `public/`.

## Run

```bash
npm install
npm run dev
```

Run the API from the repository root in another terminal:

```bash
.venv-da3/bin/python -m viewer.server --host 127.0.0.1 --port 8000
```

`vite.config.js` proxies the reconstruction API and generated export URLs to
that service. The demo keeps native 1024-pixel JPEG previews in `public/`,
including an indexed semantic class render for every bundled tile.

The full-height scene stage uses a compact floating inspector panel. Use the
scene switcher in the inspector to move through all available locations.
RGB, AGL-height, depth, and semantic-class previews are aligned per tile.
Catalog and image-relative reconstructions use a responsive 513² mesh.
Georeferenced direct reconstructions use a 1025² height grid (1024×1024
terrain cells) without altering metric elevation with semantic classes.

RGB texture mode and `0.5×` vertical exaggeration are the defaults. The exaggeration control ranges from `0×` (flat) to `3×`. During manual flight, use `W/A/S/D` to move horizontally and `Q/E` to lower or raise the camera. Movement is slow, time-based, and eased; the scene remains draggable with Three.js OrbitControls when no flight key is pressed.

For a waypoint route, enable **Set points**, click two or more terrain positions, and press **Play**. Click an existing marker to attach Three.js X/Y/Z transform arrows and move it along any axis. The camera follows a smooth Catmull–Rom path through the edited points without rotating the terrain. While the route position advances absolutely along the path, dragging the viewport changes the camera yaw and pitch independently.
