# AltiMap Scene Studio

The React/Three.js viewer uses a docked inspector with Scenes, View, Measure, Accuracy and
Context tabs. **Scenes → Demos** opens prepared model outputs with saved textures and
original elevation grids; **Reference** retains the separate GAMUS LiDAR previews. The
default HF collection loads Sikkim automatically. `VITE_API_BASE=disabled` keeps demos
available while live uploads are offline. See [DEPLOYMENT.md](../docs/DEPLOYMENT.md).

## Run

```bash
npm install
npm run dev
```

The prepared demo catalog contains seven model outputs hosted on Hugging Face, including
Sikkim and Pittsburgh bridges. Bundled GAMUS RGB, LiDAR heights and semantic-class layers
remain available in **Reference**. The interactive mesh samples the height grid at
513 × 513. Saved scenes retain their original metric grids, textures and raster exports;
the reference gallery is separately labelled and does not represent inferred heights.

RGB texture mode is the default. Metric uploads and prepared demos open at `1×` true scale;
reference previews use `0.5×`. The exaggeration control ranges from `0×` to `3×`.
During manual flight, use `W/A/S/D` to move horizontally and `Q/E` to lower or raise the
camera. The scene remains draggable with Three.js OrbitControls when no flight key is pressed.

For a waypoint route, enable **Set points**, click two or more terrain positions, and press **Play**. Click an existing marker to attach Three.js X/Y/Z transform arrows and move it along any axis. The camera follows a smooth Catmull–Rom path through the edited points without rotating the terrain. While the route position advances absolutely along the path, dragging the viewport changes the camera yaw and pitch independently.
