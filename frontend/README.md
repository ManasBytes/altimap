# GAMUS Terrain Studio

Static frontend prototype for the AltiMap / SlashCompact concept. It uses aligned GAMUS RGB, AGL-height, semantic-class, and height-preview layers extracted from the downloaded HDF5 triplets. The Three.js terrain mesh supports flythrough playback, WASD/QE camera navigation, RGB/Surface/Height/Classes switching, semantic class inspection, 512 × 512 interactive terrain segments, vertical exaggeration, scene switching, waypoint routes, height profile, calibration confidence, slope, and export feedback. The scene browser exposes all 150 aligned GAMUS tiles plus the 15 curated preview tiles.

## Run

```bash
npm install
npm run dev
```

The original 50-per-split source triplets remain in `/home/biplab-dev/GAMUS_50_each`. An additional 15 aligned triplets (45 HDF5 files) are in `/home/biplab-dev/GAMUS_extra_15`. The demo keeps native 1024-pixel JPEG previews in `public/`, including an indexed semantic class render for every tile, so the browser stays responsive while loading dense geometry.

The full-height scene stage uses a compact floating inspector panel. Use the scene switcher in the inspector to move through all available locations; building-rich tiles are grouped at the top. RGB, AGL-height, depth, and semantic-class previews are aligned per tile. The class map is also used during mesh construction: buildings retain a slightly stronger profile, while tree and low-vegetation canopy heights are reduced to keep the terrain proportional. The interactive mesh samples the height grid at 513 × 513 for smooth navigation.

RGB texture mode and `0.5×` vertical exaggeration are the defaults. The exaggeration control ranges from `0×` (flat) to `3×`. During manual flight, use `W/A/S/D` to move horizontally and `Q/E` to lower or raise the camera. Movement is slow, time-based, and eased; the scene remains draggable with Three.js OrbitControls when no flight key is pressed.

For a waypoint route, enable **Set points**, click two or more terrain positions, and press **Play**. Click an existing marker to attach Three.js X/Y/Z transform arrows and move it along any axis. The camera follows a smooth Catmull–Rom path through the edited points without rotating the terrain. While the route position advances absolutely along the path, dragging the viewport changes the camera yaw and pitch independently.
