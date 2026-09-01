# GAMUS Terrain Studio

Static frontend prototype for the AltiMap / SlashCompact concept. It uses real GAMUS RGB and AGL previews extracted from downloaded HDF5 triplets and provides a navigable Three.js terrain mesh, flythrough playback, WASD/QE camera navigation, layer switching, native-resolution displacement geometry (1023 × 1023 segments), vertical exaggeration, split selection, measurement affordance, height profile, calibration confidence, slope, and export feedback. Eighteen selected RGB scene previews (six per split) are available in the scene browser, including 15 newly downloaded scenes.

## Run

```bash
npm install
npm run dev
```

The original 50-per-split source triplets remain in `/home/biplab-dev/GAMUS_50_each`. An additional 15 aligned triplets (45 HDF5 files) are in `/home/biplab-dev/GAMUS_extra_15`. The demo keeps lightweight previews in `public/` so the browser stays responsive while loading dense geometry.

The full-height scene stage uses a compact floating inspector panel. Use the scene switcher in the inspector to move through the 18 available locations. Display textures are rendered at 2048 × 2048, while the interactive mesh samples the height grid at 513 × 513 for smooth navigation.

RGB texture mode and `0.5×` vertical exaggeration are the defaults. The exaggeration control ranges from `0×` (flat) to `3×`. During manual flight, use `W/A/S/D` to move horizontally and `Q/E` to lower or raise the camera. Movement is slow, time-based, and eased; the scene remains draggable with Three.js OrbitControls when no flight key is pressed.

For a waypoint route, enable **Set points**, click two or more terrain positions, and press **Play**. Click an existing marker to attach Three.js X/Y/Z transform arrows and move it along any axis. The camera follows a smooth Catmull–Rom path through the edited points without rotating the terrain. While the route position advances absolutely along the path, dragging the viewport changes the camera yaw and pitch independently.
