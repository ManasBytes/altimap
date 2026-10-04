# Map-assisted building reconstruction check (2026-10-04)

This is a geometry and integration check, not an independent survey accuracy benchmark.
The previous flat-ground feature only changed terrain presentation. This addition changes
building footprints, part heights and roof meshes using OSM data where available.

## Philadelphia City Hall scene

A Fast/SRTM upload of `demo/naip_philadelphia_cityhall.tif` returned:

| Result | Count |
|---|---:|
| Mapped building parts | 77 |
| Parts with their own explicit height tags | 39 |
| Parts inheriting a containing parent's height (estimate) | 19 |
| Parts using floor counts × 3.2 m (estimate) | 15 |
| Parts using floor counts plus roof height (estimate) | 3 |
| Mapped footprints using an image height estimate | 1 |
| Simple shaped roof meshes | 13 |
| Additional image-derived fallback parts, hidden by default | 37 |

The mapped City Hall tower part `way/333316166` uses its OSM height tag of **167 m**.
Other City Hall sections have separate heights, including 30, 137.77 and 155.8 m. These values
are faithfully transferred from map tags or explicitly labelled inherited estimates; they
have not been independently surveyed here. Mapped simple roofs include pyramidal sections,
cones and a dome in the scene. Unsupported complex roofs retain a labelled flat approximation.

Mapped outlines and parts are interpreted according to
[OSM Simple 3D Buildings](https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings).
The [City Hall tower object](https://www.openstreetmap.org/way/333316166) identifies the source
of the 167 m tag. OSM data are cached by the existing query helper; caches have no automatic
expiry and map data may be incomplete, outdated or inaccurate.

## Verification

Synthetic regression tests verify coordinate projection, metric footprint area, frame clipping,
multipolygon courtyards, part-vs-outline precedence, raised bases, unit conversion, height-source
labels, unknown-height abstention, floor/roof accounting, roof mesh completeness and normals,
and facility support on pitched roofs. Frontend tests verify flattened roof shape, raised bases,
marker interpolation and input immutability.

Browser checks cover a real GeoTIFF upload, mapped default view, optional image fallback,
estimated/flat ground, Height/RGB layers, Walk, Exact DSM, building inspection and GLB export.
The GLB contains mapped height provenance, roof metadata and OSM attribution. The checked
flat-ground export with image fallback disabled contains the 77 displayed mapped parts, including
the 167 m tower.

Completed checks:

- `env -u PYTHONPATH .venv/bin/python -m pytest -q`: **154 passed, 1 skipped**.
- `cd frontend && node --test src/cityDisplay.test.mjs`: **3 passed**.
- `cd frontend && npm run build`: passed, with the existing bundle-size warning.
- `git diff --check`: passed.

Pixel arrays, CRS and affine transforms of both nDSM and absolute DSM were identical to the
pre-feature Fast/SRTM upload. Mapped dimensions do not enter reference validation, raster
probe values or the [landscape accuracy benchmark](landscapes-2026-10-04.md).

## Limits

This is map-assisted massing with simple roof geometry. Roof height without a tag uses a
bounded footprint-width heuristic. Floor counts use a 3.2 m convention, and inherited heights
are not per-part measurements. Main-axis along/across roofs are supported; roof directional
bearings and complex roof types such as quadruple saltbox are not reconstructed. Facade details,
unseen surfaces, roof equipment and exact architectural shapes remain unresolved. Ground
footprints and solid roof colours also do not rectify the oblique source image's displacement.

Better-looking geometry is not proof of improved reconstruction accuracy. Further structural
evaluation needs independently aligned footprint/roof references; detailed metric reconstruction
needs stronger height evidence such as stereo-derived DSM or LiDAR.
