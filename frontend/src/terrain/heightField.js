// Palette used by the static class-map previews. JPEG compression can shift a
// color by a few values, so class decoding below uses nearest-palette matching
// rather than exact RGB equality.
const CLASS_PALETTE = [
  [31, 43, 61], // other/background
  [145, 116, 76], // ground
  [139, 205, 91], // low vegetation
  [235, 143, 57], // buildings
  [50, 157, 214], // water
  [142, 151, 163], // roads
  [47, 116, 81], // trees
];
// Buildings get a small raised floor plus their measured AGL signal, while
// tree canopies and low vegetation are compressed toward the terrain plane so
// a tall tree does not become a skyscraper beside a normal roof.
const CLASS_HEIGHT_BASE = [0.02, 0.03, 0.04, 0.24, 0.02, 0.04, 0.06];
const CLASS_HEIGHT_GAIN = [0.78, 0.68, 0.58, 0.95, 0.2, 0.5, 0.3];
const CLASS_HEIGHT_CAP = [0.72, 0.62, 0.42, 0.78, 0.28, 0.56, 0.34];
const HEIGHT_WORLD_SCALE = 0.62;
const TERRAIN_BASELINE = 0.32;
const BUILDING_CLASS_INDEX = 3;
const BUILDING_NORMALIZE_LOW = 0.1;
// A handful of genuinely tall structures (or noisy AGL pixels) can otherwise
// set the top of the scale, which then over-amplifies an ordinary house's
// own single highest pixel (e.g. its roof ridge) toward that same cap.
// Trimming the top percentile further keeps that ceiling closer to what
// "tall" buildings in the scene actually look like.
const BUILDING_NORMALIZE_HIGH = 0.88;

// Raw height-map pixels are single-sample estimates, so buildings render as
// bristling, knife-edge columns. A box/Gaussian blur would remove that noise
// by averaging across edges too, melting flat rooftops and vertical walls
// into rounded blobs. A median filter instead replaces each sample with the
// middle value of its neighborhood — noise spikes get discarded (a spike is
// never the median of a mostly-flat neighborhood) while flat planes and hard
// edges pass through unchanged, so buildings keep crisp geometry.
function medianFilter(src, width, height, radius) {
  const out = new Float32Array(src.length);
  const window = new Float32Array((radius * 2 + 1) * (radius * 2 + 1));
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);
      let n = 0;
      for (let ny = y0; ny <= y1; ny++)
        for (let nx = x0; nx <= x1; nx++) window[n++] = src[ny * width + nx];
      out[y * width + x] = window.subarray(0, n).sort()[n >> 1];
    }
  }
  return out;
}

// A short separable Gaussian pass removes the thin vertical walls left by a
// single-pixel AGL jump, while retaining the broad footprint of a roof or
// hillside. It runs in two linear passes so the 513² interactive grid stays
// responsive during scene changes.
function gaussianBlurField(src, width, height) {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const kernel = [1, 4, 6, 4, 1];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let total = 0;
      for (let k = -2; k <= 2; k++) {
        const sampleX = Math.min(width - 1, Math.max(0, x + k));
        total += src[y * width + sampleX] * kernel[k + 2];
      }
      tmp[y * width + x] = total / 16;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let total = 0;
      for (let k = -2; k <= 2; k++) {
        const sampleY = Math.min(height - 1, Math.max(0, y + k));
        total += tmp[sampleY * width + x] * kernel[k + 2];
      }
      out[y * width + x] = total / 16;
    }
  }
  return out;
}

// Denoises spikes with an edge-preserving median filter, then clips the
// tallest slice of the field so isolated outliers (trees, poles) too wide
// for the filter to remove don't tower over the surrounding buildings.
function buildHeightField(px, width, height) {
  const size = width * height;
  const raw = new Float32Array(size);
  for (let i = 0; i < size; i++) raw[i] = px[i * 4] / 255;
  const median = medianFilter(raw, width, height, 2);
  const smoothed = gaussianBlurField(
    gaussianBlurField(median, width, height),
    width,
    height,
  );
  const cap = Float32Array.from(smoothed).sort()[Math.floor(size * 0.985)];
  for (let i = 0; i < size; i++) if (smoothed[i] > cap) smoothed[i] = cap;
  return smoothed;
}

// Metric elevation produced by the GeoTIFF route is already a continuous,
// calibrated field. Do not run the relative-preview denoiser or class-height
// remapping over it: either would alter real slope and height differences.
function buildAbsoluteHeightField(px, width, height) {
  const raw = new Float32Array(width * height);
  for (let i = 0; i < raw.length; i++) raw[i] = px[i * 4] / 255;
  return raw;
}

function buildClassField(px, width, height) {
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) {
    const offset = i * 4;
    let best = 0;
    let distance = Infinity;
    for (let classIndex = 0; classIndex < CLASS_PALETTE.length; classIndex++) {
      const palette = CLASS_PALETTE[classIndex];
      const dr = px[offset] - palette[0];
      const dg = px[offset + 1] - palette[1];
      const db = px[offset + 2] - palette[2];
      const nextDistance = dr * dr + dg * dg + db * db;
      if (nextDistance < distance) {
        distance = nextDistance;
        best = classIndex;
      }
    }
    out[i] = best;
  }
  return out;
}

// A single clamp pass only compares a pixel to its immediate neighbors, so a
// ridge-shaped spike a few pixels wide (a roof peak, a noisy AGL cluster)
// keeps a high neighbor-mean along its whole length and survives untouched.
// Running several passes lets each one erode a bit more off the spike's
// edges, so by the last pass even a multi-pixel ridge has been ground back
// down toward its surrounding roofline instead of just its outermost rim.
function limitHeightDiscontinuities(field, width, height, passes = 4) {
  const maxJump = 0.07;
  let current = field;
  for (let pass = 0; pass < passes; pass++) {
    const out = current.slice();
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const index = y * width + x;
        let neighborTotal = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx || dy) neighborTotal += current[(y + dy) * width + x + dx];
          }
        }
        const neighborMean = neighborTotal / 8;
        out[index] = Math.min(
          neighborMean + maxJump,
          Math.max(neighborMean - maxJump, current[index]),
        );
      }
    }
    current = out;
  }
  return current;
}

function applyClassHeightScale(rawHeight, classField, width, height) {
  if (!classField) return limitHeightDiscontinuities(rawHeight, width, height);

  // Buildings need their own robust range. AGL JPEGs can use only a small
  // slice of the available grayscale range in one tile, making an entire
  // neighborhood look flat even when its roofs have useful height contrast.
  // Percentile normalization lifts that class only, while ignoring isolated
  // tree/edge pixels that would otherwise become tall outliers.
  let buildingCount = 0;
  for (let i = 0; i < classField.length; i++)
    if (classField[i] === BUILDING_CLASS_INDEX) buildingCount++;
  let buildingLow = 0;
  let buildingHigh = 1;
  if (buildingCount >= 8) {
    const buildingHeights = new Float32Array(buildingCount);
    let offset = 0;
    for (let i = 0; i < classField.length; i++) {
      if (classField[i] === BUILDING_CLASS_INDEX)
        buildingHeights[offset++] = rawHeight[i];
    }
    buildingHeights.sort();
    buildingLow =
      buildingHeights[Math.floor((buildingCount - 1) * BUILDING_NORMALIZE_LOW)];
    buildingHigh =
      buildingHeights[
        Math.floor((buildingCount - 1) * BUILDING_NORMALIZE_HIGH)
      ];
    if (buildingHigh - buildingLow < 0.04) {
      buildingLow = 0;
      buildingHigh = 1;
    }
  }
  const buildingRange = Math.max(0.04, buildingHigh - buildingLow);
  const adjusted = new Float32Array(rawHeight.length);
  for (let i = 0; i < rawHeight.length; i++) {
    const classIndex = classField[i];
    const base = CLASS_HEIGHT_BASE[classIndex] ?? 0.02;
    const gain = CLASS_HEIGHT_GAIN[classIndex] ?? 0.78;
    const cap = CLASS_HEIGHT_CAP[classIndex] ?? 0.72;
    if (classIndex === BUILDING_CLASS_INDEX && buildingCount >= 8) {
      const normalized = Math.min(
        1,
        Math.max(0, (rawHeight[i] - buildingLow) / buildingRange),
      );
      adjusted[i] = base + normalized * (cap - base);
    } else {
      adjusted[i] = Math.min(cap, base + rawHeight[i] * gain);
    }
  }
  return limitHeightDiscontinuities(adjusted, width, height);
}

// Classic "jet" depth-map palette (dark blue -> blue -> cyan -> green ->
// yellow -> orange -> dark red), the same convention used in depth/height
// estimation papers, computed from the mesh's own smoothed height data
// instead of loading the pre-baked depth.jpg.
const JET_STOPS = [
  [0.0, [0.02, 0.02, 0.35]],
  [0.2, [0.05, 0.35, 0.85]],
  [0.4, [0.05, 0.85, 0.85]],
  [0.55, [0.25, 0.85, 0.25]],
  [0.7, [0.95, 0.95, 0.15]],
  [0.85, [0.98, 0.55, 0.05]],
  [1.0, [0.75, 0.05, 0.05]],
];
function heightToColor(t, out) {
  t = Math.min(1, Math.max(0, t));
  let i = 0;
  while (i < JET_STOPS.length - 2 && t > JET_STOPS[i + 1][0]) i++;
  const [t0, c0] = JET_STOPS[i];
  const [t1, c1] = JET_STOPS[i + 1];
  const f = (t - t0) / (t1 - t0 || 1);
  out[0] = c0[0] + (c1[0] - c0[0]) * f;
  out[1] = c0[1] + (c1[1] - c0[1]) * f;
  out[2] = c0[2] + (c1[2] - c0[2]) * f;
  return out;
}

// Stretches color mapping across the scene's own min/max (rather than an
// assumed fixed band) so the full dark-blue-to-dark-red range is always used,
// even for scenes with a narrower height spread.
function buildHeightColors(heightField) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < heightField.length; i++) {
    if (heightField[i] < min) min = heightField[i];
    if (heightField[i] > max) max = heightField[i];
  }
  const range = Math.max(max - min, 1e-4);
  const colors = new Float32Array(heightField.length * 3);
  const rgb = [0, 0, 0];
  for (let i = 0; i < heightField.length; i++) {
    heightToColor((heightField[i] - min) / range, rgb);
    colors[i * 3] = rgb[0];
    colors[i * 3 + 1] = rgb[1];
    colors[i * 3 + 2] = rgb[2];
  }
  return colors;
}

export {
  HEIGHT_WORLD_SCALE,
  TERRAIN_BASELINE,
  buildHeightField,
  buildAbsoluteHeightField,
  buildClassField,
  applyClassHeightScale,
  buildHeightColors,
};
