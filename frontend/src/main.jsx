import {
  Activity,
  Download,
  Eye,
  FileImage,
  Layers3,
  Maximize2,
  Mountain,
  Pause,
  Play,
  RotateCcw,
  RotateCw,
  Ruler,
  Satellite,
  Target,
  Trash2,
  Upload,
  X,
  TriangleRight,
  Diff,
  Waves,
  PersonStanding,
  Moon,
  Sun,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import "./styles.css";
import "./light.css";
import "./enhancements.css";
import "./blender.css";
import { gamusScenes } from "./gamusScenes";

const samples = {
  train: {
    id: "DC_01_25",
    label: "Campus district",
    coord: "12.9716° N, 77.5946° E",
    rgb: "/train-DC_01_25-rgb.jpg",
    height: "/train-DC_01_25-height.jpg",
    max: "42.8 m",
  },
  val: {
    id: "DC_02_26",
    label: "Urban edge",
    coord: "12.9352° N, 77.6245° E",
    rgb: "/val-DC_02_26-rgb.jpg",
    height: "/val-DC_02_26-height.jpg",
    max: "31.4 m",
  },
  test: {
    id: "DC_03_26",
    label: "Dense blocks",
    coord: "13.0068° N, 77.5813° E",
    rgb: "/test-DC_03_26-rgb.jpg",
    height: "/test-DC_03_26-height.jpg",
    max: "56.2 m",
  },
};
const sceneCatalog = [
  { ...samples.train, split: "train", thumb: "/train-DC_01_25-rgb.jpg" },
  {
    id: "DC_02_24",
    label: "Residential grid",
    coord: "12.9821° N, 77.6012° E",
    rgb: "/train-DC_02_24-rgb.jpg",
    height: "/train-DC_02_24-height.jpg",
    max: "38.6 m",
    split: "train",
    thumb: "/train-DC_02_24-rgb.jpg",
  },
  {
    id: "DC_02_25",
    label: "Green corridor",
    coord: "12.9688° N, 77.5827° E",
    rgb: "/train-DC_02_25-rgb.jpg",
    height: "/train-DC_02_25-height.jpg",
    max: "29.1 m",
    split: "train",
    thumb: "/train-DC_02_25-rgb.jpg",
  },
  {
    id: "DC_02_27",
    label: "Industrial pocket",
    coord: "12.9534° N, 77.6123° E",
    rgb: "/train-DC_02_27-rgb.jpg",
    height: "/train-DC_02_27-height.jpg",
    max: "34.7 m",
    split: "train",
    thumb: "/train-DC_02_27-rgb.jpg",
  },
  {
    id: "DC_03_23",
    label: "Transit district",
    coord: "12.9442° N, 77.5881° E",
    rgb: "/train-DC_03_23-rgb.jpg",
    height: "/train-DC_03_23-height.jpg",
    max: "46.2 m",
    split: "train",
    thumb: "/train-DC_03_23-rgb.jpg",
  },
  { ...samples.val, split: "val", thumb: "/val-DC_02_26-rgb.jpg" },
  {
    id: "DC_04_23",
    label: "River approach",
    coord: "12.9204° N, 77.6061° E",
    rgb: "/val-DC_04_23-rgb.jpg",
    height: "/val-DC_04_23-height.jpg",
    max: "27.9 m",
    split: "val",
    thumb: "/val-DC_04_23-rgb.jpg",
  },
  {
    id: "DC_04_27",
    label: "Low-rise fabric",
    coord: "12.9281° N, 77.6352° E",
    rgb: "/val-DC_04_27-rgb.jpg",
    height: "/val-DC_04_27-height.jpg",
    max: "25.4 m",
    split: "val",
    thumb: "/val-DC_04_27-rgb.jpg",
  },
  {
    id: "DC_08_31",
    label: "Hillside fringe",
    coord: "12.9017° N, 77.5748° E",
    rgb: "/val-DC_08_31-rgb.jpg",
    height: "/val-DC_08_31-height.jpg",
    max: "51.7 m",
    split: "val",
    thumb: "/val-DC_08_31-rgb.jpg",
  },
  {
    id: "DC_09_33",
    label: "Canopy study",
    coord: "12.8894° N, 77.6217° E",
    rgb: "/val-DC_09_33-rgb.jpg",
    height: "/val-DC_09_33-height.jpg",
    max: "33.8 m",
    split: "val",
    thumb: "/val-DC_09_33-rgb.jpg",
  },
  { ...samples.test, split: "test", thumb: "/test-DC_03_26-rgb.jpg" },
  {
    id: "DC_05_28",
    label: "Civic core",
    coord: "13.0184° N, 77.6032° E",
    rgb: "/test-DC_05_28-rgb.jpg",
    height: "/test-DC_05_28-height.jpg",
    max: "44.5 m",
    split: "test",
    thumb: "/test-DC_05_28-rgb.jpg",
  },
  {
    id: "DC_05_30",
    label: "Open blocks",
    coord: "13.0272° N, 77.5722° E",
    rgb: "/test-DC_05_30-rgb.jpg",
    height: "/test-DC_05_30-height.jpg",
    max: "22.3 m",
    split: "test",
    thumb: "/test-DC_05_30-rgb.jpg",
  },
  {
    id: "DC_07_21",
    label: "Dense campus",
    coord: "13.0411° N, 77.5919° E",
    rgb: "/test-DC_07_21-rgb.jpg",
    height: "/test-DC_07_21-height.jpg",
    max: "49.1 m",
    split: "test",
    thumb: "/test-DC_07_21-rgb.jpg",
  },
  {
    id: "DC_07_29",
    label: "North ridge",
    coord: "13.0562° N, 77.6128° E",
    rgb: "/test-DC_07_29-rgb.jpg",
    height: "/test-DC_07_29-height.jpg",
    max: "56.2 m",
    split: "test",
    thumb: "/test-DC_07_29-rgb.jpg",
  },
];
const extraCatalog = [
  {
    id: "DC_10_17",
    label: "Mixed-use edge",
    coord: "13.0642° N, 77.5843° E",
    rgb: "/train-DC_10_17-rgb.jpg",
    height: "/train-DC_10_17-height.jpg",
    max: "39.8 m",
    split: "train",
    thumb: "/train-DC_10_17-rgb.jpg",
  },
  {
    id: "DC_10_18",
    label: "Canal district",
    coord: "13.0714° N, 77.6018° E",
    rgb: "/train-DC_10_18-rgb.jpg",
    height: "/train-DC_10_18-height.jpg",
    max: "28.6 m",
    split: "train",
    thumb: "/train-DC_10_18-rgb.jpg",
  },
  {
    id: "DC_10_19",
    label: "Industrial north",
    coord: "13.0831° N, 77.6195° E",
    rgb: "/train-DC_10_19-rgb.jpg",
    height: "/train-DC_10_19-height.jpg",
    max: "45.1 m",
    split: "train",
    thumb: "/train-DC_10_19-rgb.jpg",
  },
  {
    id: "DC_10_21",
    label: "Civic expansion",
    coord: "13.0942° N, 77.5726° E",
    rgb: "/train-DC_10_21-rgb.jpg",
    height: "/train-DC_10_21-height.jpg",
    max: "32.7 m",
    split: "train",
    thumb: "/train-DC_10_21-rgb.jpg",
  },
  {
    id: "DC_10_27",
    label: "Open greenfield",
    coord: "13.1021° N, 77.5942° E",
    rgb: "/train-DC_10_27-rgb.jpg",
    height: "/train-DC_10_27-height.jpg",
    max: "21.9 m",
    split: "train",
    thumb: "/train-DC_10_27-rgb.jpg",
  },
  {
    id: "DC_20_13",
    label: "West hillside",
    coord: "13.1127° N, 77.5532° E",
    rgb: "/val-DC_20_13-rgb.jpg",
    height: "/val-DC_20_13-height.jpg",
    max: "48.3 m",
    split: "val",
    thumb: "/val-DC_20_13-rgb.jpg",
  },
  {
    id: "DC_20_14",
    label: "Low-density fringe",
    coord: "13.1218° N, 77.5784° E",
    rgb: "/val-DC_20_14-rgb.jpg",
    height: "/val-DC_20_14-height.jpg",
    max: "24.8 m",
    split: "val",
    thumb: "/val-DC_20_14-rgb.jpg",
  },
  {
    id: "DC_20_18",
    label: "Creek crossing",
    coord: "13.1344° N, 77.6031° E",
    rgb: "/val-DC_20_18-rgb.jpg",
    height: "/val-DC_20_18-height.jpg",
    max: "35.6 m",
    split: "val",
    thumb: "/val-DC_20_18-rgb.jpg",
  },
  {
    id: "DC_20_19",
    label: "New development",
    coord: "13.1451° N, 77.6214° E",
    rgb: "/val-DC_20_19-rgb.jpg",
    height: "/val-DC_20_19-height.jpg",
    max: "30.2 m",
    split: "val",
    thumb: "/val-DC_20_19-rgb.jpg",
  },
  {
    id: "DC_20_29",
    label: "Forest interface",
    coord: "13.1532° N, 77.5926° E",
    rgb: "/val-DC_20_29-rgb.jpg",
    height: "/val-DC_20_29-height.jpg",
    max: "52.6 m",
    split: "val",
    thumb: "/val-DC_20_29-rgb.jpg",
  },
  {
    id: "DC_20_12",
    label: "North campus",
    coord: "13.1642° N, 77.5718° E",
    rgb: "/test-DC_20_12-rgb.jpg",
    height: "/test-DC_20_12-height.jpg",
    max: "41.5 m",
    split: "test",
    thumb: "/test-DC_20_12-rgb.jpg",
  },
  {
    id: "DC_20_15",
    label: "Warehouse belt",
    coord: "13.1728° N, 77.6061° E",
    rgb: "/test-DC_20_15-rgb.jpg",
    height: "/test-DC_20_15-height.jpg",
    max: "36.9 m",
    split: "test",
    thumb: "/test-DC_20_15-rgb.jpg",
  },
  {
    id: "DC_20_20",
    label: "Eastern ridge",
    coord: "13.1817° N, 77.6282° E",
    rgb: "/test-DC_20_20-rgb.jpg",
    height: "/test-DC_20_20-height.jpg",
    max: "55.7 m",
    split: "test",
    thumb: "/test-DC_20_20-rgb.jpg",
  },
  {
    id: "DC_20_23",
    label: "Residential north",
    coord: "13.1932° N, 77.5835° E",
    rgb: "/test-DC_20_23-rgb.jpg",
    height: "/test-DC_20_23-height.jpg",
    max: "33.4 m",
    split: "test",
    thumb: "/test-DC_20_23-rgb.jpg",
  },
  {
    id: "DC_20_25",
    label: "Outer ring edge",
    coord: "13.2041° N, 77.6147° E",
    rgb: "/test-DC_20_25-rgb.jpg",
    height: "/test-DC_20_25-height.jpg",
    max: "46.8 m",
    split: "test",
    thumb: "/test-DC_20_25-rgb.jpg",
  },
];
// Keep the curated labels at the front, then expose every aligned tile from
// the 50-per-split download. Each scene gets a semantic class-map URL so the
// viewer can switch between RGB, height, depth, and land-cover classes.
const curatedScenes = [...sceneCatalog, ...extraCatalog];
const curatedBuildingCoverage = {
  "train:DC_01_25": 19.7,
  "train:DC_02_24": 15.7,
  "train:DC_02_25": 27.3,
  "train:DC_02_27": 19.2,
  "train:DC_03_23": 11.2,
  "train:DC_10_17": 25.6,
  "train:DC_10_18": 22.0,
  "train:DC_10_19": 26.6,
  "train:DC_10_21": 9.2,
  "train:DC_10_27": 13.1,
  "val:DC_02_26": 22.6,
  "val:DC_04_23": 4.4,
  "val:DC_04_27": 22.5,
  "val:DC_08_31": 23.3,
  "val:DC_09_33": 24.4,
  "val:DC_20_13": 38.9,
  "val:DC_20_14": 22.8,
  "val:DC_20_18": 20.9,
  "val:DC_20_19": 19.1,
  "val:DC_20_29": 30.4,
  "test:DC_03_26": 19.3,
  "test:DC_05_28": 26.8,
  "test:DC_05_30": 20.5,
  "test:DC_07_21": 24.8,
  "test:DC_07_29": 28.5,
  "test:DC_20_12": 35.5,
  "test:DC_20_15": 27.5,
  "test:DC_20_20": 19.6,
  "test:DC_20_23": 0,
  "test:DC_20_25": 7.6,
};
const GAMUS_CITIES = { DC: "Washington DC", PHL: "Philadelphia", NYC: "New York" };
const GAMUS_GSD_M = 0.33;
const displayedScenes = [...curatedScenes, ...gamusScenes].map((scene) => {
  const buildingCoverage =
    scene.buildingCoverage ??
    curatedBuildingCoverage[`${scene.split}:${scene.id}`];
  const city = GAMUS_CITIES[scene.id.split("_")[0]] ?? "GAMUS";
  return {
    ...scene,
    // Provenance, not invented coordinates: these are GAMUS aerial tiles.
    // These heights and classes are the dataset's LiDAR/label layers, not model output.
    coord: `GAMUS · ${city} · ${scene.split} tile · ${GAMUS_GSD_M} m/px · LiDAR reference heights, not model output`,
    groundWidthM: 1024 * GAMUS_GSD_M,
    // Preview JPEGs are non-linearly encoded, so pixel -> metres is not
    // recoverable; the probe reports metres only for uploads (linear encoding).
    metricHeights: false,
    rgb: scene.rgb || `/${scene.split}-${scene.id}-rgb.jpg`,
    height: scene.height || `/${scene.split}-${scene.id}-height.jpg`,
    depth: scene.depth || `/${scene.split}-${scene.id}-depth.jpg`,
    classes: scene.classes || `/${scene.split}-${scene.id}-classes.jpg`,
    thumb: scene.thumb || scene.rgb || `/${scene.split}-${scene.id}-rgb.jpg`,
    buildingCoverage,
    urban: scene.urban ?? (buildingCoverage != null && buildingCoverage >= 20),
  };
});
// Nothing is preloaded: the workspace starts empty until imagery is imported or
// a catalog scene is picked. Text fields fall back to this placeholder.
const EMPTY_SAMPLE = {
  id: "",
  label: "No imagery loaded",
  coord: "Import a PNG, JPG or GeoTIFF to begin",
  max: "—",
  sourceLabel: "no source",
};
const urbanScenes = displayedScenes.filter((scene) => scene.urban);
const otherScenes = displayedScenes.filter((scene) => !scene.urban);

// Terrain detail controls. Keep samples one larger than segments because a grid
// with N segments contains N + 1 vertices along that axis.
// Mesh detail, chosen once at load. 1024 segments (~1 vertex per preview pixel, 2M
// triangles) shows the model's detail in Exact DSM; software renderers, mobile GPUs and
// Intel integrated graphics get 512 (0.5M triangles) so navigation stays smooth. Override
// with ?detail=high or ?detail=standard. The server's grids are 1025 samples a side
// (GRID_SIDE in viewer/server.py); 513 is every other one of them.
function pickMeshSegments() {
  const asked = new URLSearchParams(window.location.search).get("detail");
  if (asked === "high") return 1024;
  if (asked === "standard") return 512;
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) return 512;
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const name = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return /swiftshader|llvmpipe|softpipe|software|mali|adreno|powervr|intel(?!.*arc)/i.test(name)
      ? 512
      : 1024;
  } catch {
    return 512;
  }
}
const MESH_SEGMENTS_X = pickMeshSegments();
const MESH_SEGMENTS_Y = MESH_SEGMENTS_X;
// Weak GPUs also get a smaller shadow map and no supersampling: on software rendering the
// 513 mesh ran at 5 fps with these at full quality.
const LOW_DETAIL = MESH_SEGMENTS_X < 1024;
const HEIGHT_SAMPLE_WIDTH = MESH_SEGMENTS_X + 1;
const HEIGHT_SAMPLE_HEIGHT = MESH_SEGMENTS_Y + 1;

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

// Uploads: the model's own metric heights, drawn faithfully. Only a 3x3 median
// against single-pixel speckle; no blur, cap or discontinuity limiting, since
// those erode walls into mounds and break projection accuracy. Scaled so world
// y = metres * (8 / ground width): at exaggeration 1x the relief is true scale.
function metricDisplayField(metres, width, height, groundWidthM) {
  const clean = medianFilter(metres, width, height, 1);
  const toField = 8 / groundWidthM / HEIGHT_WORLD_SCALE;
  for (let i = 0; i < clean.length; i++)
    clean[i] = TERRAIN_BASELINE + clean[i] * toField;
  return clean;
}

// Linear 8-bit preview pixels -> metres (pixel / 255 * maxM).
function pxToMetres(px, maxM) {
  const out = new Float32Array(px.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = (px[i * 4] / 255) * maxM;
  return out;
}

// 16-bit heights from /api/estimate `grids` (metres = lo + u16 / 65535 * span), taken onto
// the mesh grid: `srcSide` samples a side, every `step`-th one (step 2 for the 513 mesh).
// The 8-bit preview PNGs step 0.1-0.3 m, which made slopes jump ~13 degrees.
function decodeGrid(g, srcSide, side) {
  if (!g) return null;
  const bin = atob(g.b64);
  const step = (srcSide - 1) / (side - 1);
  const out = new Float32Array(side * side);
  for (let r = 0; r < side; r++)
    for (let c = 0; c < side; c++) {
      const k = r * step * srcSide + c * step;
      out[r * side + c] =
        g.lo + ((bin.charCodeAt(2 * k) | (bin.charCodeAt(2 * k + 1) << 8)) / 65535) * g.span;
    }
  return out;
}

// Steep faces (building walls) get darker vertex colours. A single overhead
// photo has no facade pixels, so walls otherwise show roof-edge texture
// smeared downwards; shading them reads as a facade instead. Slope is taken at
// true 1x scale so the shading does not shift with the exaggeration slider.
function buildWallShade(field, width, height) {
  const out = new Float32Array(field.length * 3);
  const step = 8 / (width - 1); // world units between samples
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const xl = x > 0 ? i - 1 : i;
      const xr = x < width - 1 ? i + 1 : i;
      const yu = y > 0 ? i - width : i;
      const yd = y < height - 1 ? i + width : i;
      const gx = ((field[xr] - field[xl]) * HEIGHT_WORLD_SCALE) / ((xr - xl) * step || 1);
      const gy = ((field[yd] - field[yu]) * HEIGHT_WORLD_SCALE) / (((yd - yu) / width) * step || 1);
      const slope = Math.hypot(gx, gy); // rise over run; 1 = 45 degrees
      const shade = 1 - 0.45 * Math.min(1, Math.max(0, (slope - 1) / 2));
      out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = shade;
    }
  }
  return out;
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
            if (dx || dy)
              neighborTotal += current[(y + dy) * width + x + dx];
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
// The legend's gradient, built from the same stops. Vertex colours are linear, and the
// renderer outputs sRGB, so each stop is converted the same way before it reaches CSS.
const JET_CSS = `linear-gradient(90deg, ${JET_STOPS.map(
  ([t, c]) => `${new THREE.Color(c[0], c[1], c[2]).getStyle()} ${t * 100}%`,
).join(", ")})`;
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
// With `lo`/`hi` given (uploads: 0 .. nDSM max, metres) the colours mean fixed heights,
// which is what the legend prints.
// Slope colours: green (flat) -> yellow (15°) -> orange (30°) -> red (45°+), linear RGB.
const SLOPE_MAX_DEG = 45;
const SLOPE_STOPS = [
  [0, [0.03, 0.3, 0.1]],
  [1 / 3, [0.8, 0.7, 0.04]],
  [2 / 3, [0.85, 0.25, 0.02]],
  [1, [0.55, 0.01, 0.02]],
];
const SLOPE_CSS = `linear-gradient(90deg, ${SLOPE_STOPS.map(
  ([t, c]) => `${new THREE.Color(c[0], c[1], c[2]).getStyle()} ${t * 100}%`,
).join(", ")})`;

// Same stops as the server's error map (viewer/server.py:_error_png), which is sRGB.
const ERROR_CSS = "linear-gradient(90deg, rgb(33,102,172), rgb(247,247,247), rgb(178,24,43))";

// Display field (world units) -> metres of the surface the mesh shows.
function displayMetres(field, groundWidthM) {
  const k = (HEIGHT_WORLD_SCALE * groundWidthM) / 8;
  return Float32Array.from(field, (f) => (f - TERRAIN_BASELINE) * k);
}

// Per-vertex slope colours from metres on a W x H grid spanning gW x gH metres.
function buildSlopeColors(metres, W, H, gW, gH) {
  const colors = new Float32Array(W * H * 3);
  const dx = gW / (W - 1);
  const dy = gH / (H - 1);
  const rgb = [0, 0, 0];
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const zx = (metres[r * W + Math.min(W - 1, c + 1)] - metres[r * W + Math.max(0, c - 1)]) /
        (dx * (Math.min(W - 1, c + 1) - Math.max(0, c - 1)));
      const zy = (metres[Math.min(H - 1, r + 1) * W + c] - metres[Math.max(0, r - 1) * W + c]) /
        (dy * (Math.min(H - 1, r + 1) - Math.max(0, r - 1)));
      const deg = (Math.atan(Math.hypot(zx, zy)) * 180) / Math.PI;
      rampColor(SLOPE_STOPS, Math.min(deg / SLOPE_MAX_DEG, 1), rgb);
      colors.set(rgb, i * 3);
    }
  return colors;
}

function rampColor(stops, t, out) {
  let k = 1;
  while (k < stops.length - 1 && t > stops[k][0]) k++;
  const [t0, c0] = stops[k - 1];
  const [t1, c1] = stops[k];
  const f = Math.min(1, Math.max(0, (t - t0) / Math.max(t1 - t0, 1e-6)));
  for (let j = 0; j < 3; j++) out[j] = c0[j] + (c1[j] - c0[j]) * f;
  return out;
}

// A round contour spacing giving roughly 8-15 lines over the relief.
function contourInterval(reliefM) {
  const target = Math.max(reliefM, 0.5) / 10;
  return [0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 500].find((v) => v >= target) ?? 1000;
}

function buildHeightColors(heightField, lo, hi) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < heightField.length; i++) {
    if (heightField[i] < min) min = heightField[i];
    if (heightField[i] > max) max = heightField[i];
  }
  if (lo != null && hi != null) [min, max] = [lo, hi];
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

// Numbered badge sprite so a waypoint's order in the flight path is legible
// at a glance instead of needing to count markers along the route.
function createWaypointLabel(number) {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(7, 17, 30, 0.92)";
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffb55e";
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.fillStyle = "#ffe4c2";
  ctx.font = "bold 30px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(number), size / 2, size / 2 + 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, depthTest: false }),
  );
  sprite.scale.set(0.26, 0.26, 1);
  sprite.position.y = 0.34;
  sprite.renderOrder = 10;
  return sprite;
}

// A pin (stem + head + ground halo) reads far better than a bare sphere and
// gives waypoints a visible "footprint" on the terrain. A separate, larger,
// invisible hit-sphere makes markers easy to click/reselect without needing
// pixel-precise aim on the small visible head. Everything lives under one
// group so raycasting can walk back up to `userData.waypointIndex`.
function createWaypointMarker(index, point) {
  const group = new THREE.Group();
  group.position.copy(point);
  group.userData.waypointIndex = index;

  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.012, 0.012, 0.16, 8),
    new THREE.MeshBasicMaterial({ color: 0xffb55e }),
  );
  stem.position.y = 0.08;
  group.add(stem);

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.055, 20, 16),
    new THREE.MeshStandardMaterial({
      color: 0xff9f43,
      emissive: 0x552300,
      emissiveIntensity: 0.4,
      roughness: 0.35,
      metalness: 0.1,
    }),
  );
  head.position.y = 0.16;
  group.add(head);

  const halo = new THREE.Mesh(
    new THREE.RingGeometry(0.075, 0.11, 28),
    new THREE.MeshBasicMaterial({
      color: 0xffb55e,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
    }),
  );
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = 0.004;
  group.add(halo);

  const hitArea = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 12, 10),
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  hitArea.position.y = 0.1;
  group.add(hitArea);

  const label = createWaypointLabel(index + 1);
  group.add(label);

  group.userData.head = head;
  group.userData.halo = halo;
  group.userData.label = label;
  return group;
}

function setWaypointNumber(marker, number) {
  const old = marker.userData.label;
  if (old) {
    marker.remove(old);
    old.material.map.dispose();
    old.material.dispose();
  }
  const label = createWaypointLabel(number);
  marker.add(label);
  marker.userData.label = label;
}

function setWaypointSelected(marker, selected) {
  const { head, halo } = marker.userData;
  head.material.color.set(selected ? 0xfff3d6 : 0xff9f43);
  head.material.emissive.set(selected ? 0xffa64d : 0x552300);
  head.material.emissiveIntensity = selected ? 0.9 : 0.4;
  halo.material.opacity = selected ? 1 : 0.65;
  marker.scale.setScalar(selected ? 1.25 : 1);
}

function disposeWaypointMarker(marker) {
  marker.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (child.material.map) child.material.map.dispose();
      child.material.dispose();
    }
  });
}

// Finds the ancestor marker group for a raycast hit against any of a
// marker's child meshes (stem, head, halo, hit-sphere, label).
function resolveWaypointHit(hit) {
  let obj = hit?.object ?? null;
  while (obj && obj.userData.waypointIndex === undefined) obj = obj.parent;
  return obj;
}

// 3D city model: every footprint from /api/estimate (normalized u,v rings +
// height in metres) becomes a real extruded block. Roofs sample the photo at
// their own ground position; walls get a plain lit facade material, since a
// single overhead image has no facade pixels. All roofs and all walls are
// merged into two meshes (two draw calls). Built in metres; the group's y scale
// maps metres to world units, so exaggeration is one scale change.
const FACADE_COLOR = 0xd6cfc0;
const TRUNK_COLOR = 0x5b4636;
// 1x1 black height map: flat ground under the city model.
const FLAT_HEIGHT = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";
// Home view: back the camera off with the scene's height at true scale, so a 130 m downtown
// doesn't open inside its towers (the fixed view was framed for flat tiles at 0.5x).
function homeCamera(camera, controls, sample, exaggeration) {
  let k = 1;
  if (sample?.metricHeights && sample.groundWidthM) {
    const metres = Math.max(sample.ndsmMaxM ?? sample.maxM ?? 0, sample.terrain?.dsm?.relief_m ?? 0);
    k = Math.max(1, 1 + ((metres * 8) / sample.groundWidthM) * exaggeration / 2.5);
  }
  camera.position.set(0, 3.4 * k, 5.6 * k);
  controls.target.set(0, 0, 0);
}

// Critical facilities (viewer/facilities.py, from OpenStreetMap): a badge per kind, floating
// above the facility; the colour also tints the walls of the building it stands in.
const FACILITY_STYLE = {
  hospital: { colour: "#d7263d", glyph: "H", label: "Hospitals" },
  clinic: { colour: "#e4576b", glyph: "cross", label: "Clinics" },
  "fire station": { colour: "#f07f13", glyph: "flame", label: "Fire stations" },
  police: { colour: "#2d6cdf", glyph: "star", label: "Police" },
  shelter: { colour: "#1f9d61", glyph: "house", label: "Shelters" },
  "assembly point": { colour: "#1f9d61", glyph: "house", label: "Assembly points" },
  school: { colour: "#d9a400", glyph: "book", label: "Schools" },
  college: { colour: "#d9a400", glyph: "book", label: "Colleges" },
  university: { colour: "#d9a400", glyph: "book", label: "Universities" },
};
const facilityIconCache = new Map();
function facilityIcon(kind) {
  if (facilityIconCache.has(kind)) return facilityIconCache.get(kind);
  const style = FACILITY_STYLE[kind] ?? FACILITY_STYLE.shelter;
  const size = 128;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  const m = size / 2;
  g.beginPath(); // map-pin badge: a disc with a point underneath
  g.arc(m, m - 8, m - 14, Math.PI * 0.8, Math.PI * 2.2);
  g.lineTo(m, size - 4);
  g.closePath();
  g.fillStyle = style.colour;
  g.fill();
  g.lineWidth = 6;
  g.strokeStyle = "#ffffff";
  g.stroke();
  g.fillStyle = "#ffffff";
  const cy = m - 8;
  g.beginPath();
  if (style.glyph === "H") {
    g.font = "bold 58px sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText("H", m, cy + 3);
  } else if (style.glyph === "cross") {
    g.rect(m - 9, cy - 28, 18, 56);
    g.rect(m - 28, cy - 9, 56, 18);
  } else if (style.glyph === "flame") {
    g.moveTo(m, cy - 30);
    g.bezierCurveTo(m + 30, cy - 4, m + 22, cy + 26, m, cy + 28);
    g.bezierCurveTo(m - 22, cy + 26, m - 30, cy - 2, m - 8, cy - 14);
    g.bezierCurveTo(m - 6, cy - 2, m + 2, cy + 2, m, cy - 30);
  } else if (style.glyph === "star") {
    for (let i = 0; i < 10; i++) {
      const r = i % 2 ? 13 : 31;
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      g.lineTo(m + r * Math.cos(a), cy + r * Math.sin(a));
    }
  } else if (style.glyph === "book") {
    g.moveTo(m, cy - 16); g.lineTo(m - 30, cy - 24); g.lineTo(m - 30, cy + 20); g.lineTo(m, cy + 28);
    g.lineTo(m + 30, cy + 20); g.lineTo(m + 30, cy - 24); g.closePath();
    g.fill();
    g.beginPath();
    g.fillStyle = style.colour;
    g.rect(m - 2, cy - 14, 4, 40);
  } else {
    g.moveTo(m, cy - 30); g.lineTo(m + 30, cy - 2); g.lineTo(m + 20, cy - 2); g.lineTo(m + 20, cy + 26);
    g.lineTo(m - 20, cy + 26); g.lineTo(m - 20, cy - 2); g.lineTo(m - 30, cy - 2); g.closePath();
  }
  g.fill();
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  facilityIconCache.set(kind, texture);
  return texture;
}
// Sprites take their parent's scale: the city group squashes y (metres -> world), so each icon
// is stretched back by the same factor to stay a round badge.
function fitFacilityIcons(city) {
  for (const sp of city.userData.icons ?? []) sp.scale.y = sp.scale.x / city.scale.y;
}

function buildCityGroup(buildings, trees, roofTexture, groundWidthM, maxM, facilities, embankments) {
  // hcol: the Height layer's colours (jet over 0..maxM metres), swapped in by setHeightMode
  const roof = { pos: [], nrm: [], uv: [], hcol: [], ranges: [] }; // ranges: [firstTri, endTri, building]
  const wall = { pos: [], nrm: [], col: [], hcol: [], ranges: [] };
  const jet = [0, 0, 0];
  const top = Math.max(maxM || 0, 1e-3);
  const facade = new THREE.Color(FACADE_COLOR);
  const wallColour = new THREE.Color();
  const toVec = ([u, v]) => new THREE.Vector2(-4 + 8 * u, 4 - 8 * v);
  const uvGenerator = {
    generateTopUV(_geometry, vertices, a, b, c) {
      const uv = (i) =>
        new THREE.Vector2((vertices[i * 3] + 4) / 8, (vertices[i * 3 + 1] + 4) / 8);
      return [uv(a), uv(b), uv(c)];
    },
    generateSideWallUV() {
      return [0, 1, 2, 3].map(() => new THREE.Vector2());
    },
  };
  for (let bi = 0; bi < buildings.length; bi++) {
    const b = buildings[bi];
    let g;
    try {
      const shape = new THREE.Shape(b.rings[0].map(toVec));
      for (const hole of b.rings.slice(1))
        shape.holes.push(new THREE.Path(hole.map(toVec)));
      // Base = lowest ground under the footprint, top = median ground + height (metres).
      g = new THREE.ExtrudeGeometry(shape, {
        depth: Math.max(0.5, (b.t ?? b.h) - (b.b ?? 0)),
        bevelEnabled: false,
        UVGenerator: uvGenerator,
      });
    } catch {
      continue; // a degenerate footprint should not take the whole model down
    }
    g.rotateX(-Math.PI / 2); // extrusion axis -> world up; shape y -> image rows
    g.translate(0, b.b ?? 0, 0);
    // Walls take the building's own roof colour, softened toward a neutral facade.
    const c = b.c ?? [200, 195, 185];
    wallColour
      .setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace)
      .lerp(facade, 0.45);
    if (b.facility) wallColour.set(FACILITY_STYLE[b.facility.kind]?.colour ?? FACADE_COLOR);
    const pos = g.attributes.position.array;
    const nrm = g.attributes.normal.array;
    const uv = g.attributes.uv.array;
    heightToColor(b.h / top, jet);
    const roofStart = roof.pos.length / 9;
    const wallStart = wall.pos.length / 9;
    for (const group of g.groups) {
      const isCap = group.materialIndex === 0;
      const target = isCap ? roof : wall;
      for (let t = group.start; t < group.start + group.count; t += 3) {
        if (isCap && nrm[t * 3 + 1] < 0) continue; // bottom cap faces the ground: never visible
        for (let k = t; k < t + 3; k++) {
          target.pos.push(pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2]);
          target.nrm.push(nrm[k * 3], nrm[k * 3 + 1], nrm[k * 3 + 2]);
          if (isCap) target.uv.push(uv[k * 2], uv[k * 2 + 1]);
          else target.col.push(wallColour.r, wallColour.g, wallColour.b);
          target.hcol.push(jet[0], jet[1], jet[2]);
        }
      }
    }
    g.dispose();
    roof.ranges.push([roofStart, roof.pos.length / 9, bi]);
    wall.ranges.push([wallStart, wall.pos.length / 9, bi]);
  }
  const makeGeometry = (parts) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(parts.pos, 3));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute(parts.nrm, 3));
    if (parts.uv) geometry.setAttribute("uv", new THREE.Float32BufferAttribute(parts.uv, 2));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(parts.col ?? parts.hcol, 3));
    return geometry;
  };
  const photoRoof = new THREE.MeshStandardMaterial({ map: roofTexture, roughness: 0.9, metalness: 0.02 });
  const heightRoof = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0.02 });
  const roofs = new THREE.Mesh(makeGeometry(roof), photoRoof);
  const walls = new THREE.Mesh(
    makeGeometry(wall),
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 }),
  );
  roofs.userData.ranges = roof.ranges;
  walls.userData.ranges = wall.ranges;
  const group = new THREE.Group();
  const meshes = [roofs, walls];
  // Trees: one instanced crown + trunk per detected tree, crown coloured from
  // the photo. x/z are world units (metres * 8 / ground width); y is metres,
  // like the buildings, so the group's y scale applies to both.
  if (trees?.length) {
    const toWorld = 8 / groundWidthM;
    const crownGeo = new THREE.IcosahedronGeometry(1, LOW_DETAIL ? 1 : 2);
    const trunkGeo = new THREE.CylinderGeometry(1, 1, 1, 6);
    trunkGeo.translate(0, 0.5, 0); // base at y = 0
    const crowns = new THREE.InstancedMesh(
      crownGeo,
      new THREE.MeshStandardMaterial({ roughness: 0.9 }),
      trees.length,
    );
    const trunks = new THREE.InstancedMesh(
      trunkGeo,
      new THREE.MeshStandardMaterial({ color: TRUNK_COLOR, roughness: 1 }),
      trees.length,
    );
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const colour = new THREE.Color();
    const hsl = {};
    const crownHeightColours = new Float32Array(trees.length * 3);
    trees.forEach((tree, i) => {
      const x = -4 + 8 * tree.u;
      const z = -4 + 8 * tree.v;
      // crown half-height: the crown fills the top ~2/3 of the tree, trunk the rest
      const rv = Math.min(Math.max(tree.r * 0.85, tree.h * 0.33), tree.h * 0.45);
      const rw = tree.r * toWorld; // crown radius, world units
      const base = tree.b ?? 0; // ground at the trunk, metres
      m4.compose(new THREE.Vector3(x, base + tree.h - rv, z), q, new THREE.Vector3(rw, rv, rw));
      crowns.setMatrixAt(i, m4);
      // Canopy in overhead photos is dark, and lit from above it rendered near-black: lift it,
      // keeping the photo's hue (autumn and dry-season trees keep their colour).
      colour.setRGB(tree.c[0] / 255, tree.c[1] / 255, tree.c[2] / 255, THREE.SRGBColorSpace);
      colour.getHSL(hsl);
      colour.setHSL(hsl.h, Math.min(1, hsl.s * 1.15 + 0.05), Math.max(hsl.l * 1.25, 0.22));
      crowns.setColorAt(i, colour);
      heightToColor(tree.h / top, jet);
      crownHeightColours.set(jet, i * 3);
      const tw = Math.max(0.15, tree.r * 0.1) * toWorld;
      m4.compose(new THREE.Vector3(x, base, z), q, new THREE.Vector3(tw, tree.h - rv, tw));
      trunks.setMatrixAt(i, m4);
    });
    meshes.push(crowns, trunks);
    crowns.userData.colours = [Float32Array.from(crowns.instanceColor.array), crownHeightColours];
  }
  // Flood defences (OpenStreetMap levees, embankments, dams): a bright band along each crest.
  for (const e of embankments ?? []) {
    const pts = e.pts.map(([u, v]) => new THREE.Vector3(-4 + 8 * u, e.crest + 1.5, -4 + 8 * v));
    const band = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, false, "centripetal"), Math.max(8, pts.length * 6), 0.01, 6),
      new THREE.MeshBasicMaterial({ color: 0x2bd4ff }),
    );
    band.userData.noPick = true;
    meshes.push(band);
  }
  const icons = [];
  if (facilities?.length) {
    const toWorld = 8 / groundWidthM;
    const poleM = 14; // the badge floats this far above the roof or ground it marks
    const poles = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(1, 1, 1, 6).translate(0, 0.5, 0),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
      facilities.length,
    );
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    facilities.forEach((f, i) => {
      const x = -4 + 8 * f.u;
      const z = -4 + 8 * f.v;
      m4.compose(new THREE.Vector3(x, f.z, z), q, new THREE.Vector3(0.012, poleM, 0.012));
      poles.setMatrixAt(i, m4);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: facilityIcon(f.kind), transparent: true }));
      sp.position.set(x, f.z + poleM, z);
      sp.center.set(0.5, 0.03); // the badge's point sits on the pole's top
      const w = Math.max(0.12, Math.min(0.3, 40 * toWorld)); // ~40 m wide, kept legible at any scale
      sp.scale.set(w, w, 1);
      sp.userData.facility = f;
      icons.push(sp);
    });
    poles.userData.noPick = true;
    meshes.push(poles);
  }
  for (const m of meshes) {
    m.castShadow = true;
    m.receiveShadow = true;
    group.add(m);
  }
  if (icons.length) {
    const iconGroup = new THREE.Group();
    iconGroup.userData.noPick = true;
    icons.forEach((sp) => iconGroup.add(sp));
    group.add(iconGroup);
  }
  group.userData.icons = icons;
  // Height layer: roofs, walls and crowns take the jet colour of their height; photo otherwise.
  group.userData.setHeightMode = (on) => {
    roofs.material = on ? heightRoof : photoRoof;
    walls.geometry.attributes.color.array.set(on ? wall.hcol : wall.col);
    walls.geometry.attributes.color.needsUpdate = true;
    const crowns = meshes[2];
    if (crowns?.userData.colours) {
      crowns.instanceColor.array.set(crowns.userData.colours[on ? 1 : 0]);
      crowns.instanceColor.needsUpdate = true;
    }
  };
  return group;
}

// Outline of one extruded building (same placement as buildCityGroup), for highlighting a pick.
function buildingOutline(b) {
  const toVec = ([u, v]) => new THREE.Vector2(-4 + 8 * u, 4 - 8 * v);
  const shape = new THREE.Shape(b.rings[0].map(toVec));
  for (const hole of b.rings.slice(1)) shape.holes.push(new THREE.Path(hole.map(toVec)));
  const g = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.5, (b.t ?? b.h) - (b.b ?? 0)),
    bevelEnabled: false,
  });
  g.rotateX(-Math.PI / 2);
  g.translate(0, b.b ?? 0, 0);
  const edges = new THREE.EdgesGeometry(g, 20);
  g.dispose();
  const lines = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color: 0x7ff0d8, depthTest: false, transparent: true }),
  );
  lines.renderOrder = 10;
  return lines;
}

// Footprint area in m² (outer ring minus holes), u/v scaled to the ground extent.
function footprintAreaM2(b, widthM, heightM) {
  const ring = (r) => {
    let s = 0;
    for (let i = 0; i < r.length; i++) {
      const [u1, v1] = r[i];
      const [u2, v2] = r[(i + 1) % r.length];
      s += u1 * widthM * (v2 * heightM) - u2 * widthM * (v1 * heightM);
    }
    return Math.abs(s) / 2;
  };
  return ring(b.rings[0]) - b.rings.slice(1).reduce((a, r) => a + ring(r), 0);
}

// Real height profile along the scene's centre row: SVG paths + stats.
// Metres for uploads, % of scene max for non-linear preview tiles.
function profileStats(profile) {
  const v = profile?.values;
  if (!v?.length) return null;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const x of v) {
    if (x < min) min = x;
    if (x > max) max = x;
    sum += x;
  }
  const range = Math.max(max - min, 1e-6);
  const pts = v.map(
    (x, i) =>
      `${((i / (v.length - 1)) * 300).toFixed(1)},${(65 - ((x - min) / range) * 60).toFixed(1)}`,
  );
  const line = `M${pts.join(" L")}`;
  const fmt = profile.metric
    ? (x) => `${x.toFixed(1)} m`
    : (x) => `${(x * 100).toFixed(0)}%`;
  return { line, area: `${line} V70 H0Z`, min, max, avg: sum / v.length, fmt };
}

// Validation beyond the four headline numbers: what was compared, the bias, a scatter
// of model vs reference (1500 random pixels) and where the error sits by land cover.
function ValidationDetail({ v }) {
  const pts = v.scatter;
  let lo = Infinity;
  let hi = -Infinity;
  for (const [r, p] of pts) {
    lo = Math.min(lo, r, p);
    hi = Math.max(hi, r, p);
  }
  const span = Math.max(hi - lo, 1e-3);
  const X = (x) => (((x - lo) / span) * 100).toFixed(1);
  const Y = (y) => (100 - ((y - lo) / span) * 100).toFixed(1);
  return (
    <div className="validation-detail">
      <div className="validation-note">
        Compared with the {v.reference_kind === "dsm" ? "absolute DSM" : "height above ground (nDSM)"}
        {v.reprojected ? " · reference reprojected by coordinates" : " · reference resampled to the image"}
        {" "}· bias {v.bias >= 0 ? "+" : ""}
        {v.bias?.toFixed(2)} m · {(v.coverage * 100).toFixed(0)}% of pixels
      </div>
      <svg className="scatter" viewBox="-14 -4 118 118" aria-label="model vs reference scatter">
        <rect x="0" y="0" width="100" height="100" className="scatter-frame" />
        <line x1="0" y1="100" x2="100" y2="0" className="scatter-diag" />
        {pts.map(([r, p], i) => (
          <circle key={i} cx={X(r)} cy={Y(p)} r="0.9" />
        ))}
        <text x="50" y="112" textAnchor="middle">reference (m)</text>
        <text x="-9" y="50" textAnchor="middle" transform="rotate(-90 -9 50)">model (m)</text>
        <text x="0" y="108">{lo.toFixed(0)}</text>
        <text x="100" y="108" textAnchor="end">{hi.toFixed(0)}</text>
      </svg>
      {v.by_class?.length > 0 && (
        <table className="class-errors">
          <thead>
            <tr>
              <th>Class</th>
              <th>Share</th>
              <th>RMSE</th>
              <th>Bias</th>
            </tr>
          </thead>
          <tbody>
            {v.by_class.map((c) => (
              <tr key={c.class}>
                <td>{c.class}</td>
                <td>{(c.share * 100).toFixed(0)}%</td>
                <td>{c.rmse?.toFixed(2)} m</td>
                <td>
                  {c.bias >= 0 ? "+" : ""}
                  {c.bias?.toFixed(2)} m
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function TerrainCanvas({
  sample,
  exaggeration,
  layer,
  resetToken,
  onMeasure,
  onProfile,
  waypointMode,
  pathCommand,
  onWaypointChange,
  onWaypointSelect,
  onPathEnd,
  autoRotate,
  onBuildingPick,
  exportRef,
  contours = false,
  contourStep = 1,
  profilePoints = [],
  walkMode = false,
  measureMode = false,
}) {
  const ref = useRef(null);
  const state = useRef({});
  // Read inside the scene effect and render loop, which outlive a single render.
  const contourUniforms = useRef({ uContour: { value: 0 }, uInterval: { value: 1 } }).current;
  const walkRef = useRef(walkMode);
  const measureModeRef = useRef(measureMode);
  measureModeRef.current = measureMode;
  const profilePointsRef = useRef(profilePoints);
  const waypointModeRef = useRef(waypointMode);
  const pathEndRef = useRef(onPathEnd);
  const waypointSelectRef = useRef(onWaypointSelect);
  const layerRef = useRef(layer);
  const keys = useRef({});
  useEffect(() => {
    waypointModeRef.current = waypointMode;
  }, [waypointMode]);
  useEffect(() => {
    pathEndRef.current = onPathEnd;
  }, [onPathEnd]);
  useEffect(() => {
    waypointSelectRef.current = onWaypointSelect;
  }, [onWaypointSelect]);
  useEffect(() => {
    layerRef.current = layer;
  }, [layer]);
  useEffect(() => {
    const host = ref.current,
      scene = new THREE.Scene();
    scene.background = new THREE.Color("#07111e");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 3.4, 5.6);
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(LOW_DETAIL ? 1 : Math.min(window.devicePixelRatio || 1, 1.25));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 1.4;
    const transform = new TransformControls(camera, renderer.domElement);
    transform.setMode("translate");
    transform.setSize(0.72);
    const transformHelper = transform.getHelper();
    scene.add(transformHelper);
    transform.addEventListener("dragging-changed", (event) => {
      controls.enabled = !event.value;
    });
    scene.add(new THREE.HemisphereLight(0x9bc9d0, 0x172333, 1.7));
    const sun = new THREE.DirectionalLight(0xf7e4ba, 2.5);
    sun.position.set(-3, 6, 4);
    // Sun shadows: buildings and canopy shade the ground, which is what makes
    // relief readable from above.
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    sun.castShadow = true;
    sun.shadow.mapSize.set(LOW_DETAIL ? 1024 : 2048, LOW_DETAIL ? 1024 : 2048);
    Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 0.5, far: 20 });
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    scene.add(sun);
    const grid = new THREE.GridHelper(9, 18, 0x315263, 0x1b3040);
    grid.position.y = -0.55;
    scene.add(grid);
    const geo = new THREE.PlaneGeometry(8, 8, MESH_SEGMENTS_X, MESH_SEGMENTS_Y);
    geo.rotateX(-Math.PI / 2);
    geo.setAttribute(
      "color",
      new THREE.BufferAttribute(
        new Float32Array(geo.attributes.position.count * 3).fill(1),
        3,
      ),
    );
    // Metres of the shown surface per vertex, for contour lines (uploads only).
    geo.setAttribute(
      "elevM",
      new THREE.BufferAttribute(new Float32Array(geo.attributes.position.count), 1),
    );
    const tex = new THREE.TextureLoader();
    const updateGeometry = () => {
      const current = state.current;
      if (!current.rawHeightField) return;
      // Class-based rescaling is a display heuristic for the lossy catalog
      // previews; uploads keep the model's metric heights unaltered.
      const heightField = sample.metricHeights
        ? current.rawHeightField
        : applyClassHeightScale(
            current.rawHeightField,
            current.classField,
            HEIGHT_SAMPLE_WIDTH,
            HEIGHT_SAMPLE_HEIGHT,
          );
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++)
        pos.setY(
          i,
          (heightField[i] - TERRAIN_BASELINE) *
            HEIGHT_WORLD_SCALE *
            exaggeration,
        );
      current.heightField = heightField;
      // Uploads: colour = height above ground (nDSM metres, 0..max) in every mode, matching
      // the legend and the city model; catalog previews keep the relative stretch.
      // In the city model the ground is bare earth (0 m above itself): buildings and trees
      // carry the colour, not a halo of their heights painted on the ground.
      const ndsmMax = sample.ndsmMaxM ?? sample.maxM;
      current.heightColors = sample.metricHeights && sample.buildings
        ? buildHeightColors(new Float32Array(heightField.length), 0, ndsmMax)
        : sample.metricHeights && current.probeField
          ? buildHeightColors(current.probeField, 0, ndsmMax)
          : buildHeightColors(heightField);
      current.shadeColors = buildWallShade(
        heightField,
        HEIGHT_SAMPLE_WIDTH,
        HEIGHT_SAMPLE_HEIGHT,
      );
      if (sample.metricHeights) {
        // Slope and contours of the surface the mesh shows (bare ground in the city model,
        // the full DSM in Exact mode), in real metres.
        const gW = sample.groundWidthM ?? 1024 * GAMUS_GSD_M;
        const metres = displayMetres(heightField, gW);
        current.slopeColors = buildSlopeColors(
          metres,
          HEIGHT_SAMPLE_WIDTH,
          HEIGHT_SAMPLE_HEIGHT,
          gW,
          sample.groundHeightM ?? gW,
        );
        const elev = geo.attributes.elevM.array;
        const offset = sample.meshOffsetM ?? 0;
        for (let i = 0; i < elev.length; i++) elev[i] = metres[i] + offset;
        geo.attributes.elevM.needsUpdate = true;
      }
      current.exaggeration = exaggeration;
      pos.needsUpdate = true;
      geo.computeVertexNormals();
      geo.attributes.color.array.set(
        layerRef.current === "depth"
          ? current.heightColors
          : layerRef.current === "slope" && current.slopeColors
            ? current.slopeColors
            : current.shadeColors,
      );
      geo.attributes.color.needsUpdate = true;
      current.renderDirty = true;
    };
    // Probe copy of the heights, untouched by any display smoothing:
    // metres for uploads (linear encoding), 0..1 relative for previews.
    const setProbe = (px) => {
      const probe = new Float32Array(HEIGHT_SAMPLE_WIDTH * HEIGHT_SAMPLE_HEIGHT);
      const probeMax = sample.exactHeight ? sample.exactMaxM : sample.maxM;
      const scale = sample.metricHeights ? probeMax / 255 : 1 / 255;
      for (let i = 0; i < probe.length; i++) probe[i] = px[i * 4] * scale;
      setProbeField(probe);
    };
    const setProbeField = (probe) => {
      state.current.probeField = probe;
      state.current.probeMetric = Boolean(sample.metricHeights);
      state.current.groundWidthM = sample.groundWidthM ?? 1024 * GAMUS_GSD_M;
      state.current.groundHeightM = sample.groundHeightM ?? state.current.groundWidthM;
      const mid = Math.floor(HEIGHT_SAMPLE_HEIGHT / 2) * HEIGHT_SAMPLE_WIDTH;
      state.current.centreProfile = {
        values: Array.from(probe.subarray(mid, mid + HEIGHT_SAMPLE_WIDTH)),
        metric: Boolean(sample.metricHeights),
      };
      if (profilePointsRef.current.length < 2) onProfile?.(state.current.centreProfile);
    };
    const loadPixels = (url, done) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = HEIGHT_SAMPLE_WIDTH;
        c.height = HEIGHT_SAMPLE_HEIGHT;
        const context = c.getContext("2d");
        context.drawImage(img, 0, 0, HEIGHT_SAMPLE_WIDTH, HEIGHT_SAMPLE_HEIGHT);
        done(context.getImageData(0, 0, HEIGHT_SAMPLE_WIDTH, HEIGHT_SAMPLE_HEIGHT).data);
      };
      img.src = url;
    };
    const height = tex.load(sample.height, (loaded) => {
      const c = document.createElement("canvas");
      c.width = HEIGHT_SAMPLE_WIDTH;
      c.height = HEIGHT_SAMPLE_HEIGHT;
      const x = c.getContext("2d");
      x.drawImage(
        loaded.image,
        0,
        0,
        HEIGHT_SAMPLE_WIDTH,
        HEIGHT_SAMPLE_HEIGHT,
      );
      const px = x.getImageData(
        0,
        0,
        HEIGHT_SAMPLE_WIDTH,
        HEIGHT_SAMPLE_HEIGHT,
      ).data;
      // Uploads carry 16-bit grids (meshGrid/probeGrid/elevGrid); the 8-bit PNG is the fallback.
      const heightField = sample.metricHeights
        ? metricDisplayField(
            sample.meshGrid ?? pxToMetres(px, sample.maxM),
            HEIGHT_SAMPLE_WIDTH,
            HEIGHT_SAMPLE_HEIGHT,
            sample.groundWidthM,
          )
        : buildHeightField(px, HEIGHT_SAMPLE_WIDTH, HEIGHT_SAMPLE_HEIGHT);
      state.current.rawHeightField = heightField;
      // In city-model mode the mesh shows canopy only; the probe and profile
      // still read the exact nDSM, which is what the GeoTIFF export contains.
      if (sample.probeGrid) setProbeField(sample.probeGrid);
      else if (sample.exactHeight) loadPixels(sample.exactHeight, setProbe);
      else setProbe(px);
      // Absolute elevation (exported DSM, metres above the geoid) for the probe.
      state.current.elevField = sample.elevGrid ?? null;
      if (!sample.elevGrid && sample.elevation)
        loadPixels(sample.elevation.png, (dpx) => {
          const f = new Float32Array(HEIGHT_SAMPLE_WIDTH * HEIGHT_SAMPLE_HEIGHT);
          const { min_m: lo, relief_m: span } = sample.elevation;
          for (let i = 0; i < f.length; i++) f[i] = lo + (dpx[i * 4] / 255) * span;
          state.current.elevField = f;
        });
      updateGeometry();
    });
    const classTexture = tex.load(sample.classes, (loaded) => {
      const c = document.createElement("canvas");
      c.width = HEIGHT_SAMPLE_WIDTH;
      c.height = HEIGHT_SAMPLE_HEIGHT;
      const context = c.getContext("2d");
      context.drawImage(
        loaded.image,
        0,
        0,
        HEIGHT_SAMPLE_WIDTH,
        HEIGHT_SAMPLE_HEIGHT,
      );
      const px = context.getImageData(
        0,
        0,
        HEIGHT_SAMPLE_WIDTH,
        HEIGHT_SAMPLE_HEIGHT,
      ).data;
      state.current.classField = buildClassField(
        px,
        HEIGHT_SAMPLE_WIDTH,
        HEIGHT_SAMPLE_HEIGHT,
      );
      updateGeometry();
    });
    height.colorSpace = THREE.SRGBColorSpace; // display only: the field is read from a canvas
    const material = new THREE.MeshStandardMaterial({
      map: height,
      vertexColors: true,
      roughness: 0.86,
      metalness: 0.02,
      side: THREE.FrontSide,
      wireframe: false,
    });
    // Contour lines drawn per pixel from the elevM attribute: one line every uInterval
    // metres, width constant on screen (fwidth), toggled by uContour.
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, contourUniforms);
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nattribute float elevM;\nvarying float vElevM;")
        .replace("#include <begin_vertex>", "#include <begin_vertex>\nvElevM = elevM;");
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          "#include <common>\nuniform float uContour;\nuniform float uInterval;\nvarying float vElevM;",
        )
        .replace(
          "#include <dithering_fragment>",
          `#include <dithering_fragment>
          if (uContour > 0.5) {
            float c = vElevM / uInterval;
            float line = 1.0 - smoothstep(0.0, 1.2 * fwidth(c), abs(fract(c - 0.5) - 0.5));
            gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.03, 0.05, 0.07), 0.75 * line);
          }`,
        );
    };
    const mesh = new THREE.Mesh(geo, material);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    scene.add(mesh);
    let city = null;
    let roofTexture = null;
    if (sample.buildings) {
      roofTexture = tex.load(sample.rgb);
      roofTexture.colorSpace = THREE.SRGBColorSpace;
      city = buildCityGroup(
        sample.buildings,
        sample.trees,
        roofTexture,
        sample.groundWidthM ?? 1024 * GAMUS_GSD_M,
        sample.ndsmMaxM ?? sample.maxM,
        sample.facilities,
        sample.embankments,
      );
      city.scale.y =
        (exaggeration * 8) / (sample.groundWidthM ?? 1024 * GAMUS_GSD_M);
      fitFacilityIcons(city);
      scene.add(city);
    }
    const pickable = city ? [mesh, ...city.children.filter((o) => !o.userData.noPick)] : [mesh];
    // Click a building: outline it and report its facts (uses the raycaster set by the click).
    let outline = null;
    const clearOutline = () => {
      if (!outline) return;
      city.remove(outline);
      outline.geometry.dispose();
      outline.material.dispose();
      outline = null;
    };
    const pickBuilding = () => {
      if (!city) return;
      const hit = raycaster.intersectObjects(city.children)[0];
      const ranges = hit?.object.userData.ranges;
      const found = ranges?.find(([a, z]) => hit.faceIndex >= a && hit.faceIndex < z);
      clearOutline();
      if (!found) {
        onBuildingPick?.(null);
        state.current.renderDirty = true;
        return;
      }
      const b = sample.buildings[found[2]];
      outline = buildingOutline(b);
      city.add(outline);
      const widthM = sample.groundWidthM ?? 1024 * GAMUS_GSD_M;
      onBuildingPick?.({
        bridge: b.kind === "bridge",
        facility: b.facility ?? null,
        height: b.h,
        floors: Math.max(1, Math.round(b.h / 3.2)),
        areaM2: footprintAreaM2(b, widthM, sample.groundHeightM ?? widthM),
        volumeM3: footprintAreaM2(b, widthM, sample.groundHeightM ?? widthM) * b.h,
        roofElevation: sample.groundMinM != null ? sample.groundMinM + (b.t ?? b.h) : null,
      });
      state.current.renderDirty = true;
    };
    // Export the current 3D model (terrain + city) as binary glTF in metres.
    if (exportRef)
      exportRef.current = (filename) =>
        new Promise((resolve, reject) => {
          const gw = sample.groundWidthM ?? 1024 * GAMUS_GSD_M;
          const exag = state.current.exaggeration || 1;
          const wrap = new THREE.Group();
          wrap.scale.set(gw / 8, gw / (8 * exag), gw / 8); // world units -> metres
          wrap.add(mesh.clone());
          if (city) {
            const c = city.clone();
            c.children.filter((o) => o.isLineSegments).forEach((o) => c.remove(o));
            wrap.add(c);
          }
          new GLTFExporter().parse(
            wrap,
            (glb) => {
              const url = URL.createObjectURL(new Blob([glb], { type: "model/gltf-binary" }));
              const a = document.createElement("a");
              a.href = url;
              a.download = filename;
              a.click();
              setTimeout(() => URL.revokeObjectURL(url), 2000);
              resolve();
            },
            reject,
            { binary: true, maxTextureSize: 4096 },
          );
        });
    const waypointGroup = new THREE.Group();
    scene.add(waypointGroup);
    state.current = {
      scene,
      camera,
      renderer,
      controls,
      mesh,
      city,
      tex,
      height,
      classTexture,
      waypointGroup,
      transform,
      waypoints: [],
      waypointLine: null,
      selectedMarker: null,
      pathPlaying: false,
      routeYaw: 0,
      routePitch: 0,
      renderDirty: true,
      routePosition: new THREE.Vector3(),
      routeLookAt: new THREE.Vector3(),
      routeDirection: new THREE.Vector3(),
      routePitchAxis: new THREE.Vector3(),
    };
    // Walk mode: ground height under a world (x, z), eye height, and building collisions.
    const cellOf = (x, z) => {
      const W = HEIGHT_SAMPLE_WIDTH;
      const H = HEIGHT_SAMPLE_HEIGHT;
      const col = Math.min(W - 1, Math.max(0, Math.round(((x + 4) / 8) * (W - 1))));
      const row = Math.min(H - 1, Math.max(0, Math.round(((z + 4) / 8) * (H - 1))));
      return row * W + col;
    };
    const worldPerMetre = () =>
      (8 / (state.current.groundWidthM ?? 1024 * GAMUS_GSD_M)) * (state.current.exaggeration ?? 1);
    state.current.walkGround = (x, z) => {
      const i = cellOf(x, z);
      // Exact DSM: the mesh holds buildings and canopy; stand on the ground beneath them.
      const onObjects =
        sample.metricHeights && !sample.buildings && state.current.probeField
          ? state.current.probeField[i] * worldPerMetre()
          : 0;
      return geo.attributes.position.getY(i) - onObjects;
    };
    state.current.walkEye = () => 1.7 * worldPerMetre();
    state.current.walkBlocked = (x, z) => {
      if (Math.abs(x) > 3.98 || Math.abs(z) > 3.98) return true;
      const i = cellOf(x, z);
      const cf = state.current.classField;
      return Boolean(
        cf &&
          cf[i] === BUILDING_CLASS_INDEX &&
          (!state.current.probeMetric || state.current.probeField?.[i] > 2),
      );
    };
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const groundPoint = new THREE.Vector3();
    const rebuildWaypointLine = () => {
      if (state.current.waypointLine) {
        scene.remove(state.current.waypointLine);
        state.current.waypointLine.geometry.dispose();
        state.current.waypointLine.material.dispose();
        state.current.waypointLine = null;
      }
      const waypoints = state.current.waypoints;
      if (waypoints.length >= 2) {
        // Preview the actual CatmullRom flight curve (same tension used by
        // "Play route") instead of straight segments, so the drawn path
        // matches what flying it will look like.
        const liftedPoints = waypoints.map((p) =>
          p.clone().add(new THREE.Vector3(0, 0.2, 0)),
        );
        const curve = new THREE.CatmullRomCurve3(
          liftedPoints,
          false,
          "catmullrom",
          0.45,
        );
        const curvePoints = curve.getPoints(
          Math.max(16, waypoints.length * 20),
        );
        state.current.waypointLine = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(curvePoints),
          new THREE.LineBasicMaterial({
            color: 0xffb55e,
            transparent: true,
            opacity: 0.85,
          }),
        );
        scene.add(state.current.waypointLine);
      }
      state.current.renderDirty = true;
    };
    state.current.rebuildWaypointLine = rebuildWaypointLine;
    const selectMarker = (marker) => {
      if (
        state.current.selectedMarker &&
        state.current.selectedMarker !== marker
      ) {
        setWaypointSelected(state.current.selectedMarker, false);
      }
      state.current.selectedMarker = marker;
      setWaypointSelected(marker, true);
      transform.attach(marker);
      state.current.renderDirty = true;
      waypointSelectRef.current?.(marker.userData.waypointIndex);
    };
    const deselectMarker = () => {
      if (state.current.selectedMarker)
        setWaypointSelected(state.current.selectedMarker, false);
      state.current.selectedMarker = null;
      transform.detach();
      state.current.renderDirty = true;
      waypointSelectRef.current?.(null);
    };
    const deleteWaypoint = (marker) => {
      const idx = marker.userData.waypointIndex;
      waypointGroup.remove(marker);
      disposeWaypointMarker(marker);
      state.current.waypoints.splice(idx, 1);
      waypointGroup.children.forEach((m) => {
        if (m.userData.waypointIndex > idx) {
          m.userData.waypointIndex -= 1;
          setWaypointNumber(m, m.userData.waypointIndex + 1);
        }
      });
      if (state.current.selectedMarker === marker) deselectMarker();
      rebuildWaypointLine();
      state.current.renderDirty = true;
      onWaypointChange(state.current.waypoints.length);
    };
    state.current.deleteWaypoint = deleteWaypoint;
    // Height + slope under the cursor, read from the probe field at the exact
    // mesh intersection (uv), so it matches what is drawn at that spot.
    state.current.measureAt = (event) => {
      const s = state.current;
      if (!s.probeField) return null;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(pickable)[0];
      if (!hit) return null;
      const W = HEIGHT_SAMPLE_WIDTH;
      const H = HEIGHT_SAMPLE_HEIGHT;
      // World x, z in [-4, 4] map to image column, row (row 0 at z = -4).
      const col = Math.min(W - 2, Math.max(1, Math.round(((hit.point.x + 4) / 8) * (W - 1))));
      const row = Math.min(H - 2, Math.max(1, Math.round(((hit.point.z + 4) / 8) * (H - 1))));
      const f = s.probeField;
      const i = row * W + col;
      // Slope of the surface itself: the absolute DSM when georeferenced (terrain + objects),
      // else the nDSM over flat ground. Sample spacing in metres along each axis.
      const z = s.elevField ?? f;
      const dzdx = (z[i + 1] - z[i - 1]) / ((2 * s.groundWidthM) / (W - 1));
      const dzdy = (z[i + W] - z[i - W]) / ((2 * s.groundHeightM) / (H - 1));
      return {
        height: f[i],
        elevation: s.elevField ? s.elevField[i] : null,
        slopeDeg: s.probeMetric
          ? (Math.atan(Math.hypot(dzdx, dzdy)) * 180) / Math.PI
          : null,
        metric: s.probeMetric,
        row,
        col,
      };
    };
    transform.addEventListener("objectChange", () => {
      const marker = transform.object;
      if (!marker || marker.userData.waypointIndex === undefined) return;
      state.current.waypoints[marker.userData.waypointIndex].copy(
        marker.position,
      );
      rebuildWaypointLine();
    });
    const addWaypoint = (event) => {
      if (state.current.pathPlaying || transform.dragging || transform.axis)
        return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const markerHit = resolveWaypointHit(
        raycaster.intersectObjects(waypointGroup.children, true)[0],
      );
      if (markerHit) {
        selectMarker(markerHit);
        return;
      }
      if (!waypointModeRef.current) {
        deselectMarker();
        if (!measureModeRef.current) pickBuilding(); // Measure double-clicks probe, not pick
        return;
      }
      const roofHit = city ? raycaster.intersectObjects(city.children)[0] : null;
      if (roofHit) {
        // Clicked a building: put the waypoint on its roof.
        const onRoof = roofHit.point.clone();
        state.current.waypoints.push(onRoof);
        const roofMarker = createWaypointMarker(state.current.waypoints.length - 1, onRoof);
        waypointGroup.add(roofMarker);
        rebuildWaypointLine();
        selectMarker(roofMarker);
        onWaypointChange(state.current.waypoints.length);
        return;
      }
      const point = raycaster.ray.intersectPlane(groundPlane, groundPoint);
      if (!point || Math.abs(point.x) > 4 || Math.abs(point.z) > 4) return;
      if (state.current.heightField) {
        const column = Math.min(
          HEIGHT_SAMPLE_WIDTH - 1,
          Math.max(
            0,
            Math.round(((point.x + 4) / 8) * (HEIGHT_SAMPLE_WIDTH - 1)),
          ),
        );
        const row = Math.min(
          HEIGHT_SAMPLE_HEIGHT - 1,
          Math.max(
            0,
            // PlaneGeometry row 0 (image top) sits at z = -4 after rotateX(-PI/2)
            Math.round(((point.z + 4) / 8) * (HEIGHT_SAMPLE_HEIGHT - 1)),
          ),
        );
        const heightValue =
          state.current.heightField[row * HEIGHT_SAMPLE_WIDTH + column];
        point.y =
          (heightValue - TERRAIN_BASELINE) *
          HEIGHT_WORLD_SCALE *
          state.current.exaggeration;
      }
      // intersectPlane writes into and returns the shared `groundPoint`
      // scratch vector, so every waypoint must store its own clone —
      // otherwise all points alias one object and silently collapse to
      // wherever was clicked most recently.
      state.current.waypoints.push(point.clone());
      const marker = createWaypointMarker(
        state.current.waypoints.length - 1,
        point,
      );
      waypointGroup.add(marker);
      rebuildWaypointLine();
      selectMarker(marker);
      onWaypointChange(state.current.waypoints.length);
    };
    // A click picks a building only if the pointer didn't move since it went down: releasing
    // an orbit/look drag over a building must not pop its card up.
    const downAt = { x: 0, y: 0 };
    const markDown = (e) => Object.assign(downAt, { x: e.clientX, y: e.clientY });
    const onClick = (e) => {
      if (Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 5) return;
      addWaypoint(e);
    };
    renderer.domElement.addEventListener("pointerdown", markDown);
    renderer.domElement.addEventListener("click", onClick);
    const onWaypointHover = (event) => {
      if (state.current.pathPlaying || transform.dragging) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hovering = !!resolveWaypointHit(
        raycaster.intersectObjects(waypointGroup.children, true)[0],
      );
      renderer.domElement.style.cursor = hovering
        ? "grab"
        : waypointModeRef.current
          ? "crosshair"
          : "";
    };
    const onWaypointHoverEnd = () => {
      if (!state.current.pathPlaying) renderer.domElement.style.cursor = "";
    };
    renderer.domElement.addEventListener("pointermove", onWaypointHover);
    renderer.domElement.addEventListener("pointerleave", onWaypointHoverEnd);
    let routeLookDrag = null;
    const startRouteLook = (event) => {
      if (!state.current.pathPlaying || event.button !== 0) return;
      routeLookDrag = { x: event.clientX, y: event.clientY };
      renderer.domElement.style.cursor = "grabbing";
    };
    const moveRouteLook = (event) => {
      if (!routeLookDrag || !state.current.pathPlaying) return;
      const dx = event.clientX - routeLookDrag.x;
      const dy = event.clientY - routeLookDrag.y;
      state.current.routeYaw -= dx * 0.004;
      state.current.routePitch = THREE.MathUtils.clamp(
        state.current.routePitch - dy * 0.003,
        -0.9,
        0.9,
      );
      routeLookDrag = { x: event.clientX, y: event.clientY };
    };
    const endRouteLook = () => {
      routeLookDrag = null;
      renderer.domElement.style.cursor = "";
    };
    renderer.domElement.addEventListener("pointerdown", startRouteLook);
    window.addEventListener("pointermove", moveRouteLook);
    window.addEventListener("pointerup", endRouteLook);
    const onResize = () => {
      const w = host.clientWidth,
        h = host.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    onResize();
    window.addEventListener("resize", onResize);
    const keyDown = (e) => {
      keys.current.shift = e.shiftKey;
      if (["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
      if (
        (e.key === "Delete" || e.key === "Backspace") &&
        state.current.selectedMarker
      ) {
        e.preventDefault();
        state.current.deleteWaypoint(state.current.selectedMarker);
        return;
      }
      if ("wasdqe".includes(e.key.toLowerCase())) {
        keys.current[e.key.toLowerCase()] = true;
        e.preventDefault();
        if (state.current.pathPlaying) {
          state.current.pathPlaying = false;
          pathEndRef.current?.();
        }
        controls.enabled = false;
      }
    };
    const keyUp = (e) => {
      keys.current.shift = e.shiftKey;
      if (keys.current[e.key.toLowerCase()]) {
        keys.current[e.key.toLowerCase()] = false;
        if (!Object.values(keys.current).some(Boolean)) controls.enabled = true;
      }
    };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    const clock = new THREE.Clock();
    const velocity = new THREE.Vector3();
    const move = new THREE.Vector3();
    const forward = new THREE.Vector3();
    const right = new THREE.Vector3();
    const desired = new THREE.Vector3();
    const step = new THREE.Vector3();
    const worldUp = new THREE.Vector3(0, 1, 0);
    let renderedOnce = false;
    let raf;
    const loop = () => {
      const dt = Math.min(clock.getDelta(), 0.05);
      move.set(0, 0, 0);
      camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();
      right.crossVectors(forward, camera.up).normalize();
      if (keys.current.w) move.add(forward);
      if (keys.current.s) move.sub(forward);
      if (keys.current.d) move.add(right);
      if (keys.current.a) move.sub(right);
      if (keys.current.e) move.y += 1;
      if (keys.current.q) move.y -= 1;
      const walking = walkRef.current && state.current.walkGround;
      // Walk: 6 m/s on the ground (Shift: 24 m/s), no altitude keys. Fly: 1.35 units/s.
      const speed = walking
        ? (6 * (keys.current.shift ? 4 : 1) * 8) / (state.current.groundWidthM ?? 1024 * GAMUS_GSD_M)
        : 1.35;
      if (walking) move.y = 0;
      if (move.lengthSq()) desired.copy(move).normalize().multiplyScalar(speed);
      else desired.set(0, 0, 0);
      velocity.lerp(desired, 1 - Math.exp(-6 * dt));
      let hasVelocity = velocity.lengthSq() > 0.000001;
      if (hasVelocity) {
        step.copy(velocity).multiplyScalar(dt);
        if (walking) {
          // Slide along walls: try the full step, then each axis alone.
          const { x, z } = camera.position;
          const blocked = state.current.walkBlocked;
          if (blocked(x + step.x, z + step.z)) {
            if (!blocked(x + step.x, z)) step.z = 0;
            else if (!blocked(x, z + step.z)) step.x = 0;
            else step.set(0, 0, 0);
          }
        }
        camera.position.add(step);
        controls.target.add(step);
      }
      if (walking) {
        const dy =
          state.current.walkGround(camera.position.x, camera.position.z) +
          state.current.walkEye() -
          camera.position.y;
        if (Math.abs(dy) > 1e-6) {
          camera.position.y += dy;
          controls.target.y += dy;
          hasVelocity = true;
        }
      }
      let routeActive = false;
      if (state.current.pathPlaying && state.current.pathCurve) {
        routeActive = true;
        state.current.pathElapsed += dt;
        const t = Math.min(
          state.current.pathElapsed / state.current.pathDuration,
          1,
        );
        const position = state.current.routePosition;
        const lookAt = state.current.routeLookAt;
        state.current.pathCurve.getPoint(t, position);
        state.current.pathCurve.getPoint(Math.min(t + 0.012, 1), lookAt);
        position.y += 0.65;
        lookAt.y += 0.42;
        camera.position.copy(position);
        const direction = state.current.routeDirection
          .copy(lookAt)
          .sub(position)
          .normalize();
        direction.applyAxisAngle(worldUp, state.current.routeYaw);
        state.current.routePitchAxis
          .crossVectors(direction, worldUp)
          .normalize();
        direction.applyAxisAngle(
          state.current.routePitchAxis,
          state.current.routePitch,
        );
        controls.target.copy(position).add(direction);
        if (t >= 1) {
          state.current.pathPlaying = false;
          controls.enabled = true;
          pathEndRef.current?.();
        }
      }
      const controlsChanged = controls.update();
      if (
        !renderedOnce ||
        state.current.renderDirty ||
        controlsChanged ||
        hasVelocity ||
        routeActive
      ) {
        renderer.render(scene, camera);
        renderedOnce = true;
        state.current.renderDirty = false;
      }
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      renderer.domElement.removeEventListener("click", onClick);
      renderer.domElement.removeEventListener("pointerdown", markDown);
      renderer.domElement.removeEventListener("pointermove", onWaypointHover);
      renderer.domElement.removeEventListener(
        "pointerleave",
        onWaypointHoverEnd,
      );
      renderer.domElement.removeEventListener("pointerdown", startRouteLook);
      window.removeEventListener("pointermove", moveRouteLook);
      window.removeEventListener("pointerup", endRouteLook);
      controls.dispose();
      transform.dispose();
      geo.dispose();
      material.dispose();
      if (city)
        city.children.forEach((m) => {
          m.geometry.dispose();
          m.material.dispose();
        });
      roofTexture?.dispose();
      if (exportRef) exportRef.current = null;
      if (material.map && material.map !== height) material.map.dispose();
      height.dispose();
      classTexture.dispose();
      waypointGroup.children.forEach(disposeWaypointMarker);
      if (state.current.waypointLine) {
        state.current.waypointLine.geometry.dispose();
        state.current.waypointLine.material.dispose();
      }
      renderer.dispose();
      if (renderer.domElement.parentNode === host)
        host.removeChild(renderer.domElement);
    };
  }, [sample]);
  useEffect(() => {
    const s = state.current;
    if (!s.scene || !pathCommand || pathCommand.type === "idle") return;
    if (pathCommand.type === "clear") {
      s.pathPlaying = false;
      s.transform.detach();
      s.selectedMarker = null;
      waypointSelectRef.current?.(null);
      s.waypoints.length = 0;
      while (s.waypointGroup.children.length) {
        disposeWaypointMarker(s.waypointGroup.children.pop());
      }
      if (s.waypointLine) {
        s.scene.remove(s.waypointLine);
        s.waypointLine.geometry.dispose();
        s.waypointLine.material.dispose();
        s.waypointLine = null;
      }
      s.controls.enabled = true;
      onWaypointChange(0);
      return;
    }
    if (pathCommand.type === "delete-selected") {
      if (s.selectedMarker) s.deleteWaypoint(s.selectedMarker);
      return;
    }
    if (pathCommand.type === "pause") {
      s.pathPlaying = false;
      s.controls.enabled = true;
      return;
    }
    if (pathCommand.type === "play" && s.waypoints.length >= 2) {
      s.pathCurve = new THREE.CatmullRomCurve3(
        s.waypoints.map((point) => point.clone()),
        false,
        "catmullrom",
        0.45,
      );
      s.pathElapsed = 0;
      s.pathDuration = Math.max(7, s.waypoints.length * 2.8);
      s.pathPlaying = true;
      s.routeYaw = 0;
      s.routePitch = 0;
      if (s.selectedMarker) setWaypointSelected(s.selectedMarker, false);
      s.selectedMarker = null;
      waypointSelectRef.current?.(null);
      s.transform.detach();
      s.controls.enabled = false;
    }
  }, [pathCommand, onWaypointChange]);
  useEffect(() => {
    const s = state.current;
    if (!s.mesh) return;
    s.city?.userData.setHeightMode?.(layer === "depth");
    const colorAttr = s.mesh.geometry.attributes.color;
    if (layer === "depth" || layer === "slope") {
      // Dark blue-to-red heat map driven by the same smoothed height data
      // used to build the mesh, instead of loading the pre-baked depth.jpg.
      const previousMap = s.mesh.material.map;
      s.mesh.material.map = null;
      if (previousMap && previousMap !== s.height) previousMap.dispose();
      const colours = layer === "slope" ? s.slopeColors : s.heightColors;
      if (colorAttr && colours) {
        colorAttr.array.set(colours);
        colorAttr.needsUpdate = true;
      }
      s.mesh.material.needsUpdate = true;
      s.renderDirty = true;
      return;
    }
    if (colorAttr) {
      if (s.shadeColors) colorAttr.array.set(s.shadeColors);
      else colorAttr.array.fill(1);
      colorAttr.needsUpdate = true;
    }
    const source =
      layer === "texture"
        ? sample.rgb
        : layer === "classes"
          ? sample.classes
          : layer === "error" && sample.error
            ? sample.error
            : sample.height;
    const previousMap = s.mesh.material.map;
    const nextMap = s.tex.load(source, () => {
      s.renderDirty = true;
    });
    nextMap.colorSpace = THREE.SRGBColorSpace; // PNG/JPG photos and maps: untagged, they drew washed out
    s.mesh.material.map = nextMap;
    if (previousMap && previousMap !== s.height) previousMap.dispose();
    s.mesh.material.needsUpdate = true;
    s.renderDirty = true;
  }, [layer, sample]);
  useEffect(() => {
    const m = state.current.mesh,
      heightField = state.current.heightField;
    if (!m || !heightField) return;
    const p = m.geometry.attributes.position;
    for (let i = 0; i < p.count; i++)
      p.setY(
        i,
        (heightField[i] - TERRAIN_BASELINE) * HEIGHT_WORLD_SCALE * exaggeration,
      );
    p.needsUpdate = true;
    m.geometry.computeVertexNormals();
    if (state.current.city)
      state.current.city.scale.y =
        (exaggeration * 8) / (state.current.groundWidthM ?? 1024 * GAMUS_GSD_M);
    if (state.current.city) fitFacilityIcons(state.current.city);
    state.current.exaggeration = exaggeration;
    state.current.renderDirty = true;
  }, [exaggeration]);
  useEffect(() => {
    const s = state.current;
    if (!s.camera || !s.controls) return;
    homeCamera(s.camera, s.controls, sample, s.exaggeration ?? 1);
    s.controls.update();
  }, [resetToken]);
  useEffect(() => {
    contourUniforms.uContour.value = contours ? 1 : 0;
    contourUniforms.uInterval.value = contourStep;
    state.current.renderDirty = true;
  }, [contours, contourStep, contourUniforms]);
  // Height profile between two measured points (A -> B), drawn as a line on the surface.
  useEffect(() => {
    profilePointsRef.current = profilePoints;
    const s = state.current;
    if (!s.scene) return;
    if (s.profileLine) {
      s.scene.remove(s.profileLine);
      s.profileLine.geometry.dispose();
      s.profileLine.material.dispose();
      s.profileLine = null;
    }
    s.renderDirty = true;
    if (profilePoints.length !== 2 || !s.probeField) {
      if (!profilePoints.length && s.centreProfile) onProfile?.(s.centreProfile);
      return;
    }
    const [a, b] = profilePoints;
    const W = HEIGHT_SAMPLE_WIDTH;
    const H = HEIGHT_SAMPLE_HEIGHT;
    const lengthM = Math.hypot(
      ((b.col - a.col) * s.groundWidthM) / (W - 1),
      ((b.row - a.row) * s.groundHeightM) / (H - 1),
    );
    const n = Math.max(2, Math.min(400, Math.round(Math.hypot(b.col - a.col, b.row - a.row)) + 1));
    const z = s.elevField ?? s.probeField;
    const pos = s.mesh.geometry.attributes.position;
    // City model: the mesh is bare ground, so lift the line over buildings and trees.
    const lift = s.city ? (8 / s.groundWidthM) * (s.exaggeration ?? 1) : 0;
    const values = [];
    const pts = [];
    for (let k = 0; k < n; k++) {
      const t = k / (n - 1);
      const i =
        Math.round(a.row + (b.row - a.row) * t) * W + Math.round(a.col + (b.col - a.col) * t);
      values.push(z[i]);
      pts.push(new THREE.Vector3(pos.getX(i), pos.getY(i) + s.probeField[i] * lift + 0.004, pos.getZ(i)));
    }
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0xffb55e, depthTest: false }),
    );
    line.renderOrder = 10;
    s.scene.add(line);
    s.profileLine = line;
    onProfile?.({
      values,
      metric: s.probeMetric,
      lengthM,
      label: `A → B · ${lengthM < 1000 ? `${Math.round(lengthM)} m` : `${(lengthM / 1000).toFixed(2)} km`}`,
      what: s.elevField ? "elevation" : "height above ground",
    });
  }, [profilePoints, onProfile]);
  // First person: stand at the view centre at eye height; OrbitControls around a target
  // 5 cm ahead turns in place, which is mouse-look. Leaving returns to the overview.
  useEffect(() => {
    walkRef.current = walkMode;
    const s = state.current;
    if (!s.camera || !s.controls || !s.walkGround) return;
    const c = s.controls;
    c.enableZoom = !walkMode;
    c.enablePan = !walkMode;
    if (walkMode) {
      const fwd = new THREE.Vector3();
      s.camera.getWorldDirection(fwd);
      fwd.y = 0;
      if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
      fwd.normalize();
      let x = THREE.MathUtils.clamp(c.target.x, -3.9, 3.9);
      let zc = THREE.MathUtils.clamp(c.target.z, -3.9, 3.9);
      // don't start inside a building: step back along the view until clear
      for (let k = 0; k < 80 && s.walkBlocked(x, zc); k++) {
        x -= fwd.x * 0.05;
        zc -= fwd.z * 0.05;
      }
      const y = s.walkGround(x, zc) + s.walkEye();
      s.camera.position.set(x, y, zc);
      c.target.set(x + fwd.x * 0.05, y, zc + fwd.z * 0.05);
      s.camera.near = 0.002;
    } else {
      s.camera.near = 0.1;
      // uploads open at true scale (the exaggeration state may not have caught up yet)
      homeCamera(s.camera, c, sample, sample?.metricHeights ? 1 : s.exaggeration ?? 1);
    }
    s.camera.updateProjectionMatrix();
    c.update();
    s.renderDirty = true;
  }, [walkMode, sample]);
  useEffect(() => {
    const s = state.current;
    if (!s.controls) return;
    s.controls.autoRotate = autoRotate;
    s.renderDirty = true;
  }, [autoRotate]);
  return (
    <div
      ref={ref}
      onDoubleClick={(event) => onMeasure?.(state.current.measureAt?.(event))}
      className={`terrain-canvas ${waypointMode ? "waypoint-active" : ""}`}
    />
  );
}

// Upload path: the fine-tuned RS3DAda height model (viewer/estimate.py) turns
// PNG/JPG into a metric nDSM and GeoTIFF into an absolute DSM (nDSM + GLO-30
// ground). Served by `python -m viewer.server` on :8000.
// Same origin as the page: viewer.server serves both the app and /api. For `npm run dev` /
// `npm run preview`, vite.config.js proxies /api and /data-uploads to :8000.
const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const ESTIMATE_API_URL = `${API_BASE}/api/estimate`;
const CLASSIFY_CLASS_LABELS = [
  "background",
  "ground",
  "low_vegetation",
  "buildings",
  "water",
  "roads",
  "trees",
];
// Maps the classifier's plural class names to the singular .class-dot
// modifiers the terrain workspace's legend already defines in blender.css.
const CLASS_DOT_STYLE = {
  background: "",
  ground: "ground",
  low_vegetation: "vegetation",
  buildings: "building",
  water: "water",
  roads: "road",
  trees: "tree",
};

function App() {
  const [view, setView] = useState("upload"); // terrain | upload
  const [split, setSplit] = useState("train");
  const [sample, setSample] = useState(null);
  const [layer, setLayer] = useState("texture");
  const [exaggeration, setExaggeration] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [measure, setMeasure] = useState(false);
  const [toast, setToast] = useState("");
  const [theme, setTheme] = useState("dark");
  const [resetToken, setResetToken] = useState(0);
  const [profileCleared, setProfileCleared] = useState(false);
  const [profile, setProfile] = useState(null);
  const [measurePoint, setMeasurePoint] = useState(null);
  const [profilePoints, setProfilePoints] = useState([]); // Measure: A, then B -> A-B profile
  const [contours, setContours] = useState(false);
  const [walkMode, setWalkMode] = useState(false);
  useEffect(() => {
    if ((layer === "error" && !sample?.error) || (layer === "slope" && !sample?.metricHeights))
      setLayer("texture");
  }, [layer, sample]);
  const [cityMode, setCityMode] = useState(true);
  const [pickedBuilding, setPickedBuilding] = useState(null);
  const exportRef = useRef(null);
  const profileView = profileStats(profile);
  const shown = sample ?? EMPTY_SAMPLE;
  // City model: canopy-only terrain + extruded buildings; the exact DSM stays
  // attached for the probe and profile.
  const viewSample = useMemo(() => {
    if (!sample) return sample;
    const terrain = sample.terrain; // georeferenced uploads only
    // 16-bit grids (uploads): mesh relative to the same zero as its relief PNG, probe = nDSM,
    // elevation = absolute DSM. Catalog scenes have none and use their PNGs.
    const relative = (grid, lo) => grid && Float32Array.from(grid, (v) => v - lo);
    const grids = { probeGrid: sample.ndsmGrid, elevGrid: terrain ? sample.dsmGrid : null };
    if (cityMode && sample.city)
      return {
        ...sample,
        ...grids,
        meshGrid: terrain ? relative(sample.groundGrid, terrain.ground.min_m) : null,
        meshOffsetM: terrain ? terrain.ground.min_m : 0,
        // bare-earth terrain (or flat ground) under extruded buildings and trees
        height: terrain ? terrain.ground.png : FLAT_HEIGHT,
        maxM: terrain ? terrain.ground.relief_m : sample.maxM,
        exactHeight: sample.height,
        exactMaxM: sample.maxM,
        elevation: terrain?.dsm,
        groundMinM: terrain?.ground.min_m,
        buildings: sample.city.buildings,
        trees: sample.city.trees,
      };
    if (terrain)
      // exact surface of a GeoTIFF upload: the exported absolute DSM itself
      return {
        ...sample,
        ...grids,
        meshGrid: relative(sample.dsmGrid, terrain.dsm.min_m),
        meshOffsetM: terrain.dsm.min_m,
        height: terrain.dsm.png,
        maxM: terrain.dsm.relief_m,
        exactHeight: sample.height,
        exactMaxM: sample.maxM,
        elevation: terrain.dsm,
      };
    return sample.ndsmGrid ? { ...sample, ...grids, meshGrid: sample.ndsmGrid } : sample;
  }, [sample, cityMode]);
  const [waypointMode, setWaypointMode] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [waypointCount, setWaypointCount] = useState(0);
  const [selectedWaypoint, setSelectedWaypoint] = useState(null);
  const [pathCommand, setPathCommand] = useState({ type: "idle", id: 0 });
  const [uploadFileName, setUploadFileName] = useState("");
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle | loading | error
  const [uploadError, setUploadError] = useState("");
  const [uploadMeta, setUploadMeta] = useState(null);
  const geoid = uploadMeta?.dsm?.base_dem === "srtm" ? "EGM96" : "EGM2008"; // of the DSM's heights
  const [uploadGsd, setUploadGsd] = useState("");
  const [referenceFile, setReferenceFile] = useState(null);
  const [gcpFile, setGcpFile] = useState(null); // CSV lon, lat, height: corrects a GeoTIFF's DSM
  const [uploadQuality, setUploadQuality] = useState("high"); // high = 4-flip TTA
  const [baseDem, setBaseDem] = useState("srtm"); // public DEM under a GeoTIFF's absolute DSM (the brief names SRTM)
  const [lastUploadFile, setLastUploadFile] = useState(null); // re-run at a measured scale
  const [knownLengthM, setKnownLengthM] = useState(""); // real length of the measured A -> B
  const [uploadProgress, setUploadProgress] = useState(null); // {stage, progress}
  const fileRef = useRef(null);
  const uploadFileRef = useRef(null);
  const referenceFileRef = useRef(null);
  const gcpFileRef = useRef(null);
  const notify = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 1800);
  };
  const selectScene = (s) => {
    setSplit(s.split);
    setSample(s);
    setProfileCleared(false);
    setProfilePoints([]);
    setWalkMode(false);
    setWaypointCount(0);
    setSelectedWaypoint(null);
    setPlaying(false);
    notify(`Loaded ${s.id}`);
  };
  const runClassification = async (file, gsdOverride) => {
    setLastUploadFile(file);
    setUploadFileName(file.name);
    setUploadStatus("loading");
    setUploadError("");
    // Progress: the server reports real stages for this job id while the POST runs.
    const job = crypto.randomUUID();
    setUploadProgress({ stage: "Uploading", progress: 0 });
    const poll = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/progress/${job}`);
        if (r.ok) setUploadProgress(await r.json());
      } catch {
        // a missed poll is harmless; the next one catches up
      }
    }, 400);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("job", job);
      body.append("tta", uploadQuality === "high" ? "true" : "false");
      const gsdValue = gsdOverride ?? uploadGsd.trim();
      if (gsdValue) body.append("gsd", gsdValue);
      if (referenceFile) body.append("reference", referenceFile);
      if (gcpFile) body.append("gcps", gcpFile);
      body.append("base_dem", baseDem);
      const res = await fetch(ESTIMATE_API_URL, { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        const msg = detail?.detail;
        throw new Error(
          (typeof msg === "string" ? msg : JSON.stringify(msg)) ||
            `Server returned ${res.status}`,
        );
      }
      const data = await res.json();
      const gsdText = data.gsd_m
        ? `${data.gsd_m.toFixed(2)} m/px`
        : "GSD unknown (0.33 m/px assumption; experimental metric)";
      // Reuses the exact same `sample` shape as a catalog scene (rgb/height/
      // classes URLs + label/coord/max) so every existing viewer feature --
      // layer switching, waypoints, measuring -- works on it unmodified.
      setSample({
        id: "upload",
        label: file.name,
        coord: data.dsm
          ? `Georeferenced · ${gsdText} · DSM ${data.dsm.min_m.toFixed(0)}–${data.dsm.max_m.toFixed(0)} m`
          : `${data.georeferenced ? "Georeferenced" : "Non-georeferenced"} · ${gsdText}`,
        rgb: data.rgb,
        height: data.height,
        classes: data.classes,
        max: `${data.max_m.toFixed(1)} m`,
        thumb: data.rgb,
        // /api/estimate encodes height linearly: pixel/255 * max_m.
        metricHeights: true,
        maxM: data.max_m,
        ndsmMaxM: data.max_m, // viewSample overrides maxM per mode; this stays the nDSM's
        // 16-bit heights on the mesh grid, when the server's grid lines up with it
        ...(data.grids && (data.grids.side - 1) % (HEIGHT_SAMPLE_WIDTH - 1) === 0
          ? {
              ndsmGrid: decodeGrid(data.grids.ndsm, data.grids.side, HEIGHT_SAMPLE_WIDTH),
              groundGrid: decodeGrid(data.grids.ground, data.grids.side, HEIGHT_SAMPLE_WIDTH),
              dsmGrid: decodeGrid(data.grids.dsm, data.grids.side, HEIGHT_SAMPLE_WIDTH),
            }
          : {}),
        // The server centres the scene in a square at its own aspect: these span the square.
        groundWidthM: (data.view_shape ?? data.shape)[1] * (data.gsd_m || GAMUS_GSD_M),
        groundHeightM: (data.view_shape ?? data.shape)[0] * (data.gsd_m || GAMUS_GSD_M),
        areaKm2:
          (data.shape[0] * data.shape[1] * (data.gsd_m || GAMUS_GSD_M) ** 2) / 1e6,
        gsdAssumed: data.gsd_assumed,
        sourceLabel: `${data.shape[1]} × ${data.shape[0]} source`,
        city: data.city,
        facilities: data.facilities ?? [],
        embankments: data.embankments ?? [],
        terrain: data.terrain,
        error: data.error?.png ?? null, // signed model - reference map, when validated
        errorLimitM: data.error?.limit_m ?? null,
      });
      setMeasurePoint(null);
      setProfilePoints([]);
      setWalkMode(false);
      setExaggeration(1); // uploads are metric: open at true scale (0.5 suits the restyled previews)
      setUploadMeta({
        seconds: data.seconds,
        width: data.width,
        height_px: data.height_px,
        class_pixel_counts: data.class_pixel_counts,
        max_m: data.max_m,
        p99_m: data.ndsm_p99_m,
        dsm: data.dsm,
        dsm_error: data.dsm_error,
        downloads: data.downloads,
        validation: data.validation,
        gcp: data.gcp,
        checkpoint: data.checkpoint,
        models: data.models,
        work_gsd_m: data.work_gsd_m,
        gsd_m: data.gsd_m,
        height_mode: data.height_mode,
        geospatial_warnings: data.geospatial_warnings,
        base_dem_fallback: data.base_dem_fallback,
        dem_agreement: data.dem_agreement,
        facilities: data.facilities ?? [],
        facilities_error: data.facilities_error,
        flood: data.flood_defences ?? null,
        bridges: data.bridges ?? null,
      });
      setProfileCleared(false);
      setWaypointCount(0);
      setSelectedWaypoint(null);
      setPlaying(false);
      setUploadStatus("idle");
      notify(`Heights estimated for ${file.name}`);
    } catch (err) {
      setUploadError(
        err.message === "Failed to fetch"
          ? "Couldn't reach the height backend — is `python -m viewer.server` running?"
          : err.message,
      );
      setUploadStatus("error");
    } finally {
      clearInterval(poll);
      setUploadProgress(null);
    }
  };
  const onPickUpload = (e) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // picking the same file again should re-run
    if (file) runClassification(file);
  };
  const onDropFile = (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (!file || uploadStatus === "loading") return;
    if (!/\.(png|jpe?g|tiff?)$/i.test(file.name)) {
      notify("Drop a PNG, JPG or GeoTIFF");
      return;
    }
    setView("upload");
    runClassification(file);
  };
  const sceneIndex = Math.max(
    0,
    displayedScenes.findIndex((scene) => scene.id === shown.id),
  );
  const changeScene = (offset) => {
    const next =
      displayedScenes[
        (sceneIndex + offset + displayedScenes.length) % displayedScenes.length
      ];
    selectScene(next);
  };
  const togglePathPlayback = () => {
    if (playing) {
      setPathCommand({ type: "pause", id: Date.now() });
      setPlaying(false);
      return;
    }
    if (waypointCount < 2) {
      notify("Set at least two waypoints first");
      return;
    }
    setPathCommand({ type: "play", id: Date.now() });
    setPlaying(true);
    setWaypointMode(false);
    setAutoRotate(false);
  };
  const toggleAutoRotate = () => {
    const next = !autoRotate;
    setAutoRotate(next);
    if (next) setWaypointMode(false);
    notify(next ? "Auto-rotate on" : "Auto-rotate off");
  };
  const removeSelectedWaypoint = () => {
    setPathCommand({ type: "delete-selected", id: Date.now() });
    notify("Point removed");
  };
  const clearPath = () => {
    setPathCommand({ type: "clear", id: Date.now() });
    setWaypointCount(0);
    setSelectedWaypoint(null);
    setPlaying(false);
    notify("Waypoints cleared");
  };
  const importScene = (e) => {
    const f = e.target.files?.[0];
    e.target.value = ""; // allow re-importing the same file
    if (!f) return;
    setView("upload");
    runClassification(f);
  };
  const exportScene = () => {
    if (!sample) {
      notify("Nothing to export yet · import imagery first");
      return;
    }
    if (sample.metricHeights && exportRef.current) {
      const name = (sample.label || "scene").replace(/\.[^.]+$/, "");
      exportRef.current(`${name}-3d.glb`)
        .then(() => notify("3D model exported (.glb, metres)"))
        .catch((err) => notify(`Export failed: ${err?.message ?? err}`));
      return;
    }
    const blob = new Blob(
      [
        JSON.stringify(
          {
            format: "GAMUS Terrain Studio scene",
            scene: shown.id,
            split,
            layer,
            verticalExaggeration: exaggeration,
            controls: "WASDQE",
            source: "earthflow/GAMUS",
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${shown.id}-terrain-scene.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify("Scene manifest downloaded");
  };
  return (
    <div className={`app theme-${theme}`}>
      <input
        ref={fileRef}
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff"
        onChange={importScene}
        hidden
      />
      <aside className="rail">
        <div className="brandmark">
          <Mountain size={18} />
        </div>
        <div className="rail-nav">
          <button
            className={view === "terrain" ? "active" : ""}
            onClick={() => setView("terrain")}
            title="Terrain workspace"
          >
            <Layers3 size={18} />
          </button>
          <button
            className={view === "upload" ? "active" : ""}
            onClick={() => setView("upload")}
            title="Upload imagery"
          >
            <Upload size={18} />
          </button>
        </div>
        <div className="rail-bottom">
          <button
            onClick={() => {
              setTheme(theme === "dark" ? "light" : "dark");
              notify(`${theme === "dark" ? "Light" : "Dark"} mode enabled`);
            }}
            title={theme === "dark" ? "Light mode" : "Dark mode"}
          >
            {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </aside>
      <main>
        <section className="workspace">
          <div
            className={`viewport-card ${layer === "depth" || layer === "classes" ? "depth-mode" : ""}`}
          >
            <div className="viewport-top">
              <div className="scene-title">
                <span className="scene-chip">
                  <Satellite size={14} />
                </span>
                <div>
                  <strong>{shown.label}</strong>
                  <small>
                    {shown.id ? `${shown.id} · ` : ""}{shown.coord}
                  </small>
                </div>
              </div>
              <div className="view-actions">
                <button onClick={() => fileRef.current?.click()}>
                  <Upload size={14} /> Import
                </button>
                <button onClick={exportScene}>
                  <Download size={14} /> Export
                </button>
                <button onClick={togglePathPlayback}>
                  {playing ? <Pause size={15} /> : <Play size={15} />}{" "}
                  {playing ? "Pause path" : "Play route"}
                </button>
                <button
                  onClick={() =>
                    Promise.resolve(
                      document
                        .querySelector(".workspace")
                        ?.requestFullscreen?.(),
                    ).catch(() => notify("Fullscreen unavailable"))
                  }
                >
                  <Maximize2 size={15} />
                </button>
              </div>
            </div>
            <div
              className="canvas-wrap"
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDropFile}
            >
              {pickedBuilding && (
                <div className="building-card">
                  <div className="label-row">
                    <strong>{pickedBuilding.bridge ? "Bridge" : "Building"}</strong>
                    <button onClick={() => setPickedBuilding(null)} title="Close">
                      <X size={13} />
                    </button>
                  </div>
                  {pickedBuilding.bridge ? (
                  <dl>
                    <dt>Source</dt>
                    <dd>OpenStreetMap</dd>
                    {pickedBuilding.roofElevation != null && (
                      <>
                        <dt>Deck elevation</dt>
                        <dd>{pickedBuilding.roofElevation.toFixed(1)} m ({geoid})</dd>
                      </>
                    )}
                  </dl>
                  ) : (
                  <dl>
                    {pickedBuilding.facility && (
                      <>
                        <dt>{FACILITY_STYLE[pickedBuilding.facility.kind]?.label.replace(/s$/, "") ?? "Facility"}</dt>
                        <dd>{pickedBuilding.facility.name || "unnamed (OpenStreetMap)"}</dd>
                      </>
                    )}
                    <dt>Height</dt>
                    <dd>{pickedBuilding.height.toFixed(1)} m</dd>
                    <dt>Floors (≈3.2 m each)</dt>
                    <dd>~{pickedBuilding.floors}</dd>
                    <dt>Footprint</dt>
                    <dd>{Math.round(pickedBuilding.areaM2).toLocaleString()} m²</dd>
                    <dt>Volume</dt>
                    <dd>{Math.round(pickedBuilding.volumeM3).toLocaleString()} m³</dd>
                    {pickedBuilding.roofElevation != null && (
                      <>
                        <dt>Roof elevation</dt>
                        <dd>{pickedBuilding.roofElevation.toFixed(1)} m ({geoid})</dd>
                      </>
                    )}
                  </dl>
                  )}
                </div>
              )}
              {uploadStatus === "loading" && (
                <div className="processing-overlay">
                  <strong>{uploadFileName}</strong>
                  <span>{uploadProgress?.stage ?? "Working"}…</span>
                  <i className="progress-track">
                    <i
                      className="progress-fill"
                      style={{ width: `${Math.round((uploadProgress?.progress ?? 0) * 100)}%` }}
                    />
                  </i>
                  <small>{Math.round((uploadProgress?.progress ?? 0) * 100)}%</small>
                </div>
              )}
              {sample ? (
              <TerrainCanvas
                sample={viewSample}
                exaggeration={exaggeration}
                layer={layer}
                resetToken={resetToken}
                onMeasure={(info) => {
                  if (!measure) return;
                  if (!info) {
                    notify("Double-click on the terrain surface");
                    return;
                  }
                  setMeasurePoint(info);
                  setProfileCleared(false);
                  setProfilePoints((p) => (p.length === 1 ? [p[0], info] : [info]));
                }}
                onProfile={setProfile}
                profilePoints={profilePoints}
                contours={contours && Boolean(shown.metricHeights)}
                contourStep={contourInterval(viewSample?.maxM ?? 1)}
                walkMode={walkMode}
                measureMode={measure}
                waypointMode={waypointMode}
                pathCommand={pathCommand}
                onWaypointChange={setWaypointCount}
                onWaypointSelect={setSelectedWaypoint}
                onPathEnd={() => setPlaying(false)}
                autoRotate={autoRotate && !walkMode}
                onBuildingPick={setPickedBuilding}
                exportRef={exportRef}
              />
              ) : uploadStatus === "loading" ? null : (
                <div className="empty-viewport">
                  <Upload size={28} />
                  <strong>No imagery loaded</strong>
                  <span>Import a PNG, JPG or GeoTIFF to estimate heights and build the 3D terrain.</span>
                  <button onClick={() => fileRef.current?.click()}>Import imagery</button>
                </div>
              )}
              <div className="flight-help">
                {walkMode ? (
                  <>
                    <kbd>W</kbd>
                    <kbd>A</kbd>
                    <kbd>S</kbd>
                    <kbd>D</kbd> walk · <kbd>Shift</kbd> run · drag to look · buildings block
                  </>
                ) : (
                  <>
                    <kbd>W</kbd>
                    <kbd>A</kbd>
                    <kbd>S</kbd>
                    <kbd>D</kbd> move · <kbd>Q</kbd>
                    <kbd>E</kbd> altitude · drag to look on route
                  </>
                )}
              </div>
              {sample && (
                <button
                  className={`walk-toggle ${walkMode ? "active" : ""}`}
                  title="First-person: walk on the ground at eye height"
                  onClick={() => {
                    setPlaying(false);
                    setWalkMode((w) => !w);
                  }}
                >
                  <PersonStanding size={15} /> {walkMode ? "Exit walk" : "Walk"}
                </button>
              )}
              <div className="canvas-badge">
                {layer === "elevation"
                  ? "Surface"
                  : layer === "depth"
                    ? "Height estimate"
                    : layer === "classes"
                      ? "Semantic classes"
                      : layer === "slope"
                        ? "Slope"
                        : layer === "error"
                          ? "Error vs reference"
                          : "RGB texture"}
              </div>
            </div>
            <div className="viewport-footer">
              <div className="legend">
                {!sample ? null : layer === "slope" ? (
                  <>
                    <span>0°</span>
                    <b className="height-ramp" style={{ background: SLOPE_CSS }} />
                    <span>{SLOPE_MAX_DEG}°+</span>
                    <span className="legend-note">
                      slope of the {cityMode && sample.city ? "ground" : "surface"} shown
                    </span>
                  </>
                ) : layer === "error" && sample.errorLimitM ? (
                  <>
                    <span>−{sample.errorLimitM} m</span>
                    <b className="height-ramp" style={{ background: ERROR_CSS }} />
                    <span>+{sample.errorLimitM} m</span>
                    <span className="legend-note">model − reference · blue reads low, red high</span>
                  </>
                ) : layer === "depth" ? (
                  // Same stops as the Height layer's colours (JET_STOPS), so they can't drift.
                  <>
                    <span>{shown.metricHeights ? "0 m" : "low"}</span>
                    <b className="height-ramp" style={{ background: JET_CSS }} />
                    <span>
                      {shown.metricHeights ? `${(shown.ndsmMaxM ?? shown.maxM).toFixed(1)} m` : "high"}
                    </span>
                    <span className="legend-note">
                      {shown.metricHeights ? "height above ground" : "relative height (LiDAR reference)"}
                    </span>
                  </>
                ) : shown.metricHeights ? (
                  <span>
                    Height above ground 0–{(shown.ndsmMaxM ?? shown.maxM).toFixed(1)} m · Height layer shows it in colour
                  </span>
                ) : (
                  <span>LiDAR reference heights, relative · Height layer shows them in colour</span>
                )}
              </div>
              <div className="footer-note">
                {contours && shown.metricHeights && (
                  <span>contours every {contourInterval(viewSample?.maxM ?? 1)} m · </span>
                )}
                <Eye size={14} /> {shown.sourceLabel ?? "1024 × 1024 source"}{" "}
                · {HEIGHT_SAMPLE_WIDTH}² live mesh
              </div>
            </div>
          </div>
          <aside className="inspector">
            <div className="panel-heading">
              <div>
                <div className="eyebrow">Scene inspector</div>
                <h2>Reconstruction</h2>
              </div>
            </div>
            {view === "terrain" ? (
            <div className="control-section scene-switcher">
              <div className="label-row">
                <label>Scene</label>
                <span>
                  {sceneIndex + 1} / {displayedScenes.length} ·{" "}
                  {urbanScenes.length} urban
                </span>
              </div>
              <div className="scene-preview">
                {sample && (
                  <img src={shown.thumb || shown.rgb} alt={shown.label} />
                )}
                <div>
                  <strong>{shown.id}</strong>
                  <small>
                    {shown.urban ? "Building-rich · " : ""}
                    {shown.label}
                  </small>
                </div>
              </div>
              <select
                value={shown.id}
                onChange={(event) =>
                  selectScene(
                    displayedScenes.find(
                      (scene) => scene.id === event.target.value,
                    ),
                  )
                }
              >
                <option value="" disabled>
                  Choose a GAMUS preview scene…
                </option>
                <optgroup
                  label={`Urban / building-rich (${urbanScenes.length})`}
                >
                  {urbanScenes.map((scene) => (
                    <option key={`${scene.split}-${scene.id}`} value={scene.id}>
                      {scene.id} — {scene.label}
                    </option>
                  ))}
                </optgroup>
                <optgroup label={`Other GAMUS tiles (${otherScenes.length})`}>
                  {otherScenes.map((scene) => (
                    <option key={`${scene.split}-${scene.id}`} value={scene.id}>
                      {scene.id} — {scene.label}
                    </option>
                  ))}
                </optgroup>
              </select>
              <div className="photo-nav">
                <button onClick={() => changeScene(-1)}>← Previous</button>
                <button onClick={() => changeScene(1)}>Next →</button>
              </div>
            </div>
            ) : (
            <div className="control-section upload-switcher">
              <div className="label-row">
                <label>Upload imagery</label>
                <span>PNG · JPG · GeoTIFF</span>
              </div>
              <p className="upload-copy">
                A fine-tuned height model estimates metric height above
                ground (nDSM) for every pixel. GeoTIFFs with coordinates also
                get an absolute DSM on Copernicus GLO-30 or SRTM, scored
                against both. Both download as GeoTIFF.
              </p>
              <div className="upload-options">
                <label>
                  <small>Pixel size (m/px, optional)</small>
                  <input
                    type="number"
                    min="0.05"
                    max="10"
                    step="0.01"
                    placeholder="from GeoTIFF, else 0.33"
                    value={uploadGsd}
                    onChange={(e) => setUploadGsd(e.target.value)}
                  />
                </label>
                <input
                  ref={referenceFileRef}
                  type="file"
                  accept=".tif,.tiff,.png,.h5"
                  onChange={(e) => setReferenceFile(e.target.files?.[0] ?? null)}
                  hidden
                />
                <div className="upload-reference-row">
                  <button
                    className="upload-reference"
                    onClick={() => referenceFileRef.current?.click()}
                  >
                    {referenceFile
                      ? `Reference: ${referenceFile.name}`
                      : "Add reference heights (optional)"}
                  </button>
                  {referenceFile && (
                    <button
                      className="upload-reference-clear"
                      title="Remove reference"
                      onClick={() => {
                        setReferenceFile(null);
                        if (referenceFileRef.current) referenceFileRef.current.value = "";
                      }}
                    >
                      <X size={13} />
                    </button>
                  )}
                </div>
                <input
                  ref={gcpFileRef}
                  type="file"
                  accept=".csv,.txt"
                  onChange={(e) => setGcpFile(e.target.files?.[0] ?? null)}
                  hidden
                />
                <div className="upload-reference-row">
                  <button
                    className="upload-reference"
                    title="CSV rows of lon, lat, height (m above sea level). Corrects the absolute DSM of a GeoTIFF."
                    onClick={() => gcpFileRef.current?.click()}
                  >
                    {gcpFile
                      ? `Control points: ${gcpFile.name}`
                      : "Add ground control points (optional, GeoTIFF)"}
                  </button>
                  {gcpFile && (
                    <button
                      className="upload-reference-clear"
                      title="Remove control points"
                      onClick={() => {
                        setGcpFile(null);
                        if (gcpFileRef.current) gcpFileRef.current.value = "";
                      }}
                    >
                      <X size={13} />
                    </button>
                  )}
                </div>
                <div className="quality-row">
                  <small>Quality</small>
                  <div className="segmented">
                    <button
                      className={uploadQuality === "high" ? "selected" : ""}
                      onClick={() => setUploadQuality("high")}
                      title="Averages 4 flipped predictions: most accurate"
                    >
                      High
                    </button>
                    <button
                      className={uploadQuality === "fast" ? "selected" : ""}
                      onClick={() => setUploadQuality("fast")}
                      title="Single prediction: about 4x faster"
                    >
                      Fast
                    </button>
                  </div>
                </div>
                <div className="quality-row">
                  <small>Base terrain (GeoTIFF)</small>
                  <div className="segmented">
                    <button
                      className={baseDem === "glo30" ? "selected" : ""}
                      onClick={() => setBaseDem("glo30")}
                      title="Copernicus GLO-30 (2011-15, EGM2008)"
                    >
                      Copernicus
                    </button>
                    <button
                      className={baseDem === "srtm" ? "selected" : ""}
                      onClick={() => setBaseDem("srtm")}
                      title="SRTM GL1 (2000, EGM96)"
                    >
                      SRTM
                    </button>
                  </div>
                </div>
              </div>
              <input
                ref={uploadFileRef}
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff"
                onChange={onPickUpload}
                hidden
              />
              <button
                className="upload-dropzone"
                onClick={() => uploadFileRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDropFile}
                disabled={uploadStatus === "loading"}
              >
                <Upload size={20} />
                <span>
                  {uploadStatus === "loading"
                    ? `${uploadProgress?.stage ?? "Working"} · ${Math.round((uploadProgress?.progress ?? 0) * 100)}%`
                    : uploadFileName
                      ? `${uploadFileName} — drop or click to replace`
                      : "Drop an image here, or click to choose"}
                </span>
                {uploadStatus === "loading" && (
                  <i className="progress-track">
                    <i
                      className="progress-fill"
                      style={{ width: `${Math.round((uploadProgress?.progress ?? 0) * 100)}%` }}
                    />
                  </i>
                )}
              </button>
              {uploadStatus === "error" && (
                <div className="upload-error">{uploadError}</div>
              )}
              {uploadMeta && (
                <>
                  <div className="metric-grid">
                    <div>
                      <small>Processed in</small>
                      <strong>{uploadMeta.seconds.toFixed(2)}s</strong>
                    </div>
                    <div>
                      <small>Tallest object</small>
                      <strong>{uploadMeta.max_m.toFixed(1)} m</strong>
                    </div>
                    {uploadMeta.dsm && (
                      <>
                        <div>
                          <small>Ground ({uploadMeta.dsm.base_dem === "srtm" ? "SRTM" : "GLO-30"})</small>
                          <strong>
                            {uploadMeta.dsm.ground_min_m.toFixed(0)}–
                            {uploadMeta.dsm.ground_max_m.toFixed(0)} m
                          </strong>
                        </div>
                        <div>
                          <small>DSM range</small>
                          <strong>
                            {uploadMeta.dsm.min_m.toFixed(0)}–
                            {uploadMeta.dsm.max_m.toFixed(0)} m
                          </strong>
                        </div>
                      </>
                    )}
                  </div>
                  {uploadMeta.models && (
                    <div className="models-line">
                      Models: {uploadMeta.models.join(" · ")}
                      {uploadMeta.height_mode === "assumed_gsd_experimental" &&
                        " · unknown GSD: metric scale is experimental"}
                      {uploadMeta.gsd_m && uploadMeta.work_gsd_m > 0.34 &&
                        ` · processed at ${uploadMeta.work_gsd_m.toFixed(2)} m (very large scene)`}
                      {uploadMeta.models.some((m) => m.startsWith("CHMv2")) && (
                        <span className="dino-credit"> · Built with DINOv3</span>
                      )}
                    </div>
                  )}
                  {uploadMeta.height_mode !== "metadata_metric" && lastUploadFile && (
                    <div className="gcp-result scale-tool">
                      <div>
                        Pixel size used: {(uploadMeta.gsd_m ?? GAMUS_GSD_M).toFixed(3)} m
                        {uploadMeta.gsd_m == null ? " (assumed)" : ""}. Heights scale with it.
                      </div>
                      <div>
                        Set the scale: Measure, double-click both ends of something you know
                        (a road lane ≈ 3.5 m, a car ≈ 4.5 m, a building side), then enter its real length.
                      </div>
                      <div className="scale-row">
                        <span>
                          A → B {profile?.lengthM ? `${profile.lengthM.toFixed(1)} m now` : "not measured yet"} · real
                        </span>
                        <input
                          type="number"
                          min="0.1"
                          step="0.1"
                          placeholder="m"
                          value={knownLengthM}
                          onChange={(e) => setKnownLengthM(e.target.value)}
                        />
                        <button
                          disabled={!(profile?.lengthM > 0 && +knownLengthM > 0) || uploadStatus === "loading"}
                          onClick={() => {
                            const gsd = (uploadMeta.gsd_m ?? GAMUS_GSD_M) * (+knownLengthM / profile.lengthM);
                            setUploadGsd(gsd.toFixed(3));
                            setMeasure(false);
                            runClassification(lastUploadFile, gsd.toFixed(3));
                          }}
                        >
                          Reprocess
                        </button>
                      </div>
                    </div>
                  )}
                  {uploadMeta.dsm_error && (
                    <div className="upload-error">
                      Absolute DSM unavailable: {uploadMeta.dsm_error}
                    </div>
                  )}
                  {uploadMeta.base_dem_fallback && (
                    <div className="upload-error">{uploadMeta.base_dem_fallback}</div>
                  )}
                  {uploadMeta.geospatial_warnings?.map((warning) => (
                    <div className="upload-error" key={warning}>
                      GeoTIFF warning: {warning}
                    </div>
                  ))}
                  {uploadMeta.gcp &&
                    (uploadMeta.gcp.error ? (
                      <div className="upload-error">
                        Control points not applied: {uploadMeta.gcp.error}
                      </div>
                    ) : uploadMeta.gcp.model === "none" ? (
                      <div className="gcp-result">
                        Control points: {uploadMeta.gcp.n_used} of {uploadMeta.gcp.n_given} inside
                        the image · {uploadMeta.gcp.note}
                      </div>
                    ) : (
                      <div className="gcp-result">
                        Control points: {uploadMeta.gcp.n_used} of {uploadMeta.gcp.n_given} inside
                        the image · DSM shifted {uploadMeta.gcp.offset_m >= 0 ? "+" : ""}
                        {uploadMeta.gcp.offset_m.toFixed(2)} m · ground error at them{" "}
                        {uploadMeta.gcp.rmse_before_m.toFixed(2)} →{" "}
                        {uploadMeta.gcp.rmse_after_m.toFixed(2)} m
                        {uploadMeta.gcp.rmse_left_out_m != null &&
                          ` (${uploadMeta.gcp.rmse_left_out_m.toFixed(2)} m on left-out points)`}
                      </div>
                    ))}
                  {uploadMeta.dem_agreement && (
                    <div
                      className="gcp-result"
                      title="The DSM averaged over each 30 m DEM cell and compared with that DEM, the way GeoTIFF output is scored"
                    >
                      DSM vs public DEMs (30 m cells)
                      {[["glo30", "Copernicus GLO-30"], ["srtm", "SRTM"]].map(([key, label]) => {
                        const a = uploadMeta.dem_agreement[key];
                        return (
                          <div key={key}>
                            {label}
                            {uploadMeta.dsm?.base_dem === key ? " (base)" : ""}:{" "}
                            {a
                              ? `RMSE ${a.cell_rmse.toFixed(2)} m · bias ${a.cell_bias >= 0 ? "+" : ""}${a.cell_bias.toFixed(2)} m`
                              : "unavailable"}
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {uploadMeta.bridges && (uploadMeta.bridges.error || uploadMeta.bridges.deck_m2 > 0) && (
                    <div className="gcp-result bridge-line">
                      Bridges (OpenStreetMap):{" "}
                      {uploadMeta.bridges.error
                        ? `unavailable (${uploadMeta.bridges.error})`
                        : `${Math.round(uploadMeta.bridges.deck_m2).toLocaleString()} m² of deck from ${uploadMeta.bridges.ways} ways, heights from the banks`}
                    </div>
                  )}
                  {uploadMeta.flood && (uploadMeta.flood.error || uploadMeta.flood.items?.length > 0) && (
                    <div className="gcp-result facility-list">
                      Embankments and flood defences (OpenStreetMap)
                      {uploadMeta.flood.error ? (
                        <div>unavailable: {uploadMeta.flood.error}</div>
                      ) : (
                        uploadMeta.flood.items.map((e, i) => (
                          <div key={i}>
                            <i className="facility-dot" style={{ background: "#2bd4ff" }} />
                            {e.kind}{e.name ? ` "${e.name}"` : ""}: crest ≈ {e.crest_m.toFixed(1)} m ({geoid},{" "}
                            {uploadMeta.dsm?.base_dem === "glo30" ? "Copernicus" : "SRTM"})
                          </div>
                        ))
                      )}
                    </div>
                  )}
                  {(uploadMeta.facilities?.length > 0 || uploadMeta.facilities_error) && (
                    <div className="gcp-result facility-list">
                      Critical facilities (OpenStreetMap)
                      {uploadMeta.facilities_error ? (
                        <div>unavailable: {uploadMeta.facilities_error}</div>
                      ) : (
                        Object.entries(
                          uploadMeta.facilities.reduce((acc, f) => {
                            (acc[f.kind] ??= []).push(f.name);
                            return acc;
                          }, {}),
                        ).map(([kind, names]) => (
                          <div key={kind}>
                            <i
                              className="facility-dot"
                              style={{ background: FACILITY_STYLE[kind]?.colour ?? "#888" }}
                            />
                            {FACILITY_STYLE[kind]?.label ?? kind} ({names.length})
                            {names.filter(Boolean).length > 0 &&
                              `: ${names.filter(Boolean).slice(0, 4).join(", ")}${names.filter(Boolean).length > 4 ? "…" : ""}`}
                          </div>
                        ))
                      )}
                    </div>
                  )}
                  {uploadMeta.validation && (
                    <div className="metric-grid validation-grid">
                      <div>
                        <small>RMSE vs reference</small>
                        <strong>{uploadMeta.validation.rmse?.toFixed(2)} m</strong>
                      </div>
                      <div>
                        <small>MAE</small>
                        <strong>{uploadMeta.validation.mae?.toFixed(2)} m</strong>
                      </div>
                      <div>
                        <small>Correlation r</small>
                        <strong>
                          {uploadMeta.validation.pearson?.toFixed(3) ?? "n/a"}
                        </strong>
                      </div>
                      <div>
                        <small>Building RMSE</small>
                        <strong>
                          {uploadMeta.validation.building_rmse != null
                            ? `${uploadMeta.validation.building_rmse.toFixed(2)} m`
                            : "n/a"}
                        </strong>
                      </div>
                    </div>
                  )}
                  {uploadMeta.validation?.scatter && (
                    <ValidationDetail v={uploadMeta.validation} />
                  )}
                  <div className="download-row">
                    {Object.entries(uploadMeta.downloads ?? {})
                      .filter(([name]) => name.endsWith(".tif") || name.endsWith(".geojson"))
                      .map(([name, path]) => (
                        <a key={name} href={`${API_BASE}${path}`} download>
                          ⬇{" "}
                          {name === "dsm.tif"
                            ? "Absolute DSM (GeoTIFF)"
                            : name === "ndsm.tif"
                              ? "nDSM (GeoTIFF)"
                              : "3D buildings (GeoJSON)"}
                        </a>
                      ))}
                  </div>
                  <div className="class-breakdown">
                    {CLASSIFY_CLASS_LABELS.map((name) => {
                      const count = uploadMeta.class_pixel_counts?.[name] ?? 0;
                      const total = uploadMeta.width * uploadMeta.height_px;
                      const pct = total
                        ? ((count / total) * 100).toFixed(1)
                        : "0.0";
                      return (
                        <div key={name} className="class-breakdown-row">
                          <i className={`class-dot ${CLASS_DOT_STYLE[name]}`} />
                          <span>{name.replace("_", " ")}</span>
                          <strong>{pct}%</strong>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
            )}
            <div className="metric-grid">
              <div>
                <small>Surface max</small>
                <strong>{shown.max}</strong>
                <em className="neutral">
                  {shown.metricHeights ? "model estimate" : sample ? "LiDAR reference" : "scene max"}
                </em>
              </div>
              <div>
                <small>Coverage</small>
                <strong>
                  {sample
                    ? `${(shown.areaKm2 ?? (1024 * GAMUS_GSD_M) ** 2 / 1e6).toFixed(3)} km²`
                    : "—"}
                </strong>
                <em className="neutral">
                  {shown.gsdAssumed ? "GSD experimental 0.33 m" : "ground area"}
                </em>
              </div>
            </div>
            {sample?.city && (
              <div className="control-section">
                <div className="label-row">
                  <label>3D model</label>
                  <span>
                    {sample.city.buildings.length} buildings ·{" "}
                    {sample.city.trees.length} trees
                  </span>
                </div>
                <div className="segmented layer-tabs">
                  <button
                    className={cityMode ? "selected" : ""}
                    onClick={() => setCityMode(true)}
                  >
                    City model
                  </button>
                  <button
                    className={!cityMode ? "selected" : ""}
                    onClick={() => setCityMode(false)}
                  >
                    Exact DSM
                  </button>
                </div>
              </div>
            )}
            <div className="control-section">
              <label>Active layer</label>
              <div
                className={`segmented layer-tabs ${shown.metricHeights ? "wide" : ""}`}
              >
                <button
                  className={layer === "elevation" ? "selected" : ""}
                  onClick={() => setLayer("elevation")}
                >
                  <Mountain size={14} /> Surface
                </button>
                <button
                  className={layer === "depth" ? "selected" : ""}
                  onClick={() => setLayer("depth")}
                >
                  <Activity size={14} /> Height
                </button>
                <button
                  className={layer === "texture" ? "selected" : ""}
                  onClick={() => setLayer("texture")}
                >
                  <FileImage size={14} /> RGB
                </button>
                <button
                  className={layer === "classes" ? "selected" : ""}
                  onClick={() => setLayer("classes")}
                >
                  <Layers3 size={14} /> Classes
                </button>
                {shown.metricHeights && (
                  <button
                    className={layer === "slope" ? "selected" : ""}
                    onClick={() => setLayer("slope")}
                  >
                    <TriangleRight size={14} /> Slope
                  </button>
                )}
                {shown.metricHeights && (
                  <button
                    className={layer === "error" ? "selected" : ""}
                    disabled={!sample?.error}
                    title={
                      sample?.error
                        ? "Where the model reads higher or lower than the reference"
                        : "Add reference heights to the upload to see the error map"
                    }
                    onClick={() => setLayer("error")}
                  >
                    <Diff size={14} /> Error
                  </button>
                )}
              </div>
              {shown.metricHeights && (
                <label className="contour-toggle">
                  <input
                    type="checkbox"
                    checked={contours}
                    onChange={(e) => setContours(e.target.checked)}
                  />
                  <Waves size={13} /> Contour lines every {contourInterval(viewSample?.maxM ?? 1)} m
                </label>
              )}
              {layer === "classes" && (
                <div
                  className="class-legend"
                  aria-label="GAMUS semantic classes"
                >
                  <span>
                    <i className="class-dot building" /> Buildings
                  </span>
                  <span>
                    <i className="class-dot tree" /> Trees
                  </span>
                  <span>
                    <i className="class-dot road" /> Roads
                  </span>
                  <span>
                    <i className="class-dot ground" /> Ground
                  </span>
                  <span>
                    <i className="class-dot water" /> Water
                  </span>
                  <span>
                    <i className="class-dot vegetation" /> Low vegetation
                  </span>
                </div>
              )}
            </div>
            <div className="control-section">
              <div className="label-row">
                <label>Vertical exaggeration</label>
                <span>
                  {exaggeration.toFixed(1)}×
                  {shown.metricHeights && Math.abs(exaggeration - 1) < 0.05 ? " · true scale" : ""}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="3"
                step=".1"
                value={exaggeration}
                onChange={(e) => setExaggeration(+e.target.value)}
              />
              <div className="range-labels">
                <span>subtle</span>
                <span>dramatic</span>
              </div>
            </div>
            <div className="profile">
              <div className="label-row">
                <label>
                  <Activity size={14} /> Height profile
                </label>
                <button
                  className="text-btn"
                  onClick={() => {
                    setProfileCleared(true);
                    setProfilePoints([]);
                    notify("Profile cleared");
                  }}
                >
                  Clear
                </button>
              </div>
              <div className={`sparkline ${profileCleared ? "cleared" : ""}`}>
                <svg viewBox="0 0 300 70" preserveAspectRatio="none">
                  {profileView && (
                    <>
                      <path
                        d={profileView.line}
                        fill="none"
                        stroke="#8bd2c4"
                        strokeWidth="1.5"
                        vectorEffect="non-scaling-stroke"
                      />
                      <path
                        d={profileView.area}
                        fill="url(#fill)"
                        opacity=".22"
                      />
                    </>
                  )}
                  <defs>
                    <linearGradient id="fill" x1="0" x2="0" y1="0" y2="1">
                      <stop stopColor="#8bd2c4" />
                      <stop offset="1" stopColor="#8bd2c4" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <div className="profile-values">
                {profileView ? (
                  <>
                    <span>{profileView.fmt(profileView.min)}</span>
                    <strong>
                      {profileView.fmt(profileView.avg)} avg ·{" "}
                      {profile?.label ?? "centre line (Measure: double-click A, then B)"}
                    </strong>
                    <span>{profileView.fmt(profileView.max)}</span>
                  </>
                ) : (
                  <strong>{sample ? "loading…" : "—"}</strong>
                )}
              </div>
            </div>
            <div className="control-section waypoint-panel">
              <div className="label-row">
                <label>Camera route</label>
                <span>
                  {selectedWaypoint !== null
                    ? `Point ${selectedWaypoint + 1} selected`
                    : `${waypointCount} points`}
                </span>
              </div>
              <p>
                Set points on the terrain. Click a point to reselect it, drag
                its arrows to reposition, or press Delete to remove it.
              </p>
              <div className="waypoint-actions">
                <button
                  className={waypointMode ? "tool-active" : ""}
                  onClick={() => {
                    const next = !waypointMode;
                    setWaypointMode(next);
                    if (next) setAutoRotate(false);
                    notify(
                      next ? "Click terrain to add points" : "Point mode off",
                    );
                  }}
                >
                  <Target size={14} />{" "}
                  {waypointMode ? "Point mode on" : "Set points"}
                </button>
                <button
                  onClick={togglePathPlayback}
                  disabled={waypointCount < 2}
                >
                  {playing ? <Pause size={14} /> : <Play size={14} />}
                  {playing ? "Pause" : "Play"}
                </button>
                <button
                  onClick={removeSelectedWaypoint}
                  disabled={selectedWaypoint === null}
                >
                  <Trash2 size={14} /> Remove point
                </button>
                <button onClick={clearPath} disabled={!waypointCount}>
                  <X size={14} /> Clear
                </button>
              </div>
            </div>
            <div className="tool-row">
              <button
                className={autoRotate ? "tool-active" : ""}
                onClick={toggleAutoRotate}
                disabled={playing}
              >
                <RotateCw size={15} />{" "}
                {autoRotate ? "Rotating…" : "Auto-rotate"}
              </button>
              <button
                className={measure ? "tool-active" : ""}
                onClick={() => {
                  setMeasure(!measure);
                  notify(
                    measure
                      ? "Measure mode off"
                      : "Measure mode on — double-click terrain",
                  );
                }}
              >
                <Ruler size={15} /> Measure
              </button>
              <button
                onClick={() => {
                  setResetToken((x) => x + 1);
                  notify("Camera reset");
                }}
              >
                <RotateCcw size={15} /> Reset view
              </button>
            </div>
          </aside>
        </section>
      </main>
      {toast && <div className="toast">{toast}</div>}
      {measure && (
        <div className="measure-hint">
          <Ruler size={15} />{" "}
          {!measurePoint
            ? "Double-click the terrain to read height and slope"
            : measurePoint.metric
              ? `Height ${measurePoint.height.toFixed(1)} m above ground${measurePoint.elevation != null ? ` · elevation ${measurePoint.elevation.toFixed(1)} m (${geoid})` : ""} · slope ${measurePoint.slopeDeg.toFixed(0)}° · pixel (${measurePoint.col}, ${measurePoint.row})`
              : `Relative height ${(measurePoint.height * 100).toFixed(0)}% · preview tile, upload imagery for metric heights`}{" "}
          <X
            size={14}
            onClick={() => {
              setMeasure(false);
              setMeasurePoint(null);
            }}
          />
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
