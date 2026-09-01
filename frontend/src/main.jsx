import {
  Activity,
  Compass,
  Download,
  Eye,
  FileImage,
  Layers3,
  Map,
  Maximize2,
  Minus,
  Mountain,
  Pause,
  Play,
  Plus,
  RotateCcw,
  RotateCw,
  Ruler,
  Satellite,
  Settings2,
  SlidersHorizontal,
  Target,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
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
const displayedScenes = [...curatedScenes, ...gamusScenes].map((scene) => {
  const buildingCoverage =
    scene.buildingCoverage ??
    curatedBuildingCoverage[`${scene.split}:${scene.id}`];
  return {
    ...scene,
    rgb: scene.rgb || `/${scene.split}-${scene.id}-rgb.jpg`,
    height: scene.height || `/${scene.split}-${scene.id}-height.jpg`,
    depth: scene.depth || `/${scene.split}-${scene.id}-depth.jpg`,
    classes: scene.classes || `/${scene.split}-${scene.id}-classes.jpg`,
    thumb: scene.thumb || scene.rgb || `/${scene.split}-${scene.id}-rgb.jpg`,
    buildingCoverage,
    urban: scene.urban ?? (buildingCoverage != null && buildingCoverage >= 20),
  };
});
const initialSample =
  displayedScenes.find((scene) => scene.id === samples.train.id) ||
  displayedScenes[0];
const urbanScenes = displayedScenes.filter((scene) => scene.urban);
const otherScenes = displayedScenes.filter((scene) => !scene.urban);

// Terrain detail controls. Keep samples one larger than segments because a grid
// with N segments contains N + 1 vertices along that axis.
// 512 segments keeps more than half a million triangles while avoiding the
// million-vertex CPU/GPU cost of the source grid during interactive flight.
const MESH_SEGMENTS_X = 512;
const MESH_SEGMENTS_Y = 512;
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
const BUILDING_NORMALIZE_LOW = 0.08;
const BUILDING_NORMALIZE_HIGH = 0.94;

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

function limitHeightDiscontinuities(field, width, height) {
  const out = field.slice();
  const maxJump = 0.12;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      let neighborTotal = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx || dy) neighborTotal += field[(y + dy) * width + x + dx];
        }
      }
      const neighborMean = neighborTotal / 8;
      out[index] = Math.min(
        neighborMean + maxJump,
        Math.max(neighborMean - maxJump, field[index]),
      );
    }
  }
  return out;
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

function TerrainCanvas({
  sample,
  exaggeration,
  layer,
  resetToken,
  onMeasure,
  waypointMode,
  pathCommand,
  onWaypointChange,
  onWaypointSelect,
  onPathEnd,
  autoRotate,
}) {
  const ref = useRef(null);
  const state = useRef({});
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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
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
    const tex = new THREE.TextureLoader();
    const updateGeometry = () => {
      const current = state.current;
      if (!current.rawHeightField) return;
      const heightField = applyClassHeightScale(
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
      current.heightColors = buildHeightColors(heightField);
      current.exaggeration = exaggeration;
      pos.needsUpdate = true;
      geo.computeVertexNormals();
      if (layerRef.current === "depth") {
        geo.attributes.color.array.set(current.heightColors);
        geo.attributes.color.needsUpdate = true;
      }
      current.renderDirty = true;
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
      const heightField = buildHeightField(
        px,
        HEIGHT_SAMPLE_WIDTH,
        HEIGHT_SAMPLE_HEIGHT,
      );
      state.current.rawHeightField = heightField;
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
    const material = new THREE.MeshStandardMaterial({
      map: height,
      vertexColors: true,
      roughness: 0.86,
      metalness: 0.02,
      side: THREE.FrontSide,
      wireframe: false,
    });
    const mesh = new THREE.Mesh(geo, material);
    scene.add(mesh);
    const waypointGroup = new THREE.Group();
    scene.add(waypointGroup);
    state.current = {
      scene,
      camera,
      renderer,
      controls,
      mesh,
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
            Math.round(((4 - point.z) / 8) * (HEIGHT_SAMPLE_HEIGHT - 1)),
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
    renderer.domElement.addEventListener("click", addWaypoint);
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
      if (move.lengthSq()) desired.copy(move).normalize().multiplyScalar(1.35);
      else desired.set(0, 0, 0);
      velocity.lerp(desired, 1 - Math.exp(-6 * dt));
      const hasVelocity = velocity.lengthSq() > 0.000001;
      if (hasVelocity) {
        step.copy(velocity).multiplyScalar(dt);
        camera.position.add(step);
        controls.target.add(step);
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
      renderer.domElement.removeEventListener("click", addWaypoint);
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
    const colorAttr = s.mesh.geometry.attributes.color;
    if (layer === "depth") {
      // Dark blue-to-red heat map driven by the same smoothed height data
      // used to build the mesh, instead of loading the pre-baked depth.jpg.
      const previousMap = s.mesh.material.map;
      s.mesh.material.map = null;
      if (previousMap && previousMap !== s.height) previousMap.dispose();
      if (colorAttr && s.heightColors) {
        colorAttr.array.set(s.heightColors);
        colorAttr.needsUpdate = true;
      }
      s.mesh.material.needsUpdate = true;
      s.renderDirty = true;
      return;
    }
    if (colorAttr) {
      colorAttr.array.fill(1);
      colorAttr.needsUpdate = true;
    }
    const source =
      layer === "texture"
        ? sample.rgb
        : layer === "classes"
          ? sample.classes
          : sample.height;
    const previousMap = s.mesh.material.map;
    const nextMap = s.tex.load(source, () => {
      s.renderDirty = true;
    });
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
    state.current.exaggeration = exaggeration;
    state.current.renderDirty = true;
  }, [exaggeration]);
  useEffect(() => {
    const s = state.current;
    if (!s.camera || !s.controls) return;
    s.camera.position.set(0, 3.4, 5.6);
    s.controls.target.set(0, 0, 0);
    s.controls.update();
  }, [resetToken]);
  useEffect(() => {
    const s = state.current;
    if (!s.controls) return;
    s.controls.autoRotate = autoRotate;
    s.renderDirty = true;
  }, [autoRotate]);
  return (
    <div
      ref={ref}
      onDoubleClick={onMeasure}
      className={`terrain-canvas ${waypointMode ? "waypoint-active" : ""}`}
    />
  );
}

function App() {
  const [split, setSplit] = useState("train");
  const [sample, setSample] = useState(initialSample);
  const [layer, setLayer] = useState("texture");
  const [exaggeration, setExaggeration] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [measure, setMeasure] = useState(false);
  const [toast, setToast] = useState("");
  const [theme, setTheme] = useState("dark");
  const [resetToken, setResetToken] = useState(0);
  const [profileCleared, setProfileCleared] = useState(false);
  const [waypointMode, setWaypointMode] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [waypointCount, setWaypointCount] = useState(0);
  const [selectedWaypoint, setSelectedWaypoint] = useState(null);
  const [pathCommand, setPathCommand] = useState({ type: "idle", id: 0 });
  const fileRef = useRef(null);
  const notify = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 1800);
  };
  const selectScene = (s) => {
    setSplit(s.split);
    setSample(s);
    setProfileCleared(false);
    setWaypointCount(0);
    setSelectedWaypoint(null);
    setPlaying(false);
    notify(`Loaded ${s.id}`);
  };
  const sceneIndex = Math.max(
    0,
    displayedScenes.findIndex((scene) => scene.id === sample.id),
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
    if (f) notify(`Imported ${f.name} · preview ready`);
  };
  const exportScene = () => {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            format: "GAMUS Terrain Studio scene",
            scene: sample.id,
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
    a.download = `${sample.id}-terrain-scene.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify("Scene manifest downloaded");
  };
  return (
    <div className={`app theme-${theme}`}>
      <input
        ref={fileRef}
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,.geojson"
        onChange={importScene}
        hidden
      />
      <aside className="rail">
        <div className="brandmark">
          <Mountain size={18} />
        </div>
        <div className="rail-nav">
          <button
            className="active"
            onClick={() => notify("Terrain workspace active")}
          >
            <Layers3 size={18} />
          </button>
          <button onClick={() => notify("Map overview selected")}>
            <Map size={18} />
          </button>
          <button onClick={() => notify("Analytics panel selected")}>
            <Activity size={18} />
          </button>
        </div>
        <div className="rail-bottom">
          <button
            onClick={() => {
              setTheme(theme === "dark" ? "light" : "dark");
              notify(`${theme === "dark" ? "Light" : "Dark"} mode enabled`);
            }}
            title="Toggle light mode"
          >
            <Settings2 size={18} />
          </button>
          <div className="avatar">MB</div>
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
                  <strong>{sample.label}</strong>
                  <small>
                    {sample.id} · {sample.coord}
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
            <div className="canvas-wrap">
              <TerrainCanvas
                sample={sample}
                exaggeration={exaggeration}
                layer={layer}
                resetToken={resetToken}
                onMeasure={() => measure && notify("Point captured · 18.4 m")}
                waypointMode={waypointMode}
                pathCommand={pathCommand}
                onWaypointChange={setWaypointCount}
                onWaypointSelect={setSelectedWaypoint}
                onPathEnd={() => setPlaying(false)}
                autoRotate={autoRotate}
              />
              <div className="flight-help">
                <kbd>W</kbd>
                <kbd>A</kbd>
                <kbd>S</kbd>
                <kbd>D</kbd> move · <kbd>Q</kbd>
                <kbd>E</kbd> altitude · drag to look on route
              </div>
              <div className="canvas-badge">
                <span className="pulse" /> LIVE PREVIEW <i />{" "}
                {layer === "elevation"
                  ? "rDSM surface"
                  : layer === "depth"
                    ? "Height estimate"
                    : layer === "classes"
                      ? "Semantic classes"
                      : "RGB texture"}
              </div>
              <div className="compass">
                <Compass size={23} />
                <span>N</span>
              </div>
              <div className="canvas-controls">
                <button
                  onClick={() =>
                    setExaggeration(Math.max(0, exaggeration - 0.1))
                  }
                >
                  <Minus size={15} />
                </button>
                <span>{exaggeration.toFixed(1)}×</span>
                <button
                  onClick={() =>
                    setExaggeration(Math.min(3, exaggeration + 0.1))
                  }
                >
                  <Plus size={15} />
                </button>
              </div>
            </div>
            <div className="viewport-footer">
              <div className="legend">
                <span>
                  <b className="swatch low" /> Low · 0 m
                </span>
                <span>
                  <b className="swatch mid" /> Mid · 21 m
                </span>
                <span>
                  <b className="swatch high" /> High · {sample.max}
                </span>
              </div>
              <div className="footer-note">
                <Eye size={14} /> 1024 × 1024 source · 513² live mesh · 4
                aligned layers
              </div>
            </div>
          </div>
          <aside className="inspector">
            <div className="panel-heading">
              <div>
                <div className="eyebrow">Scene inspector</div>
                <h2>Reconstruction</h2>
              </div>
              <button
                className="icon-btn"
                onClick={() => notify("Inspector settings ready")}
              >
                <SlidersHorizontal size={16} />
              </button>
            </div>
            <div className="metric-grid">
              <div>
                <small>Surface max</small>
                <strong>{sample.max}</strong>
                <em>+ 4.8%</em>
              </div>
              <div>
                <small>Coverage</small>
                <strong>1.05 km²</strong>
                <em className="neutral">stable</em>
              </div>
            </div>
            <div className="control-section">
              <label>Active layer</label>
              <div className="segmented layer-tabs">
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
              </div>
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
                <span>{exaggeration.toFixed(1)}×</span>
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
                    notify("Profile cleared");
                  }}
                >
                  Clear
                </button>
              </div>
              <div className={`sparkline ${profileCleared ? "cleared" : ""}`}>
                <svg viewBox="0 0 300 70" preserveAspectRatio="none">
                  <path
                    d="M0,60 C12,45 22,51 33,42 S54,50 66,38 S86,46 99,27 S119,36 133,30 S150,46 164,19 S184,32 199,23 S214,30 230,14 S248,29 263,19 S281,20 300,5"
                    fill="none"
                    stroke="#8bd2c4"
                    strokeWidth="2"
                  />
                  <path
                    d="M0,60 C12,45 22,51 33,42 S54,50 66,38 S86,46 99,27 S119,36 133,30 S150,46 164,19 S184,32 199,23 S214,30 230,14 S248,29 263,19 S281,20 300,5 V70 H0Z"
                    fill="url(#fill)"
                    opacity=".22"
                  />
                  <defs>
                    <linearGradient id="fill" x1="0" x2="0" y1="0" y2="1">
                      <stop stopColor="#8bd2c4" />
                      <stop offset="1" stopColor="#8bd2c4" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <div className="profile-values">
                <span>0 m</span>
                <strong>24.6 m avg</strong>
                <span>42.8 m</span>
              </div>
            </div>
            <div className="control-section scene-switcher">
              <div className="label-row">
                <label>Scene</label>
                <span>
                  {sceneIndex + 1} / {displayedScenes.length} ·{" "}
                  {urbanScenes.length} urban
                </span>
              </div>
              <div className="scene-preview">
                <img src={sample.thumb || sample.rgb} alt={sample.label} />
                <div>
                  <strong>{sample.id}</strong>
                  <small>
                    {sample.urban ? "Building-rich · " : ""}
                    {sample.label}
                  </small>
                </div>
              </div>
              <select
                value={sample.id}
                onChange={(event) =>
                  selectScene(
                    displayedScenes.find(
                      (scene) => scene.id === event.target.value,
                    ),
                  )
                }
              >
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
          <Ruler size={15} /> Double-click the terrain to capture a point{" "}
          <X size={14} onClick={() => setMeasure(false)} />
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
