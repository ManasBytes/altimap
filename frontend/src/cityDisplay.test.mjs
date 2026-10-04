import test from "node:test";
import assert from "node:assert/strict";
import { buildingContains, flatCity, roofElevation } from "./cityDisplay.js";

test("flat ground retains object heights, places facilities on roofs or ground, and preserves source", () => {
  const sample = {
    city: {
      buildings: [{ h: 30, b: 8, t: 40, rings: [
        [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
      ] }],
      trees: [{ u: 0.95, v: 0.95, b: 10, h: 12 }],
    },
    facilities: [
      { kind: "hospital", u: 0.2, v: 0.2, z: 40 },
      { kind: "school", u: 0.5, v: 0.5, z: 10 }, // courtyard
      { kind: "clinic", u: 0.95, v: 0.05, z: 10 },
    ],
    embankments: [{ crest_m: 22 }],
    terrain: { ground: { min_m: 100 } },
    groundGrid: new Float32Array(9).fill(110),
  };
  const original = structuredClone(sample);
  const flat = flatCity(sample);
  assert.equal(flat.buildings[0].b, 0);
  assert.equal(flat.buildings[0].t, 30);
  assert.equal(flat.buildings[0].h, 30);
  assert.deepEqual(flat.facilities.map((f) => f.z), [30, 0, 0]);
  assert.equal(flat.trees[0].b, 0);
  assert.equal(flat.trees[0].h, 12);
  assert.deepEqual(flat.embankments, []);
  assert.deepEqual(sample, original);
});

test("bridge slab stays raised above its local ground in flat view", () => {
  const sample = {
    city: { trees: [], buildings: [{ kind: "bridge", h: 1.5, b: 18.5, t: 20,
      rings: [[[0, 0], [1, 0], [1, 1], [0, 1]]] }] },
    terrain: { ground: { min_m: 100 } },
    groundGrid: new Float32Array(9).fill(110),
  };
  const [bridge] = flatCity(sample).buildings;
  assert.equal(bridge.b, 8.5);
  assert.equal(bridge.t, 10);
  assert.equal(bridge.h, 1.5);
});

test("mapped roofs preserve their shape and raised bases on flat ground; markers follow the roof", () => {
  const building = { h: 30, b: 18, t: 40, min_height_m: 8,
    rings: [[[0, 0], [1, 0], [1, 1], [0, 1]]],
    roof: { vertices: [[0, 0, 30], [1, 0, 30], [0.5, 1, 40]], faces: [[0, 1, 2]] } };
  const sample = { city: { buildings: [building], trees: [] },
    facilities: [{ u: 0.5, v: 0.25, z: 32.5 }] };
  const flat = flatCity(sample);
  assert.equal(flat.buildings[0].b, 8);
  assert.equal(flat.buildings[0].t, 30);
  assert.deepEqual(flat.buildings[0].roof.vertices.map((p) => p[2]), [20, 20, 30]);
  assert.equal(flat.facilities[0].z, 22.5);
  assert.equal(roofElevation(building, 0.5, 0.25), 32.5);
  assert.equal(building.roof.vertices[2][2], 40);
  assert.equal(buildingContains(building, 0.2, 0.2), true);
});
