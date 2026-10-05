import test from "node:test";
import assert from "node:assert/strict";
import { demoCatalog, resultAssets } from "./hosting.js";

test("split-host uploads resolve downloads against the API, preserving data URIs and geometry", () => {
  const input = { rgb: "data:image/png;base64,abc", height: "/height.png", classes: "https://cdn.example/classes.png",
    downloads: { "ndsm.tif": "/data-uploads/scenes/one/ndsm.tif" }, city: { buildings: [] } };
  const original = structuredClone(input);
  const output = resultAssets(input, "https://api.example/api/estimate");
  assert.equal(output.rgb, input.rgb);
  assert.equal(output.height, "https://api.example/height.png");
  assert.equal(output.classes, input.classes);
  assert.equal(output.downloads["ndsm.tif"], "https://api.example/data-uploads/scenes/one/ndsm.tif");
  assert.equal(output.city, input.city);
  assert.deepEqual(input, original);
});

test("HF scene assets and catalog links stay independent of the VM", () => {
  const catalogUrl = "https://huggingface.co/datasets/Dilavesh/altimap-demo/resolve/main/index.json";
  const [scene] = demoCatalog({ scenes: [{ id: "one", label: "City", result: "one/result.json" }] }, catalogUrl);
  const input = { rgb: "rgb.png", terrain: { ground: { png: "ground.png", min_m: 100 }, dsm: { png: "dsm.png" } },
    error: { png: "error.png", limit_m: 10 }, downloads: { "ndsm.tif": "ndsm.tif" } };
  const output = resultAssets(input, scene.result);
  assert.equal(output.rgb, catalogUrl.replace("index.json", "one/rgb.png"));
  assert.equal(output.terrain.ground.png, catalogUrl.replace("index.json", "one/ground.png"));
  assert.equal(output.terrain.ground.min_m, 100);
  assert.equal(output.error.png, catalogUrl.replace("index.json", "one/error.png"));
  assert.equal(output.downloads["ndsm.tif"], catalogUrl.replace("index.json", "one/ndsm.tif"));
});
