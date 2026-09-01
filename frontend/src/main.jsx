import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
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
  Ruler,
  Satellite,
  Settings2,
  SlidersHorizontal,
  Target,
  X,
  AlertTriangle,
  BarChart3,
} from "lucide-react";
import "./styles.css";
import "./light.css";
import "./enhancements.css";
import "./blender.css";

const N = 512,
  S = N + 1;
const EMPTY = { mae: null, mse: null, rmse: null, bias: null };
const fallbackPairs = [
  ["train", "DC_01_25"],
  ["train", "DC_02_24"],
  ["train", "DC_02_25"],
  ["train", "DC_02_27"],
  ["train", "DC_03_23"],
  ["train", "DC_10_17"],
  ["train", "DC_10_18"],
  ["train", "DC_10_19"],
  ["train", "DC_10_21"],
  ["train", "DC_10_27"],
  ["val", "DC_02_26"],
  ["val", "DC_04_23"],
  ["val", "DC_04_27"],
  ["val", "DC_08_31"],
  ["val", "DC_09_33"],
  ["val", "DC_20_13"],
  ["val", "DC_20_14"],
  ["val", "DC_20_18"],
  ["val", "DC_20_19"],
  ["val", "DC_20_29"],
  ["test", "DC_03_26"],
  ["test", "DC_05_28"],
  ["test", "DC_05_30"],
  ["test", "DC_07_21"],
  ["test", "DC_07_29"],
  ["test", "DC_20_12"],
  ["test", "DC_20_15"],
  ["test", "DC_20_20"],
  ["test", "DC_20_23"],
  ["test", "DC_20_25"],
];
const fallback = fallbackPairs.map(([split, id]) => ({
  id,
  split,
  label: `${split} scene ${id}`,
  previewOnly: true,
  localGrid: true,
  crs: null,
  metrics: EMPTY,
  layers: {
    rgb: `/${split}-${id}-rgb.jpg`,
    depth: `/${split}-${id}-depth.jpg`,
    predictedHeight: `/${split}-${id}-height.jpg`,
    referenceHeight: `/${split}-${id}-height.jpg`,
    errorHeatmap: null,
  },
  geometry: {
    predicted: `/${split}-${id}-height.jpg`,
    reference: `/${split}-${id}-height.jpg`,
    error: `/${split}-${id}-height.jpg`,
  },
}));
function sceneOf(input) {
  const l = input.layers || {},
    g = input.geometry || {},
    reference = g.reference || l.referenceHeight || input.height,
    predicted = g.predicted || l.predictedHeight || reference;
  return {
    ...input,
    label: input.label || input.id,
    split: input.split || "unknown",
    localGrid: input.localGrid !== false,
    crs: input.crs || null,
    previewOnly: Boolean(input.previewOnly || !g.predicted || !input.metrics),
    layers: {
      rgb: l.rgb || input.rgb,
      depth: l.depth || input.depth,
      predictedHeight: l.predictedHeight || predicted,
      referenceHeight: l.referenceHeight || reference,
      errorHeatmap: l.errorHeatmap || l.error || null,
    },
    geometry: { predicted, reference, error: g.error || predicted },
    metrics: { ...EMPTY, ...(input.metrics || {}) },
    classMetrics: input.classMetrics || {},
  };
}
function metric(v, unit) {
  return Number.isFinite(Number(v)) ? `${Number(v).toFixed(2)} ${unit}` : "—";
}
function median(values, width, height) {
  const out = new Float32Array(values.length),
    win = [];
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      win.length = 0;
      for (let yy = Math.max(0, y - 2); yy <= Math.min(height - 1, y + 2); yy++)
        for (
          let xx = Math.max(0, x - 2);
          xx <= Math.min(width - 1, x + 2);
          xx++
        )
          win.push(values[yy * width + xx]);
      win.sort((a, b) => a - b);
      out[y * width + x] = win[Math.floor(win.length / 2)];
    }
  return out;
}
function decode(image, encoding = "grayscale8") {
  const canvas = document.createElement("canvas");
  canvas.width = S;
  canvas.height = S;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data,
    values = new Float32Array(S * S);
  for (let i = 0; i < values.length; i++) {
    values[i] = encoding === "rg16-linear"
      ? (px[i * 4] * 256 + px[i * 4 + 1]) / 65535
      : px[i * 4] / 255;
  }
  return median(values, S, S);
}

function Terrain({
  data,
  mode,
  layer,
  exaggeration,
  reset,
  waypointsEnabled,
  command,
  onWaypoints,
  onPathEnd,
  onMeasure,
}) {
  const host = useRef(null),
    state = useRef({}),
    keys = useRef({}),
    enabled = useRef(waypointsEnabled),
    ended = useRef(onPathEnd);
  useEffect(() => {
    enabled.current = waypointsEnabled;
  }, [waypointsEnabled]);
  useEffect(() => {
    ended.current = onPathEnd;
  }, [onPathEnd]);
  useEffect(() => {
    const root = host.current,
      scene = new THREE.Scene();
    scene.background = new THREE.Color("#07111e");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 3.4, 5.6);
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.25));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    root.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    const transform = new TransformControls(camera, renderer.domElement);
    transform.setMode("translate");
    transform.setSize(0.72);
    scene.add(transform.getHelper());
    transform.addEventListener("dragging-changed", (e) => {
      controls.enabled = !e.value;
    });
    scene.add(new THREE.HemisphereLight(0x9bc9d0, 0x172333, 1.7));
    const sun = new THREE.DirectionalLight(0xf7e4ba, 2.5);
    sun.position.set(-3, 6, 4);
    scene.add(sun);
    const grid = new THREE.GridHelper(9, 18, 0x315263, 0x1b3040);
    grid.position.y = -0.55;
    scene.add(grid);
    const geometry = new THREE.PlaneGeometry(8, 8, N, N);
    geometry.rotateX(-Math.PI / 2);
    const material = new THREE.MeshStandardMaterial({
      roughness: 0.86,
      metalness: 0.02,
      side: THREE.FrontSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    const loader = new THREE.TextureLoader();
    const markerGroup = new THREE.Group();
    scene.add(markerGroup);
    state.current = {
      scene,
      camera,
      controls,
      transform,
      geometry,
      material,
      mesh,
      loader,
      markerGroup,
      waypoints: [],
      line: null,
      heights: null,
      exaggeration,
      dirty: true,
      playing: false,
      routeYaw: 0,
      routePitch: 0,
      pos: new THREE.Vector3(),
      look: new THREE.Vector3(),
      direction: new THREE.Vector3(),
      pitchAxis: new THREE.Vector3(),
    };
    const ray = new THREE.Raycaster(),
      pointer = new THREE.Vector2(),
      point = new THREE.Vector3();
    const heightAt = (p) =>
      state.current.heights
        ? state.current.heights[
            Math.min(N, Math.max(0, Math.round(((4 - p.z) / 8) * N))) * S +
              Math.min(N, Math.max(0, Math.round(((p.x + 4) / 8) * N)))
          ]
        : 0;
    const redraw = () => {
      if (state.current.line) {
        scene.remove(state.current.line);
        state.current.line.geometry.dispose();
        state.current.line.material.dispose();
      }
      state.current.line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(
          state.current.waypoints.map((p) =>
            p.clone().add(new THREE.Vector3(0, 0.085, 0)),
          ),
        ),
        new THREE.LineBasicMaterial({ color: 0xffb55e }),
      );
      scene.add(state.current.line);
      state.current.dirty = true;
    };
    const click = (e) => {
      if (state.current.playing || transform.dragging || !enabled.current)
        return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      ray.setFromCamera(pointer, camera);
      if (
        !ray.ray.intersectPlane(
          new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
          point,
        ) ||
        Math.abs(point.x) > 4 ||
        Math.abs(point.z) > 4
      )
        return;
      point.y = heightAt(point);
      state.current.waypoints.push(point.clone());
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.09, 18, 14),
        new THREE.MeshBasicMaterial({ color: 0xff9f43 }),
      );
      marker.position.copy(point).add(new THREE.Vector3(0, 0.08, 0));
      markerGroup.add(marker);
      transform.attach(marker);
      redraw();
      onWaypoints(state.current.waypoints.length);
    };
    renderer.domElement.addEventListener("click", click);
    const resize = () => {
      const w = root.clientWidth,
        h = root.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener("resize", resize);
    const down = (e) => {
      const k = e.key.toLowerCase();
      if (
        !["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) &&
        "wasdqe".includes(k)
      ) {
        keys.current[k] = true;
        e.preventDefault();
        controls.enabled = false;
      }
    };
    const up = (e) => {
      delete keys.current[e.key.toLowerCase()];
      if (!Object.keys(keys.current).length) controls.enabled = true;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    const velocity = new THREE.Vector3(),
      movement = new THREE.Vector3(),
      forward = new THREE.Vector3(),
      right = new THREE.Vector3(),
      desired = new THREE.Vector3(),
      step = new THREE.Vector3(),
      upVector = new THREE.Vector3(0, 1, 0),
      clock = new THREE.Clock();
    let raf,
      first = true;
    const loop = () => {
      const dt = Math.min(clock.getDelta(), 0.05);
      movement.set(0, 0, 0);
      camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();
      right.crossVectors(forward, camera.up).normalize();
      if (keys.current.w) movement.add(forward);
      if (keys.current.s) movement.sub(forward);
      if (keys.current.d) movement.add(right);
      if (keys.current.a) movement.sub(right);
      if (keys.current.e) movement.y += 1;
      if (keys.current.q) movement.y -= 1;
      desired.copy(movement);
      if (desired.lengthSq()) desired.normalize().multiplyScalar(1.35);
      else desired.set(0, 0, 0);
      velocity.lerp(desired, 1 - Math.exp(-6 * dt));
      if (velocity.lengthSq() > 1e-6) {
        step.copy(velocity).multiplyScalar(dt);
        camera.position.add(step);
        controls.target.add(step);
      }
      let route = false;
      if (state.current.playing && state.current.curve) {
        route = true;
        state.current.elapsed += dt;
        const t = Math.min(state.current.elapsed / state.current.duration, 1);
        state.current.curve.getPoint(t, state.current.pos);
        state.current.curve.getPoint(
          Math.min(t + 0.012, 1),
          state.current.look,
        );
        state.current.pos.y += 0.65;
        state.current.look.y += 0.42;
        camera.position.copy(state.current.pos);
        const direction = state.current.direction
          .copy(state.current.look)
          .sub(camera.position)
          .normalize()
          .applyAxisAngle(upVector, state.current.routeYaw);
        state.current.pitchAxis.crossVectors(direction, upVector).normalize();
        direction.applyAxisAngle(
          state.current.pitchAxis,
          state.current.routePitch,
        );
        controls.target.copy(camera.position).add(direction);
        if (t >= 1) {
          state.current.playing = false;
          controls.enabled = true;
          ended.current?.();
        }
      }
      const changed = controls.update();
      if (
        first ||
        state.current.dirty ||
        changed ||
        route ||
        velocity.lengthSq() > 1e-6
      ) {
        renderer.render(scene, camera);
        first = false;
        state.current.dirty = false;
      }
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("click", click);
      window.removeEventListener("resize", resize);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      controls.dispose();
      transform.dispose();
      geometry.dispose();
      material.map?.dispose();
      material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === root)
        root.removeChild(renderer.domElement);
    };
  }, [data.id]);
  useEffect(() => {
    const s = state.current,
      url = data.geometry[mode] || data.geometry.predicted;
    if (!s.mesh || !url) return;
    const image = new Image();
    image.onload = () => {
      s.heights = decode(image, data.heightEncoding || "grayscale8");
      const p = s.geometry.attributes.position;
      for (let i = 0; i < p.count; i++)
        p.setY(i, (s.heights[i] - 0.32) * 1.5 * s.exaggeration);
      p.needsUpdate = true;
      s.geometry.computeVertexNormals();
      s.dirty = true;
    };
    image.onerror = () => {
      s.heights = new Float32Array(S * S);
      s.dirty = true;
    };
    image.src = url;
  }, [data, mode]);
  useEffect(() => {
    const s = state.current,
      url =
        layer === "texture"
          ? data.layers.rgb
          : layer === "depth"
            ? data.layers.depth
            : layer === "error"
              ? data.layers.errorHeatmap
              : layer === "classes"
                ? data.layers.classes
              : mode === "reference"
                ? data.layers.referenceHeight
                : data.layers.predictedHeight;
    if (!s.mesh) return;
    if (!url) {
      s.material.map = null;
      s.material.color.set(mode === "error" ? "#db755e" : "#5f8f88");
      s.material.needsUpdate = true;
      s.dirty = true;
      return;
    }
    const texture = s.loader.load(url, () => {
      s.dirty = true;
    });
    texture.colorSpace = THREE.SRGBColorSpace;
    const previous = s.material.map;
    s.material.map = texture;
    s.material.color.set("#fff");
    s.material.needsUpdate = true;
    if (previous) previous.dispose();
    s.dirty = true;
  }, [data, mode, layer]);
  useEffect(() => {
    const s = state.current;
    if (!s.heights) return;
    const p = s.geometry.attributes.position;
    for (let i = 0; i < p.count; i++)
      p.setY(i, (s.heights[i] - 0.32) * 1.5 * exaggeration);
    p.needsUpdate = true;
    s.geometry.computeVertexNormals();
    s.exaggeration = exaggeration;
    s.dirty = true;
  }, [exaggeration]);
  useEffect(() => {
    const s = state.current;
    if (s.camera) {
      s.camera.position.set(0, 3.4, 5.6);
      s.controls.target.set(0, 0, 0);
      s.controls.update();
    }
  }, [reset]);
  useEffect(() => {
    const s = state.current;
    if (!s.mesh || !command) return;
    if (command.type === "clear") {
      s.transform.detach();
      s.waypoints.length = 0;
      while (s.markerGroup.children.length) {
        const m = s.markerGroup.children.pop();
        m.geometry.dispose();
        m.material.dispose();
      }
      if (s.line) {
        s.scene.remove(s.line);
        s.line.geometry.dispose();
        s.line.material.dispose();
        s.line = null;
      }
      onWaypoints(0);
    }
    if (command.type === "pause") {
      s.playing = false;
      s.controls.enabled = true;
    }
    if (command.type === "play" && s.waypoints.length >= 2) {
      s.curve = new THREE.CatmullRomCurve3(
        s.waypoints.map((p) => p.clone()),
        false,
        "catmullrom",
        0.45,
      );
      s.elapsed = 0;
      s.duration = Math.max(7, s.waypoints.length * 2.8);
      s.playing = true;
      s.transform.detach();
      s.controls.enabled = false;
    }
  }, [command]);
  return (
    <div
      ref={host}
      onDoubleClick={onMeasure}
      className={`terrain-canvas ${waypointsEnabled ? "waypoint-active" : ""}`}
    />
  );
}

function App() {
  const [scenes, setScenes] = useState([]),
    [manifest, setManifest] = useState("loading"),
    [split, setSplit] = useState("all"),
    [sample, setSample] = useState(null),
    [mode, setMode] = useState("predicted"),
    [layer, setLayer] = useState("texture"),
    [exaggeration, setExaggeration] = useState(0.5),
    [playing, setPlaying] = useState(false),
    [measure, setMeasure] = useState(false),
    [toast, setToast] = useState(""),
    [theme, setTheme] = useState("dark"),
    [reset, setReset] = useState(0),
    [cleared, setCleared] = useState(false),
    [waypoints, setWaypoints] = useState(false),
    [count, setCount] = useState(0),
    [command, setCommand] = useState(null);
  const notify = (text) => {
    setToast(text);
    window.setTimeout(() => setToast(""), 1800);
  };
  useEffect(() => {
    fetch("/gamus/scenes.json")
      .then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      })
      .then((j) => {
        const list = Array.isArray(j) ? j : j.scenes;
        const ready = (list || []).map(sceneOf);
        setScenes(ready);
        setSample(ready[0] || null);
        setManifest(ready.length ? "ready" : "empty");
      })
      .catch(() => {
        setScenes(fallback);
        setSample(fallback[0]);
        setManifest("fallback");
      });
  }, []);
  const filtered = useMemo(
    () => (split === "all" ? scenes : scenes.filter((s) => s.split === split)),
    [scenes, split],
  );
  useEffect(() => {
    if (sample && !filtered.some((s) => s.id === sample.id))
      setSample(filtered[0] || null);
  }, [filtered, sample]);
  const index = Math.max(
    0,
    filtered.findIndex((s) => s.id === sample?.id),
  );
  const choose = (s) => {
    if (!s) return;
    setSample(s);
    setCleared(false);
    setCount(0);
    setPlaying(false);
    setCommand({ type: "clear", id: Date.now() });
    notify(`Loaded ${s.id}`);
  };
  const change = (n) =>
    filtered.length &&
    choose(filtered[(index + n + filtered.length) % filtered.length]);
  const togglePath = () => {
    if (playing) {
      setCommand({ type: "pause", id: Date.now() });
      setPlaying(false);
    } else if (count < 2) notify("Set at least two waypoints first");
    else {
      setCommand({ type: "play", id: Date.now() });
      setPlaying(true);
      setWaypoints(false);
    }
  };
  const clearPath = () => {
    setCommand({ type: "clear", id: Date.now() });
    setCount(0);
    setPlaying(false);
    notify("Waypoints cleared");
  };
  const exportScene = () => {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            scene: sample.id,
            split: sample.split,
            mode,
            layer,
            metrics: sample.metrics,
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${sample.id}-terrain-scene.json`;
    a.click();
    notify("Scene manifest downloaded");
  };
  if (manifest === "loading")
    return (
      <div className="app loading-state">Loading GAMUS scene manifest…</div>
    );
  if (!sample)
    return (
      <div className="app empty-state">
        <AlertTriangle size={30} />
        <h1>No GAMUS scenes available</h1>
        <p>
          Add <code>frontend/public/gamus/scenes.json</code> with browser-ready
          assets.
        </p>
      </div>
    );
  const m = sample.metrics || EMPTY;
  const classMetrics = Object.entries(sample.classMetrics || {});
  return (
    <div
      className={`app theme-${theme} ${mode === "error" ? "error-mode" : ""}`}
    >
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
          <button onClick={() => notify("Local-grid gallery")}>
            <Map size={18} />
          </button>
          <button onClick={() => notify("Metrics are read from the manifest")}>
            <Activity size={18} />
          </button>
        </div>
        <div className="rail-bottom">
          <button
            onClick={() => {
              setTheme(theme === "dark" ? "light" : "dark");
              notify("Theme changed");
            }}
          >
            <Settings2 size={18} />
          </button>
          <div className="avatar">MB</div>
        </div>
      </aside>
      <main>
        <section className="workspace">
          <div className="viewport-card">
            <div className="viewport-top">
              <div className="scene-title">
                <span className="scene-chip">
                  <Satellite size={14} />
                </span>
                <div>
                  <strong>{sample.label}</strong>
                  <small>
                    {sample.id} ·{" "}
                    {sample.localGrid ? "Local grid · no CRS" : sample.crs}
                  </small>
                </div>
              </div>
              <div className="view-actions">
                <button onClick={exportScene}>
                  <Download size={14} /> Export
                </button>
                <button onClick={togglePath}>
                  {playing ? <Pause size={15} /> : <Play size={15} />}{" "}
                  {playing ? "Pause path" : "Play route"}
                </button>
                <button
                  onClick={() =>
                    document.querySelector(".workspace")?.requestFullscreen?.()
                  }
                >
                  <Maximize2 size={15} />
                </button>
              </div>
            </div>
            <div className="canvas-wrap">
              <Terrain
                data={sample}
                mode={mode}
                layer={layer}
                exaggeration={exaggeration}
                reset={reset}
                waypointsEnabled={waypoints}
                command={command}
                onWaypoints={setCount}
                onPathEnd={() => setPlaying(false)}
                onMeasure={() =>
                  measure &&
                  notify("Numeric probe unavailable for preview assets")
                }
              />
              <div className="flight-help">
                <kbd>W</kbd>
                <kbd>A</kbd>
                <kbd>S</kbd>
                <kbd>D</kbd> move · <kbd>Q</kbd>
                <kbd>E</kbd> altitude · drag to look on route
              </div>
              <div className="canvas-badge">
                <span className="pulse" />{" "}
                {sample.previewOnly ? "PREVIEW ONLY" : "GAMUS RUN"}
                <i />{" "}
                {mode === "predicted"
                  ? "Predicted 3D"
                  : mode === "reference"
                    ? "Reference 3D"
                    : "Error 3D"}
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
                  <b className="swatch low" /> Low
                </span>
                <span>
                  <b className="swatch mid" /> Mid
                </span>
                <span>
                  <b className="swatch high" /> High
                </span>
              </div>
              <div className="footer-note">
                <Eye size={14} /> 512 × 512 interactive mesh ·{" "}
                {sample.localGrid
                  ? "local grid, no global coordinates"
                  : sample.crs}
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
                onClick={() => notify("Manifest-driven scene metadata")}
              >
                <SlidersHorizontal size={16} />
              </button>
            </div>
            {sample.previewOnly && (
              <div className="preview-notice">
                <AlertTriangle size={14} />
                <span>
                  Preview-only assets. Prediction metrics are not available.
                </span>
              </div>
            )}
            <div className="metric-grid">
              <div>
                <small>MAE</small>
                <strong>{metric(m.mae, "m")}</strong>
                <em>{m.mae == null ? "not reported" : "manifest value"}</em>
              </div>
              <div>
                <small>MSE</small>
                <strong>{metric(m.mse, "m²")}</strong>
                <em>{m.mse == null ? "not reported" : "manifest value"}</em>
              </div>
              <div>
                <small>RMSE</small>
                <strong>{metric(m.rmse, "m")}</strong>
                <em>{m.rmse == null ? "not reported" : "manifest value"}</em>
              </div>
              <div>
                <small>Bias</small>
                <strong>{metric(m.bias, "m")}</strong>
                <em>{m.bias == null ? "not reported" : "manifest value"}</em>
              </div>
            </div>
            {classMetrics.length > 0 && (
              <div className="class-metrics">
                {classMetrics.map(([name, values]) => (
                  <div key={name}>
                    <span>{name}</span>
                    <strong>{metric(values?.mae ?? values, "m")}</strong>
                  </div>
                ))}
              </div>
            )}
            <div className="control-section">
              <label>Geometry mode</label>
              <div className="segmented mode-tabs">
                <button
                  className={mode === "predicted" ? "selected" : ""}
                  onClick={() => setMode("predicted")}
                >
                  <BarChart3 size={13} /> Predicted
                </button>
                <button
                  className={mode === "reference" ? "selected" : ""}
                  onClick={() => setMode("reference")}
                >
                  <Mountain size={13} /> Reference
                </button>
                <button
                  className={mode === "error" ? "selected" : ""}
                  onClick={() => setMode("error")}
                >
                  <Activity size={13} /> Error
                </button>
              </div>
              <small className="mode-description">
                {mode === "error"
                  ? "Predicted geometry coloured by absolute error."
                  : mode === "reference"
                    ? "GAMUS AGL reference surface."
                    : "Prediction height surface."}
              </small>
            </div>
            <div className="control-section">
              <label>Active layer</label>
              <div className="segmented layer-tabs">
                <button
                  className={layer === "surface" ? "selected" : ""}
                  onClick={() => setLayer("surface")}
                >
                  <Mountain size={14} /> Surface
                </button>
                <button
                  className={layer === "depth" ? "selected" : ""}
                  onClick={() => setLayer("depth")}
                >
                  <Activity size={14} /> Depth
                </button>
                <button
                  className={layer === "texture" ? "selected" : ""}
                  onClick={() => setLayer("texture")}
                >
                  <FileImage size={14} /> RGB
                </button>
                <button
                  className={layer === "error" ? "selected" : ""}
                  onClick={() => setLayer("error")}
                >
                  <AlertTriangle size={14} /> Error
                </button>
                <button
                  className={layer === "classes" ? "selected" : ""}
                  onClick={() => setLayer("classes")}
                >
                  <Layers3 size={14} /> Classes
                </button>
              </div>
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
                    setCleared(true);
                    notify("Profile cleared");
                  }}
                >
                  Clear
                </button>
              </div>
              <div className={`sparkline ${cleared ? "cleared" : ""}`}>
                <svg viewBox="0 0 300 70" preserveAspectRatio="none">
                  <path
                    d="M0,60 C12,45 22,51 33,42 S54,50 66,38 S86,46 99,27 S119,36 133,30 S150,46 164,19 S184,32 199,23 S214,30 230,14 S248,29 263,19 S281,20 300,5"
                    fill="none"
                    stroke="#8bd2c4"
                    strokeWidth="2"
                  />
                </svg>
              </div>
              <div className="profile-values">
                <span>0 m</span>
                <strong>
                  {m.mae == null ? "profile unavailable" : metric(m.mae, "MAE")}
                </strong>
                <span>—</span>
              </div>
            </div>
            <div className="control-section scene-switcher">
              <div className="label-row">
                <label>Scene gallery</label>
                <span>
                  {index + 1} / {filtered.length}
                </span>
              </div>
              <div className="scene-preview">
                <img src={sample.layers.rgb} alt={sample.label} />
                <div>
                  <strong>{sample.id}</strong>
                  <small>{sample.label}</small>
                </div>
              </div>
              <select
                value={sample.id}
                onChange={(e) =>
                  choose(filtered.find((s) => s.id === e.target.value))
                }
              >
                {filtered.map((s) => (
                  <option key={`${s.split}-${s.id}`} value={s.id}>
                    {s.id} — {s.label}
                  </option>
                ))}
              </select>
              <div className="photo-nav">
                <button onClick={() => change(-1)}>← Previous</button>
                <button onClick={() => change(1)}>Next →</button>
              </div>
            </div>
            <div className="control-section split-filter">
              <div className="label-row">
                <label>Dataset split</label>
                <span>
                  {manifest === "fallback"
                    ? "development fallback"
                    : "manifest"}
                </span>
              </div>
              <div className="segmented">
                <button
                  className={split === "all" ? "selected" : ""}
                  onClick={() => setSplit("all")}
                >
                  All
                </button>
                <button
                  className={split === "train" ? "selected" : ""}
                  onClick={() => setSplit("train")}
                >
                  Train
                </button>
                <button
                  className={split === "val" ? "selected" : ""}
                  onClick={() => setSplit("val")}
                >
                  Val
                </button>
                <button
                  className={split === "test" ? "selected" : ""}
                  onClick={() => setSplit("test")}
                >
                  Test
                </button>
              </div>
            </div>
            <div className="control-section waypoint-panel">
              <div className="label-row">
                <label>Camera route</label>
                <span>{count} points</span>
              </div>
              <p>
                Set points on the terrain. Click a marker to reveal its X/Y/Z
                move arrows.
              </p>
              <div className="waypoint-actions">
                <button
                  className={waypoints ? "tool-active" : ""}
                  onClick={() => {
                    setWaypoints(!waypoints);
                    notify(
                      waypoints
                        ? "Point mode off"
                        : "Click terrain to add points",
                    );
                  }}
                >
                  <Target size={14} />{" "}
                  {waypoints ? "Point mode on" : "Set points"}
                </button>
                <button onClick={togglePath} disabled={count < 2}>
                  {playing ? <Pause size={14} /> : <Play size={14} />}{" "}
                  {playing ? "Pause" : "Play"}
                </button>
                <button onClick={clearPath} disabled={!count}>
                  <X size={14} /> Clear
                </button>
              </div>
            </div>
            <div className="tool-row">
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
                  setReset((v) => v + 1);
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
