import React, { useEffect, useRef, useState } from "react";
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
  Satellite,
  Settings2,
  SlidersHorizontal,
  Target,
  Upload,
  X,
} from "lucide-react";
import "./styles.css";
import "./light.css";
import "./enhancements.css";
import "./blender.css";

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
const displayedScenes = [...sceneCatalog.slice(0, 3), ...extraCatalog];

// Terrain detail controls. Keep samples one larger than segments because a grid
// with N segments contains N + 1 vertices along that axis.
const MESH_SEGMENTS_X = 1023;
const MESH_SEGMENTS_Y = 1023;
const HEIGHT_SAMPLE_WIDTH = MESH_SEGMENTS_X + 1;
const HEIGHT_SAMPLE_HEIGHT = MESH_SEGMENTS_Y + 1;
const HEIGHT_WORLD_SCALE = 0.75;

function TerrainCanvas({
  sample,
  exaggeration,
  layer,
  resetToken,
  waypointMode,
  pathCommand,
  onWaypointChange,
  onPathEnd,
}) {
  const ref = useRef(null);
  const state = useRef({});
  const waypointModeRef = useRef(waypointMode);
  const pathEndRef = useRef(onPathEnd);
  const keys = useRef({});
  useEffect(() => {
    waypointModeRef.current = waypointMode;
  }, [waypointMode]);
  useEffect(() => {
    pathEndRef.current = onPathEnd;
  }, [onPathEnd]);
  useEffect(() => {
    const host = ref.current,
      scene = new THREE.Scene();
    scene.background = new THREE.Color("#07111e");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 3.4, 5.6);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
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
    const tex = new THREE.TextureLoader();
    const displacementSource = sample.height.replace(
      "-height.jpg",
      "-displacement.png",
    );
    const height = tex.load(displacementSource, (loaded) => {
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
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const v = px[i * 4] / 255;
        pos.setY(i, v * HEIGHT_WORLD_SCALE * exaggeration);
      }
      state.current.heightPixels = px;
      state.current.exaggeration = exaggeration;
      pos.needsUpdate = true;
      geo.computeVertexNormals();
    });
    const material = new THREE.MeshStandardMaterial({
      map: height,
      roughness: 0.86,
      metalness: 0.02,
      side: THREE.DoubleSide,
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
      waypointGroup,
      transform,
      waypoints: [],
      waypointLine: null,
      pathPlaying: false,
      routeYaw: 0,
      routePitch: 0,
    };
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const rebuildWaypointLine = () => {
      if (state.current.waypointLine) {
        scene.remove(state.current.waypointLine);
        state.current.waypointLine.geometry.dispose();
        state.current.waypointLine.material.dispose();
      }
      const linePoints = state.current.waypoints.map((p) =>
        p.clone().add(new THREE.Vector3(0, 0.085, 0)),
      );
      state.current.waypointLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(linePoints),
        new THREE.LineBasicMaterial({ color: 0xffb55e }),
      );
      scene.add(state.current.waypointLine);
    };
    transform.addEventListener("objectChange", () => {
      const marker = transform.object;
      if (!marker || marker.userData.waypointIndex === undefined) return;
      state.current.waypoints[marker.userData.waypointIndex]
        .copy(marker.position)
        .add(new THREE.Vector3(0, -0.08, 0));
      rebuildWaypointLine();
    });
    const addWaypoint = (event) => {
      if (state.current.pathPlaying || transform.dragging || transform.axis)
        return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const markerHit = raycaster.intersectObjects(
        waypointGroup.children,
        false,
      )[0];
      if (markerHit) {
        transform.attach(markerHit.object);
        return;
      }
      if (!waypointModeRef.current) {
        transform.detach();
        return;
      }
      const point = raycaster.ray.intersectPlane(
        new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
        new THREE.Vector3(),
      );
      if (!point || Math.abs(point.x) > 4 || Math.abs(point.z) > 4) return;
      if (state.current.heightPixels) {
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
          state.current.heightPixels[(row * HEIGHT_SAMPLE_WIDTH + column) * 4] /
          255;
        point.y = heightValue * HEIGHT_WORLD_SCALE * state.current.exaggeration;
      }
      state.current.waypoints.push(point);
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.09, 18, 14),
        new THREE.MeshBasicMaterial({ color: 0xff9f43 }),
      );
      marker.position.copy(point).add(new THREE.Vector3(0, 0.08, 0));
      marker.userData.waypointIndex = state.current.waypoints.length - 1;
      waypointGroup.add(marker);
      transform.attach(marker);
      rebuildWaypointLine();
      onWaypointChange(state.current.waypoints.length);
    };
    renderer.domElement.addEventListener("click", addWaypoint);
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
    let raf;
    const loop = () => {
      const dt = Math.min(clock.getDelta(), 0.05);
      const move = new THREE.Vector3();
      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();
      const right = new THREE.Vector3()
        .crossVectors(forward, camera.up)
        .normalize();
      if (keys.current.w) move.add(forward);
      if (keys.current.s) move.sub(forward);
      if (keys.current.d) move.add(right);
      if (keys.current.a) move.sub(right);
      if (keys.current.e) move.y += 1;
      if (keys.current.q) move.y -= 1;
      const desired = move.lengthSq()
        ? move.normalize().multiplyScalar(1.35)
        : new THREE.Vector3();
      velocity.lerp(desired, 1 - Math.exp(-6 * dt));
      if (velocity.lengthSq() > 0.000001) {
        const step = velocity.clone().multiplyScalar(dt);
        camera.position.add(step);
        controls.target.add(step);
      }
      if (state.current.pathPlaying && state.current.pathCurve) {
        state.current.pathElapsed += dt;
        const t = Math.min(
          state.current.pathElapsed / state.current.pathDuration,
          1,
        );
        const position = state.current.pathCurve.getPoint(t);
        const lookAt = state.current.pathCurve.getPoint(Math.min(t + 0.012, 1));
        position.y += 0.65;
        lookAt.y += 0.42;
        camera.position.copy(position);
        const direction = lookAt.sub(position).normalize();
        direction.applyAxisAngle(
          new THREE.Vector3(0, 1, 0),
          state.current.routeYaw,
        );
        const pitchAxis = new THREE.Vector3()
          .crossVectors(direction, new THREE.Vector3(0, 1, 0))
          .normalize();
        direction.applyAxisAngle(pitchAxis, state.current.routePitch);
        controls.target.copy(position).add(direction);
        if (t >= 1) {
          state.current.pathPlaying = false;
          controls.enabled = true;
          pathEndRef.current?.();
        }
      }
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      renderer.domElement.removeEventListener("click", addWaypoint);
      renderer.domElement.removeEventListener("pointerdown", startRouteLook);
      window.removeEventListener("pointermove", moveRouteLook);
      window.removeEventListener("pointerup", endRouteLook);
      controls.dispose();
      transform.dispose();
      geo.dispose();
      material.dispose();
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
      s.waypoints.length = 0;
      while (s.waypointGroup.children.length) {
        const marker = s.waypointGroup.children.pop();
        marker.geometry.dispose();
        marker.material.dispose();
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
      s.transform.detach();
      s.controls.enabled = false;
    }
  }, [pathCommand, onWaypointChange]);
  useEffect(() => {
    const s = state.current;
    if (!s.mesh) return;
    const source =
      layer === "texture"
        ? sample.rgb
        : layer === "depth"
          ? sample.height.replace("-height.jpg", "-depth.jpg")
          : sample.height;
    s.mesh.material.map = s.tex.load(source);
    s.mesh.material.needsUpdate = true;
  }, [layer, sample]);
  useEffect(() => {
    const m = state.current.mesh,
      h = state.current.height;
    if (!m || !h?.image || !h.image.complete) return;
    const p = m.geometry.attributes.position;
    const c = document.createElement("canvas");
    c.width = HEIGHT_SAMPLE_WIDTH;
    c.height = HEIGHT_SAMPLE_HEIGHT;
    const x = c.getContext("2d");
    x.drawImage(h.image, 0, 0, HEIGHT_SAMPLE_WIDTH, HEIGHT_SAMPLE_HEIGHT);
    const px = x.getImageData(
      0,
      0,
      HEIGHT_SAMPLE_WIDTH,
      HEIGHT_SAMPLE_HEIGHT,
    ).data;
    for (let i = 0; i < p.count; i++)
      p.setY(i, (px[i * 4] / 255) * HEIGHT_WORLD_SCALE * exaggeration);
    p.needsUpdate = true;
    m.geometry.computeVertexNormals();
    state.current.exaggeration = exaggeration;
  }, [exaggeration]);
  useEffect(() => {
    const s = state.current;
    if (!s.camera || !s.controls) return;
    s.camera.position.set(0, 3.4, 5.6);
    s.controls.target.set(0, 0, 0);
    s.controls.update();
  }, [resetToken]);
  return (
    <div
      ref={ref}
      className={`terrain-canvas ${waypointMode ? "waypoint-active" : ""}`}
    />
  );
}

function App() {
  const [split, setSplit] = useState("train");
  const [sample, setSample] = useState(samples.train);
  const [layer, setLayer] = useState("texture");
  const [exaggeration, setExaggeration] = useState(0.3);
  const [playing, setPlaying] = useState(false);
  const [toast, setToast] = useState("");
  const [theme, setTheme] = useState("dark");
  const [resetToken, setResetToken] = useState(0);
  const [profileCleared, setProfileCleared] = useState(false);
  const [waypointMode, setWaypointMode] = useState(false);
  const [waypointCount, setWaypointCount] = useState(0);
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
  };
  const clearPath = () => {
    setPathCommand({ type: "clear", id: Date.now() });
    setWaypointCount(0);
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
            className={`viewport-card ${layer === "depth" ? "depth-mode" : ""}`}
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
                waypointMode={waypointMode}
                pathCommand={pathCommand}
                onWaypointChange={setWaypointCount}
                onPathEnd={() => setPlaying(false)}
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
                    : "RGB texture"}
              </div>
              <div className="compass">
                <Compass size={23} />
                <span>N</span>
              </div>
              <div className="canvas-controls">
                <button
                  onClick={() =>
                    setExaggeration(Math.max(0.3, exaggeration - 0.1))
                  }
                >
                  <Minus size={15} />
                </button>
                <span>{exaggeration.toFixed(1)}×</span>
                <button
                  onClick={() =>
                    setExaggeration(Math.min(2, exaggeration + 0.1))
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
                <Eye size={14} /> 2048 × 2048 texture · 1024 height grid
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
              </div>
            </div>
            <div className="control-section">
              <div className="label-row">
                <label>Vertical exaggeration</label>
                <span>{exaggeration.toFixed(1)}×</span>
              </div>
              <input
                type="range"
                min=".3"
                max="2"
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
                  {sceneIndex + 1} / {displayedScenes.length}
                </span>
              </div>
              <div className="scene-preview">
                <img src={sample.thumb || sample.rgb} alt={sample.label} />
                <div>
                  <strong>{sample.id}</strong>
                  <small>{sample.label}</small>
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
                {displayedScenes.map((scene) => (
                  <option key={`${scene.split}-${scene.id}`} value={scene.id}>
                    {scene.id} — {scene.label}
                  </option>
                ))}
              </select>
              <div className="photo-nav">
                <button onClick={() => changeScene(-1)}>← Previous</button>
                <button onClick={() => changeScene(1)}>Next →</button>
              </div>
            </div>
            <div className="control-section waypoint-panel">
              <div className="label-row">
                <label>Camera route</label>
                <span>{waypointCount} points</span>
              </div>
              <p>
                Set points on the terrain. Click a marker to reveal its X/Y/Z
                move arrows.
              </p>
              <div className="waypoint-actions">
                <button
                  className={waypointMode ? "tool-active" : ""}
                  onClick={() => {
                    setWaypointMode(!waypointMode);
                    notify(
                      waypointMode
                        ? "Point mode off"
                        : "Click terrain to add points",
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
                <button onClick={clearPath} disabled={!waypointCount}>
                  <X size={14} /> Clear
                </button>
              </div>
            </div>
            <div className="tool-row">
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
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
