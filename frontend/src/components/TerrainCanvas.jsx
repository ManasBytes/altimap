import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { HEIGHT_SAMPLE_WIDTH } from "../terrain/settings";
import {
  HEIGHT_WORLD_SCALE,
  TERRAIN_BASELINE,
  buildHeightField,
  buildAbsoluteHeightField,
  buildClassField,
  applyClassHeightScale,
  buildHeightColors,
} from "../terrain/heightField";
import {
  createWaypointMarker,
  setWaypointNumber,
  setWaypointSelected,
  disposeWaypointMarker,
  resolveWaypointHit,
} from "../terrain/waypoints";
export default function TerrainCanvas({
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
  flightSpeed = 0.7,
  routeSpeed = 0.6,
  wireframe = false,
  showGrid = true,
  theme = "dark",
  measureMode = false,
  onRouteChange,
  onSurfaceStats,
  onRouteProgress,
}) {
  const ref = useRef(null);
  const state = useRef({});
  const waypointModeRef = useRef(waypointMode);
  const pathEndRef = useRef(onPathEnd);
  const waypointSelectRef = useRef(onWaypointSelect);
  const layerRef = useRef(layer);
  const keys = useRef({});
  const live = useRef({});
  live.current = {
    flightSpeed,
    routeSpeed,
    measureMode,
    onMeasure,
    onRouteChange,
    onSurfaceStats,
    onRouteProgress,
    exaggeration,
    autoRotate,
  };
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
    // Catalog and image-relative scenes stay at the responsive 513² grid.
    // A direct GeoTIFF response explicitly requests 1025² samples, yielding
    // 1024×1024 terrain cells (four times the previous 512×512 cell count).
    const requestedMeshResolution = Math.round(
      Number(sample.meshResolution) || HEIGHT_SAMPLE_WIDTH,
    );
    const meshSegments = Math.min(
      1024,
      Math.max(64, requestedMeshResolution - 1),
    );
    const aspect = sample.aspectRatio || 1;
    const terrainWidth = 8 * Math.min(1, aspect);
    const terrainDepth = 8 / Math.max(1, aspect);
    const heightSampleWidth = Math.max(
      2,
      Math.round((meshSegments * terrainWidth) / 8) + 1,
    );
    const heightSampleHeight = Math.max(
      2,
      Math.round((meshSegments * terrainDepth) / 8) + 1,
    );
    let disposed = false;
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
    const geo = new THREE.PlaneGeometry(
      terrainWidth,
      terrainDepth,
      heightSampleWidth - 1,
      heightSampleHeight - 1,
    );
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
      if (disposed || !current.rawHeightField) return;
      const heightField =
        current.heightMode === "relative"
          ? applyClassHeightScale(
              current.rawHeightField,
              current.classField,
              heightSampleWidth,
              heightSampleHeight,
            )
          : current.rawHeightField;
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++)
        pos.setY(
          i,
          (heightField[i] - current.heightBaseline) * current.heightWorldScale,
        );
      current.heightField = heightField;
      current.heightColors = buildHeightColors(heightField);
      current.exaggeration = live.current.exaggeration;
      current.mesh.scale.y = live.current.exaggeration;
      pos.needsUpdate = true;
      geo.computeVertexNormals();
      geo.computeBoundingSphere();
      const profile = Array.from(
        { length: 80 },
        (_, i) =>
          heightField[
            Math.floor(heightSampleHeight / 2) * heightSampleWidth +
              Math.round((i / 79) * (heightSampleWidth - 1))
          ],
      );
      let low = Infinity,
        high = -Infinity,
        sum = 0;
      for (const h of heightField) {
        low = Math.min(low, h);
        high = Math.max(high, h);
        sum += h;
      }
      live.current.onSurfaceStats?.({
        min: low,
        max: high,
        mean: sum / heightField.length,
        profile,
        vertices: geo.attributes.position.count,
      });
      if (layerRef.current === "depth") {
        geo.attributes.color.array.set(current.heightColors);
        geo.attributes.color.needsUpdate = true;
      }
      current.renderDirty = true;
    };
    const height = tex.load(sample.height, (loaded) => {
      if (disposed) return;
      if (sample.heightFloat32) {
        const bytes = Uint8Array.from(atob(sample.heightFloat32), (c) =>
          c.charCodeAt(0),
        );
        const floats = new Float32Array(bytes.buffer);
        const field = new Float32Array(heightSampleWidth * heightSampleHeight);
        const sw = sample.gridWidth,
          sh = sample.gridHeight;
        for (let y = 0; y < heightSampleHeight; y++)
          for (let x = 0; x < heightSampleWidth; x++) {
            const sx = (x / (heightSampleWidth - 1)) * (sw - 1),
              sy = (y / (heightSampleHeight - 1)) * (sh - 1);
            const x0 = Math.floor(sx),
              y0 = Math.floor(sy),
              x1 = Math.min(sw - 1, x0 + 1),
              y1 = Math.min(sh - 1, y0 + 1);
            const tx = sx - x0,
              ty = sy - y0;
            field[y * heightSampleWidth + x] =
              (floats[y0 * sw + x0] * (1 - tx) + floats[y0 * sw + x1] * tx) *
                (1 - ty) +
              (floats[y1 * sw + x0] * (1 - tx) + floats[y1 * sw + x1] * tx) *
                ty;
          }
        state.current.rawHeightField = field;
        updateGeometry();
        return;
      }
      const c = document.createElement("canvas");
      c.width = heightSampleWidth;
      c.height = heightSampleHeight;
      const x = c.getContext("2d");
      x.drawImage(loaded.image, 0, 0, heightSampleWidth, heightSampleHeight);
      const px = x.getImageData(
        0,
        0,
        heightSampleWidth,
        heightSampleHeight,
      ).data;
      const heightField =
        sample.heightMode === "absolute"
          ? buildAbsoluteHeightField(px, heightSampleWidth, heightSampleHeight)
          : buildHeightField(px, heightSampleWidth, heightSampleHeight);
      state.current.rawHeightField = heightField;
      updateGeometry();
    });
    const classTexture = tex.load(sample.classes, (loaded) => {
      if (
        disposed ||
        sample.heightMode === "absolute" ||
        sample.heightMode === "relative-final"
      )
        return;
      const c = document.createElement("canvas");
      c.width = heightSampleWidth;
      c.height = heightSampleHeight;
      const context = c.getContext("2d");
      context.drawImage(
        loaded.image,
        0,
        0,
        heightSampleWidth,
        heightSampleHeight,
      );
      const px = context.getImageData(
        0,
        0,
        heightSampleWidth,
        heightSampleHeight,
      ).data;
      state.current.classField = buildClassField(
        px,
        heightSampleWidth,
        heightSampleHeight,
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
      grid,
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
      heightMode: sample.heightMode || "relative",
      heightBaseline: sample.heightBaseline ?? TERRAIN_BASELINE,
      heightWorldScale: sample.heightWorldScale ?? HEIGHT_WORLD_SCALE,
    };
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
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
        const liftedPoints = waypoints.map((p) => p.clone());
        const curve = new THREE.CatmullRomCurve3(
          liftedPoints,
          false,
          "centripetal",
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
      state.current.pathCurve = null;
      state.current.pathElapsed = 0;
      live.current.onRouteProgress?.(0);
      live.current.onRouteChange?.(waypoints.map((p) => p.toArray()));
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
    state.current.selectMarker = selectMarker;
    state.current.addRoutePoint = (coordinates) => {
      const point = new THREE.Vector3(...coordinates);
      state.current.waypoints.push(point);
      const marker = createWaypointMarker(
        state.current.waypoints.length - 1,
        point,
      );
      waypointGroup.add(marker);
      rebuildWaypointLine();
      onWaypointChange(state.current.waypoints.length);
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
      if (
        pointerDown &&
        Math.hypot(
          event.clientX - pointerDown.x,
          event.clientY - pointerDown.y,
        ) > 4
      )
        return;
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
      mesh.updateMatrixWorld();
      const hit = raycaster.intersectObject(mesh)[0];
      if (!hit) return;
      const point = hit.point.clone();
      point.y += 0.25;
      // Store a clone for each node so later surface hits cannot alias the
      // route and collapse every point onto the latest click.
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
    let pointerDown = null;
    const rememberPointer = (e) => {
      pointerDown = { x: e.clientX, y: e.clientY };
    };
    renderer.domElement.addEventListener("pointerdown", rememberPointer);
    let lastMeasured = null;
    const measurePoint = (event) => {
      if (!live.current.measureMode) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      mesh.updateMatrixWorld();
      const hit = raycaster.intersectObject(mesh)[0];
      if (!hit) return;
      const x = Math.round(hit.uv.x * (heightSampleWidth - 1)),
        y = Math.round((1 - hit.uv.y) * (heightSampleHeight - 1));
      const normalized = state.current.heightField[y * heightSampleWidth + x];
      const metric = sample.heightMode === "absolute",
        range = sample.heightRange;
      const height =
        metric && range
          ? range.min + normalized * (range.max - range.min)
          : normalized;
      // Horizontal distance is independent of display exaggeration.
      const distance = lastMeasured
        ? Math.hypot(
            hit.point.x - lastMeasured.x,
            hit.point.z - lastMeasured.z,
          ) * (sample.groundWidth ? sample.groundWidth / terrainWidth : 1)
        : null;
      lastMeasured = hit.point.clone();
      live.current.onMeasure?.({
        height,
        unit: metric ? "m" : "relative",
        distance,
        distanceUnit: sample.groundWidth ? "m" : "scene units",
      });
    };
    renderer.domElement.addEventListener("dblclick", measurePoint);
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
      if (
        ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(e.target.tagName) ||
        e.target.isContentEditable ||
        e.ctrlKey ||
        e.metaKey ||
        e.altKey
      )
        return;
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
    const releaseKeys = () => {
      keys.current = {};
      if (!state.current.pathPlaying) controls.enabled = true;
    };
    window.addEventListener("blur", releaseKeys);
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
      if (move.lengthSq())
        desired.copy(move).normalize().multiplyScalar(live.current.flightSpeed);
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
        state.current.pathElapsed += dt * live.current.routeSpeed;
        const t = Math.min(
          state.current.pathElapsed / state.current.pathDuration,
          1,
        );
        const position = state.current.routePosition;
        const lookAt = state.current.routeLookAt;
        state.current.pathCurve.getPointAt(t, position);
        state.current.pathCurve.getTangentAt(Math.min(t, 0.9999), lookAt);
        lookAt.add(position);
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
        if (
          !state.current.progressTime ||
          clock.elapsedTime - state.current.progressTime > 0.1
        ) {
          live.current.onRouteProgress?.(t);
          state.current.progressTime = clock.elapsedTime;
        }
        if (t >= 1) {
          state.current.pathPlaying = false;
          controls.enabled = true;
          pathEndRef.current?.();
          live.current.onRouteProgress?.(1);
        }
      }
      controls.autoRotate =
        live.current.autoRotate && !hasVelocity && !routeActive;
      // OrbitControls.update applies damping even when disabled. During route
      // playback position and aim must belong exclusively to the route camera.
      if (routeActive) camera.lookAt(controls.target);
      const controlsChanged = routeActive ? false : controls.update();
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
      disposed = true;
      keys.current = {};
      cancelAnimationFrame(raf);
      window.removeEventListener("blur", releaseKeys);
      renderer.domElement.removeEventListener("pointerdown", rememberPointer);
      renderer.domElement.removeEventListener("dblclick", measurePoint);
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
      grid.geometry.dispose();
      grid.material.dispose();
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
    if (pathCommand.type === "view") {
      const poses = {
        top: [0, 8, 0.001],
        front: [0, 1, 7],
        perspective: [0, 3.4, 5.6],
      };
      s.pathPlaying = false;
      s.controls.enabled = true;
      pathEndRef.current?.();
      s.camera.position.set(...(poses[pathCommand.view] || poses.perspective));
      s.controls.target.set(0, 0, 0);
      s.controls.update();
      s.renderDirty = true;
      return;
    }
    if (pathCommand.type === "select") {
      const marker = s.waypointGroup.children[pathCommand.index];
      if (marker) s.selectMarker(marker);
      return;
    }
    if (pathCommand.type === "move" && s.selectedMarker) {
      s.selectedMarker.position.set(...pathCommand.coordinates);
      s.waypoints[s.selectedMarker.userData.waypointIndex].copy(
        s.selectedMarker.position,
      );
      s.rebuildWaypointLine();
      return;
    }
    if (pathCommand.type === "load") {
      s.pathPlaying = false;
      s.controls.enabled = true;
      s.transform.detach();
      s.selectedMarker = null;
      waypointSelectRef.current?.(null);
      for (const marker of [...s.waypointGroup.children]) {
        s.waypointGroup.remove(marker);
        disposeWaypointMarker(marker);
      }
      s.waypoints = [];
      for (const p of pathCommand.points) s.addRoutePoint(p);
      s.rebuildWaypointLine();
      onWaypointChange(s.waypoints.length);
      return;
    }
    if (pathCommand.type === "clear") {
      s.pathPlaying = false;
      s.transform.detach();
      s.selectedMarker = null;
      waypointSelectRef.current?.(null);
      s.waypoints.length = 0;
      while (s.waypointGroup.children.length) {
        const marker = s.waypointGroup.children[0];
        s.waypointGroup.remove(marker);
        disposeWaypointMarker(marker);
      }
      if (s.waypointLine) {
        s.scene.remove(s.waypointLine);
        s.waypointLine.geometry.dispose();
        s.waypointLine.material.dispose();
        s.waypointLine = null;
      }
      s.controls.enabled = true;
      s.pathCurve = null;
      s.pathElapsed = 0;
      live.current.onRouteChange?.([]);
      live.current.onRouteProgress?.(0);
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
      if (!s.pathCurve || s.pathElapsed >= s.pathDuration) {
        s.pathCurve = new THREE.CatmullRomCurve3(
          s.waypoints.map((point) => point.clone()),
          false,
          "centripetal",
          0.45,
        );
        s.pathElapsed = 0;
        s.pathDuration = Math.max(0.01, s.pathCurve.getLength());
        s.routeYaw = 0;
        s.routePitch = 0;
      }
      s.pathPlaying = true;
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
    m.scale.y = exaggeration;
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
  useEffect(() => {
    const s = state.current;
    if (!s.mesh) return;
    s.mesh.material.wireframe = wireframe;
    s.grid.visible = showGrid;
    s.scene.background.set(theme === "light" ? "#dce5eb" : "#10151c");
    s.renderDirty = true;
  }, [wireframe, showGrid, theme, sample]);
  return (
    <div
      ref={ref}
      className={`terrain-canvas ${waypointMode ? "waypoint-active" : ""}`}
    />
  );
}
