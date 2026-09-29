import {
  Activity,
  Compass,
  Download,
  Eye,
  FileImage,
  Layers3,
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
import TerrainCanvas from "./components/TerrainCanvas";
import RoutePanel from "./components/RoutePanel";
import SurfaceProfile from "./components/SurfaceProfile";
import ModelReport from "./components/ModelReport";
import {
  samples,
  SUPPORTED_UPLOAD_EXTENSIONS,
  displayedScenes,
  initialSample,
  urbanScenes,
  otherScenes,
} from "./data/scenes";
import { HEIGHT_SAMPLE_WIDTH } from "./terrain/settings";
// Vite proxies the two-route reconstruction API to the local backend. A
// georeferenced GeoTIFF follows the direct elevation path; PNG/JPEG follows
// relative depth plus semantic classification.
const CLASSIFY_API_URL = "/api/reconstruct";
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

export default function App() {
  const [view, setView] = useState("terrain"); // terrain | upload
  const [split, setSplit] = useState("train");
  const [sample, setSample] = useState(initialSample);
  const [layer, setLayer] = useState("texture");
  const [exaggeration, setExaggeration] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [measure, setMeasure] = useState(false);
  const [toast, setToast] = useState("");
  const [theme, setTheme] = useState("dark");
  const [resetToken, setResetToken] = useState(0);
  const [surfaceStats, setSurfaceStats] = useState(null);
  const [routePoints, setRoutePoints] = useState([]);
  const [routeProgress, setRouteProgress] = useState(0);
  const [flightSpeed, setFlightSpeed] = useState(0.7);
  const [routeSpeed, setRouteSpeed] = useState(0.6);
  const [wireframe, setWireframe] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [inspectorVisible, setInspectorVisible] = useState(true);
  const [showValidation, setShowValidation] = useState(false);
  const [measurement, setMeasurement] = useState(null);
  const [waypointMode, setWaypointMode] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [waypointCount, setWaypointCount] = useState(0);
  const [selectedWaypoint, setSelectedWaypoint] = useState(null);
  const [pathCommand, setPathCommand] = useState({ type: "idle", id: 0 });
  const [uploadFileName, setUploadFileName] = useState("");
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle | loading | error
  const [uploadError, setUploadError] = useState("");
  const [uploadMeta, setUploadMeta] = useState(null);
  const [isUploadDragActive, setIsUploadDragActive] = useState(false);
  const [elevationBand, setElevationBand] = useState("");
  const fileRef = useRef(null);
  const uploadFileRef = useRef(null);
  useEffect(() => {
    if (view === "upload") {
      setInspectorVisible(true);
      requestAnimationFrame(() => document.querySelector(".upload-switcher")?.scrollIntoView({ block: "start", behavior: "smooth" }));
    }
  }, [view]);
  const notify = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 1800);
  };
  const selectScene = (s) => {
    setSplit(s.split);
    setSample(s);
    setSurfaceStats(null);
    setRoutePoints([]);
    setRouteProgress(0);
    setMeasurement(null);
    setWaypointCount(0);
    setSelectedWaypoint(null);
    setPlaying(false);
    notify(`Loaded ${s.id}`);
  };
  const runClassification = async (file) => {
    setUploadFileName(file.name);
    setUploadStatus("loading");
    setUploadError("");
    try {
      const body = new FormData();
      body.append("file", file);
      if (elevationBand) body.append("elevation_band", elevationBand);
      const res = await fetch(CLASSIFY_API_URL, { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Server returned ${res.status}`);
      }
      const data = await res.json();
      // Reuses the exact same `sample` shape as a catalog scene (rgb/height/
      // classes URLs + label/coord/max) so every existing viewer feature --
      // layer switching, waypoints, measuring -- works on it unmodified.
      setSample({
        id: `upload-${file.name}-${file.size}`,
        label: file.name,
        coord: data.geo?.georeferenced
          ? `${data.geo.crs} · ${data.geo.res_m?.[0]?.toFixed(2)} m/px`
          : "No coordinates · non-georeferenced",
        rgb: data.rgb,
        height: data.height,
        classes: data.classes,
        max:
          data.height_range?.unit === "m"
            ? `${data.height_range.max.toFixed(1)} m`
            : "relative estimate",
        thumb: data.rgb,
        heightMode: data.height_mode,
        heightBaseline: data.height_baseline,
        heightWorldScale: data.height_world_scale,
        heightRange: data.height_range,
        reconstructionMode: data.mode,
        meshResolution: data.mesh_resolution,
        heightFloat32: data.height_float32,
        gridWidth: data.width,
        gridHeight: data.height_px,
        aspectRatio: data.aspect_ratio,
        groundWidth: data.geo?.ground_m?.[0],
        geo: data.geo,
        glb: data.terrain_glb_url,
      });
      setUploadMeta({
        seconds: data.seconds,
        width: data.width,
        height_px: data.height_px,
        meshResolution: data.mesh_resolution,
        source_width: data.source_width,
        source_height: data.source_height,
        class_pixel_counts: data.class_pixel_counts,
        geo: data.geo,
        maskGeoTiffUrl: data.mask_geo_tiff_url,
        elevationGeoTiffUrl: data.elevation_geo_tiff_url,
        terrainGlbUrl: data.terrain_glb_url,
        classIdsUrl: data.class_ids_url,
        estimatedAglUrl: data.estimated_agl_url,
        confidenceUrl: data.confidence_url,
        confidence: data.classification_confidence,
        inferenceWidth: data.inference_width,
        inferenceHeight: data.inference_height,
        heightRange: data.height_range,
        elevationSource: data.elevation_source,
        buildingRefinement: data.building_refinement,
        mode: data.mode,
      });
      setSurfaceStats(null);
      setRoutePoints([]);
      setRouteProgress(0);
      setMeasurement(null);
      setWaypointCount(0);
      setSelectedWaypoint(null);
      setPlaying(false);
      // Mirrors the direct GeoTIFF viewer: its reported world scale already
      // encodes elevation-range / ground-footprint, so 1× is the true-scale
      // view. Relative image reconstructions retain the normal 0.5× default.
      setExaggeration(data.mode === "geotiff-direct" ? 1 : 0.5);
      setLayer("texture");
      setUploadStatus("idle");
      notify(
        data.mode === "geotiff-direct"
          ? `Built georeferenced terrain for ${file.name}`
          : `Built relative terrain for ${file.name}`,
      );
    } catch (err) {
      setUploadError(
        err.message === "Failed to fetch"
          ? `Couldn't reach the classifier backend at ${CLASSIFY_API_URL}`
          : err.message,
      );
      setUploadStatus("error");
    }
  };
  const submitUpload = (file) => {
    if (!file || uploadStatus === "loading") return;
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!SUPPORTED_UPLOAD_EXTENSIONS.has(extension)) {
      setUploadError("Use a PNG, JPG, JPEG, TIF, TIFF, or GeoTIFF image.");
      setUploadStatus("error");
      return;
    }
    runClassification(file);
  };
  const onPickUpload = (e) => {
    submitUpload(e.target.files?.[0]);
    e.target.value = "";
  };
  const onUploadDragEnter = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (uploadStatus !== "loading") setIsUploadDragActive(true);
  };
  const onUploadDragLeave = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setIsUploadDragActive(false);
    }
  };
  const onUploadDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsUploadDragActive(false);
    if (uploadStatus !== "loading") submitUpload(event.dataTransfer.files?.[0]);
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
    if (f) {
      setView("upload");
      submitUpload(f);
    }
    e.target.value = "";
  };
  const exportScene = () => {
    if (sample.glb) {
      const a = document.createElement("a");
      a.href = sample.glb;
      a.download = "terrain.glb";
      a.click();
      return;
    }
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
            waypoints: routePoints,
            flightSpeed,
            routeSpeed,
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
    <div
      className={`app theme-${theme} ${inspectorVisible ? "" : "inspector-hidden"}`}
    >
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
            title="Upload & reconstruct"
          >
            <Upload size={18} />
          </button>
          <button
            title="Model validation"
            onClick={() => {
              setShowValidation((v) => !v);
              setInspectorVisible(true);
            }}
          >
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
                <button disabled={uploadStatus === "loading"} onClick={() => fileRef.current?.click()}>
                  <Upload size={14} /> {uploadStatus === "loading" ? "Processing…" : "Import"}
                </button>
                <button onClick={exportScene}>
                  <Download size={14} /> Export
                </button>
                <button onClick={togglePathPlayback}>
                  {playing ? <Pause size={15} /> : <Play size={15} />}{" "}
                  {playing ? "Pause path" : "Play route"}
                </button>
                <button
                  title="Toggle fullscreen"
                  onClick={() =>
                    Promise.resolve(
                      document.fullscreenElement
                        ? document.exitFullscreen()
                        : document
                            .querySelector(".workspace")
                            ?.requestFullscreen?.(),
                    ).catch(() => notify("Fullscreen unavailable"))
                  }
                >
                  <Maximize2 size={15} />
                </button>
              </div>
            </div>
            <div className="workspace-tools">
              {["perspective", "top", "front"].map((view) => (
                <button
                  key={view}
                  onClick={() =>
                    setPathCommand({
                      type: "view",
                      view,
                      id: performance.now(),
                    })
                  }
                >
                  {view[0].toUpperCase() + view.slice(1)}
                </button>
              ))}
              <button
                title="Toggle inspector"
                onClick={() => setInspectorVisible((v) => !v)}
              >
                <SlidersHorizontal size={14} />
              </button>
            </div>
            <div className="canvas-wrap">
              <TerrainCanvas
                sample={sample}
                exaggeration={exaggeration}
                layer={layer}
                resetToken={resetToken}
                onMeasure={setMeasurement}
                measureMode={measure}
                flightSpeed={flightSpeed}
                routeSpeed={routeSpeed}
                wireframe={wireframe}
                showGrid={showGrid}
                theme={theme}
                onRouteChange={setRoutePoints}
                onSurfaceStats={setSurfaceStats}
                onRouteProgress={setRouteProgress}
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
                  ? sample.heightMode === "absolute"
                    ? "Metric elevation"
                    : "rDSM surface"
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
                  <b className="swatch low" /> Low ·{" "}
                  {sample.heightRange?.unit === "m"
                    ? `${sample.heightRange.min.toFixed(1)} m`
                    : "relative"}
                </span>
                <span>
                  <b className="swatch mid" /> Mid ·{" "}
                  {sample.heightRange?.unit === "m"
                    ? `${((sample.heightRange.min + sample.heightRange.max) / 2).toFixed(1)} m`
                    : "estimate"}
                </span>
                <span>
                  <b className="swatch high" /> High ·{" "}
                  {sample.heightRange?.unit === "m"
                    ? sample.max
                    : `${surfaceStats?.max.toFixed(2) || "—"} relative`}
                </span>
              </div>
              <div className="footer-note">
                <Eye size={14} /> {sample.gridWidth || HEIGHT_SAMPLE_WIDTH} ×{" "}
                {sample.gridHeight || HEIGHT_SAMPLE_WIDTH}
                live mesh · 4 aligned layers
              </div>
            </div>
          </div>
          <aside className="inspector">
            <div className="panel-heading">
              <div>
                <div className="eyebrow">Scene inspector</div>
                <h2>Surface workspace</h2>
              </div>
              <button
                className="icon-btn"
                title="Hide inspector"
                onClick={() => setInspectorVisible(false)}
              >
                <SlidersHorizontal size={16} />
              </button>
            </div>
            <div className="metric-grid">
              <div>
                <small>Surface data</small>
                <strong>
                  {sample.heightMode === "absolute"
                    ? "Metric elevation"
                    : sample.reconstructionMode
                      ? "RGB estimate"
                      : "GAMUS preview"}
                </strong>
              </div>
              <div>
                <small>Mesh vertices</small>
                <strong>
                  {surfaceStats?.vertices.toLocaleString() || "Loading…"}
                </strong>
              </div>
            </div>
            <section className="control-section">
              <div className="viewport-preferences">
                <label>
                  <input
                    type="checkbox"
                    checked={wireframe}
                    onChange={(e) => setWireframe(e.target.checked)}
                  />
                  Wireframe
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={showGrid}
                    onChange={(e) => setShowGrid(e.target.checked)}
                  />
                  Grid
                </label>
              </div>
              <div className="label-row">
                <label htmlFor="flight-speed">Navigation speed</label>
                <span>{flightSpeed.toFixed(2)} u/s</span>
              </div>
              <input
                id="flight-speed"
                type="range"
                min=".1"
                max="3"
                step=".05"
                value={flightSpeed}
                onChange={(e) => setFlightSpeed(+e.target.value)}
              />
            </section>
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
            <SurfaceProfile stats={surfaceStats} sample={sample} />
            {showValidation && <ModelReport />}
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
                      <option
                        key={`${scene.split}-${scene.id}`}
                        value={scene.id}
                      >
                        {scene.id} — {scene.label}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label={`Other GAMUS tiles (${otherScenes.length})`}>
                    {otherScenes.map((scene) => (
                      <option
                        key={`${scene.split}-${scene.id}`}
                        value={scene.id}
                      >
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
                  <label>Upload &amp; reconstruct</label>
                  <span>
                    {uploadMeta?.geo?.georeferenced
                      ? "georeferenced"
                      : "non-georeferenced"}
                  </span>
                </div>
                <p className="upload-copy">
                  GeoTIFF: direct metric terrain, preserving its CRS, grid, and
                  true height scale. It uses an embedded elevation band or a
                  reference DEM, then drapes the original RGB image over it.
                  Semantic classes are labels only; they do not alter metric
                  heights. PNG/JPG: jointly predicts seven land-cover classes
                  and surface height with the GAMUS-trained model. These heights
                  are uncalibrated estimates, not survey measurements.
                </p>
                <input
                  ref={uploadFileRef}
                  type="file"
                  accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff,.geotif,.geotiff"
                  onChange={onPickUpload}
                  hidden
                />
                <label className="elevation-band-field">
                  <span>Elevation band (optional)</span>
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={elevationBand}
                    onChange={(event) => setElevationBand(event.target.value)}
                    placeholder="Auto-detect"
                  />
                </label>
                <button
                  className={`upload-dropzone ${isUploadDragActive ? "is-dragging" : ""}`}
                  onClick={() => uploadFileRef.current?.click()}
                  onDragEnter={onUploadDragEnter}
                  onDragOver={onUploadDragEnter}
                  onDragLeave={onUploadDragLeave}
                  onDrop={onUploadDrop}
                  disabled={uploadStatus === "loading"}
                  type="button"
                >
                  <Upload size={20} />
                  <span>
                    {uploadStatus === "loading"
                      ? `Reconstructing ${uploadFileName}…`
                      : uploadFileName
                        ? `${uploadFileName} — click to replace`
                        : "Drop an image here or click to browse"}
                  </span>
                  <small>PNG · JPG · TIF · TIFF · GeoTIFF</small>
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
                        <small>Resolution</small>
                        <strong>
                          {uploadMeta.width}×{uploadMeta.height_px}
                        </strong>
                      </div>
                    </div>
                    <div className="class-breakdown">
                      {CLASSIFY_CLASS_LABELS.map((name) => {
                        const count =
                          uploadMeta.class_pixel_counts?.[name] ?? 0;
                        const total = uploadMeta.width * uploadMeta.height_px;
                        const pct = total
                          ? ((count / total) * 100).toFixed(1)
                          : "0.0";
                        return (
                          <div key={name} className="class-breakdown-row">
                            <i
                              className={`class-dot ${CLASS_DOT_STYLE[name]}`}
                            />
                            <span>{name.replace("_", " ")}</span>
                            <strong>{pct}%</strong>
                          </div>
                        );
                      })}
                    </div>
                    {uploadMeta.geo?.georeferenced && (
                      <div className="geo-output">
                        <strong>{uploadMeta.geo.crs}</strong>
                        <span>
                          {uploadMeta.geo.width}×{uploadMeta.geo.height} native
                          grid · {uploadMeta.geo.res_m?.[0]?.toFixed(2)} m/px
                        </span>
                        <span>{uploadMeta.elevationSource}</span>
                        {uploadMeta.buildingRefinement && (
                          <span>
                            {uploadMeta.buildingRefinement.n_extruded > 0
                              ? `${uploadMeta.buildingRefinement.n_extruded} buildings extruded from ${uploadMeta.buildingRefinement.n_with_height} known heights`
                              : uploadMeta.buildingRefinement.note ||
                                "No measured building heights available for this footprint"}
                          </span>
                        )}
                        {uploadMeta.heightRange?.unit === "m" && (
                          <span>
                            {uploadMeta.heightRange.min.toFixed(1)}–
                            {uploadMeta.heightRange.max.toFixed(1)} m elevation
                          </span>
                        )}
                        <a href={uploadMeta.maskGeoTiffUrl} download>
                          Download georeferenced class mask
                        </a>
                        <a href={uploadMeta.elevationGeoTiffUrl} download>
                          Download georeferenced elevation model
                        </a>
                        <a href={uploadMeta.terrainGlbUrl} download>
                          Download textured 3D model (.glb)
                        </a>
                      </div>
                    )}
                    {!uploadMeta.geo?.georeferenced &&
                      uploadMeta.terrainGlbUrl && (
                        <div className="geo-output">
                          <span>Relative terrain estimate</span>
                          <span>
                            Inference: {uploadMeta.inferenceWidth} ×{" "}
                            {uploadMeta.inferenceHeight} · mean class
                            probability{" "}
                            {(100 * uploadMeta.confidence).toFixed(1)}% (not
                            accuracy)
                          </span>
                          {uploadMeta.classIdsUrl && (
                            <a href={uploadMeta.classIdsUrl} download>
                              Download class IDs (.png)
                            </a>
                          )}
                          {uploadMeta.estimatedAglUrl && (
                            <a href={uploadMeta.estimatedAglUrl} download>
                              Download raw estimated AGL (.npy)
                            </a>
                          )}
                          {uploadMeta.confidenceUrl && (
                            <a href={uploadMeta.confidenceUrl} download>
                              Download class confidence (.png)
                            </a>
                          )}
                          <a href={uploadMeta.terrainGlbUrl} download>
                            Download textured 3D model (.glb)
                          </a>
                        </div>
                      )}
                  </>
                )}
              </div>
            )}
            <RoutePanel
              points={routePoints}
              selected={selectedWaypoint}
              playing={playing}
              marking={waypointMode}
              onMark={() => {
                setWaypointMode((v) => !v);
                setAutoRotate(false);
                setMeasure(false);
              }}
              onPlay={togglePathPlayback}
              onClear={clearPath}
              onDelete={removeSelectedWaypoint}
              command={setPathCommand}
              speed={routeSpeed}
              setSpeed={setRouteSpeed}
              progress={routeProgress}
              sceneId={sample.id}
              notify={notify}
            />
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
                  setWaypointMode(false);
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
          {measurement
            ? `Height ${measurement.height.toFixed(3)} ${measurement.unit}${measurement.distance === null ? "" : ` · Horizontal distance ${measurement.distance.toFixed(2)} ${measurement.distanceUnit}`}`
            : "Double-click two surface locations to inspect height and distance"}{" "}
          <X size={14} onClick={() => setMeasure(false)} />
        </div>
      )}
    </div>
  );
}
