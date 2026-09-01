import { useCallback, useEffect, useRef, useState } from 'react';
import DropZone from './components/DropZone';
import SceneList from './components/SceneList';
import Viewer3D from './components/Viewer3D';
import SurfaceControls from './components/SurfaceControls';
import GeoInfo from './components/GeoInfo';
import ProbeInfo from './components/ProbeInfo';
import SceneMetrics from './components/SceneMetrics';
import { deleteScene, getScene, listScenes, pollScene, uploadScene } from './api';
import { metricParams } from './three/probe';
import './App.css';

export default function App() {
  const viewerRef = useRef(null);

  const [scenes, setScenes] = useState([]);
  const [currentId, setCurrentId] = useState(null);
  const [meta, setMeta] = useState(null);         // meta.json returned by loadScene()
  const [glbUrl, setGlbUrl] = useState(null);
  const [stageStatus, setStageStatus] = useState('upload an image to begin');

  const [uploadStatus, setUploadStatus] = useState('');
  const [uploadError, setUploadError] = useState(false);
  const [busy, setBusy] = useState(false);

  const [probe, setProbe] = useState(null);
  const [mode, setMode] = useState('raw');
  const [exaggeration, setExaggeration] = useState(0.15);
  const [colormap, setColormap] = useState('0');
  const [navMode, setNavMode] = useState('orbit');
  const [navHint, setNavHint] = useState('drag to orbit, scroll to zoom');

  const refresh = useCallback(async () => {
    const list = await listScenes();
    setScenes(list);
    return list;
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Without this the browser navigates away to a file dropped outside #drop.
  useEffect(() => {
    const stop = (e) => e.preventDefault();
    window.addEventListener('dragover', stop);
    window.addEventListener('drop', stop);
    return () => {
      window.removeEventListener('dragover', stop);
      window.removeEventListener('drop', stop);
    };
  }, []);

  // Fetches the scene fresh from the API rather than reading local `scenes`
  // state -- this is called right after an upload's own refresh(), before
  // that state update has necessarily landed, and a stale closure here would
  // read the pre-upload record.
  const selectScene = useCallback(async (id) => {
    setCurrentId(id);
    setProbe(null);
    setMeta(null);
    setGlbUrl(null);

    const record = await getScene(id);
    if (record.status !== 'done') {
      setStageStatus(record.status === 'failed' ? (record.error_message || 'processing failed') : 'processing…');
      return;
    }

    setStageStatus('loading…');
    try {
      const m = await viewerRef.current?.loadScene(record.base_url);
      setMeta(m);
      setGlbUrl(record.glb_url);
      setStageStatus('');
    } catch (err) {
      setStageStatus(`failed to load: ${err.message}`);
    }
  }, []);

  const removeScene = useCallback(async (id) => {
    await deleteScene(id);
    if (currentId === id) {
      setCurrentId(null);
      setMeta(null);
      setStageStatus('upload an image to begin');
    }
    await refresh();
  }, [currentId, refresh]);

  const handleUpload = useCallback(async (file) => {
    setUploadError(false);
    setBusy(true);
    setUploadStatus(`uploading ${file.name} (${(file.size / 1e6).toFixed(1)} MB)…`);
    try {
      const created = await uploadScene(file);
      await refresh();
      setUploadStatus(`${created.original_filename}: queued…`);

      const finished = await pollScene(created.id, {
        onUpdate: (s) => {
          setScenes((prev) => prev.map((p) => (p.id === s.id ? s : p)));
          if (s.status === 'processing') setUploadStatus(`${s.original_filename}: processing…`);
        },
      });

      if (finished.status === 'failed') {
        throw new Error(finished.error_message || 'processing failed');
      }
      setUploadStatus(`${finished.original_filename}: done in ${finished.seconds ?? '?'}s`
        + (finished.georeferenced ? ' · georeferenced' : ''));
      await refresh();
      await selectScene(finished.id);
    } catch (err) {
      setUploadStatus(`upload failed — ${err.message}`);
      setUploadError(true);
    } finally {
      setBusy(false);
    }
  }, [refresh, selectScene]);

  const handleNavChange = (m) => {
    const applied = viewerRef.current?.setNavMode(m) ?? m;
    setNavMode(applied);
    setNavHint(applied === 'fly'
      ? 'pointer locked · WASD move · QE up/down · Shift fast · Esc release'
      : (m === 'fly'
        ? 'this browser refused pointer lock — staying in orbit mode'
        : 'drag to orbit, scroll to zoom'));
  };

  const metric = !!metricParams(meta);

  return (
    <div id="app">
      <aside id="scenes">
        <header>
          <h1>AltiMap</h1>
          <p className="sub">GeoTIFF, TIFF, PNG or JPG → 3D</p>
          <DropZone onUpload={handleUpload} statusText={uploadStatus} error={uploadError} busy={busy} />
        </header>
        <SceneList scenes={scenes} currentId={currentId} onSelect={selectScene} onDelete={removeScene} />
      </aside>

      <main id="stage">
        <Viewer3D ref={viewerRef} onProbe={setProbe} />
        {stageStatus && (
          <div id="overlay"><p id="status">{stageStatus}</p></div>
        )}
      </main>

      <aside id="controls">
        <SurfaceControls
          mode={mode} onModeChange={(m) => { setMode(m); viewerRef.current?.setMode(m); }}
          exaggeration={exaggeration} onExagChange={(v) => { setExaggeration(v); viewerRef.current?.setExaggeration(v); }}
          colormap={colormap} onColormapChange={(c) => { setColormap(c); viewerRef.current?.setColormap(c); }}
          navMode={navMode} onNavChange={handleNavChange} navHint={navHint}
        />
        <GeoInfo meta={meta} />
        <ProbeInfo probe={probe} metric={metric} />
        <SceneMetrics meta={meta} glbUrl={glbUrl} />
      </aside>
    </div>
  );
}
