import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { initViewer } from '../three/viewer';
import { pickTerrain } from '../three/probe';

/**
 * Thin React wrapper around the imperative three.js viewer (src/three/viewer.js).
 * The renderer owns its own animation loop and a WebGLRenderer/mesh that must
 * survive scene swaps untouched, so this stays a single init-on-mount +
 * ref-driven API rather than a declarative scene graph.
 */
const Viewer3D = forwardRef(function Viewer3D({ onProbe }, ref) {
  const canvasRef = useRef(null);
  const viewerRef = useRef(null);
  const [initError, setInitError] = useState(null);

  useEffect(() => {
    try {
      viewerRef.current = initViewer(canvasRef.current);
    } catch (err) {
      setInitError(err.message);
    }
    return () => {
      viewerRef.current = null;
    };
  }, []);

  useImperativeHandle(ref, () => ({
    loadScene: (base, opts) => viewerRef.current?.loadScene(base, opts),
    setExaggeration: (v) => viewerRef.current?.setExaggeration(v),
    setColormap: (c) => viewerRef.current?.setColormap(c),
    setMode: (m) => viewerRef.current?.setMode(m),
    setNavMode: (m) => viewerRef.current?.setNavMode(m),
  }), []);

  const handleClick = (event) => {
    if (!onProbe) return;
    onProbe(pickTerrain(event, canvasRef.current, viewerRef.current));
  };

  return (
    <>
      <canvas id="view" ref={canvasRef} onClick={handleClick} />
      {initError && (
        <div id="overlay"><p id="status">{initError}</p></div>
      )}
    </>
  );
});

export default Viewer3D;
