export default function SurfaceControls({
  mode, onModeChange,
  exaggeration, onExagChange,
  colormap, onColormapChange,
  navMode, onNavChange, navHint,
}) {
  return (
    <section>
      <h2>Surface</h2>
      <div className="seg">
        <button className={mode === 'raw' ? 'on' : ''} onClick={() => onModeChange('raw')}>raw</button>
        <button className={mode === 'detrended' ? 'on' : ''} onClick={() => onModeChange('detrended')}>detrended</button>
      </div>
      <label>
        exaggeration <output>{exaggeration.toFixed(2)}</output>
        <input
          type="range" min="0" max="1" step="0.01" value={exaggeration}
          onChange={(e) => onExagChange(Number(e.target.value))}
        />
      </label>
      <label>
        colormap
        <select value={colormap} onChange={(e) => onColormapChange(e.target.value)}>
          <option value="0">satellite RGB</option>
          <option value="1">height (turbo)</option>
          <option value="2">slope shade</option>
        </select>
      </label>
      <div className="seg">
        <button className={navMode === 'orbit' ? 'on' : ''} onClick={() => onNavChange('orbit')}>orbit</button>
        <button className={navMode === 'fly' ? 'on' : ''} onClick={() => onNavChange('fly')}>fly</button>
      </div>
      <p className="hint">{navHint}</p>
    </section>
  );
}
