import { initViewer } from './terrain.js';
import { metricParams, pickTerrain } from './probe.js';

const DATA = './data-inria';

const el = (id) => document.getElementById(id);
const els = {
  list: el('scene-list'), classSel: el('class-sel'), sort: el('sort'),
  status: el('status'), overlay: el('overlay'), canvas: el('view'),
  sub: el('ds-sub'),
  gStatus: el('g-status'), gCrs: el('g-crs'), gGsd: el('g-gsd'),
  gExtent: el('g-extent'), gNote: el('g-note'),
  pHeight: el('p-height'), pSlope: el('p-slope'),
  slopeLabel: el('slope-label'), pWarn: el('p-warn'),
  mR2: el('m-r2'), mRelief: el('m-relief'), mStruct: el('m-struct'),
  mContext: el('m-context'), caveat: el('m-caveat'), glb: el('glb'),
  modeRefined: el('mode-refined'), modeHint: el('mode-hint'),
  rFp: el('r-fp'), rEdge: el('r-edge'), rCal: el('r-cal'),
};

const state = { index: null, byId: new Map(), cls: null, current: null, mode: 'raw' };
const fmt = (v, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(d);

let viewer = null;
try {
  viewer = initViewer(els.canvas);
  window.__viewer = viewer;
} catch (err) {
  els.status.textContent = err.message;
}

function scenesForClass() {
  const rows = state.index.scenes.filter((s) => s.c === state.cls);
  const key = els.sort.value;
  if (key === 'structure_alignment') return [...rows].sort((a, b) => (b.st ?? -Infinity) - (a.st ?? -Infinity));
  if (key === 'num') return [...rows].sort((a, b) => a.n - b.n);
  return [...rows].sort((a, b) => (a.r2 ?? Infinity) - (b.r2 ?? Infinity));
}

function renderList() {
  const rows = scenesForClass();
  els.list.replaceChildren(...rows.map((s) => {
    const li = document.createElement('li');
    li.dataset.id = s.id;
    li.innerHTML = `
      <img alt="" loading="lazy" src="${DATA}/scenes/${s.id}/rgb.jpg">
      <div>
        <div class="name">tile ${s.n} <span class="badge">geo</span></div>
        <div class="meta">R² ${fmt(s.r2, 2)} · str ${fmt(s.st, 2)}</div>
      </div>`;
    li.addEventListener('click', () => selectScene(s.id));
    return li;
  }));
  if (rows.length) {
    const stillThere = rows.some((r) => r.id === state.current);
    selectScene(stillThere ? state.current : rows[0].id);
  }
}

function markActive() {
  for (const li of els.list.children) li.classList.toggle('on', li.dataset.id === state.current);
}

async function selectScene(id) {
  state.current = id;
  markActive();
  const row = state.byId.get(id);
  showMetrics(row);
  els.pHeight.textContent = '—';
  els.pSlope.textContent = '—';
  els.overlay.classList.remove('hidden');
  els.status.textContent = `loading ${state.cls} tile ${row.n}…`;
  try {
    const meta = await viewer?.loadScene(`${DATA}/scenes/${id}`,
      { variant: state.mode === 'refined' ? 'refined' : 'depth' });
    showGeo(meta);
    showRefinement(meta);
    els.overlay.classList.add('hidden');
  } catch (err) {
    els.status.textContent = `failed to load ${id}: ${err.message}`;
  }
}

function showGeo(meta) {
  const geo = meta?.geo;
  const abs = meta?.absolute;
  const metric = metricParams(meta);

  els.gStatus.textContent = geo?.georeferenced ? 'georeferenced' : 'none';
  els.gCrs.textContent = geo?.crs ? geo.crs.replace('EPSG:', 'EPSG ') : '—';
  els.gGsd.textContent = geo?.res_m ? `${geo.res_m[0].toFixed(2)} m/px` : '—';
  els.gExtent.textContent = geo?.ground_m
    ? `${geo.ground_m[0].toFixed(0)} × ${geo.ground_m[1].toFixed(0)} m` : '—';

  if (!abs) {
    els.gNote.textContent = 'No DEM patch retrieved for this footprint.';
    els.gNote.classList.remove('alert');
  } else {
    // Inria's reference is Copernicus GLO-30 -- a global DSM, not bare earth.
    // It nominally includes building/canopy tops, but at 30 m posting against
    // 0.5 m imagery individual structures are still unresolved.
    const kind = abs.reference_is_bare_earth ? 'bare earth' : 'DSM (not bare earth)';
    if (metric) {
      els.gNote.textContent = `Calibrated to ${abs.source} (${abs.reference_posting_m} m, ${kind}): `
        + `${abs.scale_m.toFixed(1)} m range, fit R² ${abs.fit_r2.toFixed(2)}.`;
      els.gNote.classList.remove('alert');
    } else {
      els.gNote.textContent = `Elevation calibration REJECTED (${abs.reject_reason}; `
        + `fit R² ${fmt(abs.fit_r2, 2)}, scale ${fmt(abs.scale_m, 1)} m). Height stays relative.`;
      els.gNote.classList.add('alert');
    }
  }

  if (metric) {
    els.slopeLabel.textContent = 'ground slope';
    els.pWarn.textContent = 'Elevation calibrated against a 30 m reference DSM: terrain-scale '
      + 'relief is metric; individual buildings are not resolved by that reference.';
  } else {
    els.slopeLabel.textContent = 'display slope';
    els.pWarn.textContent = 'Relative and unitless. Display slope depends on the '
      + 'exaggeration setting — it is not a ground slope.';
  }
}

function showRefinement(meta) {
  const r = meta?.refined;
  els.modeRefined.disabled = !r;
  els.modeRefined.title = r ? '' : 'no building footprints for this tile';

  if (!r) {
    els.rFp.textContent = els.rEdge.textContent = els.rCal.textContent = '—';
    if (state.mode === 'refined') {
      state.mode = 'raw';
      document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b.dataset.mode === 'raw'));
    }
    return;
  }
  els.rFp.textContent = `${r.n_footprints} (${r.n_with_height} with height)`;
  els.rEdge.textContent = (r.edge_sharpness_before == null || r.edge_sharpness_after == null)
    ? 'n/a' : `${r.edge_sharpness_before.toFixed(3)} → ${r.edge_sharpness_after.toFixed(3)}`;
  const m = r.metric;
  els.rCal.textContent = !m ? 'no anchors'
    : m.usable ? `${m.scale_m.toFixed(1)} m · R² ${m.fit_r2.toFixed(2)} · n=${m.n_anchors}`
    : `rejected (${m.reject_reason})`;
}

function showMetrics(row) {
  els.mR2.textContent = fmt(row.r2);
  els.mRelief.textContent = fmt(row.rr);
  els.mStruct.textContent = fmt(row.st);

  const agg = state.index.class_aggregates?.[state.cls];
  els.mContext.textContent = agg
    ? `${state.cls}: n=${agg.n}, median R² ${fmt(agg.plane_r2.median, 2)} `
      + `(IQR ${fmt(agg.plane_r2.q1, 2)}–${fmt(agg.plane_r2.q3, 2)})`
    : '';

  const noisy = Number.isFinite(row.st) && Math.abs(row.st) < 0.05;
  if (state.mode === 'refined') {
    els.caveat.textContent = 'Footprint shapes are real map data (Overture). Roof heights are '
      + 'model-derived — verify against validation_summary.json before treating them as accurate.';
    els.caveat.classList.add('alert');
  } else {
    els.caveat.textContent = noisy
      ? 'Residual barely aligns with image structure — detrended view is mostly amplified noise.'
      : 'Detrended amplitude is renormalized per scene; relief height is not comparable between scenes.';
    els.caveat.classList.toggle('alert', noisy);
  }

  if (row.glb) {
    els.glb.href = `${DATA}/scenes/${row.id}/terrain.glb`;
    els.glb.classList.remove('off');
  } else {
    els.glb.classList.add('off');
  }
}

els.canvas.addEventListener('click', (event) => {
  const hit = pickTerrain(event, els.canvas, viewer);
  if (!hit) { els.pHeight.textContent = els.pSlope.textContent = '—'; return; }
  els.pHeight.textContent = hit.elevationM !== undefined ? `${hit.elevationM.toFixed(1)} m` : `${hit.height.toFixed(3)} rel`;
  els.pSlope.textContent = hit.slopeDegMetric !== undefined ? `${hit.slopeDegMetric.toFixed(1)}°` : `${hit.slopeDeg.toFixed(1)}°`;
});

els.classSel.addEventListener('change', () => { state.cls = els.classSel.value; state.current = null; renderList(); });
els.sort.addEventListener('change', renderList);

el('exag').addEventListener('input', (e) => {
  const v = Number(e.target.value);
  el('exag-out').textContent = v.toFixed(2);
  viewer?.setExaggeration(v);
});
el('cmap').addEventListener('change', (e) => viewer?.setColormap(e.target.value));

for (const btn of document.querySelectorAll('#mode button')) {
  btn.addEventListener('click', () => {
    if (btn.disabled) return;
    document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b === btn));
    state.mode = btn.dataset.mode;
    el('mode-hint').textContent = {
      raw: 'raw depth — the fitted ramp is included',
      detrended: 'fitted ramp removed — what is left is residual relief',
      refined: 'ground fitted off-footprint, roofs flattened to one level each',
    }[state.mode];
    if (state.mode === 'refined' || viewer?.variant === 'refined') selectScene(state.current);
    else viewer?.setMode(state.mode);
  });
}
for (const btn of document.querySelectorAll('#nav button')) {
  btn.addEventListener('click', () => {
    const applied = viewer?.setNavMode(btn.dataset.nav) ?? btn.dataset.nav;
    document.querySelectorAll('#nav button').forEach((b) => b.classList.toggle('on', b.dataset.nav === applied));
    el('nav-hint').textContent = applied === 'fly'
      ? 'pointer locked · WASD move · QE up/down · Shift fast · Esc release'
      : (btn.dataset.nav === 'fly' ? 'this browser refused pointer lock — staying in orbit mode'
        : 'drag to orbit, scroll to zoom');
  });
}

async function boot() {
  try {
    const res = await fetch(`${DATA}/index.json`);
    if (!res.ok) throw new Error(`index.json ${res.status}`);
    state.index = await res.json();
  } catch (err) {
    els.status.textContent = `could not load ${DATA}/index.json — run viewer/export_inria.py first (${err.message})`;
    return;
  }
  for (const s of state.index.scenes) state.byId.set(s.id, s);
  const idx = state.index;
  els.sub.textContent = `Inria HR 0.5m · ${idx.total} tiles, all georeferenced · `
    + `${idx.calibration_usable ?? 0}/${idx.calibration_attempted ?? 0} elevation-calibrated`;
  els.classSel.replaceChildren(...idx.classes.map((c) => {
    const o = document.createElement('option'); o.value = c; o.textContent = c; return o;
  }));
  state.cls = idx.classes[0];
  els.classSel.value = state.cls;
  renderList();
}
boot();
