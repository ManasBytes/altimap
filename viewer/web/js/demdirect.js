import { initViewer } from './terrain.js';
import { pickTerrain } from './probe.js';

const DATA = './data-demdirect';

const el = (id) => document.getElementById(id);
const els = {
  list: el('scene-list'), status: el('status'), overlay: el('overlay'), canvas: el('view'),
  sub: el('ds-sub'),
  gDem: el('g-dem'), gCrs: el('g-crs'), gGsd: el('g-gsd'), gExtent: el('g-extent'),
  gRange: el('g-range'), gNote: el('g-note'),
  pHeight: el('p-height'), pSlope: el('p-slope'),
  glb: el('glb'), exag: el('exag'), exagOut: el('exag-out'), trueScale: el('true-scale'),
  modeBuildings: el('mode-buildings'), modeHint: el('mode-hint'),
  bFp: el('b-fp'), bKnown: el('b-known'), bExt: el('b-ext'),
};

const state = { index: null, current: null, mode: 'dem' };

let viewer = null;
try {
  // terrain.js's default mesh density (768 segments) is enough that buildings
  // here -- which carry real sub-metre footprint precision from Overture, not
  // a model guess -- render as recognisable raised blocks rather than a soft
  // bump. This does NOT make individual cars resolvable: that ceiling is the
  // data (no source catalogs vehicle positions), not mesh density.
  viewer = initViewer(els.canvas);
  window.__viewer = viewer;
} catch (err) {
  els.status.textContent = err.message;
}

function renderList() {
  els.list.replaceChildren(...state.index.scenes.map((s) => {
    const li = document.createElement('li');
    li.dataset.id = s.id;
    li.innerHTML = `
      <img alt="" loading="lazy" src="${DATA}/scenes/${s.id}/rgb.jpg">
      <div>
        <div class="name">${s.label}${s.n_extruded ? ` <span class="badge">${s.n_extruded} bldg</span>` : ''}</div>
        <div class="meta">relief ${s.elevation_range_m.toFixed(0)} m · ${s.crs.replace('EPSG:', 'EPSG ')}</div>
      </div>`;
    li.addEventListener('click', () => selectScene(s.id));
    return li;
  }));
  if (state.index.scenes.length) {
    const stillThere = state.index.scenes.some((s) => s.id === state.current);
    selectScene(stillThere ? state.current : state.index.scenes[0].id);
  }
}

function markActive() {
  for (const li of els.list.children) li.classList.toggle('on', li.dataset.id === state.current);
}

async function selectScene(id) {
  state.current = id;
  markActive();
  els.pHeight.textContent = els.pSlope.textContent = '—';
  els.overlay.classList.remove('hidden');
  els.status.textContent = `loading ${id}…`;
  try {
    // terrain.js's shared loader calls this variant 'refined' internally
    // (it was built for the footprint-refined model-depth pages); reusing
    // that exact mechanism here means zero further changes to terrain.js.
    const meta = await viewer?.loadScene(`${DATA}/scenes/${id}`,
      { variant: state.mode === 'buildings' ? 'refined' : 'depth' });
    showMeta(meta);
    // Every tile has its own true-scale exaggeration (elevation range / footprint),
    // and the buildings variant has a DIFFERENT range (extrusions raise the max) so
    // a different true scale -- pick the one matching what's actually displayed.
    const trueScale = (state.mode === 'buildings' && meta.refined
      ? meta.refined.true_scale_exaggeration : meta.true_scale_exaggeration) ?? 1;
    els.exag.value = String(Math.min(trueScale, Number(els.exag.max)));
    els.exagOut.textContent = trueScale.toFixed(3);
    viewer?.setExaggeration(trueScale);
    els.overlay.classList.add('hidden');
  } catch (err) {
    els.status.textContent = `failed to load ${id}: ${err.message}`;
  }
}

function showMeta(meta) {
  els.gDem.textContent = meta.dem_source;
  els.gDem.title = meta.dem_item ?? '';
  els.gCrs.textContent = meta.geo.crs.replace('EPSG:', 'EPSG ');
  els.gGsd.textContent = `${meta.geo.res_m[0].toFixed(2)} m/px`;
  els.gExtent.textContent = `${meta.geo.ground_m[0].toFixed(0)} × ${meta.geo.ground_m[1].toFixed(0)} m`;
  const e = meta.elevation_m;
  els.gRange.textContent = `${e.min.toFixed(1)} – ${e.max.toFixed(1)} m (Δ${e.range.toFixed(1)} m)`;
  els.gNote.textContent = state.index.scenes.find((s) => s.id === meta.id)?.note ?? '';

  const r = meta.refined;
  els.modeBuildings.disabled = !r;
  els.modeBuildings.title = r ? '' : 'no Overture footprints for this tile';
  if (r) {
    els.bFp.textContent = r.n_footprints;
    els.bKnown.textContent = `${r.n_with_height} (${(100 * r.n_with_height / r.n_footprints).toFixed(0)}%)`;
    els.bExt.textContent = r.n_extruded;
  } else {
    els.bFp.textContent = els.bKnown.textContent = els.bExt.textContent = '—';
    if (state.mode === 'buildings') {
      state.mode = 'dem';
      document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b.dataset.mode === 'dem'));
    }
  }

  const usingBuildings = state.mode === 'buildings' && !!r;
  const glbFile = usingBuildings ? 'terrain_buildings.glb' : 'terrain.glb';
  if (usingBuildings ? meta.has_buildings_glb : meta.has_glb) {
    els.glb.href = `${DATA}/scenes/${meta.id}/${glbFile}`;
    els.glb.classList.remove('off');
  } else {
    els.glb.classList.add('off');
  }
}

els.canvas.addEventListener('click', (event) => {
  const hit = pickTerrain(event, els.canvas, viewer);
  if (!hit) { els.pHeight.textContent = els.pSlope.textContent = '—'; return; }
  // metricParams() always resolves here (absolute.usable is unconditionally
  // true in this exporter's meta -- see export_dem_direct.py), so elevationM /
  // slopeDegMetric are always present: this page never falls back to relative.
  els.pHeight.textContent = `${hit.elevationM.toFixed(1)} m`;
  els.pSlope.textContent = `${hit.slopeDegMetric.toFixed(1)}°`;
});

els.exag.addEventListener('input', (e) => {
  const v = Number(e.target.value);
  els.exagOut.textContent = v.toFixed(3);
  viewer?.setExaggeration(v);
});
els.trueScale.addEventListener('click', () => selectScene(state.current));

for (const btn of document.querySelectorAll('#mode button')) {
  btn.addEventListener('click', () => {
    if (btn.disabled) return;
    document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b === btn));
    state.mode = btn.dataset.mode;
    els.modeHint.textContent = state.mode === 'buildings'
      ? 'ground from the real DEM, buildings extruded to their real Overture height'
      : '30 m reference terrain — buildings are not individually resolved at this posting.';
    selectScene(state.current);
  });
}

el('cmap').addEventListener('change', (e) => viewer?.setColormap(e.target.value));
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
    els.status.textContent = `could not load ${DATA}/index.json — run viewer/export_dem_direct.py first (${err.message})`;
    return;
  }
  els.sub.textContent = `${state.index.scenes.length} tiles · terrain = ${state.index.dem_source} · no depth model`;
  renderList();
}
boot();
