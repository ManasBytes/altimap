import { initViewer } from './terrain.js';
import { metricParams, pickTerrain } from './probe.js';

const DATA = './data-uploads';

const el = (id) => document.getElementById(id);
const els = {
  drop: el('drop'), file: el('file'), upStatus: el('up-status'),
  list: el('scene-list'), status: el('status'), overlay: el('overlay'),
  canvas: el('view'),
  gStatus: el('g-status'), gCrs: el('g-crs'), gGsd: el('g-gsd'),
  gExtent: el('g-extent'), gNote: el('g-note'),
  pHeight: el('p-height'), pSlope: el('p-slope'),
  slopeLabel: el('slope-label'), pWarn: el('p-warn'),
  mR2: el('m-r2'), mRelief: el('m-relief'), mStruct: el('m-struct'),
  mSize: el('m-size'), caveat: el('m-caveat'), glb: el('glb'),
};

const state = { scenes: [], current: null };
const fmt = (v, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(d);

let viewer = null;
try {
  viewer = initViewer(els.canvas);
  window.__viewer = viewer;
} catch (err) {
  els.status.textContent = err.message;
}

function renderList() {
  els.list.replaceChildren(...state.scenes.map((s) => {
    const li = document.createElement('li');
    li.dataset.id = s.id;
    const badges = [
      s.georeferenced ? '<span class="badge">geo</span>' : '',
      s.calibrated ? '<span class="badge">metric</span>' : '',
    ].join(' ');
    li.innerHTML = `
      <img alt="" loading="lazy" src="${DATA}/scenes/${s.id}/rgb.jpg">
      <div>
        <div class="name">${s.source_image ?? s.id} ${badges}</div>
        <div class="meta">R² ${fmt(s.plane_r2, 2)} · str ${fmt(s.structure_alignment, 2)}</div>
      </div>
      <button class="x" title="remove">×</button>`;
    li.addEventListener('click', (e) => {
      if (e.target.classList.contains('x')) { removeScene(s.id); return; }
      selectScene(s.id);
    });
    return li;
  }));
  for (const li of els.list.children) li.classList.toggle('on', li.dataset.id === state.current);
}

async function removeScene(id) {
  await fetch(`/api/uploads/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (state.current === id) state.current = null;
  await refresh();
}

async function refresh() {
  const res = await fetch('/api/uploads');
  const index = await res.json();
  state.scenes = index.scenes;
  renderList();
}

function showGeo(meta) {
  const geo = meta?.geo;
  const abs = meta?.absolute;
  const metric = metricParams(meta);

  if (!geo?.georeferenced) {
    els.gStatus.textContent = 'none';
    els.gCrs.textContent = els.gGsd.textContent = els.gExtent.textContent = '—';
    const hints = geo?.partial_hints;
    els.gNote.textContent = hints && (hints.gcps || hints.rpcs)
      ? 'Has GCPs/RPCs but no CRS+transform — not enough for a map-grid footprint.'
      : 'No CRS or geotransform in this file — output stays relative and unitless.';
    els.gNote.classList.remove('alert');
  } else {
    els.gStatus.textContent = 'georeferenced';
    els.gCrs.textContent = geo.crs.replace('EPSG:', 'EPSG ');
    els.gGsd.textContent = `${geo.res_m[0].toFixed(3)} m/px`;
    els.gExtent.textContent = `${geo.ground_m[0].toFixed(0)} × ${geo.ground_m[1].toFixed(0)} m`;
    if (!abs) {
      els.gNote.textContent = 'Georeferenced, but no reference DEM covers this footprint.';
      els.gNote.classList.remove('alert');
    } else if (metric) {
      els.gNote.textContent = `Calibrated to ${abs.source} (${abs.reference_posting_m} m, bare earth): `
        + `${abs.scale_m.toFixed(1)} m range, fit R² ${abs.fit_r2.toFixed(2)}.`;
      els.gNote.classList.remove('alert');
    } else {
      els.gNote.textContent = `Elevation calibration REJECTED (${abs.reject_reason}; `
        + `fit R² ${fmt(abs.fit_r2, 2)}). Height stays relative.`;
      els.gNote.classList.add('alert');
    }
  }

  if (metric) {
    els.slopeLabel.textContent = 'ground slope';
    els.pWarn.textContent = 'Elevation calibrated against a 10 m bare-earth DEM: '
      + 'terrain level is metric, individual buildings are not resolved by that reference.';
  } else {
    els.slopeLabel.textContent = 'display slope';
    els.pWarn.textContent = 'Relative and unitless. Display slope depends on the '
      + 'exaggeration setting — it is not a ground slope.';
  }
}

async function selectScene(id) {
  state.current = id;
  renderList();
  els.pHeight.textContent = els.pSlope.textContent = '—';
  els.overlay.classList.remove('hidden');
  els.status.textContent = 'loading…';
  try {
    const meta = await viewer?.loadScene(`${DATA}/scenes/${id}`);
    showGeo(meta);
    els.mR2.textContent = fmt(meta.plane_r2);
    els.mRelief.textContent = fmt(meta.residual_relief);
    els.mStruct.textContent = fmt(meta.structure_alignment);
    els.mSize.textContent = meta.source_size
      ? `${meta.source_size[0]}×${meta.source_size[1]}` : '—';
    const noisy = Number.isFinite(meta.structure_alignment)
      && Math.abs(meta.structure_alignment) < 0.05;
    els.caveat.textContent = noisy
      ? 'Residual barely aligns with image structure — detrended view is mostly amplified noise.'
      : 'Detrended amplitude is renormalized per scene.';
    els.caveat.classList.toggle('alert', noisy);
    if (meta.has_glb) {
      els.glb.href = `${DATA}/scenes/${id}/terrain.glb`;
      els.glb.classList.remove('off');
    } else {
      els.glb.classList.add('off');
    }
    els.overlay.classList.add('hidden');
  } catch (err) {
    els.status.textContent = `failed to load: ${err.message}`;
  }
}

async function send(file) {
  els.drop.classList.add('busy');
  els.upStatus.textContent = `uploading ${file.name} (${(file.size / 1e6).toFixed(1)} MB)…`;
  const body = new FormData();
  body.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body });
    if (!res.ok) {
      // FastAPI reports the reason in `detail`; surfacing it beats "failed".
      let detail = `HTTP ${res.status}`;
      try { detail = (await res.json()).detail ?? detail; } catch { /* keep status */ }
      throw new Error(detail);
    }
    const meta = await res.json();
    els.upStatus.textContent = `${meta.source_image}: done in ${meta.seconds}s`
      + (meta.geo?.georeferenced ? ' · georeferenced' : '');
    await refresh();
    await selectScene(meta.id);
  } catch (err) {
    els.upStatus.textContent = `upload failed — ${err.message}`;
    els.upStatus.classList.add('alert');
    setTimeout(() => els.upStatus.classList.remove('alert'), 6000);
  } finally {
    els.drop.classList.remove('busy');
  }
}

els.drop.addEventListener('click', () => els.file.click());
els.file.addEventListener('change', () => {
  if (els.file.files[0]) send(els.file.files[0]);
  els.file.value = '';   // re-selecting the same file must fire change again
});
for (const type of ['dragenter', 'dragover']) {
  els.drop.addEventListener(type, (e) => { e.preventDefault(); els.drop.classList.add('over'); });
}
for (const type of ['dragleave', 'drop']) {
  els.drop.addEventListener(type, (e) => { e.preventDefault(); els.drop.classList.remove('over'); });
}
els.drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) send(f);
});
// Without this the browser navigates away to the dropped file.
for (const type of ['dragover', 'drop']) {
  window.addEventListener(type, (e) => { if (e.target !== els.drop) e.preventDefault(); });
}

els.canvas.addEventListener('click', (event) => {
  const hit = pickTerrain(event, els.canvas, viewer);
  if (!hit) { els.pHeight.textContent = els.pSlope.textContent = '—'; return; }
  els.pHeight.textContent = hit.elevationM !== undefined
    ? `${hit.elevationM.toFixed(1)} m` : `${hit.height.toFixed(3)} rel`;
  els.pSlope.textContent = hit.slopeDegMetric !== undefined
    ? `${hit.slopeDegMetric.toFixed(1)}°` : `${hit.slopeDeg.toFixed(1)}°`;
});

el('exag').addEventListener('input', (e) => {
  const v = Number(e.target.value);
  el('exag-out').textContent = v.toFixed(2);
  viewer?.setExaggeration(v);
});
el('cmap').addEventListener('change', (e) => viewer?.setColormap(e.target.value));
for (const btn of document.querySelectorAll('#mode button')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b === btn));
    viewer?.setMode(btn.dataset.mode);
  });
}
for (const btn of document.querySelectorAll('#nav button')) {
  btn.addEventListener('click', () => {
    const applied = viewer?.setNavMode(btn.dataset.nav) ?? btn.dataset.nav;
    document.querySelectorAll('#nav button').forEach(
      (b) => b.classList.toggle('on', b.dataset.nav === applied));
    el('nav-hint').textContent = applied === 'fly'
      ? 'pointer locked · WASD move · QE up/down · Shift fast · Esc release'
      : (btn.dataset.nav === 'fly'
        ? 'this browser refused pointer lock — staying in orbit mode'
        : 'drag to orbit, scroll to zoom');
  });
}

refresh().then(() => {
  if (state.scenes.length) selectScene(state.scenes[state.scenes.length - 1].id);
});
