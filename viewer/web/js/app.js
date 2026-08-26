import { initViewer } from './terrain.js';
import { pickTerrain } from './probe.js';

const DATA = './data';

const els = {
  list: document.getElementById('scene-list'),
  sort: document.getElementById('sort'),
  status: document.getElementById('status'),
  overlay: document.getElementById('overlay'),
  canvas: document.getElementById('view'),
  mR2: document.getElementById('m-r2'),
  mRelief: document.getElementById('m-relief'),
  mStruct: document.getElementById('m-struct'),
  mConf: document.getElementById('m-conf'),
  mContext: document.getElementById('m-context'),
  caveat: document.getElementById('m-caveat'),
  glb: document.getElementById('glb'),
  pHeight: document.getElementById('p-height'),
  pSlope: document.getElementById('p-slope'),
};

const state = { index: null, current: null };

const fmt = (v, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(d);

let viewer = null;
try {
  viewer = initViewer(els.canvas);
  // Exposed for console inspection and headless checks -- e.g. verifying that
  // heightAt(u,v) at a fixed uv is independent of the exaggeration setting.
  window.__viewer = viewer;
} catch (err) {
  els.status.textContent = err.message;
}

function sortScenes(key) {
  const s = [...state.index.scenes];
  if (key === 'class') return s.sort((a, b) => a.class.localeCompare(b.class));
  if (key === 'structure_alignment') {
    return s.sort((a, b) =>
      (b.structure_alignment ?? -Infinity) - (a.structure_alignment ?? -Infinity));
  }
  return s.sort((a, b) => (a.plane_r2 ?? Infinity) - (b.plane_r2 ?? Infinity));
}

function markActive() {
  for (const li of els.list.children) li.classList.toggle('on', li.dataset.id === state.current);
}

function renderList() {
  const scenes = sortScenes(els.sort.value);
  els.list.replaceChildren(...scenes.map((scene) => {
    const li = document.createElement('li');
    li.dataset.id = scene.id;
    li.innerHTML = `
      <img alt="" loading="lazy" src="${DATA}/scenes/${scene.id}/rgb.jpg">
      <div>
        <div class="name">${scene.class}</div>
        <div class="meta">R² ${fmt(scene.plane_r2, 2)} · str ${fmt(scene.structure_alignment, 2)}</div>
      </div>`;
    li.addEventListener('click', () => selectScene(scene.id));
    return li;
  }));
  if (!state.current && scenes.length) selectScene(scenes[0].id);
  else markActive();
}

function showMetrics(scene) {
  els.mR2.textContent = fmt(scene.plane_r2);
  els.mRelief.textContent = fmt(scene.residual_relief);
  els.mStruct.textContent = fmt(scene.structure_alignment);
  els.mConf.textContent = fmt(scene.conf_texture_corr);

  // Situate this scene against its class distribution from the full-dataset
  // run, so a single number is never read without its spread.
  const agg = state.index.class_aggregates?.[scene.class];
  els.mContext.textContent = agg
    ? `${scene.class}: n=${agg.n}, class median R² ${fmt(agg.plane_r2.median, 2)} `
      + `(IQR ${fmt(agg.plane_r2.q1, 2)}–${fmt(agg.plane_r2.q3, 2)})`
    : '';

  // Both modes renormalize to [0,1], which is what makes the exaggeration
  // slider behave the same everywhere -- but it also means a scene whose
  // residual is 0.2% of the signal renders at the same visual amplitude as one
  // whose residual is 40%. Without this warning the detrended view of a
  // pure-ramp scene reads as terrain when it is amplified noise.
  const struct = scene.structure_alignment;
  const noisy = Number.isFinite(struct) && Math.abs(struct) < 0.05;
  els.caveat.textContent = noisy
    ? 'Residual barely aligns with image structure — in detrended view this is '
      + 'mostly amplified noise, not relief.'
    : 'Detrended amplitude is renormalized per scene; visual relief height is '
      + 'not comparable between scenes.';
  els.caveat.classList.toggle('alert', noisy);

  if (state.index.has_glb) {
    els.glb.href = `${DATA}/scenes/${scene.id}/terrain.glb`;
    els.glb.classList.remove('off');
  } else {
    els.glb.classList.add('off');
  }
}

async function selectScene(id) {
  state.current = id;
  markActive();
  const scene = state.index.scenes.find((s) => s.id === id);
  showMetrics(scene);
  els.pHeight.textContent = '—';
  els.pSlope.textContent = '—';
  els.overlay.classList.remove('hidden');
  els.status.textContent = `loading ${scene.class}…`;
  try {
    await viewer?.loadScene(`${DATA}/scenes/${id}`);
    els.overlay.classList.add('hidden');
  } catch (err) {
    els.status.textContent = `failed to load ${id}: ${err.message}`;
    els.list.querySelector(`[data-id="${id}"]`)?.classList.add('err');
  }
}

els.canvas.addEventListener('click', (event) => {
  const hit = pickTerrain(event, els.canvas, viewer);
  els.pHeight.textContent = hit ? `${hit.height.toFixed(3)} rel` : '—';
  els.pSlope.textContent = hit ? `${hit.slopeDeg.toFixed(1)}°` : '—';
});

document.getElementById('exag').addEventListener('input', (e) => {
  const v = Number(e.target.value);
  document.getElementById('exag-out').textContent = v.toFixed(2);
  viewer?.setExaggeration(v);
});

document.getElementById('cmap').addEventListener('change', (e) => viewer?.setColormap(e.target.value));

for (const btn of document.querySelectorAll('#mode button')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#mode button').forEach((b) => b.classList.toggle('on', b === btn));
    viewer?.setMode(btn.dataset.mode);
    document.getElementById('mode-hint').textContent = btn.dataset.mode === 'detrended'
      ? 'fitted ramp removed — what is left is residual relief'
      : 'raw depth — the fitted ramp is included';
  });
}

for (const btn of document.querySelectorAll('#nav button')) {
  btn.addEventListener('click', () => {
    // setNavMode reports the mode actually in effect -- pointer lock can be
    // refused, in which case the buttons must not claim fly mode.
    const applied = viewer?.setNavMode(btn.dataset.nav) ?? btn.dataset.nav;
    document.querySelectorAll('#nav button').forEach(
      (b) => b.classList.toggle('on', b.dataset.nav === applied));
    document.getElementById('nav-hint').textContent = applied === 'fly'
      ? 'pointer locked · WASD move · QE up/down · Shift fast · Esc release'
      : (btn.dataset.nav === 'fly'
        ? 'this browser refused pointer lock — staying in orbit mode'
        : 'drag to orbit, scroll to zoom');
  });
}

async function boot() {
  try {
    const res = await fetch(`${DATA}/index.json`);
    if (!res.ok) throw new Error(`index.json ${res.status}`);
    state.index = await res.json();
  } catch (err) {
    els.status.textContent =
      `could not load ${DATA}/index.json — run viewer/export_scenes.py first (${err.message})`;
    return;
  }
  els.sort.addEventListener('change', renderList);
  renderList();
}

boot();
