import { initViewer } from './terrain.js';
import { metricParams, pickTerrain } from './probe.js';

const DATA = './data-offnadir';

const el = (id) => document.getElementById(id);
const els = {
  list: el('scene-list'), classSel: el('class-sel'), sort: el('sort'),
  status: el('status'), overlay: el('overlay'), canvas: el('view'),
  sub: el('ds-sub'), oa: el('oa'), oaOut: el('oa-out'), oaHint: el('oa-hint'),
  gStatus: el('g-status'), gCrs: el('g-crs'), gGsd: el('g-gsd'),
  gExtent: el('g-extent'), gNote: el('g-note'),
  pHeight: el('p-height'), pSlope: el('p-slope'),
  slopeLabel: el('slope-label'), pWarn: el('p-warn'),
  mR2: el('m-r2'), mRelief: el('m-relief'), mStruct: el('m-struct'),
  mContext: el('m-context'), caveat: el('m-caveat'), glb: el('glb'),
  modeRefined: el('mode-refined'), modeHint: el('mode-hint'),
  rFp: el('r-fp'), rEdge: el('r-edge'), rCal: el('r-cal'),
};

const state = {
  index: null,
  byId: new Map(),
  groups: new Map(),   // "class|scene" -> { angle -> id }
  cls: null,
  scene: null,         // scene number within the class
  angleIdx: 0,
  mode: 'raw',
};

const fmt = (v, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(d);

let viewer = null;
try {
  viewer = initViewer(els.canvas);
  window.__viewer = viewer;
} catch (err) {
  els.status.textContent = err.message;
}

const groupKey = (cls, scene) => `${cls}|${scene}`;

/** Scenes of the current class, one entry per scene number. */
function scenesForClass() {
  const rows = [];
  for (const [key, byAngle] of state.groups) {
    const [cls, scene] = key.split('|');
    if (cls !== state.cls) continue;
    // Represent a scene by its most-nadir angle -- that is the closest thing
    // this dataset has to the nadir view the brief is written around.
    const angles = Object.keys(byAngle).map(Number).sort((a, b) => a - b);
    const rep = state.byId.get(byAngle[String(angles[0])]);
    if (rep) rows.push({ scene: Number(scene), rep, angles, byAngle });
  }
  const key = els.sort.value;
  rows.sort((a, b) => {
    if (key === 'plane_r2') return (a.rep.r2 ?? Infinity) - (b.rep.r2 ?? Infinity);
    if (key === 'structure_alignment') return (b.rep.st ?? -Infinity) - (a.rep.st ?? -Infinity);
    if (key === 'geo') {
      const ga = a.angles.some((x) => state.byId.get(a.byAngle[String(x)])?.g);
      const gb = b.angles.some((x) => state.byId.get(b.byAngle[String(x)])?.g);
      if (ga !== gb) return gb - ga;
    }
    return a.scene - b.scene;
  });
  return rows;
}

function renderList() {
  const rows = scenesForClass();
  els.list.replaceChildren(...rows.map((row) => {
    const li = document.createElement('li');
    li.dataset.scene = row.scene;
    const geoCount = row.angles.filter(
      (a) => state.byId.get(row.byAngle[String(a)])?.g).length;
    li.innerHTML = `
      <img alt="" loading="lazy" src="${DATA}/scenes/${row.rep.id}/rgb.jpg">
      <div>
        <div class="name">scene ${row.scene}${geoCount ? ' <span class="badge">geo</span>' : ''}</div>
        <div class="meta">R² ${fmt(row.rep.r2, 2)} · str ${fmt(row.rep.st, 2)} · ${row.angles.length} angles</div>
      </div>`;
    li.addEventListener('click', () => selectScene(row.scene, 0));
    return li;
  }));
  if (rows.length) {
    const stillThere = rows.some((r) => r.scene === state.scene);
    selectScene(stillThere ? state.scene : rows[0].scene, stillThere ? state.angleIdx : 0);
  }
}

function markActive() {
  for (const li of els.list.children) {
    li.classList.toggle('on', Number(li.dataset.scene) === state.scene);
  }
}

function currentGroup() {
  return state.groups.get(groupKey(state.cls, state.scene));
}

function anglesOf(group) {
  return Object.keys(group).map(Number).sort((a, b) => a - b);
}

async function selectScene(sceneNum, angleIdx) {
  state.scene = sceneNum;
  const group = currentGroup();
  if (!group) return;
  const angles = anglesOf(group);
  state.angleIdx = Math.min(Math.max(angleIdx, 0), angles.length - 1);

  els.oa.max = String(angles.length - 1);
  els.oa.value = String(state.angleIdx);
  markActive();
  await loadCurrent();
}

async function loadCurrent() {
  const group = currentGroup();
  const angles = anglesOf(group);
  const oa = angles[state.angleIdx];
  const id = group[String(oa)];
  const row = state.byId.get(id);

  els.oaOut.textContent = `OA${oa}°`;
  showMetrics(row);
  els.pHeight.textContent = '—';
  els.pSlope.textContent = '—';
  els.overlay.classList.remove('hidden');
  els.status.textContent = `loading ${state.cls} scene ${state.scene} @ OA${oa}°…`;
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

  if (!geo?.georeferenced) {
    els.gStatus.textContent = 'none';
    els.gCrs.textContent = '—';
    els.gGsd.textContent = '—';
    els.gExtent.textContent = '—';
    // 4580 of 5200 tiles are plain rasters in a .tif container. Say so, rather
    // than letting the file extension imply spatial metadata that is absent.
    els.gNote.textContent = 'Plain raster in a .tif container — no CRS, transform, GCPs or RPCs.';
    els.gNote.classList.remove('alert');
  } else {
    els.gStatus.textContent = 'georeferenced';
    els.gCrs.textContent = geo.crs.replace('EPSG:', 'EPSG ');
    els.gGsd.textContent = `${geo.res_m[0].toFixed(2)} m/px`;
    els.gExtent.textContent = `${geo.ground_m[0].toFixed(0)} × ${geo.ground_m[1].toFixed(0)} m`;
    if (!abs) {
      els.gNote.textContent = 'No DEM patch retrieved for this footprint.';
      els.gNote.classList.remove('alert');
    } else if (metric) {
      els.gNote.textContent = `Calibrated to ${abs.source} (${abs.reference_posting_m} m, bare earth): `
        + `${abs.scale_m.toFixed(1)} m range, fit R² ${abs.fit_r2.toFixed(2)}. `
        + 'Bare-earth reference constrains terrain level, not building height.';
      els.gNote.classList.remove('alert');
    } else {
      els.gNote.textContent = `Elevation calibration REJECTED (${abs.reject_reason}; `
        + `fit R² ${fmt(abs.fit_r2, 2)}, scale ${fmt(abs.scale_m, 1)} m). Height stays relative.`;
      els.gNote.classList.add('alert');
    }
  }

  // The probe labels have to follow the calibration, not the georeferencing:
  // a real ground scale with a rejected height fit still cannot yield metres.
  if (metric) {
    els.slopeLabel.textContent = 'ground slope';
    els.pWarn.textContent = 'Elevation is calibrated against a 10 m bare-earth DEM. '
      + 'Terrain level is metric; individual buildings are not resolved by that reference.';
  } else {
    els.slopeLabel.textContent = 'display slope';
    els.pWarn.textContent = 'Relative and unitless. Display slope depends on the '
      + 'exaggeration setting — it is not a ground slope.';
  }
}

function showRefinement(meta) {
  const r = meta?.refined;
  // The refined surface only exists where footprints could be rasterised, i.e.
  // georeferenced tiles inside the Overture extract. Disable rather than hide,
  // so it is visible that the mode exists and why it is unavailable here.
  els.modeRefined.disabled = !r;
  els.modeRefined.title = r ? '' : 'no building footprints for this tile';

  if (!r) {
    els.rFp.textContent = els.rEdge.textContent = els.rCal.textContent = '—';
    if (state.mode === 'refined') {
      state.mode = 'raw';
      document.querySelectorAll('#mode button').forEach(
        (b) => b.classList.toggle('on', b.dataset.mode === 'raw'));
    }
    return;
  }

  els.rFp.textContent = `${r.n_footprints} (${r.n_with_height} with height)`;
  const before = r.edge_sharpness_before, after = r.edge_sharpness_after;
  els.rEdge.textContent = (before == null || after == null)
    ? 'n/a'
    : `${before.toFixed(3)} → ${after.toFixed(3)}`;

  const m = r.metric;
  if (!m) {
    els.rCal.textContent = 'no anchors';
  } else if (m.usable) {
    els.rCal.textContent = `${m.scale_m.toFixed(1)} m · R² ${m.fit_r2.toFixed(2)} · n=${m.n_anchors}`;
  } else {
    els.rCal.textContent = `rejected (${m.reject_reason})`;
  }
}

function showMetrics(row) {
  els.mR2.textContent = fmt(row.r2);
  els.mRelief.textContent = fmt(row.rr);
  els.mStruct.textContent = fmt(row.st);

  const agg = state.index.class_aggregates?.[state.cls];
  const angleAgg = state.index.angle_aggregates?.[String(row.a)];
  const parts = [];
  if (agg?.plane_r2) {
    parts.push(`${state.cls}: n=${agg.n}, median R² ${fmt(agg.plane_r2.median, 2)}`);
  }
  if (angleAgg?.plane_r2) {
    parts.push(`OA${row.a}° across all classes: median R² ${fmt(angleAgg.plane_r2.median, 2)}`
      + (angleAgg.structure_alignment ? `, struct ${fmt(angleAgg.structure_alignment.median, 2)}` : ''));
  }
  els.mContext.textContent = parts.join(' · ');

  // Refined geometry is far more convincing than the height data behind it, so
  // its warning takes precedence. Footprint SHAPES are real map data (Overture);
  // roof HEIGHTS are model-derived and measured at +0.004 median correlation
  // against reference heights over 28,960 buildings.
  const noisy = Number.isFinite(row.st) && Math.abs(row.st) < 0.05;
  if (state.mode === 'refined') {
    els.caveat.textContent = 'Footprint shapes are real map data (Overture). Roof '
      + 'heights are model-derived and do NOT correlate with reference heights '
      + '(median r = +0.004 over 28,960 buildings) — treat building heights here '
      + 'as unreliable.';
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
  if (!hit) {
    els.pHeight.textContent = '—';
    els.pSlope.textContent = '—';
    return;
  }
  els.pHeight.textContent = hit.elevationM !== undefined
    ? `${hit.elevationM.toFixed(1)} m` : `${hit.height.toFixed(3)} rel`;
  els.pSlope.textContent = hit.slopeDegMetric !== undefined
    ? `${hit.slopeDegMetric.toFixed(1)}°` : `${hit.slopeDeg.toFixed(1)}°`;
});

els.oa.addEventListener('input', () => {
  state.angleIdx = Number(els.oa.value);
  loadCurrent();
});

els.classSel.addEventListener('change', () => {
  state.cls = els.classSel.value;
  state.scene = null;
  renderList();
});
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
    els.modeHint.textContent = {
      raw: 'raw depth — the fitted ramp is included',
      detrended: 'fitted ramp removed — what is left is residual relief',
      refined: 'ground fitted off-footprint, roofs flattened to one level each',
    }[state.mode];

    // 'refined' is a different height field on disk, so it needs a reload;
    // 'raw' and 'detrended' are a uniform flip on the field already loaded.
    if (state.mode === 'refined' || viewer?.variant === 'refined') {
      loadCurrent();
    } else {
      viewer?.setMode(state.mode);
    }
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

async function boot() {
  try {
    const res = await fetch(`${DATA}/index.json`);
    if (!res.ok) throw new Error(`index.json ${res.status}`);
    state.index = await res.json();
  } catch (err) {
    els.status.textContent =
      `could not load ${DATA}/index.json — run viewer/export_offnadir.py first (${err.message})`;
    return;
  }

  for (const row of state.index.scenes) state.byId.set(row.id, row);
  for (const g of state.index.groups) {
    state.groups.set(groupKey(g.class, g.scene), g.by_angle);
  }

  const idx = state.index;
  els.sub.textContent = `Atlanta · ${idx.total} tiles · ${idx.georeferenced} georeferenced · `
    + `${idx.calibration_usable ?? 0} elevation-calibrated`;
  els.oaHint.textContent = `same ground, ${idx.angles.length} view angles `
    + `(OA${idx.angles[0]}–OA${idx.angles[idx.angles.length - 1]}) — scrub to see depth degrade`;

  els.classSel.replaceChildren(...idx.classes.map((c) => {
    const o = document.createElement('option');
    o.value = c; o.textContent = c;
    return o;
  }));
  state.cls = idx.classes[0];
  els.classSel.value = state.cls;
  renderList();
}

boot();
