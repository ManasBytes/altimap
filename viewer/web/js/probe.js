import * as THREE from 'three';

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

/**
 * Pick a point on the displaced terrain.
 *
 * The mesh is displaced in the vertex shader, so a CPU raycast hits the FLAT
 * plane -- at high exaggeration that is visibly not the surface under the
 * cursor. So: cast the ray, then march it against the height field and bisect
 * the crossing.
 */
export function pickTerrain(event, canvas, viewer) {
  if (!viewer?.meta) return null;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, viewer.camera);

  const origin = raycaster.ray.origin.clone();
  const dir = raycaster.ray.direction.clone();
  const exag = viewer.exaggeration;

  const toUv = (p) => [p.x + 0.5, 0.5 - p.z];
  const inside = ([u, v]) => u >= 0 && u <= 1 && v >= 0 && v <= 1;

  /** Signed height of the ray above the surface at parameter t, or null off-grid. */
  const above = (t) => {
    const p = origin.clone().addScaledVector(dir, t);
    const uv = toUv(p);
    if (!inside(uv)) return null;
    return p.y - viewer.heightAt(uv[0], uv[1]) * exag;
  };

  const STEP = 0.01;
  const MAX_T = 12.0;
  let prevT = 0;
  let prevSign = above(0);
  let hitT = null;
  for (let t = STEP; t < MAX_T; t += STEP) {
    const s = above(t);
    if (s === null) { prevT = t; prevSign = null; continue; }
    if (prevSign !== null && prevSign > 0 && s <= 0) {
      let lo = prevT;
      let hi = t;
      for (let i = 0; i < 24; i++) {           // bisect the crossing
        const mid = (lo + hi) / 2;
        const sm = above(mid);
        if (sm === null) break;
        if (sm > 0) lo = mid; else hi = mid;
      }
      hitT = (lo + hi) / 2;
      break;
    }
    prevT = t;
    prevSign = s;
  }
  if (hitT === null) return null;

  const hit = origin.clone().addScaledVector(dir, hitT);
  const uv = toUv(hit);
  if (!inside(uv)) return null;
  const [u, v] = uv;

  const height = viewer.heightAt(u, v);

  // Slope from central differences on the height field. Horizontal extent is
  // one world unit across the grid, so a one-texel step is 1/width. Vertical
  // scale is the current exaggeration -- which is exactly why this is reported
  // as a DISPLAY slope, not a ground slope. A real slope needs metric height.
  const du = 1 / viewer.meta.width;
  const dv = 1 / viewer.meta.height;
  const dhdu = ((viewer.heightAt(u + du, v) - viewer.heightAt(u - du, v)) * exag) / (2 * du);
  const dhdv = ((viewer.heightAt(u, v + dv) - viewer.heightAt(u, v - dv)) * exag) / (2 * dv);
  const slopeDeg = (Math.atan(Math.hypot(dhdu, dhdv)) * 180) / Math.PI;

  const out = { u, v, height, slopeDeg };

  // Metric readouts, only where the tile is georeferenced AND the
  // relative->absolute fit passed its quality gate. Both conditions matter: a
  // real ground scale with an inverted height fit still yields nonsense metres.
  const metric = metricParams(viewer.meta);
  if (metric) {
    const { groundM, scaleM, offsetM } = metric;
    out.elevationM = scaleM * height + offsetM;
    // Now that BOTH axes are metres, slope is a physical gradient rather than a
    // function of the exaggeration slider -- note exag is deliberately absent.
    const gu = ((viewer.heightAt(u + du, v) - viewer.heightAt(u - du, v)) * scaleM)
      / (2 * du * groundM);
    const gv = ((viewer.heightAt(u, v + dv) - viewer.heightAt(u, v - dv)) * scaleM)
      / (2 * dv * groundM);
    out.slopeDegMetric = (Math.atan(Math.hypot(gu, gv)) * 180) / Math.PI;
  }
  return out;
}

/** Ground scale + calibrated height mapping, or null if either is missing. */
export function metricParams(meta) {
  const abs = meta?.absolute;
  const ground = meta?.geo?.ground_m?.[0];
  if (!abs || !abs.usable || !ground) return null;
  if (!Number.isFinite(abs.scale_m) || !Number.isFinite(abs.offset_m)) return null;
  return { groundM: ground, scaleM: abs.scale_m, offsetM: abs.offset_m };
}
