// Display-only flat ground. Original terrain, heights and geographic coordinates stay intact.
function inRing(u, v, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if ((yi > v) !== (yj > v) && u < ((xj - xi) * (v - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

export function buildingContains(b, u, v) {
  return inRing(u, v, b.rings[0]) && !b.rings.slice(1).some((r) => inRing(u, v, r));
}

export function roofElevation(building, u, v) {
  for (const face of building.roof?.faces ?? []) {
    const [a, b, c] = face.map((i) => building.roof.vertices[i]);
    const denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if (Math.abs(denominator) < 1e-15) continue;
    const wa = ((b[1] - c[1]) * (u - c[0]) + (c[0] - b[0]) * (v - c[1])) / denominator;
    const wb = ((c[1] - a[1]) * (u - c[0]) + (a[0] - c[0]) * (v - c[1])) / denominator;
    const wc = 1 - wa - wb;
    if (Math.min(wa, wb, wc) >= -1e-8) return wa * a[2] + wb * b[2] + wc * c[2];
  }
  return building.t;
}

export function flatCity(sample) {
  const ground = sample.groundGrid;
  const side = Math.round(Math.sqrt(ground?.length ?? 0));
  const groundAt = (u, v) => {
    if (!side) return 0;
    const col = Math.max(0, Math.min(side - 1, Math.round(u * (side - 1))));
    const row = Math.max(0, Math.min(side - 1, Math.round(v * (side - 1))));
    return ground[row * side + col] - sample.terrain.ground.min_m;
  };
  const buildings = sample.city.buildings.map((b) => {
    let base = b.min_height_m ?? 0;
    let top = b.h;
    if (b.kind === "bridge") {
      const ring = b.rings[0];
      const u = ring.reduce((sum, p) => sum + p[0], 0) / ring.length;
      const v = ring.reduce((sum, p) => sum + p[1], 0) / ring.length;
      base = Math.max(0, b.b - groundAt(u, v));
      top = base + b.h;
    }
    const roof = b.roof && { ...b.roof,
      vertices: b.roof.vertices.map(([u, v, z]) => [u, v, z - ((b.t ?? b.h) - b.h)]) };
    return { ...b, b: base, t: top, ...(roof ? { roof } : {}) };
  });
  const facilities = (sample.facilities ?? []).map((f) => {
    const candidates = buildings.filter((b) => b.kind !== "bridge" && buildingContains(b, f.u, f.v));
    return { ...f, z: candidates.length ? Math.max(...candidates.map((b) => roofElevation(b, f.u, f.v))) : 0 };
  });
  return {
    buildings,
    trees: sample.city.trees.map((t) => ({ ...t, b: 0 })),
    facilities,
    embankments: [], // absolute crest elevations belong to the estimated-terrain view
  };
}
