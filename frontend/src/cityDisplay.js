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
    let base = 0;
    if (b.kind === "bridge") {
      const ring = b.rings[0];
      const u = ring.reduce((sum, p) => sum + p[0], 0) / ring.length;
      const v = ring.reduce((sum, p) => sum + p[1], 0) / ring.length;
      base = Math.max(0, b.b - groundAt(u, v));
    }
    return { ...b, b: base, t: base + b.h };
  });
  const facilities = (sample.facilities ?? []).map((f) => {
    const home = buildings.find((b) => b.kind !== "bridge" &&
      inRing(f.u, f.v, b.rings[0]) && !b.rings.slice(1).some((r) => inRing(f.u, f.v, r)));
    return { ...f, z: home ? home.t : 0 };
  });
  return {
    buildings,
    trees: sample.city.trees.map((t) => ({ ...t, b: 0 })),
    facilities,
    embankments: [], // absolute crest elevations belong to the estimated-terrain view
  };
}
