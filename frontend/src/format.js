export function fmt(v, d = 3) {
  return v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(d);
}
