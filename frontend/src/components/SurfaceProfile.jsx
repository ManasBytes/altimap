export default function SurfaceProfile({ stats, sample }) {
  const metric = sample.heightRange?.unit === "m";
  const convert = (v) =>
    metric
      ? sample.heightRange.min +
        v * (sample.heightRange.max - sample.heightRange.min)
      : v;
  const profile = stats?.profile || [];
  const lo = profile.length ? Math.min(...profile) : 0,
    hi = profile.length ? Math.max(...profile) : 1;
  const points = profile
    .map(
      (v, i) =>
        `${(i / (profile.length - 1)) * 300},${60 - ((v - lo) / Math.max(hi - lo, 0.0001)) * 50}`,
    )
    .join(" ");
  return (
    <section className="control-section profile">
      <div className="label-row">
        <label>Surface transect</label>
        <span>{metric ? "m" : "relative"}</span>
      </div>
      <div className="sparkline">
        <svg
          viewBox="0 0 300 70"
          role="img"
          aria-label="Height along the center row of the surface"
        >
          <polyline
            points={points}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
          />
        </svg>
      </div>
      <div className="profile-values">
        <span>{stats ? convert(stats.min).toFixed(2) : "—"} min</span>
        <span>{stats ? convert(stats.mean).toFixed(2) : "—"} mean</span>
        <span>{stats ? convert(stats.max).toFixed(2) : "—"} max</span>
      </div>
      <small>
        Center-row profile · statistics across the displayed surface
      </small>
    </section>
  );
}
