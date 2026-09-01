export default function ProbeInfo({ probe, metric }) {
  return (
    <section>
      <h2>Probe</h2>
      <p className="hint">click the terrain</p>
      <dl>
        <dt>height</dt>
        <dd>
          {probe
            ? (probe.elevationM !== undefined ? `${probe.elevationM.toFixed(1)} m` : `${probe.height.toFixed(3)} rel`)
            : '—'}
        </dd>
        <dt>{metric ? 'ground slope' : 'display slope'}</dt>
        <dd>
          {probe
            ? (probe.slopeDegMetric !== undefined ? `${probe.slopeDegMetric.toFixed(1)}°` : `${probe.slopeDeg.toFixed(1)}°`)
            : '—'}
        </dd>
      </dl>
      <p className="warn">
        {metric
          ? 'Elevation calibrated against a 10 m bare-earth DEM: terrain level is metric, individual buildings are not resolved by that reference.'
          : 'Relative and unitless. Display slope depends on the exaggeration setting — it is not a ground slope.'}
      </p>
    </section>
  );
}
