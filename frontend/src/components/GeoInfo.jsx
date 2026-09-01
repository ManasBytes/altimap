import { metricParams } from '../three/probe';
import { fmt } from '../format';

/** Mirrors upload.js's showGeo(): note text + alert state depend on whether
 * the tile is georeferenced, has a DEM patch, and whether that patch passed
 * the fit-quality gate. */
function geoNote(meta) {
  const geo = meta?.geo;
  const abs = meta?.absolute;
  const metric = metricParams(meta);

  if (!geo?.georeferenced) {
    const hints = geo?.partial_hints;
    return {
      alert: false,
      text: hints && (hints.gcps || hints.rpcs)
        ? 'Has GCPs/RPCs but no CRS+transform — not enough for a map-grid footprint.'
        : 'No CRS or geotransform in this file — output stays relative and unitless.',
    };
  }
  if (!abs) {
    return { alert: false, text: 'Georeferenced, but no reference DEM covers this footprint.' };
  }
  if (metric) {
    return {
      alert: false,
      text: `Calibrated to ${abs.source} (${abs.reference_posting_m} m, bare earth): `
        + `${abs.scale_m.toFixed(1)} m range, fit R² ${abs.fit_r2.toFixed(2)}.`,
    };
  }
  return {
    alert: true,
    text: `Elevation calibration REJECTED (${abs.reject_reason}; fit R² ${fmt(abs.fit_r2, 2)}). Height stays relative.`,
  };
}

export default function GeoInfo({ meta }) {
  const geo = meta?.geo;
  const note = meta ? geoNote(meta) : null;

  return (
    <section>
      <h2>Georeferencing</h2>
      <dl>
        <dt>status</dt><dd>{geo ? (geo.georeferenced ? 'georeferenced' : 'none') : '—'}</dd>
        <dt>CRS</dt><dd>{geo?.georeferenced ? geo.crs.replace('EPSG:', 'EPSG ') : '—'}</dd>
        <dt>ground sample</dt><dd>{geo?.georeferenced ? `${geo.res_m[0].toFixed(3)} m/px` : '—'}</dd>
        <dt>footprint</dt><dd>{geo?.georeferenced ? `${geo.ground_m[0].toFixed(0)} × ${geo.ground_m[1].toFixed(0)} m` : '—'}</dd>
      </dl>
      {note && <p className={`hint${note.alert ? ' alert' : ''}`}>{note.text}</p>}
    </section>
  );
}
