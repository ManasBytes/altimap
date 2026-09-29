import { useEffect, useState } from "react";
export default function ModelReport() {
  const [state, setState] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch("/api/model-status")
      .then((r) => {
        if (!r.ok) throw Error("Unavailable");
        return r.json();
      })
      .then((r) => alive && setState(r))
      .catch(() => alive && setState({ error: true }));
    return () => {
      alive = false;
    };
  }, []);
  const report = state?.report,
    test = report?.test;
  return (
    <section className="control-section model-report">
      <div className="label-row">
        <label>Model validation</label>
        <span>{state?.surface_available ? "Trained" : "Not loaded"}</span>
      </div>
      {!test ? (
        <p>
          {state?.error
            ? "Validation report unavailable."
            : "No trained surface report available yet."}
        </p>
      ) : (
        <>
          <p>
            ResNet34 U-Net · joint land cover and surface height.
          </p>
          <p>
            GAMUS held-out test tiles. Heights are predictions; uploaded RGB has
            no absolute elevation reference.
          </p>
          <div className="metric-grid">
            <div>
              <small>Test mIoU</small>
              <strong>{(test.miou * 100).toFixed(1)}%</strong>
            </div>
            <div>
              <small>AGL MAE</small>
              <strong>{test.agl_mae_m.toFixed(2)} m</strong>
            </div>
          </div>
          {Object.entries(test.iou).map(([name, v]) => (
            <div className="model-score" key={name}>
              <span>{name.replace("_", " ")}</span>
              <meter min="0" max="1" value={v || 0} />
              <span>{v === null ? "absent" : `${(v * 100).toFixed(1)}%`}</span>
            </div>
          ))}
          <p>
            {report.split_ids?.train.length} training ·{" "}
            {report.split_ids?.val.length} validation ·{" "}
            {report.split_ids?.test.length} test tiles. Native pixels,
            overlapping inference.
          </p>
        </>
      )}
    </section>
  );
}
