import { useRef } from "react";
import { Play, Pause, Plus, Trash2, Download, Upload, X } from "lucide-react";

export default function RoutePanel({
  points,
  selected,
  playing,
  marking,
  onMark,
  onPlay,
  onClear,
  onDelete,
  command,
  speed,
  setSpeed,
  progress,
  sceneId,
  notify,
}) {
  const input = useRef(null);
  const send = (type, payload = {}) =>
    command({ type, id: performance.now(), ...payload });
  const save = () => {
    const url = URL.createObjectURL(
      new Blob(
        [
          JSON.stringify(
            {
              version: 1,
              scene: sceneId,
              coordinateSystem: "scene-world-xyz",
              points,
            },
            null,
            2,
          ),
        ],
        { type: "application/json" },
      ),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sceneId}-route.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const load = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (
        data.version !== 1 ||
        !Array.isArray(data.points) ||
        data.points.length > 200 ||
        !data.points.every(
          (p) =>
            Array.isArray(p) &&
            p.length === 3 &&
            p.every(
              (n) =>
                typeof n === "number" &&
                Number.isFinite(n) &&
                Math.abs(n) < 100,
            ),
        )
      )
        throw Error("Invalid route: expected up to 200 finite XYZ points.");
      if (data.scene !== sceneId)
        throw Error("This route belongs to a different scene.");
      send("load", { points: data.points });
      notify("Route loaded");
    } catch (error) {
      notify(error.message);
    }
  };
  return (
    <section className="control-section route-editor" id="route-panel">
      <div className="label-row">
        <label>Camera route</label>
        <span>{points.length} nodes</span>
      </div>
      <p>
        Place nodes on the surface. Drag the axis handles or edit XYZ below.
        Camera position follows the route; drag the viewport to look around.
      </p>
      <div className="route-actions">
        <button
          className={marking ? "tool-active" : ""}
          onClick={onMark}
          disabled={playing}
        >
          <Plus size={14} />
          {marking ? "Placing nodes" : "Add nodes"}
        </button>
        <button onClick={onPlay} disabled={points.length < 2}>
          {playing ? <Pause size={14} /> : <Play size={14} />}{" "}
          {playing ? "Pause" : "Play / resume"}
        </button>
      </div>
      <div className="label-row">
        <label htmlFor="route-speed">Route speed</label>
        <span>{speed.toFixed(2)} u/s</span>
      </div>
      <input
        id="route-speed"
        type="range"
        min=".1"
        max="2"
        step=".05"
        value={speed}
        onChange={(e) => setSpeed(+e.target.value)}
      />
      <progress aria-label="Route playback" value={progress} max="1" />
      <ol className="route-nodes">
        {points.map((p, i) => (
          <li key={i} className={i === selected ? "selected" : ""}>
            <button
              onClick={() => send("select", { index: i })}
              disabled={playing}
            >
              <span className="node-index">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span>{p.map((n) => n.toFixed(2)).join(" / ")}</span>
            </button>
          </li>
        ))}
      </ol>
      {selected !== null && points[selected] && (
        <div className="xyz-fields">
          {["X", "Y", "Z"].map((axis, i) => (
            <label key={axis}>
              {axis}
              <input
                aria-label={`Waypoint ${axis}`}
                type="number"
                step=".05"
                value={Number(points[selected][i].toFixed(3))}
                disabled={playing}
                onChange={(e) => {
                  if (e.target.value === "") return;
                  const p = [...points[selected]];
                  p[i] = Math.max(-99, Math.min(99, +e.target.value));
                  send("move", { coordinates: p });
                }}
              />
            </label>
          ))}
        </div>
      )}
      <div className="route-actions">
        <button onClick={onDelete} disabled={selected === null || playing}>
          <Trash2 size={13} />
          Delete
        </button>
        <button onClick={onClear} disabled={!points.length}>
          <X size={13} />
          Clear
        </button>
      </div>
      <div className="route-actions">
        <button onClick={save} disabled={!points.length}>
          <Download size={13} />
          Save route
        </button>
        <button onClick={() => input.current.click()} disabled={playing}>
          <Upload size={13} />
          Load route
        </button>
      </div>
      <input ref={input} type="file" accept=".json" hidden onChange={load} />
    </section>
  );
}
