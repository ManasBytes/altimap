import { fmt } from '../format';

export default function SceneList({ scenes, currentId, onSelect, onDelete }) {
  return (
    <ul id="scene-list">
      {scenes.map((s) => (
        <li
          key={s.id}
          className={s.id === currentId ? 'on' : ''}
          onClick={() => onSelect(s.id)}
        >
          {s.status === 'done' && s.rgb_url
            ? <img alt="" loading="lazy" src={s.rgb_url} />
            : <div className={`thumb-status ${s.status}`}>{s.status === 'failed' ? '!' : '…'}</div>}
          <div>
            <div className="name">
              {s.original_filename ?? s.id}{' '}
              {s.georeferenced && <span className="badge">geo</span>}{' '}
              {s.calibrated && <span className="badge">metric</span>}
            </div>
            <div className="meta">
              {s.status === 'done'
                ? `R² ${fmt(s.plane_r2, 2)} · str ${fmt(s.structure_alignment, 2)}`
                : s.status === 'failed' ? (s.error_message || 'failed') : s.status}
            </div>
          </div>
          <button
            className="x"
            title="remove"
            onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  );
}
