import { fmt } from '../format';

export default function SceneMetrics({ meta, glbUrl }) {
  const noisy = meta && Number.isFinite(meta.structure_alignment) && Math.abs(meta.structure_alignment) < 0.05;

  return (
    <section>
      <h2>Scene metrics</h2>
      <dl>
        <dt>plane R²</dt><dd>{meta ? fmt(meta.plane_r2) : '—'}</dd>
        <dt>residual relief</dt><dd>{meta ? fmt(meta.residual_relief) : '—'}</dd>
        <dt>structure alignment</dt><dd>{meta ? fmt(meta.structure_alignment) : '—'}</dd>
        <dt>source size</dt>
        <dd>{meta?.source_size ? `${meta.source_size[0]}×${meta.source_size[1]}` : '—'}</dd>
      </dl>
      {meta && (
        <p className={`hint${noisy ? ' alert' : ''}`}>
          {noisy
            ? 'Residual barely aligns with image structure — detrended view is mostly amplified noise.'
            : 'Detrended amplitude is renormalized per scene.'}
        </p>
      )}
      <a className={`dl${glbUrl ? '' : ' off'}`} href={glbUrl || '#'} download>download terrain.glb</a>
    </section>
  );
}
