const BASE = '/api/scenes';

async function asJson(res) {
  if (!res.ok) {
    // DRF reports the reason in `detail`; surfacing it beats "failed".
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail ?? detail; } catch { /* keep status */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

export function uploadScene(file) {
  const body = new FormData();
  body.append('file', file);
  return fetch(`${BASE}/upload/`, { method: 'POST', body }).then(asJson);
}

export function listScenes() {
  return fetch(`${BASE}/`).then(asJson);
}

export function getScene(id) {
  return fetch(`${BASE}/${encodeURIComponent(id)}/`).then(asJson);
}

export function deleteScene(id) {
  return fetch(`${BASE}/${encodeURIComponent(id)}/`, { method: 'DELETE' }).then(asJson);
}

/**
 * Upload is async: the API returns status="pending" immediately and the
 * pipeline runs on a background thread. Poll the detail endpoint until it
 * settles into "done" or "failed" -- a 15 minute cap keeps a stuck job from
 * polling forever in an open tab.
 */
export async function pollScene(id, { intervalMs = 1500, maxWaitMs = 15 * 60 * 1000, onUpdate } = {}) {
  const deadline = Date.now() + maxWaitMs;
  for (;;) {
    const scene = await getScene(id);
    onUpdate?.(scene);
    if (scene.status === 'done' || scene.status === 'failed') return scene;
    if (Date.now() > deadline) throw new Error('timed out waiting for processing to finish');
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
