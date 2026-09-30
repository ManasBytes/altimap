// Dev/preview servers forward API calls to viewer.server; the built app is served by it directly.
const api = { "/api": "http://127.0.0.1:8000", "/data-uploads": "http://127.0.0.1:8000" };
export default { server: { proxy: api }, preview: { proxy: api } };
