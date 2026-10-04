"""OpenStreetMap features for a georeferenced scene, through the Overpass API.

Public Overpass servers are volunteer-run and often busy (504s, timeouts), so a query tries
several within BUDGET_S overall, and every answer is kept on disk (next to the DEM patches):
a scene that worked once keeps its OSM features without the network. No answer gives None,
and callers carry on without the feature, never failing the upload.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SERVERS = ("https://overpass-api.de/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
           "https://overpass.private.coffee/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter")
BUDGET_S = 60.0


def _cache_path(query: str) -> Path:
    root = Path(os.environ.get("ALTIMAP_DEM_CACHE", Path(__file__).resolve().parent / "cache" / "dem"))
    return root / "osm" / f"{hashlib.sha256(query.encode()).hexdigest()}.json"


def query(body: str, bounds_lonlat) -> list[dict] | None:
    """Overpass elements for `body`: QL statements with a {bbox} placeholder and their own
    `out` statement, over a (west, south, east, north) box. None when no server answers."""
    import requests

    w, s, e, n = (round(float(v), 5) for v in bounds_lonlat)
    q = "[out:json][timeout:25];" + body.replace("{bbox}", f"{s},{w},{n},{e}")
    cache = _cache_path(q)
    data = None
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
        except (OSError, ValueError):
            data = None
    deadline = time.monotonic() + BUDGET_S
    # Busy spells pass in seconds: go round the servers more than once within the budget.
    for i, url in enumerate((SERVERS * 3) if data is None else ()):
        if i and i % len(SERVERS) == 0:
            time.sleep(2)
        left = deadline - time.monotonic()
        if left < 5:
            break
        try:
            r = requests.post(url, data={"data": q}, timeout=(min(10, left), min(30, left)),
                              headers={"User-Agent": "AltiMap (SIH 2026 PS 26175)"})
            r.raise_for_status()
            data = r.json()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
            break
        except Exception:  # busy server, rate limit, bad JSON: try the next one
            continue
    return None if data is None else data.get("elements", [])
