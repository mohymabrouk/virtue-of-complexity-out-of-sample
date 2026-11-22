"""Download stage: fetch the raw files once into ~/.cache/quant-research/p2/raw and write a manifest.

A file that already exists is never downloaded again; its manifest entry keeps the original download date.
"""

import hashlib
import json
import logging
import urllib.request
from datetime import datetime, timezone

from voc import config

log = logging.getLogger(__name__)

SOURCES = [
    ("goyal_predictors_2025", config.GOYAL_URL, config.GOYAL_FILE),
    ("french_ff3_monthly", config.FRENCH_URL, config.FRENCH_FILE),
    ("jkp_mkt_all_countries", config.JKP_MKT_URL, config.JKP_MKT_FILE),
    ("jkp_themes_all_countries", config.JKP_THEMES_URL, config.JKP_THEMES_FILE),
]
MANIFEST = config.RAW / "manifest.json"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def run() -> dict:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for name, url, dest in SOURCES:
        if dest.exists():
            log.info("cached, not re-downloaded: %s", dest)
            entry = manifest.get(name, {})
            entry.setdefault("downloaded_utc", datetime.fromtimestamp(dest.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"))
        else:
            log.info("downloading %s", url)
            fetch(url, dest)
            entry = {"downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        entry.update({"url": url, "file": str(dest.relative_to(config.RAW)).replace("\\", "/"),
                      "bytes": dest.stat().st_size, "sha256": _sha256(dest)})
        manifest[name] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("manifest: %s", MANIFEST)
    return manifest
