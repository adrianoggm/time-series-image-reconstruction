"""Freeze the M0 reference: hash every result, model and aggregate array.

Writes ``results/_baseline/manifest.json`` with sha256, size and (for .npy) shape
of each artefact, plus the code commit and library versions it was produced
with. Later milestones diff against it to prove WISDM still reproduces.

    python -m scripts._freeze_baseline            # write the manifest
    python -m scripts._freeze_baseline --check    # exit 1 if anything changed

Per-window reconstructions (``<subject>x<i>.npy``) and PNGs are skipped: they
are fully determined by the aggregate ``X_all_*.npy`` and the window arrays.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from src.config import DEFAULT_DATA_DIR

MANIFEST = Path("results/_baseline/manifest.json")
PER_WINDOW = re.compile(r"^\d+x\d+\.npy$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artefacts():
    data = Path(DEFAULT_DATA_DIR)
    yield from (p for p in Path("results").rglob("*.json") if "_baseline" not in p.parts)
    yield from (p for p in (data / "tseries").rglob("*.npy") if not PER_WINDOW.match(p.name))
    yield from (p for p in Path("models").glob("*") if p.is_file())


def key(path):
    """Repo-relative path, or ``$TFG_DATA_DIR/...`` so manifests compare across machines."""
    if path.is_absolute():
        return "$TFG_DATA_DIR/" + path.relative_to(DEFAULT_DATA_DIR).as_posix()
    return path.as_posix()


def describe(path):
    entry = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if path.suffix == ".npy":
        entry["shape"] = list(np.load(path, mmap_mode="r", allow_pickle=False).shape)
    if "_originals" in path.parts:
        entry["note"] = "cache, not a source (see CLAUDE.md)"
    return entry


def build():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"],
                                capture_output=True, text=True).stdout.strip())
    return {
        "code": {"commit": commit or "unknown", "dirty_src_or_scripts": dirty},
        "env": {"python": platform.python_version(), "platform": platform.platform(),
                **{lib: version(lib) for lib in ("numpy", "scipy", "scikit-learn", "pyts", "tensorflow")}},
        "data_dir": str(DEFAULT_DATA_DIR),
        "artefacts": {key(p): describe(p) for p in sorted(artefacts())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare against the stored manifest")
    args = ap.parse_args()

    current = build()
    if not args.check:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(current, indent=1) + "\n")
        print(f"wrote {MANIFEST}: {len(current['artefacts'])} artefacts at {current['code']['commit'][:7]}")
        return 0

    old = json.loads(MANIFEST.read_text())["artefacts"]
    new = current["artefacts"]
    changed = [p for p in old.keys() & new.keys() if old[p]["sha256"] != new[p]["sha256"]]
    for label, paths in (("CHANGED", changed), ("MISSING", old.keys() - new.keys()), ("NEW", new.keys() - old.keys())):
        for p in sorted(paths):
            print(label, p)
    return 1 if changed or old.keys() - new.keys() else 0


if __name__ == "__main__":
    sys.exit(main())
