"""Report \\ref targets with no \\label (BROKEN) and labels never referenced (UNUSED).

    python scripts/_check_refs.py                 # main2.tex, then supplementary.tex
    python scripts/_check_refs.py supplementary   # one document

supplementary.tex pulls main2's labels in via \\externaldocument{main2}, so its refs
are checked against the union of both documents' labels.
"""
import re
import sys
from pathlib import Path

DOC = Path("doc")


def load_with_inputs(path: Path, seen: set | None = None) -> str:
    """Read `path` and recursively splice in any ``\\input{file}`` references
    relative to the same directory (with optional ``.tex`` suffix)."""
    seen = seen or set()
    if path in seen:
        return ""
    seen.add(path)
    txt = path.read_text(encoding="utf-8")

    def _splice(m):
        name = m.group(1).strip()
        if not name.endswith(".tex"):
            name += ".tex"
        sub = DOC / name
        if not sub.exists():
            return m.group(0)
        return load_with_inputs(sub, seen)

    return re.sub(r"\\input\{([^}]+)\}", _splice, txt)


def _refs_labels(name):
    text = load_with_inputs(DOC / f"{name}.tex")
    return (set(re.findall(r"\\ref\{([^}]+)\}", text)),
            set(re.findall(r"\\label\{([^}]+)\}", text)))


docs = sys.argv[1:] or ["main2", "supplementary"]
all_labels = {}
for name in ("main2", "supplementary"):
    if (DOC / f"{name}.tex").exists():
        all_labels[name] = _refs_labels(name)[1]

status = 0
for name in docs:
    p = DOC / f"{name}.tex"
    if not p.exists():
        print(f"{name}.tex: MISSING")
        status = 1
        continue
    refs, labels = _refs_labels(name)
    # \externaldocument{main2} makes main2's labels resolvable from the supplement.
    visible = set(labels)
    if name == "supplementary":
        visible |= all_labels.get("main2", set())
    broken = sorted(refs - visible)
    print(f"=== {name}.tex ===")
    print("  BROKEN:", broken or "(none)")
    print("  UNUSED:", sorted(labels - refs) or "(none)")
    if broken:
        status = 1
sys.exit(status)
