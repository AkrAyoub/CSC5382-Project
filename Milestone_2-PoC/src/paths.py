from __future__ import annotations

from pathlib import Path


M2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = M2_ROOT.parent
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
UNCAPOPT_PATH = DATA_RAW_DIR / "uncapopt.txt"


def list_instance_files() -> list[Path]:
    items = sorted(DATA_RAW_DIR.glob("cap*.txt"))
    for name in ("capa.txt", "capb.txt", "capc.txt"):
        path = DATA_RAW_DIR / name
        if path.exists():
            items.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in items:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped
