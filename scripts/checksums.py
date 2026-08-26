#!/usr/bin/env python3
"""Generate and verify SHA-256 checksums for the distributed corpus.

The manifest is small and IS tracked in git, so collaborators can prove the
files they downloaded match the ones the results were produced from.

    python scripts/checksums.py write            # hash ijf-datasets/ -> checksums.sha256
    python scripts/checksums.py verify           # check local files against manifest
    python scripts/checksums.py write --dir clean
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ijf_config as cfg  # noqa: E402

MANIFEST = cfg.REPO_ROOT / "checksums.sha256"
BUF = 8 << 20  # 8 MiB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(BUF):
            h.update(block)
    return h.hexdigest()


def target_dir(name: str) -> Path:
    return {"raw": cfg.RAW, "clean": cfg.CLEAN, "samples": cfg.SAMPLES}[name]


def cmd_write(args) -> int:
    d = target_dir(args.dir)
    files = sorted(p for p in d.glob("*.csv"))
    if not files:
        print(f"No CSVs in {d}", file=sys.stderr)
        return 1
    lines = []
    for p in files:
        digest = sha256(p)
        size = p.stat().st_size
        print(f"  {digest[:12]}  {size / 1e6:9.1f} MB  {p.name}")
        lines.append(f"{digest}  {p.name}")
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {MANIFEST} ({len(files)} files)")
    return 0


def cmd_verify(args) -> int:
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}; run 'checksums.py write' first", file=sys.stderr)
        return 1
    d = target_dir(args.dir)
    expected = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.strip()] = digest

    ok = missing = bad = 0
    for name, digest in sorted(expected.items()):
        p = d / name
        if not p.exists():
            print(f"  MISSING  {name}")
            missing += 1
        elif sha256(p) != digest:
            print(f"  MISMATCH {name}")
            bad += 1
        else:
            print(f"  ok       {name}")
            ok += 1

    print(f"\n{ok} ok, {missing} missing, {bad} mismatched  ({d})")
    return 0 if (missing == 0 and bad == 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("write", cmd_write), ("verify", cmd_verify)):
        p = sub.add_parser(name)
        p.add_argument("--dir", choices=["raw", "clean", "samples"], default="raw")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
