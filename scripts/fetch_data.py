#!/usr/bin/env python3
"""Download the raw corpus from Zenodo into the local data root.

Set the record id once it is published (the numeric id in the Zenodo URL,
e.g. https://zenodo.org/records/1234567 -> 1234567), either by editing
ZENODO_RECORD below or via the IJF_ZENODO_RECORD environment variable.

    python scripts/fetch_data.py                 # download missing files
    python scripts/fetch_data.py --force         # re-download everything
    python scripts/fetch_data.py --list          # show what the record contains

Downloads are resumable: a partial file is written as <name>.part and only
renamed into place once complete, so an interrupted run can be re-run safely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ijf_config as cfg  # noqa: E402

ZENODO_RECORD = os.environ.get("IJF_ZENODO_RECORD", "")
API = "https://zenodo.org/api/records/{record}"
CHUNK = 8 << 20  # 8 MiB


def record_files(record: str) -> list[dict]:
    url = API.format(record=record)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            meta = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Zenodo returned {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach Zenodo: {exc.reason}") from exc
    return meta.get("files", [])


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, size: int | None) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as fh:
        while block := resp.read(CHUNK):
            fh.write(block)
            done += len(block)
            if size:
                pct = 100 * done / size
                print(f"\r    {dest.name:<24} {pct:5.1f}%  {human(done)}", end="", flush=True)
            else:
                print(f"\r    {dest.name:<24} {human(done)}", end="", flush=True)
    print()
    part.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record id")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--list", action="store_true", help="list record contents and exit")
    args = ap.parse_args()

    if not args.record:
        print(
            "No Zenodo record id.\n"
            "  Set IJF_ZENODO_RECORD, pass --record <id>, or edit ZENODO_RECORD\n"
            "  in this file once the dataset is published.",
            file=sys.stderr,
        )
        return 1

    files = record_files(args.record)
    if not files:
        print(f"Record {args.record} lists no files.", file=sys.stderr)
        return 1

    if args.list:
        total = 0
        for f in files:
            size = f.get("size", 0)
            total += size
            print(f"  {human(size):>10}  {f['key']}")
        print(f"  {human(total):>10}  TOTAL ({len(files)} files)")
        return 0

    cfg.RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading -> {cfg.RAW}")

    for f in files:
        name = f["key"]
        dest = cfg.RAW / name
        size = f.get("size")
        if dest.exists() and not args.force:
            print(f"    {name:<24} skip (exists)")
            continue
        download(f["links"]["self"], dest, size)

    print("\ndone. Next:")
    print("  python scripts/checksums.py verify")
    print("  python pipeline/clean_datasets.py")
    print("  python scripts/make_samples.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
