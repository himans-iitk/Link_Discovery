"""Build the two caches every later stage reads (in WORK/):

    entities.pkl   raw name -> entity id, members, display names
    evidence.pkl   per-entity lobbying/award calendars + national null reference

    python3 -m pipeline.fav13.data.build            # build whatever is missing
    python3 -m pipeline.fav13.data.build --rebuild  # discard and rebuild
"""

import argparse
import os

from pipeline import paths
from .. import common
from . import entities, evidence


def load(rebuild=False):
    """-> (entities, evidence), building whichever cache is missing."""
    ep, vp = paths.WORK / "entities.pkl", paths.WORK / "evidence.pkl"
    if rebuild:
        for p in (ep, vp):
            if os.path.exists(p):
                os.remove(p)
    if not os.path.exists(ep):
        entities.save()
    ents = common.load_entities()
    if not os.path.exists(vp):
        evidence.save(ents["raw2ent"])
    return ents, common.load("evidence.pkl")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build entity and evidence caches.")
    ap.add_argument("--rebuild", action="store_true", help="discard and rebuild")
    a = ap.parse_args(argv)
    ents, evi = load(a.rebuild)
    print(f"entities.pkl  {len(ents['raw2ent']):,} names -> "
          f"{len(set(ents['raw2ent'].values())):,} entities")
    print(f"evidence.pkl  {len(evi['evidence']):,} entities, "
          f"{len(evi['dual']):,} on both sides")


if __name__ == "__main__":
    main()
