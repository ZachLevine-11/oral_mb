"""Draw the mediation Sankey on this Mac from the iCloud copy of the run.

Usage:
    python -m scripts.sankey_local                      # species, iCloud copy
    python -m scripts.sankey_local --mediation path/to/mediation.csv --out fig.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from oral_mb.sankey import mediation_sankey

LOCAL_RUN = (Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"
             / "Segal Lab/oral_mb/species")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mediation", type=Path, default=LOCAL_RUN / "mediation.csv")
    ap.add_argument("--out", type=Path, default=LOCAL_RUN / "fig_mediation_sankey.png")
    ap.add_argument("--q-max", type=float, default=0.05)
    args = ap.parse_args()
    if not args.mediation.exists():
        raise SystemExit(f"no mediation.csv at {args.mediation}")
    mediation_sankey(pd.read_csv(args.mediation), args.out, q_max=args.q_max)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
