"""Top-level runner: chain the whole microbiome analysis end-to-end.

Runs, in order, into subdirectories of --out:
  1. run_pipeline -> <out>/pipeline  (discovery + sensitivity + prediction
                                      + mediation + plots, all levels)
  2. run_revision -> <out>/revision  (missingness, mediation, clinical,
                                      prediction revision stages)

External replication is intentionally NOT run here (no external cohort).

Each sub-pipeline is launched as its own `python -m scripts.<name>` process so
behaviour matches running them individually (same env, same isolation). Fails
fast: a nonzero exit in any stage aborts the rest. Runs end-to-end with no
args: `python -m scripts.run_all`.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from oral_mb.config import MICROBIOME_LEVELS, OUT_DIR

log = logging.getLogger("run_all")


def _run(module: str, argv: list[str]) -> None:
    """Launch `python -m scripts.<module>` with argv; raise on nonzero exit."""
    cmd = [sys.executable, "-m", f"scripts.{module}", *argv]
    log.info("=== %s ===\n  %s", module, " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR,
                    help="Base output dir; each stage writes a subdir under it.")
    ap.add_argument("--levels", nargs="+", default=list(MICROBIOME_LEVELS),
                    help="Microbiome levels for run_pipeline (default: all).")
    ap.add_argument("--level", default="species",
                    help="Single level for the revision stage.")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["pipeline", "revision"],
                    help="Stage(s) to skip.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if "pipeline" not in args.skip:
        _run("run_pipeline",
             ["--out", str(args.out / "pipeline"),
              "--levels", *args.levels])

    if "revision" not in args.skip:
        _run("run_revision",
             ["--level", args.level, "--stage", "all",
              "--out", str(args.out / "revision")])

    log.info("run_all done -> %s", args.out)


if __name__ == "__main__":
    main()
