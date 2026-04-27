"""Phase 3 — Cluster entry point.

Calls phase3_scale_correlation.py with cluster-optimised settings:
  - All three methods: VG + IG + SARFA
  - Larger IG chunk size (200) to exploit 24GB VRAM

Usage (see run_attention_experiments_post_hoc.md for full setup):
    conda activate vmax
    python phase3_cluster.py --model complete
    python phase3_cluster.py --model minimal

To re-generate figures only (no recomputation):
    python phase3_cluster.py --model complete --figures-only
    python phase3_cluster.py --model minimal  --figures-only
"""

import sys
import subprocess
from pathlib import Path

_HERE = Path(__file__).parent
_BASE = _HERE / "phase3_scale_correlation.py"

CLUSTER_METHODS    = ["vg", "ig", "sarfa"]
CLUSTER_CHUNK_SIZE = 200   # safe for 24GB; reduce to 100 if OOM


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 cluster run")
    parser.add_argument("--model",    required=True, choices=["complete", "minimal"])
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable, str(_BASE),
        "--model",       args.model,
        "--n-scenarios", str(args.n_scenarios),
        "--methods",     *CLUSTER_METHODS,
        "--chunk-size",  str(CLUSTER_CHUNK_SIZE),
    ]
    if args.figures_only:
        cmd.append("--figures-only")

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
