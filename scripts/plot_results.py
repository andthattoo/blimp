from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BLiMP summary metrics.")
    parser.add_argument("run_dir", help="Directory containing summary.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib or `pip install -e .[plot]`.") from exc

    variants = [row["variant"] for row in summary]
    success = [row["success_rate"] for row in summary]
    branch_actions = [row["mean_total_branch_expanded_actions"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    axes[0].bar(variants, success, color="#2671b8")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Variant")
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Task success")

    axes[1].bar(variants, branch_actions, color="#bd6b2f")
    axes[1].set_xlabel("Variant")
    axes[1].set_ylabel("Mean expanded env actions")
    axes[1].set_title("Compute used")

    out = run_dir / "summary.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
