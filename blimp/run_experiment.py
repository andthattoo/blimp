from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from blimp.policies import make_policy
from blimp.rollout import Variant, run_variant, summarize_runs, write_jsonl, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BLiMP A-D rollout ablations.")
    parser.add_argument("--env", choices=["tiny", "hard", "textworld"], default="tiny")
    parser.add_argument("--game-file", default=None, help="TextWorld game file.")
    parser.add_argument("--game-dir", default=None, help="Directory of TextWorld .ulx/.z8 games.")
    parser.add_argument(
        "--policy",
        choices=["random", "scripted-tiny", "scripted-hard", "hf"],
        default="random",
    )
    parser.add_argument("--model", default=None, help="HF model name for --policy hf.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--variants", default="A,B,C,D")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-len", type=int, default=5)
    parser.add_argument("--flat-steps", type=int, default=40)
    parser.add_argument("--chain-blocks", type=int, default=8)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--branch-depth", type=int, default=8)
    parser.add_argument("--branch-action-budget", type=int, default=None)
    parser.add_argument("--memory-words", type=int, default=160)
    parser.add_argument("--no-stop-on-solved-depth", action="store_true")
    parser.add_argument("--out", default="runs/latest")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    game_files = find_game_files(args.game_file, args.game_dir)
    policy = make_policy(
        args.policy,
        model_name=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["game_files"] = game_files
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    rows = []
    for episode_id in range(args.episodes):
        game_file = game_files[episode_id % len(game_files)] if game_files else args.game_file
        for variant in variants:
            result = run_variant(
                episode_id=episode_id,
                variant=variant,
                env_name=args.env,
                policy=policy,
                game_file=game_file,
                seed=args.seed,
                block_len=args.block_len,
                flat_steps=args.flat_steps,
                chain_blocks=args.chain_blocks,
                branch_factor=args.branch_factor,
                branch_depth=args.branch_depth,
                branch_action_budget=args.branch_action_budget,
                memory_words=args.memory_words,
                stop_on_solved_depth=not args.no_stop_on_solved_depth,
            )
            rows.append(result)
            print(
                f"episode={episode_id} variant={variant} solved={int(result.solved)} "
                f"root_steps={result.root_to_leaf_steps} "
                f"branch_actions={result.total_branch_expanded_actions} "
                f"blocks={result.total_blocks}"
            )

    summary = summarize_runs(rows)
    write_jsonl(out_dir / "trajectories.jsonl", rows)
    write_summary(out_dir / "summary.json", summary)
    write_summary_csv(out_dir / "summary.csv", summary)
    print(json.dumps(summary, indent=2))


def parse_variants(raw: str) -> list[Variant]:
    variants = [part.strip().upper() for part in raw.split(",") if part.strip()]
    allowed = {"A", "B", "C", "D"}
    bad = [variant for variant in variants if variant not in allowed]
    if bad:
        raise ValueError(f"Unknown variants: {bad}")
    return variants  # type: ignore[return-value]


def find_game_files(game_file: str | None, game_dir: str | None) -> list[str]:
    if game_file:
        return [game_file]
    if not game_dir:
        return []
    root = Path(game_dir)
    suffixes = {".ulx", ".z8"}
    files = sorted(str(path) for path in root.rglob("*") if path.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f"No TextWorld game files found in {game_dir}")
    return files


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    if not summary:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
