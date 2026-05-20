# BLiMP

Branch-Local Memory Planning for short-context, long-horizon reinforcement learning.

BLiMP studies whether an agent can learn long-horizon behavior under tight context and compute budgets by splitting interaction into short blocks, branching from compact memory snapshots, and assigning terminal reward only to the shortest successful root-to-leaf path.

See [METHOD.md](METHOD.md) for the current method note.

## Ablation

The current harness runs the four submission-critical variants:

```text
A. 40-step flat rollout, no memory
B. 8 x 5-step chain, memory, no branching
C. 5-step blocks, branching, branch-local memory
D. 5-step blocks, branching, shuffled or corrupted memory
```

For branching variants, the output reports both the solved root-to-leaf horizon and total branch-expanded environment actions. Do not compare C/D to A/B without this compute accounting.

## Quick Smoke Test

The built-in `tiny` environment is only for validating basic rollout mechanics before running TextWorld or Minecraft-style tasks. The `hard` environment is a deterministic local stress test with a passphrase clue, tool chain, locked gate, moon lock, and final memory-dependent vault action.

```bash
python -m blimp.run_experiment \
  --env tiny \
  --policy scripted-tiny \
  --episodes 3 \
  --variants A,B,C,D \
  --out runs/tiny-smoke
```

Harder local memory smoke:

```bash
python -m blimp.run_experiment \
  --env hard \
  --policy scripted-hard \
  --episodes 1 \
  --variants A,B,C,D \
  --flat-steps 60 \
  --chain-blocks 8 \
  --branch-factor 2 \
  --branch-depth 8 \
  --branch-action-budget 120 \
  --memory-words 240 \
  --out runs/hard-smoke
```

Outputs:

- `runs/<name>/trajectories.jsonl`
- `runs/<name>/summary.json`
- `runs/<name>/summary.csv`

Optional plot:

```bash
python scripts/plot_results.py runs/tiny-smoke
```

## Small HF Model Run

On a GPU machine:

```bash
pip install -e .
pip install -r requirements-gpu.txt

MODEL=Qwen/Qwen2.5-1.5B-Instruct OUT=runs/hf-tiny scripts/run_hf_tiny.sh
```

For real TextWorld games, install TextWorld and pass either `--game-file` or `--game-dir`:

```bash
pip install textworld

python -m blimp.run_experiment \
  --env textworld \
  --game-dir /path/to/textworld-games \
  --policy hf \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --episodes 50 \
  --variants A,B,C,D \
  --branch-factor 2 \
  --branch-depth 8 \
  --branch-action-budget 320 \
  --out runs/textworld-qwen15b
```

## JarvisLabs Sketch

Use an existing running machine:

```bash
jl upload . <machine_id>:/home/blimp
jl exec <machine_id> -- bash -lc 'cd /home/blimp && pip install -e . && pip install -r requirements-gpu.txt'
jl exec <machine_id> -- bash -lc 'cd /home/blimp && MODEL=Qwen/Qwen2.5-1.5B-Instruct OUT=runs/hf-tiny scripts/run_hf_tiny.sh'
```
