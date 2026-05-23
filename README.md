# BLiMP

BLiMP is an experimental training stack for GPU-poor, long-horizon reinforcement learning in text environments. The current method trains a causal language model with online policy-gradient updates plus token-native auxiliary losses compiled from the same trajectories.

The short version:

- Use TextWorld first because it is cheap, procedural, and has exact rewards.
- Compare full-history RL against short block-context RL.
- Add ECHO-style next-observation prediction and score-progress prediction.
- Optionally add structured `THINK:/ACTION:` rollouts, future-consistency loss, and branch-contrast loss.
- Keep the method causal-LM native: no extra architecture is required.

See [METHOD.md](METHOD.md) for the current method note.

## Repository Map

```text
blimp/train_reinforce.py          online REINFORCE trainer and auxiliary losses
blimp/envs.py                     tiny, hard, recall, MiniGrid, TextWorld, and ScienceWorld wrappers
blimp/sglang_rollout.py           SGLang branch-rollout utility
scripts/generate_textworld_custom_games.sh
scripts/run_full_reinforce_textworld_standard.sh
scripts/run_full_reinforce_textworld_blimp.sh
```

## Setup

On a GPU machine, `uv` is the least painful path:

```bash
cd ~/blimp
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e . -r requirements-textworld.txt
```

For ScienceWorld experiments:

```bash
uv pip install -e . -r requirements-scienceworld.txt
```

ScienceWorld is currently a stretch environment. The raw valid-action surface is much noisier than TextWorld, so TextWorld should remain the first evidence environment.

For MiniGrid memory/navigation tasks:

```bash
uv pip install -e . -r requirements-minigrid.txt
```

## Memory Diagnostic Task

`--env recall` is a controlled memory task. The first observation contains a passphrase, then the agent walks through a long corridor, and the final gate asks for that passphrase. Full history should help by construction; short history without memory should lose the clue.

Run these three controls before spending GPU time on larger environments:

```bash
# Full history: should recover the first-room clue from transcript.
uv run --active python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env recall \
  --mode standard \
  --lora-rank 0 \
  --updates 0 \
  --eval-episodes 32 \
  --max-steps 20 \
  --history-limit 0 \
  --score-batch-size 16 \
  --no-save-model \
  --out runs/eval-recall-full-history

# Short history, no memory: should lose the first-room clue.
uv run --active python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env recall \
  --mode standard \
  --lora-rank 0 \
  --updates 0 \
  --eval-episodes 32 \
  --max-steps 20 \
  --history-limit 3 \
  --score-batch-size 16 \
  --no-save-model \
  --out runs/eval-recall-short-history

# Short blocks with memory: this is the memory mechanism test.
uv run --active python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env recall \
  --mode blimp \
  --lora-rank 0 \
  --updates 0 \
  --eval-episodes 32 \
  --max-steps 20 \
  --block-len 3 \
  --history-limit 3 \
  --score-batch-size 16 \
  --no-save-model \
  --out runs/eval-recall-blimp-memory
```

Memory is only helping if the BLiMP run beats the short-history no-memory run and approaches full history.

## MiniGrid Memory Tasks

MiniGrid gives known partially observable memory and navigation tasks without Terminal-Bench's shell/tooling noise. The wrapper exposes the mission, facing direction, carried object, and local egocentric view as text. It does not reveal coordinates or the full map.

Start with `MiniGrid-MemoryS17Random-v0`:

```bash
# Full history control.
uv run --active python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env minigrid \
  --game-file MiniGrid-MemoryS17Random-v0 \
  --mode standard \
  --lora-rank 0 \
  --updates 0 \
  --eval-episodes 32 \
  --max-steps 120 \
  --history-limit 0 \
  --score-batch-size 8 \
  --no-save-model \
  --out runs/eval-minigrid-memory-full-history

# Short history, no memory.
uv run --active python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env minigrid \
  --game-file MiniGrid-MemoryS17Random-v0 \
  --mode standard \
  --lora-rank 0 \
  --updates 0 \
  --eval-episodes 32 \
  --max-steps 120 \
  --history-limit 3 \
  --score-batch-size 8 \
  --no-save-model \
  --out runs/eval-minigrid-memory-short-history

# Short blocks with memory.
uv run --active python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env minigrid \
  --game-file MiniGrid-MemoryS17Random-v0 \
  --mode blimp \
  --lora-rank 0 \
  --updates 0 \
  --eval-episodes 32 \
  --max-steps 120 \
  --block-len 3 \
  --history-limit 3 \
  --score-batch-size 8 \
  --no-save-model \
  --out runs/eval-minigrid-memory-blimp
```

If the BLiMP run does not beat short-history no-memory, the memory channel is not carrying useful state yet.

To run all three MiniGrid evals in one process, use:

```bash
VENV_PATH=/home/ubuntu/.venvs/blimp \
scripts/run_minigrid_memory_evals.sh
```

For a single systemd-managed run on a GPU box:

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/blimp-minigrid-evals.service <<'EOF'
[Unit]
Description=BLiMP MiniGrid memory ablation evals
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/ubuntu/blimp
Environment=VENV_PATH=/home/ubuntu/.venvs/blimp
ExecStart=/home/ubuntu/blimp/scripts/run_minigrid_memory_evals.sh
Restart=no

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user start blimp-minigrid-evals.service
journalctl --user -u blimp-minigrid-evals.service -o cat -f
```

## Generate TextWorld Data

The q8 pilot uses procedural TextWorld games with longer quests:

```bash
TRAIN_N=2048 \
EVAL_N=256 \
WORLD_SIZE=6 \
NB_OBJECTS=12 \
QUEST_LENGTH=8 \
OUT=data/textworld-custom-q8 \
scripts/generate_textworld_custom_games.sh
```

The train split is written to:

```text
data/textworld-custom-q8/train
```

The held-out eval split is written to:

```text
data/textworld-custom-q8/eval
```

## Baseline: Standard Full-History RL

This is the flat baseline. It uses full history by default in the launcher below and does not use block state.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GAME_DIR=data/textworld-custom-q8/train \
EVAL_GAME_DIR=data/textworld-custom-q8/eval \
OUT=runs/textworld-standard-q8 \
UPDATES=30 \
EPISODES_PER_UPDATE=4 \
EVAL_EPISODES=2 \
EVAL_EVERY=10 \
EVAL_MAX_STEPS=25 \
SKIP_INITIAL_EVAL=1 \
MAX_STEPS=50 \
HISTORY_LIMIT=0 \
SCORE_BATCH_SIZE=16 \
SAVE_MODEL=1 \
LOG_EPISODES=1 \
scripts/run_full_reinforce_textworld_standard.sh
```

## BLiMP: Block-Context RL

This run uses short block history plus compact block state.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GAME_DIR=data/textworld-custom-q8/train \
EVAL_GAME_DIR=data/textworld-custom-q8/eval \
OUT=runs/textworld-blimp-q8 \
UPDATES=30 \
EPISODES_PER_UPDATE=4 \
EVAL_EPISODES=2 \
EVAL_EVERY=10 \
EVAL_MAX_STEPS=25 \
SKIP_INITIAL_EVAL=1 \
MAX_STEPS=50 \
BLOCK_LEN=5 \
HISTORY_LIMIT=16 \
SCORE_BATCH_SIZE=16 \
SAVE_MODEL=1 \
LOG_EPISODES=1 \
scripts/run_full_reinforce_textworld_blimp.sh
```

## BLiMP + ECHO/Score Losses

The shell launcher exposes the transition auxiliary losses used in the first pilot:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GAME_DIR=data/textworld-custom-q8/train \
EVAL_GAME_DIR=data/textworld-custom-q8/eval \
OUT=runs/textworld-blimp-echo-q8 \
UPDATES=30 \
EPISODES_PER_UPDATE=4 \
EVAL_EPISODES=2 \
EVAL_EVERY=10 \
EVAL_MAX_STEPS=25 \
SKIP_INITIAL_EVAL=1 \
MAX_STEPS=50 \
BLOCK_LEN=5 \
HISTORY_LIMIT=16 \
SCORE_BATCH_SIZE=16 \
ECHO_WEIGHT=0.02 \
SCORE_WEIGHT=0.01 \
AUX_MAX_ITEMS=24 \
ECHO_MAX_WORDS=160 \
SAVE_MODEL=1 \
LOG_EPISODES=1 \
scripts/run_full_reinforce_textworld_blimp.sh
```

## BLiMP + Trainable Memory Writes

To test whether memory can become useful, train block-boundary memory as a policy output. This adds policy-gradient loss on each generated memory write, with credit assigned from rewards after that block boundary.

Use short blocks first:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GAME_DIR=data/textworld-custom-q8/train \
EVAL_GAME_DIR=data/textworld-custom-q8/eval \
OUT=runs/textworld-blimp-memory-policy-b3-q8 \
UPDATES=30 \
EPISODES_PER_UPDATE=4 \
EVAL_EPISODES=2 \
EVAL_EVERY=10 \
EVAL_MAX_STEPS=25 \
SKIP_INITIAL_EVAL=1 \
MAX_STEPS=50 \
BLOCK_LEN=3 \
HISTORY_LIMIT=3 \
SCORE_BATCH_SIZE=16 \
MEMORY_POLICY_WEIGHT=0.1 \
SAVE_MODEL=1 \
LOG_EPISODES=1 \
scripts/run_full_reinforce_textworld_blimp.sh
```

Compare this against untrained `--mode blimp` and untrained `--mode standard --history-limit 3`. Memory only counts as working if it beats short-history no-memory under the same tight context budget.

## Structured Auxiliary Run

The newest method knobs are available directly on `blimp.train_reinforce.py`. Use this path for the structured ablation until the shell launchers expose every flag.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env textworld \
  --game-dir data/textworld-custom-q8/train \
  --eval-game-dir data/textworld-custom-q8/eval \
  --mode blimp \
  --lora-rank 0 \
  --updates 30 \
  --episodes-per-update 4 \
  --eval-episodes 2 \
  --eval-every 10 \
  --eval-max-steps 25 \
  --skip-initial-eval \
  --max-steps 50 \
  --block-len 5 \
  --history-limit 16 \
  --score-batch-size 16 \
  --gradient-checkpointing \
  --temperature 1.2 \
  --epsilon 0.2 \
  --learning-rate 1e-6 \
  --structured-think-action \
  --echo-weight 0.02 \
  --score-weight 0.01 \
  --action-good-weight 0.02 \
  --future-weight 0.01 \
  --future-horizon 2 \
  --thought-weight 0 \
  --aux-max-items 24 \
  --echo-max-words 160 \
  --out runs/textworld-blimp-structured-q8
```

Keep `--thought-weight 0` for the first serious run. Thought imitation is not automatically trustworthy.

## Branch Contrast

Branch contrast requires branch-expanded rollout JSONL:

```bash
uv run python -u -m blimp.train_reinforce \
  --model Qwen/Qwen3-1.7B \
  --env textworld \
  --game-dir data/textworld-custom-q8/train \
  --eval-game-dir data/textworld-custom-q8/eval \
  --mode blimp \
  --lora-rank 0 \
  --branch-contrast-jsonl runs/branch-rollouts/results.jsonl \
  --branch-contrast-weight 0.01 \
  --aux-max-items 24 \
  --out runs/textworld-blimp-branch-contrast-q8
```

Do not enable this loss unless the JSONL contains real sibling branches from the same prefix. Randomly mixing unrelated trajectories would make the preference labels meaningless.

## Evaluation

Saved checkpoints can be evaluated by loading the checkpoint path or Hugging Face repo as `MODEL` and setting `UPDATES=0`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL=andthattoo/blimp-textworld-blimp-q8 \
GAME_DIR=data/textworld-custom-q8/train \
EVAL_GAME_DIR=data/textworld-custom-q8/eval \
OUT=runs/eval-blimp-q8-final-32 \
UPDATES=0 \
EVAL_EPISODES=32 \
EVAL_EVERY=1 \
MAX_STEPS=50 \
HISTORY_LIMIT=16 \
SCORE_BATCH_SIZE=16 \
SAVE_MODEL=0 \
LOG_EPISODES=1 \
scripts/run_full_reinforce_textworld_blimp.sh
```

Use the standard launcher for standard checkpoints:

```bash
MODEL=andthattoo/blimp-textworld-standard-q8 \
OUT=runs/eval-standard-q8-final-32 \
UPDATES=0 \
EVAL_EPISODES=32 \
SAVE_MODEL=0 \
scripts/run_full_reinforce_textworld_standard.sh
```

## Pilot TextWorld Results

Held-out TextWorld q8 eval, 32 episodes, max 50 steps, Qwen3-1.7B full-parameter checkpoints:

| Run | Checkpoint | Solved | Success | Mean score | Mean reward | Mean steps | Eval wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard full-history RL | `andthattoo/blimp-textworld-standard-q8` | 12/32 | 0.375 | 0.40625 | 0.78125 | 35.375 | 7355.4s |
| BLiMP block RL | `andthattoo/blimp-textworld-blimp-q8` | 17/32 | 0.53125 | 0.59375 | 1.125 | 33.25 | 1531.9s |
| BLiMP + ECHO/score | `andthattoo/blimp-textworld-blimp-echo-q8` | 16/32 | 0.5 | 0.53125 | 1.03125 | 33.71875 | 1618.9s |

Pilot takeaway: BLiMP block RL beat this standard full-history run on held-out success in this 32-episode eval and was much faster to evaluate in the current implementation. BLiMP + ECHO/score beat standard but did not beat plain BLiMP in this pilot.

This is not enough for a final claim. We have not yet shown parity with a well-tuned full-history RL baseline across seeds, larger validation sets, and equalized compute budgets. The current evidence should be described as a compute-limited pilot showing a promising short-context training signal.

## Current Limitations

- The final eval is only 32 held-out TextWorld games.
- There is one main seed per trained condition.
- The full-history baseline was expensive and may be under-tuned.
- The eval wall-time comparison reflects this implementation's valid-action scoring cost, not a universal property of the method.
- TextWorld q8 is useful but narrow; it does not establish transfer to broader long-horizon environments.
- BLiMP + ECHO/score did not beat plain BLiMP in the pilot, so the auxiliary losses need more ablation before being presented as an improvement.
- We still need an untrained checkpoint eval under the exact same final eval script and reward semantics.

Before making a strong paper claim, rerun with larger validation, multiple seeds, and at least one additional environment.

## Outputs

Training writes:

```text
runs/<name>/config.json
runs/<name>/metrics.jsonl
runs/<name>/train_traces.jsonl
runs/<name>/eval_traces.jsonl
runs/<name>/adapter
```

When `--lora-rank 0` is used, the `adapter` directory name is historical: it contains a full model checkpoint, not a LoRA adapter.

## Current Critical Path

For a clean paper table, run:

```text
A. standard RL
B. BLiMP block RL
C. BLiMP + ECHO/score
D. BLiMP + structured action-good/future
E. BLiMP + branch contrast, only after branch JSONL exists
```

Minimum next ablations:

```text
F. untrained Qwen3-1.7B under the exact final eval script
G. standard RL with matched seeds and longer/full eval
H. BLiMP across 3+ seeds
I. BLiMP on a harder TextWorld split, such as q10 or q12
J. one non-TextWorld environment, preferably a curated ScienceWorld subset before raw ScienceWorld
K. trainable memory writes with block_len=3 and matched no-memory short-history control
```

Report held-out success, mean score, mean reward, mean steps, eval wall time, total environment calls, train wall time, and GPU type. Do not claim the structured or branch losses help until they beat the matching BLiMP baseline on held-out games.
