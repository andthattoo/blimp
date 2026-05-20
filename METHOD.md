# BLiMP: Branch-Local Memory Planning

## One-line idea

BLiMP trains long-horizon behavior from short policy blocks by treating memory as branch-local agent state and propagating terminal reward only along the shortest successful path in a rollout tree.

## Motivation

Long-horizon language-agent tasks can require hundreds of environment interactions, but training with huge context windows is expensive and noisy. A GPU-poor setup needs the environment horizon to be long while the model context remains short.

BLiMP separates these two horizons:

- The policy acts for short blocks, such as 5 steps in TextWorld or 20 steps in CTF-like tasks.
- At the end of each block, the agent writes compact memory.
- Future blocks continue from branch-local memory snapshots instead of the full transcript.
- Search branches over memory states, not only over raw environment states.
- If a branch solves the task, terminal reward is assigned only to the upstream path that causally produced the solution.

## Rollout Structure

A rollout is a tree of fixed-length blocks.

```text
root block
  -> child block 1
  -> child block 2
  -> ...
      -> grandchild block
```

Each block contains a short interaction sequence:

```text
memory_in, observation_0
  -> action_1, observation_1
  -> action_2, observation_2
  -> ...
  -> action_K, observation_K
  -> memory_out
```

For the first mechanism test, use small settings:

```text
block_len: 5 actions
branching_factor: 2-4
depth: 4-8 blocks
model: small language model
primary environment: TextWorld
stretch environment: Minecraft or MineDojo-style tasks
memory: branch-local textual state
reward: terminal success propagated to shortest solved path
```

The near-term paper should prioritize a result that can be completed reliably. TextWorld is the primary environment because it is cheap, resettable, and gives exact success signals. Minecraft or MineDojo-style tasks are valuable only if the environment wrapper is already stable enough to run the same ablation without turning the submission into an infrastructure paper.

## Branch-Local Memory

The key design choice is that memory is local to a branch.

```text
parent memory snapshot
  -> branch A memory
  -> branch B memory
  -> branch C memory
```

Each child receives a copy of the parent memory and writes only to its own branch memory. Sibling branches do not share discoveries during the same frontier expansion.

This keeps credit assignment clean. If branch A discovers a useful fact and branch B solves, global shared memory would make it unclear which path deserves reward. Branch-local memory avoids this contamination.

Useful memory should contain compact, durable state:

- current location and inventory
- discovered map edges or locked doors
- goals and subgoals
- failed actions to avoid
- promising hypotheses
- exact symbolic facts needed for later continuation

## Reward Assignment

Let each solved leaf define a root-to-leaf path. BLiMP chooses the shortest successful path according to a cost function such as:

```text
cost(path) = number_of_blocks + alpha * environment_steps + beta * token_count
```

Terminal reward is assigned only to blocks on the selected path.

```text
reward(block) = 1 if block lies on shortest solved path
reward(block) = 0 otherwise
```

Non-winning branches can still receive local shaping, but they should not receive terminal success credit.

Local shaping can include:

- invalid action penalty
- repeated action penalty
- memory quality reward
- progress reward from environment score
- penalty for bloated or contradictory memory

## Why This Is Long-Horizon

The block boundary is only a training and context boundary. It should not be the environment, state, or credit boundary.

BLiMP is long-horizon when:

- environment state persists across blocks, or can be faithfully restored from branch snapshots
- memory is treated as part of the agent state
- reward propagates across the solved block chain
- evaluation counts full root-to-leaf trajectory success

It is not long-horizon if every block is trained and evaluated as an isolated episode.

## Core Hypothesis

Short-context policies can learn long-horizon behavior when they learn to construct, preserve, branch from, and act on compact memory states.

More concretely:

```text
branched block rollouts + branch-local memory + shortest-path credit
  > chained block rollouts + memory
  > flat rollouts without memory
```

under a fixed total environment-action budget.

## Minimal Ablation Suite

Run the following with equal total environment-action budgets:

```text
A. 40-step flat rollout, no memory
B. 8 x 5-step chain, memory, no branching
C. 5-step blocks, branching, branch-local memory
D. 5-step blocks, branching, shuffled or corrupted memory
```

The main budget unit is environment actions. For A and B this is exactly 40 actions per attempted trajectory. For C and D, report both:

- root-to-leaf horizon, capped at 8 blocks or 40 actions
- total branch-expanded actions, which can exceed 40 because branching buys exploration with extra compute

This distinction is essential. C should not be claimed better merely because it spends more total environment actions. The clean claim is that, under a fixed branch-expansion budget, branch-local memory produces better solved paths than corrupted memory and better search efficiency than chain-only memory.

The mechanism is supported if branched branch-local memory outperforms chained memory and flat rollouts, while corrupted memory collapses performance.

## Model Choice

The first experiment should use a small instruction-tuned model that is cheap enough to run many rollouts and weak enough that memory matters. Candidate sizes:

- 0.5B-1.5B for local smoke tests and harness debugging
- 3B-4B for the paper result if latency and budget allow
- avoid large frontier models for the main ablation, because they can hide whether BLiMP itself is doing anything

The model should produce both actions and memory writes. A minimal output format is:

```text
ACTION: <environment action>
MEMORY: <compact state update>
```

For the corrupted-memory ablation, keep actions sampled from the same model and replace only the memory input with shuffled, stale, or contradictory memory. This isolates whether continuation depends on the branch-local memory state.

## Metrics

Primary:

- success rate under fixed action budget
- environment steps to success
- solved-path depth
- total block executions per solved task

Secondary:

- memory length
- memory factuality or consistency
- repeated-action rate
- invalid-action rate
- branch efficiency: solved leaves per total branch expansions
- context tokens per successful trajectory

Useful plots:

```text
success rate vs total environment actions
success rate vs context or memory budget
success rate vs branching factor
steps to success vs block length
```

## Initial TextWorld Experiment

TextWorld is a mechanism test, not the final target domain. It is useful because it gives cheap long-horizon symbolic tasks with exact success signals.

Suggested first pass:

```text
env: TextWorld generated tasks
block_len: 5
branching_factor: 2 or 3
depth: 8
memory_budget: 100-300 tokens
models: small instruction-tuned language model
reward: terminal task success to shortest solved path
```

The paper should claim only that TextWorld validates the algorithmic shape: branch-local memory, block continuation, rollout-tree credit assignment, and memory-dependent continuation.

## Minecraft-Style Experiment

Minecraft or MineDojo-style tasks are a useful secondary test because they better match the "big worlds" framing: partial observability, large action space, sparse reward, and durable world state. For a three-day workshop deadline, this should be treated as a stretch result unless a stable lightweight wrapper already exists.

A tractable version should use short, symbolic, resettable tasks rather than open-ended Minecraft:

- navigate to a visible object
- collect a named item from nearby resources
- execute a short crafting or inventory sequence
- remember a discovered location across blocks

The same A-D ablation should be used. If this experiment is incomplete, the paper can still include it as motivation or an environment plan, but the empirical claim should rest on TextWorld.

## Big-World Framing

BLiMP targets settings where the environment can be too large to fully observe or remember. The agent has limited context and compute, so it must construct a useful agent state from its own history.

In this framing:

- memory is the constructed agent state
- branch expansion is exploration over agent state
- block rollout is a temporal abstraction
- shortest-path credit is sparse-reward credit assignment over a search tree
- bounded memory/context makes the benchmark resource-aware

## Open Questions

- Should memory writes be explicit policy actions or deterministic post-block summaries?
- How should partial progress rewards be calibrated without leaking task solutions?
- When should branch-local memories be merged, if ever?
- Does the learned memory policy transfer from TextWorld to CTF-like environments?
- What is the best cost function for shortest successful path: blocks, actions, tokens, wall time, or a mixture?
