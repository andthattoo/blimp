# BLiMP: Token-Native Block RL

## One-line idea

BLiMP trains a causal language model for long-horizon text environments by combining sparse policy-gradient updates with auxiliary token losses compiled from the same trajectories. The method stays language-model native: no extra heads are required, and observation, score, action, thought, future-state, and branch-preference targets are all represented as text completions.

## Motivation

Long-horizon RL for language agents is expensive because the environment horizon, prompt length, and credit-assignment horizon grow together. A GPU-poor setup needs to decouple these:

- The environment trajectory can be 40-50 steps or longer.
- The policy can act from short local blocks.
- The model can still learn useful long-horizon behavior from transition-level supervision extracted from every rollout.
- Failed trajectories should not be thrown away; they contain dense world-model and progress-prediction labels.

The current method is therefore not primarily about a Markdown memory file. The memory or block-state text is just one possible state representation. The core claim is that a small model can learn better long-horizon behavior when the RL objective is augmented with verifiable, token-level trajectory losses.

## Training Signal

For each environment transition, the trainer records:

```text
observation_t
thinking_t
action_t
observation_{t+1}
reward_t
score_delta_t
score_t
valid_t
done_t
won_t
```

The total objective is:

```text
L_total =
  L_policy
  + lambda * L_next_observation
  + theta * L_score_progress
  + alpha * L_action_good
  + beta * L_future_consistency
  + rho * L_branch_contrast
  + tau * L_thought_imitation
```

All terms are implemented as causal-LM log-probability losses over textual completions, except the policy term, which is REINFORCE over the selected action completion.

## Policy Loss

The base policy loss is online REINFORCE over environment actions:

```text
L_policy = - advantage_t * log p(action_t | prompt_t)
```

In the default action-scoring path, the model scores every valid action as:

```text
ACTION: <valid environment action>
```

The chosen action is sampled from those scores with temperature and epsilon exploration. This keeps the action interface comparable across runs because every sampled action is guaranteed to be one of the environment's valid actions.

## Structured THINK/ACTION Mode

The optional structured mode asks the model to generate:

```text
THINK: <brief state and consequence reasoning>
ACTION: <one environment action>
```

The generated action is resolved against the environment's valid-action list. This mode lets later losses train over the model's own explicit state/consequence reasoning, but it changes the inference interface: action selection becomes generation plus action resolution rather than exact valid-action scoring. For clean ablations, structured mode should be compared against a matching structured baseline, not silently mixed with the old action-scoring baseline.

## Next-Observation Loss

The ECHO-style transition loss trains the model to predict the next observation from the current observation and action:

```text
OBSERVATION:
observation_t

THINK:
thinking_t

ACTION: action_t
NEXT_OBSERVATION:
observation_{t+1}
```

This converts every rollout step, including failed steps, into a world-model training example. It is useful because TextWorld and ScienceWorld produce exact next observations without human labels.

## Score-Progress Loss

The score-progress loss predicts verifiable environment metadata after the next observation:

```text
valid: true|false
score_delta: <number>
score: <number>
done: true|false
won: true|false
```

This is not a learned reward model in the usual opaque sense. It is supervised by environment-provided labels, so it can teach the model which transitions are progress, terminal success, invalid action, or dead end.

## Action-Good Loss

The action-good loss imitates actions only when a transition has evidence of being useful:

- the episode solved,
- the transition won,
- the score increased,
- or the reward was positive.

The current implementation weights fully successful transitions more strongly and gives partial weight to local progress. This is deliberately narrower than behavior cloning from all traces; it should not teach the model to imitate arbitrary failed behavior.

## Future-Consistency Loss

The future-consistency loss asks the model to predict a future observation several steps ahead from the current prefix and thought:

```text
OBS_0 ... ACTION_{t-1}
OBS_t
THINK_t
FUTURE_OBSERVATION_{t+h}:
observation_{t+h}
```

This gives the thought/state text a consequence: a useful thought should help predict where the trajectory is going, not merely paraphrase the current observation.

## Branch-Contrast Loss

Branch contrast is the most BLiMP-specific auxiliary loss. Given sibling continuations from the same branch prefix, the trainer constructs a preference pair:

```text
preferred branch > rejected branch
```

Preference is ordered by:

```text
solved first, then higher score, then shorter path
```

The loss is:

```text
L_branch_contrast = -log sigmoid(log p(preferred) - log p(rejected))
```

This is how branch-expanded rollouts can train the model to prefer better local continuations without needing a separate critic or reward model. It requires a branch JSONL file. If no branch file is provided, this term is absent.

## Thought-Imitation Loss

The thought-imitation loss trains the model to reproduce its own generated `THINK:` text on useful transitions. This is optional and risky. It can preserve bad reasoning if the model's thoughts are merely correlated with success rather than causally useful.

The recommended default is:

```text
thought_weight = 0
```

until there is evidence that the generated thoughts are clean enough to imitate.

## Block State

In `blimp` mode, the trainer runs with short blocks. The action prompt receives only the current block history plus a compact block-state string. At block boundaries the model may update that state, and the next block uses the updated state instead of the full transcript.

Important limitation: in the current online trainer, block-state generation is not directly optimized by the policy loss. The policy-gradient items are action completions, and the auxiliary examples are compiled from transitions. Block state can still help downstream actions through the prompt, but claiming a trained memory-writing policy requires either:

- including state-writing completions in the loss,
- deriving branch-contrast examples that include state updates,
- or adding a separate verifiable state-quality objective.

This caveat matters. The present method should be described as token-native trajectory supervision with optional block state, not as proven direct optimization of a free-form memory file.

The trainer now supports the first option with `--memory-policy-weight`. Each block-boundary memory write is stored with its prompt and completion. During update, it receives the discounted return starting after that boundary, so memory is credited only for future block behavior. This is the first clean test of whether the model can learn to write useful memory rather than merely consuming an untrained note.

## Standard vs BLiMP Runs

The current clean comparison is:

```text
standard:
  full or configured history in the prompt
  no block state
  same environment, model, reward, and optimizer

blimp:
  short block history
  compact block state in the prompt
  same environment horizon
  optional auxiliary losses from the same rollouts
```

This is the comparison that matters for GPU-poor long-horizon RL: does a short-context block policy with trajectory-supervised auxiliary losses match or beat a full-history policy while using less context and wall time?

## Recommended Ablation Ladder

Run the ablation in this order:

```text
A. Standard RL
B. BLiMP block RL
C. BLiMP + next-observation + score-progress losses
D. BLiMP + structured THINK/ACTION + action-good + future-consistency losses
E. BLiMP + branch-contrast loss from branch-expanded rollouts
```

Keep the environment, model, train split, validation split, step cap, and optimizer settings fixed. The main validation metric is held-out success rate. Secondary metrics are mean score, mean environment reward, mean steps, context tokens, wall time, and environment calls per solved task.

## Suggested First Weights

These are conservative starting values for TextWorld q8-scale runs:

```text
echo_weight: 0.02
score_weight: 0.01
action_good_weight: 0.02
future_weight: 0.01
future_horizon: 2
thought_weight: 0
branch_contrast_weight: 0 unless branch JSONL exists
aux_max_items: 24
echo_max_words: 160
```

These are not claimed optimal. They are meant to make the auxiliary gradients visible without overwhelming the RL signal.

## What Would Count As Evidence

The strongest near-term evidence would be:

- Standard RL does not improve much on held-out TextWorld.
- BLiMP block RL improves or matches standard RL with shorter prompts and lower eval wall time.
- Adding next-observation and score-progress losses improves sample efficiency or stability.
- Structured action/future losses improve held-out success without increasing invalid-action rate.
- Branch contrast improves branch continuation quality on held-out games.

The honest paper claim should be limited to what the ablations support. A small `n=32` eval is a pilot, not a final statistical result.

## Pilot TextWorld Evidence

Current held-out TextWorld q8 pilot results:

| Run | Solved | Success | Mean score | Mean reward | Mean steps | Eval wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard full-history RL | 12/32 | 0.375 | 0.40625 | 0.78125 | 35.375 | 7355.4s |
| BLiMP block RL | 17/32 | 0.53125 | 0.59375 | 1.125 | 33.25 | 1531.9s |
| BLiMP + ECHO/score | 16/32 | 0.5 | 0.53125 | 1.03125 | 33.71875 | 1618.9s |

The strongest supported statement is narrow: in this pilot, short block-context BLiMP improved held-out TextWorld success over this standard full-history RL run and reduced eval wall time in this implementation. The ECHO/score auxiliary run improved over standard but did not improve over plain BLiMP. This should motivate the next structured and branch-contrast ablations rather than be treated as a final result.

## Compute-Limited Claim

The paper should be explicit that this is a compute-constrained study. The intended claim is not that the current BLiMP checkpoint already equals or beats a fully tuned full-context RL system. It is:

```text
Under a small-model, single-GPU, limited-rollout budget, short-context block RL
with trajectory-derived token losses can produce a useful long-horizon training
signal, and may be more practical than full-history RL in this regime.
```

This distinction matters because a stronger full-history baseline could improve with more rollouts, better batching, better exploration, or a larger model. The current data show a promising pilot, not a universal dominance result.

## Required Next Evidence

Before making a strong empirical claim, collect:

- Larger held-out TextWorld evals, preferably 128-256 games.
- At least 3 seeds for standard RL, BLiMP, and BLiMP plus auxiliary losses.
- An untrained Qwen3-1.7B eval under the exact same final eval script and reward semantics.
- A stronger full-history RL baseline with comparable tuning effort.
- A controlled long-history diagnostic where full history helps and short history loses the clue.
- A harder TextWorld split, such as longer quests or more objects.
- One non-TextWorld environment. A curated ScienceWorld subset is the next best target; raw ScienceWorld should not be used without action-space cleanup.
- Matched reporting of train wall time, eval wall time, GPU type, environment calls, context length, and candidate-action scoring batch size.

The clean comparison is not just success rate. It is success per unit of scarce compute.

## Long-History Diagnostic

The repo includes `--env recall` for isolating memory. The first observation contains a final passphrase, then the agent must advance through enough bland steps that a short recent-history prompt loses the clue. At the final gate, the valid actions are passphrase choices.

This environment should produce the desired ordering before larger benchmarks:

```text
full history, no memory
  > short history, no memory

short block + memory
  > short history, no memory
```

If memory cannot improve over short-history no-memory here, moving to Terminal-Bench, R2E-Gym, or raw ScienceWorld will not clarify the mechanism. Those environments are useful later, but only after the state channel is known to carry reward-relevant information.

## Known Risks

- Structured generation changes the policy interface and can break comparability with valid-action scoring.
- Thought imitation can lock in spurious reasoning.
- Score-progress prediction is only as good as the environment metadata.
- Branch contrast is only meaningful if sibling branches share the same prefix and differ in outcome.
- ScienceWorld's raw action space is noisier than TextWorld and may need curriculum or action filtering before it is a fair test of the method.

## Current Implementation Map

Main trainer:

```text
blimp/train_reinforce.py
```

Implemented knobs:

```text
--mode standard|blimp
--structured-think-action
--echo-weight
--score-weight
--action-good-weight
--thought-weight
--future-weight
--future-horizon
--branch-contrast-weight
--branch-contrast-jsonl
--memory-policy-weight
--aux-max-items
--echo-max-words
```

The method should be kept aligned with these flags. If a future paper draft claims a loss term, there should be a corresponding flag, metric, and ablation run.
