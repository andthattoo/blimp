from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from blimp.envs import TextEnv, replay_actions
from blimp.policies import Policy, corrupt_memory, truncate_memory

Variant = Literal["A", "B", "C", "D"]


@dataclass
class StepTrace:
    t: int
    observation: str
    valid_actions: list[str]
    action: str
    next_observation: str
    reward: float
    done: bool
    score: float
    memory_before: str
    memory_after: str
    raw_policy_text: str = ""


@dataclass
class BlockTrace:
    block_id: str
    variant: str
    depth: int
    parent_block_id: str | None
    branch_index: int
    memory_in: str
    memory_used: str
    memory_out: str
    path_actions_before: list[str]
    actions: list[str]
    steps: list[StepTrace]
    done: bool
    score: float
    terminal_reward: float = 0.0


@dataclass
class RunResult:
    episode_id: int
    variant: str
    solved: bool
    score: float
    winning_block_id: str | None
    winning_depth: int | None
    root_to_leaf_steps: int
    total_branch_expanded_actions: int
    total_blocks: int
    solved_path_actions: list[str]
    blocks: list[BlockTrace] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return asdict(self)


@dataclass
class BranchState:
    state_id: str
    depth: int
    path_actions: list[str]
    memory: str
    parent_block_id: str | None
    done: bool
    score: float


def run_variant(
    *,
    episode_id: int,
    variant: Variant,
    env_name: str,
    policy: Policy,
    game_file: str | None,
    seed: int,
    block_len: int = 5,
    flat_steps: int = 40,
    chain_blocks: int = 8,
    branch_factor: int = 2,
    branch_depth: int = 8,
    branch_action_budget: int | None = None,
    memory_words: int = 160,
    stop_on_solved_depth: bool = True,
) -> RunResult:
    rng = random.Random(seed + episode_id * 1009 + ord(variant))
    if variant == "A":
        return _run_flat(
            episode_id=episode_id,
            variant=variant,
            env_name=env_name,
            policy=policy,
            game_file=game_file,
            seed=seed,
            max_steps=flat_steps,
            rng=rng,
            memory_words=memory_words,
        )
    if variant == "B":
        return _run_chain(
            episode_id=episode_id,
            variant=variant,
            env_name=env_name,
            policy=policy,
            game_file=game_file,
            seed=seed,
            block_len=block_len,
            chain_blocks=chain_blocks,
            rng=rng,
            memory_words=memory_words,
        )
    if variant in {"C", "D"}:
        return _run_branching(
            episode_id=episode_id,
            variant=variant,
            env_name=env_name,
            policy=policy,
            game_file=game_file,
            seed=seed,
            block_len=block_len,
            branch_factor=branch_factor,
            branch_depth=branch_depth,
            branch_action_budget=branch_action_budget,
            rng=rng,
            memory_words=memory_words,
            corrupt=(variant == "D"),
            stop_on_solved_depth=stop_on_solved_depth,
        )
    raise ValueError(f"Unknown variant: {variant}")


def _run_flat(
    *,
    episode_id: int,
    variant: str,
    env_name: str,
    policy: Policy,
    game_file: str | None,
    seed: int,
    max_steps: int,
    rng: random.Random,
    memory_words: int,
) -> RunResult:
    env, observation, _, _ = replay_actions(env_name, [], game_file, seed)
    block = _run_block(
        block_id=f"{episode_id}-{variant}-b0",
        variant=variant,
        depth=1,
        parent_block_id=None,
        branch_index=0,
        env=env,
        observation=observation,
        policy=policy,
        path_actions_before=[],
        memory_in="",
        block_len=max_steps,
        rng=rng,
        memory_enabled=False,
        memory_words=memory_words,
        branch_hint="Flat baseline: solve directly without using memory.",
    )
    blocks = [block]
    if block.done:
        block.terminal_reward = 1.0
    return RunResult(
        episode_id=episode_id,
        variant=variant,
        solved=block.done,
        score=block.score,
        winning_block_id=block.block_id if block.done else None,
        winning_depth=1 if block.done else None,
        root_to_leaf_steps=len(block.actions),
        total_branch_expanded_actions=len(block.actions),
        total_blocks=1,
        solved_path_actions=block.actions if block.done else [],
        blocks=blocks,
        config={"flat_steps": max_steps, "memory": False},
    )


def _run_chain(
    *,
    episode_id: int,
    variant: str,
    env_name: str,
    policy: Policy,
    game_file: str | None,
    seed: int,
    block_len: int,
    chain_blocks: int,
    rng: random.Random,
    memory_words: int,
) -> RunResult:
    env, observation, _, _ = replay_actions(env_name, [], game_file, seed)
    memory = ""
    path_actions: list[str] = []
    blocks: list[BlockTrace] = []
    done = False
    score = 0.0

    for depth in range(1, chain_blocks + 1):
        block = _run_block(
            block_id=f"{episode_id}-{variant}-b{depth}",
            variant=variant,
            depth=depth,
            parent_block_id=blocks[-1].block_id if blocks else None,
            branch_index=0,
            env=env,
            observation=observation,
            policy=policy,
            path_actions_before=list(path_actions),
            memory_in=memory,
            block_len=block_len,
            rng=rng,
            memory_enabled=True,
            memory_words=memory_words,
            branch_hint="Chain baseline: continue from memory, no branching.",
        )
        blocks.append(block)
        memory = block.memory_out
        path_actions.extend(block.actions)
        observation = block.steps[-1].next_observation if block.steps else observation
        done = block.done
        score = block.score
        if done:
            break

    if done:
        for block in blocks:
            block.terminal_reward = 1.0

    return RunResult(
        episode_id=episode_id,
        variant=variant,
        solved=done,
        score=score,
        winning_block_id=blocks[-1].block_id if done and blocks else None,
        winning_depth=len(blocks) if done else None,
        root_to_leaf_steps=len(path_actions),
        total_branch_expanded_actions=len(path_actions),
        total_blocks=len(blocks),
        solved_path_actions=path_actions if done else [],
        blocks=blocks,
        config={
            "block_len": block_len,
            "chain_blocks": chain_blocks,
            "memory": True,
        },
    )


def _run_branching(
    *,
    episode_id: int,
    variant: str,
    env_name: str,
    policy: Policy,
    game_file: str | None,
    seed: int,
    block_len: int,
    branch_factor: int,
    branch_depth: int,
    branch_action_budget: int | None,
    rng: random.Random,
    memory_words: int,
    corrupt: bool,
    stop_on_solved_depth: bool,
) -> RunResult:
    root = BranchState(
        state_id="root",
        depth=0,
        path_actions=[],
        memory="",
        parent_block_id=None,
        done=False,
        score=0.0,
    )
    frontier = [root]
    blocks: list[BlockTrace] = []
    solved_states: list[BranchState] = []
    total_actions = 0
    state_counter = 0

    for depth in range(1, branch_depth + 1):
        next_frontier: list[BranchState] = []
        solved_this_depth = False
        for parent in frontier:
            if parent.done:
                next_frontier.append(parent)
                continue
            for branch_index in range(branch_factor):
                if branch_action_budget is not None and total_actions >= branch_action_budget:
                    break

                env, observation, replay_done, replay_score = replay_actions(
                    env_name, parent.path_actions, game_file, seed
                )
                if replay_done:
                    child = BranchState(
                        state_id=f"s{state_counter}",
                        depth=depth,
                        path_actions=list(parent.path_actions),
                        memory=parent.memory,
                        parent_block_id=parent.parent_block_id,
                        done=True,
                        score=replay_score,
                    )
                    state_counter += 1
                    next_frontier.append(child)
                    solved_states.append(child)
                    solved_this_depth = True
                    continue

                memory_used = corrupt_memory(parent.memory, rng) if corrupt else parent.memory
                current_block_len = block_len
                if branch_action_budget is not None:
                    current_block_len = min(block_len, branch_action_budget - total_actions)
                if current_block_len <= 0:
                    break

                block = _run_block(
                    block_id=f"{episode_id}-{variant}-b{len(blocks)}",
                    variant=variant,
                    depth=depth,
                    parent_block_id=parent.parent_block_id,
                    branch_index=branch_index,
                    env=env,
                    observation=observation,
                    policy=policy,
                    path_actions_before=list(parent.path_actions),
                    memory_in=parent.memory,
                    memory_override=memory_used,
                    block_len=current_block_len,
                    rng=rng,
                    memory_enabled=True,
                    memory_words=memory_words,
                    branch_hint=(
                        f"Branch {branch_index}: try a distinct plausible continuation "
                        "from the current memory state."
                    ),
                )
                blocks.append(block)
                total_actions += len(block.actions)
                child = BranchState(
                    state_id=f"s{state_counter}",
                    depth=depth,
                    path_actions=parent.path_actions + block.actions,
                    memory=block.memory_out,
                    parent_block_id=block.block_id,
                    done=block.done,
                    score=block.score,
                )
                state_counter += 1
                next_frontier.append(child)
                if child.done:
                    solved_states.append(child)
                    solved_this_depth = True

            if branch_action_budget is not None and total_actions >= branch_action_budget:
                break

        frontier = next_frontier
        if solved_this_depth and stop_on_solved_depth:
            break
        if branch_action_budget is not None and total_actions >= branch_action_budget:
            break
        if not frontier:
            break

    winner = _choose_winner(solved_states)
    winning_path_actions: list[str] = []
    winning_depth: int | None = None
    winning_block_id: str | None = None
    score = max((state.score for state in frontier + solved_states), default=0.0)

    if winner is not None:
        winning_path_actions = winner.path_actions
        winning_depth = winner.depth
        winning_block_id = winner.parent_block_id
        _assign_terminal_rewards(blocks, winning_block_id)
        score = winner.score

    return RunResult(
        episode_id=episode_id,
        variant=variant,
        solved=winner is not None,
        score=score,
        winning_block_id=winning_block_id,
        winning_depth=winning_depth,
        root_to_leaf_steps=len(winning_path_actions),
        total_branch_expanded_actions=total_actions,
        total_blocks=len(blocks),
        solved_path_actions=winning_path_actions,
        blocks=blocks,
        config={
            "block_len": block_len,
            "branch_factor": branch_factor,
            "branch_depth": branch_depth,
            "branch_action_budget": branch_action_budget,
            "corrupt_memory": corrupt,
            "stop_on_solved_depth": stop_on_solved_depth,
        },
    )


def _run_block(
    *,
    block_id: str,
    variant: str,
    depth: int,
    parent_block_id: str | None,
    branch_index: int,
    env: TextEnv,
    observation: str,
    policy: Policy,
    path_actions_before: list[str],
    memory_in: str,
    block_len: int,
    rng: random.Random,
    memory_enabled: bool,
    memory_words: int,
    branch_hint: str,
    memory_override: str | None = None,
) -> BlockTrace:
    memory_used = memory_override if memory_override is not None else memory_in
    if not memory_enabled:
        memory_used = ""
    memory_current = truncate_memory(memory_used, memory_words)
    steps: list[StepTrace] = []
    actions: list[str] = []
    # Block boundaries are the context boundary. Prior trajectory information
    # must flow through memory, not through a hidden action-history side channel.
    history: list[str] = []
    done = False
    score = 0.0

    for t in range(block_len):
        valid_actions = env.valid_actions()
        memory_before = memory_current
        output = policy.act(
            observation,
            memory_current if memory_enabled else "",
            valid_actions,
            history=history,
            branch_hint=branch_hint,
            rng=rng,
        )
        action = output.action.strip() or "look"
        result = env.step(action)
        actions.append(action)
        history.append(f"ACTION: {action}")

        memory_current = output.memory if memory_enabled else ""
        memory_current = truncate_memory(memory_current, memory_words)
        score = float(result.info.get("score", result.reward))
        done = result.done
        steps.append(
            StepTrace(
                t=t,
                observation=observation,
                valid_actions=valid_actions,
                action=action,
                next_observation=result.observation,
                reward=result.reward,
                done=result.done,
                score=score,
                memory_before=memory_before,
                memory_after=memory_current,
                raw_policy_text=output.raw_text,
            )
        )
        observation = result.observation
        if done:
            break

    return BlockTrace(
        block_id=block_id,
        variant=variant,
        depth=depth,
        parent_block_id=parent_block_id,
        branch_index=branch_index,
        memory_in=memory_in,
        memory_used=memory_used,
        memory_out=memory_current,
        path_actions_before=path_actions_before,
        actions=actions,
        steps=steps,
        done=done,
        score=score,
    )


def _choose_winner(states: list[BranchState]) -> BranchState | None:
    if not states:
        return None
    return min(states, key=lambda state: (state.depth, len(state.path_actions)))


def _assign_terminal_rewards(blocks: list[BlockTrace], winning_block_id: str | None) -> None:
    by_id = {block.block_id: block for block in blocks}
    current = winning_block_id
    while current is not None and current in by_id:
        block = by_id[current]
        block.terminal_reward = 1.0
        current = block.parent_block_id


def write_jsonl(path: Path, rows: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json_dict()) + "\n")


def summarize_runs(rows: list[RunResult]) -> list[dict[str, object]]:
    variants = sorted({row.variant for row in rows})
    summary: list[dict[str, object]] = []
    for variant in variants:
        group = [row for row in rows if row.variant == variant]
        solved = [row for row in group if row.solved]
        success_rate = len(solved) / len(group) if group else 0.0
        mean_root_steps = _mean([row.root_to_leaf_steps for row in solved])
        mean_branch_actions = _mean([row.total_branch_expanded_actions for row in group])
        mean_blocks = _mean([row.total_blocks for row in group])
        summary.append(
            {
                "variant": variant,
                "episodes": len(group),
                "solved": len(solved),
                "success_rate": success_rate,
                "mean_solved_root_to_leaf_steps": mean_root_steps,
                "mean_total_branch_expanded_actions": mean_branch_actions,
                "mean_total_blocks": mean_blocks,
            }
        )
    return summary


def write_summary(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
