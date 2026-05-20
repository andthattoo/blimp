from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from blimp.envs import replay_actions
from blimp.policies import make_policy, resolve_action, truncate_memory


class TextAgent(Protocol):
    def generate_action(
        self,
        *,
        observation: str,
        valid_actions: list[str],
        memory: str,
        history: list[str],
        branch_hint: str,
        memory_enabled: bool,
    ) -> str:
        ...

    def generate_memory_note(
        self,
        *,
        memory: str,
        history: list[str],
        observation: str,
        valid_actions: list[str],
        branch_hint: str,
    ) -> str:
        ...


@dataclass
class SGLangConfig:
    server_url: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 128
    memory_max_tokens: int = 256
    disable_thinking: bool = True
    timeout: float = 120.0


@dataclass
class AgentResponse:
    action: str
    raw_text: str


@dataclass
class ToolStep:
    depth: int
    branch_index: int
    t: int
    observation: str
    valid_actions: list[str]
    memory_before: str
    raw_response: str
    parsed_memory: str | None
    memory_after: str
    action: str
    next_observation: str
    reward: float
    done: bool
    score: float


@dataclass
class BranchNode:
    node_id: str
    parent_id: str | None
    depth: int
    branch_index: int
    path_actions: list[str]
    memory: str
    done: bool
    score: float
    steps: list[ToolStep] = field(default_factory=list)
    memory_prompt: str = ""
    memory_response: str = ""


@dataclass
class SGLangRunResult:
    mode: str
    env_name: str
    episode_id: int
    solved: bool
    score: float
    winning_node_id: str | None
    root_to_leaf_steps: int
    total_expanded_env_steps: int
    total_model_calls: int
    total_nodes: int
    solved_path_actions: list[str]
    nodes: list[BranchNode]
    config: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class SGLangAgent:
    def __init__(self, config: SGLangConfig) -> None:
        self.config = config
        self.url = config.server_url.rstrip("/") + "/v1/chat/completions"

    def _post(self, prompt: str, *, max_tokens: int | None = None) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if self.config.disable_thinking:
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang HTTP {exc.code}: {detail}") from exc
        return body["choices"][0]["message"]["content"]

    def generate_action(
        self,
        *,
        observation: str,
        valid_actions: list[str],
        memory: str,
        history: list[str],
        branch_hint: str,
        memory_enabled: bool,
    ) -> str:
        prompt = build_action_prompt(
            observation=observation,
            valid_actions=valid_actions,
            memory=memory,
            history=history,
            branch_hint=branch_hint,
            memory_enabled=memory_enabled,
        )
        return self._post(prompt, max_tokens=self.config.max_tokens)

    def generate_memory_note(
        self,
        *,
        memory: str,
        history: list[str],
        observation: str,
        valid_actions: list[str],
        branch_hint: str,
    ) -> str:
        prompt = build_memory_prompt(
            memory=memory,
            history=history,
            observation=observation,
            valid_actions=valid_actions,
            branch_hint=branch_hint,
        )
        return self._post(prompt, max_tokens=self.config.memory_max_tokens)


class MockPolicyAgent:
    def __init__(self, policy_name: str, seed: int = 0) -> None:
        self.policy = make_policy(policy_name)
        self.rng = random.Random(seed)

    def generate_action(
        self,
        *,
        observation: str,
        valid_actions: list[str],
        memory: str,
        history: list[str],
        branch_hint: str,
        memory_enabled: bool,
    ) -> str:
        visible_memory = _mock_visible_memory(memory if memory_enabled else "", history)
        output = self.policy.act(
            observation,
            visible_memory,
            valid_actions,
            history=history,
            branch_hint=branch_hint,
            rng=self.rng,
        )
        return f"ACTION: {output.action}"

    def generate_memory_note(
        self,
        *,
        memory: str,
        history: list[str],
        observation: str,
        valid_actions: list[str],
        branch_hint: str,
    ) -> str:
        visible_memory = _mock_visible_memory(memory, history)
        output = self.policy.act(
            observation,
            visible_memory,
            valid_actions,
            history=history,
            branch_hint=branch_hint,
            rng=self.rng,
        )
        return output.memory


def _mock_visible_memory(memory: str, history: list[str]) -> str:
    transcript = "\n".join(history[-16:])
    return "\n".join(part for part in [memory, transcript] if part.strip())


def build_action_prompt(
    *,
    observation: str,
    valid_actions: list[str],
    memory: str,
    history: list[str],
    branch_hint: str,
    memory_enabled: bool,
) -> str:
    actions = "\n".join(f"- {action}" for action in valid_actions) or "- look"
    transcript = "\n".join(history[-16:]) if history else "None."
    if memory_enabled:
        memory_text = memory.strip() or "empty"
        return f"""You are controlling a text-environment agent.
Your job is to complete the environment goal.

You have a branch-local Markdown note written by your previous self. It belongs
only to this branch. Use it as durable state, but trust the current observation
when they conflict.

Output exactly:
ACTION: <one valid action>

{branch_hint}

Previous-self note:
{memory_text}

Recent block transcript:
{transcript}

Observation:
{observation}

Valid actions:
{actions}
"""
    return f"""You are controlling a text-environment agent.
Your job is to complete the environment goal.
Choose exactly one valid action.

Output exactly:
ACTION: <one valid action>

Full trajectory transcript:
{transcript}

Observation:
{observation}

Valid actions:
{actions}
"""


def build_memory_prompt(
    *,
    memory: str,
    history: list[str],
    observation: str,
    valid_actions: list[str],
    branch_hint: str,
) -> str:
    actions = "\n".join(f"- {action}" for action in valid_actions) or "- look"
    transcript = "\n".join(history) if history else "None."
    memory_text = memory.strip() or "empty"
    return f"""You are about to lose the transcript.
Write anything that would help your next self continue from scratch.

This is branch-local memory: it will be copied only to continuations of this
branch, not to sibling branches. Write a concise free-form Markdown note. Include
durable facts, map, inventory, hypotheses, failed actions, and the next useful
steps. Do not invent facts.

{branch_hint}

Previous note:
{memory_text}

Block transcript being lost:
{transcript}

Current observation:
{observation}

Current valid actions:
{actions}

Markdown note for next self:
"""


def parse_agent_response(text: str) -> AgentResponse:
    text = strip_reasoning(text)
    action = ""
    action_match = re.search(r"(?im)^\s*ACTION\s*:\s*(.+)$", text)
    if action_match:
        action = action_match.group(1).strip()
    else:
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        action = nonempty[-1] if nonempty else "look"
        action = action.removeprefix("-").strip()

    return AgentResponse(action=action, raw_text=text.strip())


def strip_reasoning(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<think>.*$", "", text)
    return text.strip()


def run_standard(
    *,
    episode_id: int,
    env_name: str,
    game_file: str | None,
    seed: int,
    agent: TextAgent,
    max_steps: int,
) -> SGLangRunResult:
    env, observation, _, _ = replay_actions(env_name, [], game_file, seed)
    history: list[str] = []
    steps: list[ToolStep] = []
    actions: list[str] = []
    done = False
    score = 0.0

    for t in range(max_steps):
        valid_actions = env.valid_actions()
        raw = agent.generate_action(
            observation=observation,
            valid_actions=valid_actions,
            memory="",
            history=history,
            branch_hint="Standard baseline: use the full trajectory transcript.",
            memory_enabled=False,
        )
        parsed = parse_agent_response(raw)
        action = resolve_action(parsed.action, valid_actions) or (
            valid_actions[0] if valid_actions else "look"
        )
        result = env.step(action)
        score = float(result.info.get("score", result.reward))
        step = ToolStep(
            depth=1,
            branch_index=0,
            t=t,
            observation=observation,
            valid_actions=valid_actions,
            memory_before="",
            raw_response=raw,
            parsed_memory=None,
            memory_after="",
            action=action,
            next_observation=result.observation,
            reward=result.reward,
            done=result.done,
            score=score,
        )
        steps.append(step)
        actions.append(action)
        history.append(f"OBSERVATION: {observation}\nACTION: {action}")
        observation = result.observation
        done = result.done
        if done:
            break

    node = BranchNode(
        node_id="standard",
        parent_id=None,
        depth=1,
        branch_index=0,
        path_actions=actions,
        memory="",
        done=done,
        score=score,
        steps=steps,
    )
    return SGLangRunResult(
        mode="standard",
        env_name=env_name,
        episode_id=episode_id,
        solved=done,
        score=score,
        winning_node_id="standard" if done else None,
        root_to_leaf_steps=len(actions) if done else 0,
        total_expanded_env_steps=len(actions),
        total_model_calls=len(actions),
        total_nodes=1,
        solved_path_actions=actions if done else [],
        nodes=[node],
        config={"max_steps": max_steps, "memory": False},
    )


def run_blimp_tree(
    *,
    episode_id: int,
    env_name: str,
    game_file: str | None,
    seed: int,
    agent: TextAgent,
    block_len: int,
    max_depth: int,
    branch_factor: int,
    branch_action_budget: int,
    memory_words: int,
    stop_on_solved_depth: bool,
) -> SGLangRunResult:
    frontier = [
        BranchNode(
            node_id="root",
            parent_id=None,
            depth=0,
            branch_index=0,
            path_actions=[],
            memory="",
            done=False,
            score=0.0,
        )
    ]
    nodes: list[BranchNode] = []
    solved: list[BranchNode] = []
    total_steps = 0
    total_calls = 0
    node_counter = 0

    for depth in range(1, max_depth + 1):
        next_frontier: list[BranchNode] = []
        solved_this_depth = False
        for parent in frontier:
            if parent.done:
                next_frontier.append(parent)
                continue
            for branch_index in range(branch_factor):
                if total_steps >= branch_action_budget:
                    break
                env, observation, replay_done, replay_score = replay_actions(
                    env_name, parent.path_actions, game_file, seed
                )
                if replay_done:
                    solved.append(parent)
                    next_frontier.append(parent)
                    solved_this_depth = True
                    continue

                memory = truncate_memory(parent.memory, memory_words)
                history: list[str] = []
                steps: list[ToolStep] = []
                actions: list[str] = []
                done = False
                score = replay_score
                for t in range(block_len):
                    if total_steps >= branch_action_budget:
                        break
                    valid_actions = env.valid_actions()
                    memory_before = memory
                    raw = agent.generate_action(
                        observation=observation,
                        valid_actions=valid_actions,
                        memory=memory,
                        history=history,
                        branch_hint=_branch_hint(branch_index),
                        memory_enabled=True,
                    )
                    total_calls += 1
                    parsed = parse_agent_response(raw)
                    action = resolve_action(parsed.action, valid_actions) or (
                        valid_actions[0] if valid_actions else "look"
                    )
                    result = env.step(action)
                    total_steps += 1
                    score = float(result.info.get("score", result.reward))
                    step = ToolStep(
                        depth=depth,
                        branch_index=branch_index,
                        t=t,
                        observation=observation,
                        valid_actions=valid_actions,
                        memory_before=memory_before,
                        raw_response=raw,
                        parsed_memory=None,
                        memory_after=memory,
                        action=action,
                        next_observation=result.observation,
                        reward=result.reward,
                        done=result.done,
                        score=score,
                    )
                    steps.append(step)
                    actions.append(action)
                    history.append(f"OBSERVATION: {observation}\nACTION: {action}")
                    observation = result.observation
                    done = result.done
                    if done:
                        break

                memory_prompt = ""
                memory_response = ""
                if not done and actions:
                    valid_actions = env.valid_actions()
                    memory_prompt = build_memory_prompt(
                        memory=memory,
                        history=history,
                        observation=observation,
                        valid_actions=valid_actions,
                        branch_hint=_branch_hint(branch_index),
                    )
                    memory_response = strip_reasoning(
                        agent.generate_memory_note(
                            memory=memory,
                            history=history,
                            observation=observation,
                            valid_actions=valid_actions,
                            branch_hint=_branch_hint(branch_index),
                        )
                    ).strip()
                    total_calls += 1
                    if memory_response:
                        memory = truncate_memory(memory_response, memory_words)

                node = BranchNode(
                    node_id=f"n{node_counter}",
                    parent_id=parent.node_id,
                    depth=depth,
                    branch_index=branch_index,
                    path_actions=parent.path_actions + actions,
                    memory=memory,
                    done=done,
                    score=score,
                    steps=steps,
                    memory_prompt=memory_prompt,
                    memory_response=memory_response,
                )
                node_counter += 1
                nodes.append(node)
                next_frontier.append(node)
                print(
                    f"block episode={episode_id} depth={depth} branch={branch_index} "
                    f"done={int(done)} score={score:.3f} actions={len(actions)} "
                    f"total_env_steps={total_steps}",
                    flush=True,
                )
                if done:
                    solved.append(node)
                    solved_this_depth = True

            if total_steps >= branch_action_budget:
                break
        frontier = next_frontier
        if solved_this_depth and stop_on_solved_depth:
            break
        if total_steps >= branch_action_budget or not frontier:
            break

    winner = min(solved, key=lambda node: (len(node.path_actions), node.depth), default=None)
    score = winner.score if winner is not None else max(
        (node.score for node in nodes + frontier), default=0.0
    )
    return SGLangRunResult(
        mode="blimp",
        env_name=env_name,
        episode_id=episode_id,
        solved=winner is not None,
        score=score,
        winning_node_id=winner.node_id if winner else None,
        root_to_leaf_steps=len(winner.path_actions) if winner else 0,
        total_expanded_env_steps=total_steps,
        total_model_calls=total_calls,
        total_nodes=len(nodes),
        solved_path_actions=winner.path_actions if winner else [],
        nodes=nodes,
        config={
            "block_len": block_len,
            "max_depth": max_depth,
            "branch_factor": branch_factor,
            "branch_action_budget": branch_action_budget,
            "memory_words": memory_words,
            "memory": "block-boundary branch-local markdown dead-man note",
            "stop_on_solved_depth": stop_on_solved_depth,
        },
    )


def _branch_hint(branch_index: int) -> str:
    if branch_index == 0:
        return "Branch hint: pursue the most direct plausible route."
    if branch_index == 1:
        return "Branch hint: try a different plausible route or missing prerequisite."
    return f"Branch hint: branch {branch_index}, explore a distinct continuation."


def write_jsonl(path: Path, rows: list[SGLangRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json_dict()) + "\n")


def write_summary(path: Path, rows: list[SGLangRunResult]) -> None:
    summary = []
    for row in rows:
        summary.append(
            {
                "mode": row.mode,
                "episode_id": row.episode_id,
                "solved": int(row.solved),
                "score": row.score,
                "root_to_leaf_steps": row.root_to_leaf_steps,
                "total_expanded_env_steps": row.total_expanded_env_steps,
                "total_model_calls": row.total_model_calls,
                "total_nodes": row.total_nodes,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SGLang-backed BLiMP dead-man-note rollouts."
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-policy", choices=["scripted-tiny", "scripted-hard"], default=None)
    parser.add_argument("--env", choices=["tiny", "hard", "textworld"], default="hard")
    parser.add_argument("--game-file", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--modes", default="standard,blimp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--block-len", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--branch-action-budget", type=int, default=120)
    parser.add_argument("--memory-words", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--memory-max-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-stop-on-solved-depth", action="store_true")
    parser.add_argument("--out", default="runs/sglang-latest")
    args = parser.parse_args()

    if args.mock_policy:
        agent: TextAgent = MockPolicyAgent(args.mock_policy, seed=args.seed)
    else:
        agent = SGLangAgent(
            SGLangConfig(
                server_url=args.server_url,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                memory_max_tokens=args.memory_max_tokens,
                disable_thinking=not args.enable_thinking,
                timeout=args.timeout,
            )
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    modes = [mode.strip().lower() for mode in args.modes.split(",") if mode.strip()]
    rows: list[SGLangRunResult] = []
    started = time.time()
    for episode_id in range(args.episodes):
        for mode in modes:
            if mode == "standard":
                result = run_standard(
                    episode_id=episode_id,
                    env_name=args.env,
                    game_file=args.game_file,
                    seed=args.seed,
                    agent=agent,
                    max_steps=args.max_steps,
                )
            elif mode == "blimp":
                result = run_blimp_tree(
                    episode_id=episode_id,
                    env_name=args.env,
                    game_file=args.game_file,
                    seed=args.seed,
                    agent=agent,
                    block_len=args.block_len,
                    max_depth=args.max_depth,
                    branch_factor=args.branch_factor,
                    branch_action_budget=args.branch_action_budget,
                    memory_words=args.memory_words,
                    stop_on_solved_depth=not args.no_stop_on_solved_depth,
                )
            else:
                raise ValueError(f"unknown mode: {mode}")
            rows.append(result)
            write_jsonl(out_dir / "trajectories.jsonl", rows)
            write_summary(out_dir / "summary.json", rows)
            print(
                f"episode={episode_id} mode={mode} solved={int(result.solved)} "
                f"score={result.score:.3f} root_steps={result.root_to_leaf_steps} "
                f"expanded_steps={result.total_expanded_env_steps} "
                f"model_calls={result.total_model_calls}",
                flush=True,
            )
    print(json.dumps(json.load((out_dir / "summary.json").open()), indent=2))
    print(f"elapsed_seconds={time.time() - started:.1f}")


if __name__ == "__main__":
    main()
