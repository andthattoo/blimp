from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from blimp.envs import make_env
from blimp.sglang_rollout import build_action_prompt, build_memory_prompt, strip_reasoning


@dataclass
class Transition:
    prompt: str
    action: str
    reward: float
    done: bool
    score: float


@dataclass
class EpisodeTrace:
    episode_id: int
    mode: str
    solved: bool
    score: float
    total_reward: float
    actions: list[str]
    memories: list[str]
    transitions: list[Transition]

    def json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transitions"] = [asdict(t) for t in self.transitions]
        return data


class ValidActionPolicy:
    def __init__(
        self,
        model_name: str,
        *,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        learning_rate: float,
        device: str,
        seed: int,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.config.use_cache = False

        self.uses_lora = lora_rank > 0
        if self.uses_lora:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=learning_rate,
        )
        torch.manual_seed(seed)

    def render_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt.rstrip() + "\n"

    @torch.no_grad()
    def action_scores(self, prompt: str, valid_actions: list[str]) -> list[float]:
        self.model.eval()
        return [
            float(self.completion_logprob(prompt, action_completion(action), grad=False))
            for action in valid_actions
        ]

    def choose_action(
        self,
        prompt: str,
        valid_actions: list[str],
        *,
        temperature: float,
        epsilon: float,
        greedy: bool,
        rng: random.Random,
    ) -> tuple[str, float]:
        if not valid_actions:
            return "look", 0.0
        scores = self.action_scores(prompt, valid_actions)
        if greedy:
            index = max(range(len(valid_actions)), key=lambda i: scores[i])
            return valid_actions[index], scores[index]
        if rng.random() < epsilon:
            index = rng.randrange(len(valid_actions))
            return valid_actions[index], scores[index]
        logits = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=0)
        index = int(torch.multinomial(probs, 1).item())
        return valid_actions[index], scores[index]

    def generate_note(self, prompt: str, *, max_new_tokens: int) -> str:
        self.model.eval()
        rendered = self.render_prompt(prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return strip_reasoning(text).strip()

    def completion_logprob(self, prompt: str, completion: str, *, grad: bool) -> torch.Tensor:
        rendered = self.render_prompt(prompt)
        prompt_ids = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        full_ids = self.tokenizer(
            rendered + completion,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        if full_ids.shape[1] <= prompt_ids.shape[1]:
            return torch.zeros((), device=self.device)

        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = self.model(full_ids[:, :-1]).logits[0]
            targets = full_ids[0, 1:]
            token_logprobs = F.log_softmax(logits.float(), dim=-1)
            gathered = token_logprobs.gather(1, targets[:, None]).squeeze(1)
            first_completion_target = max(prompt_ids.shape[1] - 1, 0)
            completion_logprobs = gathered[first_completion_target:]
            return completion_logprobs.mean()

    def update(
        self,
        traces: list[EpisodeTrace],
        *,
        gamma: float,
        normalize_advantages: bool,
        grad_clip: float,
    ) -> dict[str, float]:
        items: list[tuple[str, str, float]] = []
        for trace in traces:
            returns: list[float] = []
            running = 0.0
            for transition in reversed(trace.transitions):
                running = transition.reward + gamma * running
                returns.append(running)
            returns.reverse()
            for transition, ret in zip(trace.transitions, returns):
                items.append((transition.prompt, transition.action, ret))

        if not items:
            return {"loss": 0.0, "mean_return": 0.0, "num_items": 0.0}

        returns_tensor = torch.tensor([item[2] for item in items], dtype=torch.float32)
        advantages = returns_tensor.clone()
        if normalize_advantages and len(items) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for (prompt, action, _), advantage in zip(items, advantages):
            logprob = self.completion_logprob(prompt, action_completion(action), grad=True)
            losses.append(-advantage.to(self.device) * logprob)
        loss = torch.stack(losses).mean()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                grad_clip,
            )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "mean_return": float(returns_tensor.mean()),
            "num_items": float(len(items)),
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def action_completion(action: str) -> str:
    return f"ACTION: {action}"


def run_episode(
    *,
    policy: ValidActionPolicy,
    episode_id: int,
    env_name: str,
    game_file: str | None,
    seed: int,
    mode: str,
    max_steps: int,
    block_len: int,
    memory_words: int,
    memory_max_tokens: int,
    temperature: float,
    epsilon: float,
    greedy: bool,
    rng: random.Random,
) -> EpisodeTrace:
    env = make_env(env_name, game_file)
    observation = env.reset(seed=seed + episode_id)
    full_history: list[str] = []
    block_history: list[str] = []
    actions: list[str] = []
    memories: list[str] = []
    transitions: list[Transition] = []
    memory = ""
    score = 0.0
    solved = False
    total_reward = 0.0

    for t in range(max_steps):
        valid_actions = env.valid_actions()
        memory_enabled = mode == "blimp"
        history = block_history if memory_enabled else full_history
        prompt = build_action_prompt(
            observation=observation,
            valid_actions=valid_actions,
            memory=memory,
            history=history,
            branch_hint="Training rollout: choose the action most likely to complete the task.",
            memory_enabled=memory_enabled,
        )
        action, _ = policy.choose_action(
            prompt,
            valid_actions,
            temperature=temperature,
            epsilon=epsilon,
            greedy=greedy,
            rng=rng,
        )
        result = env.step(action)
        score = float(result.info.get("score", result.reward))
        reward = float(result.reward)
        if result.done:
            reward += 1.0
        total_reward += reward
        transitions.append(
            Transition(
                prompt=prompt,
                action=action,
                reward=reward,
                done=result.done,
                score=score,
            )
        )
        actions.append(action)
        history_line = f"OBSERVATION: {observation}\nACTION: {action}"
        full_history.append(history_line)
        block_history.append(history_line)
        observation = result.observation
        solved = result.done
        if solved:
            break

        if memory_enabled and (t + 1) % block_len == 0:
            note_prompt = build_memory_prompt(
                memory=memory,
                history=block_history,
                observation=observation,
                valid_actions=env.valid_actions(),
                branch_hint="Training rollout: preserve only facts useful for the next block.",
            )
            note = policy.generate_note(note_prompt, max_new_tokens=memory_max_tokens)
            if note:
                memory = truncate_words(note, memory_words)
                memories.append(memory)
            block_history = []

    return EpisodeTrace(
        episode_id=episode_id,
        mode=mode,
        solved=solved,
        score=score,
        total_reward=total_reward,
        actions=actions,
        memories=memories,
        transitions=transitions,
    )


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:]).strip()


def summarize(traces: list[EpisodeTrace]) -> dict[str, float]:
    if not traces:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "mean_score": 0.0,
            "mean_reward": 0.0,
            "mean_steps": 0.0,
        }
    return {
        "episodes": len(traces),
        "success_rate": sum(t.solved for t in traces) / len(traces),
        "mean_score": sum(t.score for t in traces) / len(traces),
        "mean_reward": sum(t.total_reward for t in traces) / len(traces),
        "mean_steps": sum(len(t.actions) for t in traces) / len(traces),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online REINFORCE over valid text-environment actions."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env", choices=["tiny", "hard", "textworld"], default="hard")
    parser.add_argument("--game-file", default=None)
    parser.add_argument("--mode", choices=["standard", "blimp"], default="blimp")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--block-len", type=int, default=5)
    parser.add_argument("--memory-words", type=int, default=240)
    parser.add_argument("--memory-max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/reinforce-latest")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
        handle.write("\n")

    rng = random.Random(args.seed)
    policy = ValidActionPolicy(
        args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )

    started = time.time()
    global_episode = 0
    for update in range(args.updates + 1):
        if update % args.eval_every == 0:
            eval_traces = [
                run_episode(
                    policy=policy,
                    episode_id=10_000 + update * 100 + i,
                    env_name=args.env,
                    game_file=args.game_file,
                    seed=args.seed,
                    mode=args.mode,
                    max_steps=args.max_steps,
                    block_len=args.block_len,
                    memory_words=args.memory_words,
                    memory_max_tokens=args.memory_max_tokens,
                    temperature=1.0,
                    epsilon=0.0,
                    greedy=True,
                    rng=rng,
                )
                for i in range(args.eval_episodes)
            ]
            eval_summary = summarize(eval_traces)
            row = {
                "phase": "eval",
                "update": update,
                **eval_summary,
                "elapsed_seconds": time.time() - started,
            }
            write_jsonl(out_dir / "metrics.jsonl", [row])
            write_jsonl(
                out_dir / "eval_traces.jsonl",
                [trace.json_dict() for trace in eval_traces],
            )
            print(json.dumps(row), flush=True)

        if update == args.updates:
            break

        train_traces = [
            run_episode(
                policy=policy,
                episode_id=global_episode + i,
                env_name=args.env,
                game_file=args.game_file,
                seed=args.seed,
                mode=args.mode,
                max_steps=args.max_steps,
                block_len=args.block_len,
                memory_words=args.memory_words,
                memory_max_tokens=args.memory_max_tokens,
                temperature=args.temperature,
                epsilon=args.epsilon,
                greedy=False,
                rng=rng,
            )
            for i in range(args.episodes_per_update)
        ]
        global_episode += args.episodes_per_update
        update_stats = policy.update(
            train_traces,
            gamma=args.gamma,
            normalize_advantages=not args.no_normalize_advantages,
            grad_clip=args.grad_clip,
        )
        row = {
            "phase": "train",
            "update": update + 1,
            **summarize(train_traces),
            **update_stats,
            "elapsed_seconds": time.time() - started,
        }
        write_jsonl(out_dir / "metrics.jsonl", [row])
        write_jsonl(
            out_dir / "train_traces.jsonl",
            [trace.json_dict() for trace in train_traces],
        )
        print(json.dumps(row), flush=True)

    policy.save(out_dir / "adapter")
    summary_rows = [
        json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()
    ]
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows[-10:], handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
