from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from blimp.envs import make_env
from blimp.policies import resolve_action
from blimp.sglang_rollout import build_action_prompt, build_memory_prompt, strip_reasoning


SCIENCEWORLD_TASK_VARIATIONS: dict[str, int] = {
    "boil": 30,
    "melt": 30,
    "freeze": 30,
    "use-thermometer": 540,
    "measure-melting-point-known-substance": 300,
    "measure-melting-point-unknown-substance": 300,
    "test-conductivity": 900,
    "test-conductivity-of-unknown-substances": 900,
    "find-living-thing": 300,
    "find-non-living-thing": 300,
    "find-plant": 300,
    "find-animal": 300,
    "grow-plant": 300,
    "grow-fruit": 300,
    "lifespan-longest-lived": 300,
    "lifespan-shortest-lived": 300,
    "inclined-plane-determine-angle": 180,
    "inclined-plane-friction-unnamed-surfaces": 180,
    "mendelian-genetics-known-plant": 300,
}

DEFAULT_SCIENCEWORLD_TASKS = (
    "boil,melt,freeze,use-thermometer,measure-melting-point-known-substance,"
    "test-conductivity,find-living-thing,find-non-living-thing,find-plant,"
    "find-animal,grow-plant,grow-fruit,lifespan-longest-lived,"
    "inclined-plane-determine-angle"
)


@dataclass
class Transition:
    prompt: str
    action: str
    reward: float
    done: bool
    score: float
    observation: str = ""
    thinking: str = ""
    raw_completion: str = ""
    policy_completion: str = ""
    next_observation: str = ""
    score_delta: float = 0.0
    won: bool = False
    valid: bool = True


@dataclass
class MemoryWrite:
    prompt: str
    completion: str
    after_step: int


@dataclass
class EpisodeTrace:
    episode_id: int
    mode: str
    game_file: str | None
    solved: bool
    score: float
    total_reward: float
    actions: list[str]
    memories: list[str]
    transitions: list[Transition]
    memory_writes: list[MemoryWrite] = field(default_factory=list)

    def json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transitions"] = [asdict(t) for t in self.transitions]
        data["memory_writes"] = [asdict(m) for m in self.memory_writes]
        return data


@dataclass
class ParsedThinkAction:
    thinking: str
    action: str
    raw_text: str


@dataclass
class AuxLossExample:
    loss_type: str
    prompt: str
    completion: str
    weight: float


@dataclass
class BranchPreferenceExample:
    prompt: str
    preferred: str
    rejected: str
    weight: float


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
        score_batch_size: int,
        gradient_checkpointing: bool,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.score_batch_size = max(1, score_batch_size)
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
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("gradient checkpointing: enabled", flush=True)

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
        else:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(
                f"trainable params: {trainable:,} || all params: {total:,} "
                f"|| trainable%: {100 * trainable / total:.4f}",
                flush=True,
            )

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
        completions = [action_completion(action) for action in valid_actions]
        scores = []
        start = 0
        batch_size = self.score_batch_size
        while start < len(completions):
            batch = completions[start : start + batch_size]
            try:
                scores.extend(
                    float(logprob.detach().cpu())
                    for logprob in self.completion_logprobs(prompt, batch)
                )
                start += len(batch)
            except torch.OutOfMemoryError:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.score_batch_size = min(self.score_batch_size, batch_size)
                print(
                    f"action scoring OOM; retrying with score_batch_size={batch_size}",
                    flush=True,
                )
        return scores

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

    def generate_think_action(
        self,
        prompt: str,
        valid_actions: list[str],
        *,
        temperature: float,
        epsilon: float,
        greedy: bool,
        rng: random.Random,
        max_new_tokens: int,
    ) -> ParsedThinkAction:
        if valid_actions and not greedy and rng.random() < epsilon:
            action = rng.choice(valid_actions)
            thinking = "Exploration sample."
            return ParsedThinkAction(
                thinking=thinking,
                action=action,
                raw_text=structured_completion(thinking, action).strip(),
            )

        self.model.eval()
        rendered = self.render_prompt(prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if greedy:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = max(temperature, 1e-4)
        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        text = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return parse_think_action_response(text)

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
            completion_tokens = full_ids.shape[1] - prompt_ids.shape[1]
            logits = self._completion_logits(
                full_ids[:, :-1],
                logits_to_keep=completion_tokens,
            )[0].float()
            completion_targets = full_ids[0, 1:][-completion_tokens:]
            token_logprobs = F.log_softmax(logits, dim=-1)
            gathered = token_logprobs.gather(1, completion_targets[:, None]).squeeze(1)
            return gathered.mean()

    @torch.no_grad()
    def completion_logprobs(self, prompt: str, completions: list[str]) -> torch.Tensor:
        if not completions:
            return torch.empty(0, device=self.device)
        rendered = self.render_prompt(prompt)
        prompt_ids = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        full = self.tokenizer(
            [rendered + completion for completion in completions],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.device)
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        if input_ids.shape[1] <= prompt_ids.shape[1]:
            return torch.zeros((len(completions),), device=self.device)

        completion_tokens = input_ids.shape[1] - prompt_ids.shape[1]
        logits = self._completion_logits(
            input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            logits_to_keep=completion_tokens,
        ).float()
        targets = input_ids[:, 1:][:, -completion_tokens:]
        target_mask = attention_mask[:, 1:][:, -completion_tokens:].bool()
        token_logprobs = F.log_softmax(logits, dim=-1)
        gathered = token_logprobs.gather(2, targets[:, :, None]).squeeze(2)
        masked = gathered.masked_fill(~target_mask, 0.0)
        denominators = target_mask.sum(dim=1).clamp_min(1)
        return masked.sum(dim=1) / denominators

    def _completion_logits(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Return only logits needed for completion tokens when the model supports it."""
        logits_to_keep = max(1, int(logits_to_keep))
        try:
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
            )
            logits = output.logits
        except TypeError:
            output = self.model(input_ids, attention_mask=attention_mask)
            logits = output.logits[:, -logits_to_keep:]

        if logits.shape[1] != logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return logits

    def update(
        self,
        traces: list[EpisodeTrace],
        *,
        gamma: float,
        normalize_advantages: bool,
        grad_clip: float,
        echo_weight: float,
        score_weight: float,
        action_good_weight: float,
        thought_weight: float,
        future_weight: float,
        future_horizon: int,
        branch_contrast_weight: float,
        branch_contrast_jsonl: str | None,
        memory_policy_weight: float,
        aux_max_items: int,
        echo_max_words: int,
    ) -> dict[str, float]:
        items: list[tuple[str, str, float]] = []
        memory_items: list[tuple[str, str, float]] = []
        for trace in traces:
            returns = discounted_returns(trace.transitions, gamma)
            for transition, ret in zip(trace.transitions, returns):
                completion = transition.policy_completion or action_completion(
                    transition.action
                )
                items.append((transition.prompt, completion, ret))
            if memory_policy_weight > 0:
                memory_items.extend(memory_policy_items(trace, returns))

        if not items:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "echo_loss": 0.0,
                "score_loss": 0.0,
                "action_good_loss": 0.0,
                "thought_loss": 0.0,
                "future_loss": 0.0,
                "branch_contrast_loss": 0.0,
                "memory_policy_loss": 0.0,
                "mean_return": 0.0,
                "num_items": 0.0,
                "aux_items": 0.0,
                "branch_pairs": 0.0,
                "memory_policy_items": 0.0,
            }

        returns_tensor = torch.tensor([item[2] for item in items], dtype=torch.float32)
        advantages = returns_tensor.clone()
        return_mean = returns_tensor.mean()
        return_std = returns_tensor.std() + 1e-6
        if normalize_advantages and len(items) > 1:
            advantages = (advantages - return_mean) / return_std

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        policy_loss_value = 0.0
        echo_loss_value = 0.0
        score_loss_value = 0.0
        action_good_loss_value = 0.0
        thought_loss_value = 0.0
        future_loss_value = 0.0
        branch_contrast_loss_value = 0.0
        memory_policy_loss_value = 0.0
        scale = 1.0 / len(items)
        for (prompt, completion, _), advantage in zip(items, advantages):
            logprob = self.completion_logprob(prompt, completion, grad=True)
            loss = -advantage.to(self.device) * logprob * scale
            detached = float(loss.detach().cpu())
            loss_value += detached
            policy_loss_value += detached
            loss.backward()

        aux_examples = compile_aux_loss_examples(
            traces,
            echo_weight=echo_weight,
            score_weight=score_weight,
            action_good_weight=action_good_weight,
            thought_weight=thought_weight,
            future_weight=future_weight,
            future_horizon=future_horizon,
            echo_max_words=echo_max_words,
        )
        aux_examples = select_aux_examples(aux_examples, aux_max_items)
        aux_scale = 1.0 / max(1, len(aux_examples))
        for example in aux_examples:
            logprob = self.completion_logprob(
                example.prompt,
                example.completion,
                grad=True,
            )
            loss = -example.weight * logprob * aux_scale
            detached = float(loss.detach().cpu())
            loss_value += detached
            if example.loss_type == "echo":
                echo_loss_value += detached
            elif example.loss_type == "score":
                score_loss_value += detached
            elif example.loss_type == "action_good":
                action_good_loss_value += detached
            elif example.loss_type == "thought":
                thought_loss_value += detached
            elif example.loss_type == "future":
                future_loss_value += detached
            loss.backward()

        branch_pairs = (
            load_branch_contrast_examples(
                Path(branch_contrast_jsonl),
                weight=branch_contrast_weight,
                max_pairs=aux_max_items,
            )
            if branch_contrast_weight > 0 and branch_contrast_jsonl
            else []
        )
        branch_scale = 1.0 / max(1, len(branch_pairs))
        for pair in branch_pairs:
            preferred_logprob = self.completion_logprob(
                pair.prompt,
                pair.preferred,
                grad=True,
            )
            rejected_logprob = self.completion_logprob(
                pair.prompt,
                pair.rejected,
                grad=True,
            )
            loss = (
                -pair.weight
                * F.logsigmoid(preferred_logprob - rejected_logprob)
                * branch_scale
            )
            detached = float(loss.detach().cpu())
            loss_value += detached
            branch_contrast_loss_value += detached
            loss.backward()

        if memory_items:
            memory_returns_tensor = torch.tensor(
                [item[2] for item in memory_items],
                dtype=torch.float32,
            )
            memory_advantages = memory_returns_tensor.clone()
            if normalize_advantages and len(items) > 1:
                memory_advantages = (memory_advantages - return_mean) / return_std
            memory_scale = memory_policy_weight / len(memory_items)
            for (prompt, completion, _), advantage in zip(
                memory_items,
                memory_advantages,
            ):
                logprob = self.completion_logprob(prompt, completion, grad=True)
                loss = -advantage.to(self.device) * logprob * memory_scale
                detached = float(loss.detach().cpu())
                loss_value += detached
                memory_policy_loss_value += detached
                loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                grad_clip,
            )
        self.optimizer.step()
        return {
            "loss": loss_value,
            "policy_loss": policy_loss_value,
            "echo_loss": echo_loss_value,
            "score_loss": score_loss_value,
            "action_good_loss": action_good_loss_value,
            "thought_loss": thought_loss_value,
            "future_loss": future_loss_value,
            "branch_contrast_loss": branch_contrast_loss_value,
            "memory_policy_loss": memory_policy_loss_value,
            "mean_return": float(returns_tensor.mean()),
            "num_items": float(len(items)),
            "aux_items": float(len(aux_examples)),
            "branch_pairs": float(len(branch_pairs)),
            "memory_policy_items": float(len(memory_items)),
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def action_completion(action: str) -> str:
    return f"ACTION: {action}"


def discounted_returns(transitions: list[Transition], gamma: float) -> list[float]:
    returns: list[float] = []
    running = 0.0
    for transition in reversed(transitions):
        running = transition.reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def memory_policy_items(
    trace: EpisodeTrace,
    returns: list[float],
) -> list[tuple[str, str, float]]:
    items: list[tuple[str, str, float]] = []
    for write in trace.memory_writes:
        completion = write.completion.strip()
        if not completion:
            continue
        future_index = max(0, min(write.after_step, len(returns)))
        future_return = returns[future_index] if future_index < len(returns) else 0.0
        items.append((write.prompt, completion, future_return))
    return items


def structured_completion(thinking: str, action: str) -> str:
    thinking = thinking.strip() or "No additional reasoning."
    return f"THINK: {thinking}\nACTION: {action}"


def parse_think_action_response(text: str) -> ParsedThinkAction:
    cleaned = strip_reasoning(text).strip()
    thinking = ""
    action = ""
    think_match = re.search(
        r"(?ims)^\s*THINK\s*:\s*(.*?)(?=^\s*ACTION\s*:|\Z)",
        cleaned,
    )
    if think_match:
        thinking = think_match.group(1).strip()
    action_match = re.search(r"(?im)^\s*ACTION\s*:\s*(.+)$", cleaned)
    if action_match:
        action = action_match.group(1).strip()
    else:
        nonempty = [line.strip() for line in cleaned.splitlines() if line.strip()]
        action = nonempty[-1].removeprefix("-").strip() if nonempty else "look"
    return ParsedThinkAction(thinking=thinking, action=action, raw_text=cleaned)


def select_aux_examples(
    examples: list[AuxLossExample],
    max_items: int,
) -> list[AuxLossExample]:
    if max_items <= 0 or len(examples) <= max_items:
        return examples
    if max_items == 1:
        return [examples[-1]]
    last = len(examples) - 1
    indexes = sorted({round(i * last / (max_items - 1)) for i in range(max_items)})
    return [examples[index] for index in indexes]


def observation_prediction_example(
    transition: Transition,
    *,
    max_words: int,
) -> tuple[str, str]:
    observation = truncate_words(transition.next_observation, max_words)
    prompt = (
        f"{transition_context(transition, include_thinking=True)}\n"
        f"{action_completion(transition.action)}\n"
        "NEXT_OBSERVATION:\n"
    )
    return prompt, observation.strip()


def score_prediction_example(
    transition: Transition,
    *,
    max_observation_words: int,
) -> tuple[str, str]:
    observation = truncate_words(transition.next_observation, max_observation_words)
    prompt = (
        f"{transition_context(transition, include_thinking=True)}\n"
        f"{action_completion(transition.action)}\n"
        f"OBSERVATION:\n{observation.strip()}\n"
        "PROGRESS:\n"
    )
    target = (
        f"valid: {str(transition.valid).lower()}\n"
        f"score_delta: {transition.score_delta:g}\n"
        f"score: {transition.score:g}\n"
        f"done: {str(transition.done).lower()}\n"
        f"won: {str(transition.won).lower()}"
    )
    return prompt, target


def build_structured_action_prompt(
    *,
    observation: str,
    valid_actions: list[str],
    memory: str,
    history: list[str],
    branch_hint: str,
    memory_enabled: bool,
    history_limit: int | None = 16,
) -> str:
    base_prompt = build_action_prompt(
        observation=observation,
        valid_actions=valid_actions,
        memory=memory,
        history=history,
        branch_hint=branch_hint,
        memory_enabled=memory_enabled,
        history_limit=history_limit,
    )
    return base_prompt.replace(
        "ACTION: <one valid action>",
        "THINK: <brief state and consequence reasoning>\nACTION: <one valid action>",
    )


def format_history_line(observation: str, thinking: str, action: str) -> str:
    if thinking.strip():
        return (
            f"OBSERVATION: {observation}\n"
            f"THINK: {thinking.strip()}\n"
            f"ACTION: {action}"
        )
    return f"OBSERVATION: {observation}\nACTION: {action}"


def transition_context(transition: Transition, *, include_thinking: bool) -> str:
    parts = [f"OBSERVATION:\n{transition.observation.strip()}"]
    if include_thinking and transition.thinking.strip():
        parts.append(f"THINK:\n{transition.thinking.strip()}")
    return "\n\n".join(parts)


def prefix_context(trace: EpisodeTrace, index: int) -> str:
    rows: list[str] = []
    for i, transition in enumerate(trace.transitions[: index + 1]):
        rows.append(f"OBS_{i}:\n{transition.observation.strip()}")
        if i < index:
            if transition.thinking.strip():
                rows.append(f"THINK_{i}:\n{transition.thinking.strip()}")
            rows.append(f"ACTION_{i}:\n{transition.action.strip()}")
    return "\n\n".join(rows)


def consequence_weight(trace: EpisodeTrace, transition: Transition) -> float:
    if not transition.valid:
        return 0.0
    if trace.solved or transition.won:
        return 1.0
    if transition.score_delta > 0:
        return 0.5
    if transition.reward > 0:
        return 0.5
    return 0.0


def action_good_example(
    trace: EpisodeTrace,
    index: int,
    transition: Transition,
) -> tuple[str, str]:
    prompt = (
        f"{prefix_context(trace, index)}\n\n"
        f"THINK_{index}:\n{transition.thinking.strip() or 'No additional reasoning.'}\n\n"
        f"ACTION_{index}:\n"
    )
    return prompt, transition.action.strip()


def thought_example(
    trace: EpisodeTrace,
    index: int,
    transition: Transition,
) -> tuple[str, str]:
    prompt = f"{prefix_context(trace, index)}\n\nTHINK_{index}:\n"
    return prompt, transition.thinking.strip()


def future_consistency_example(
    trace: EpisodeTrace,
    index: int,
    transition: Transition,
    *,
    horizon: int,
    max_words: int,
) -> tuple[str, str] | None:
    future_index = index + max(1, horizon)
    if future_index >= len(trace.transitions):
        return None
    future = trace.transitions[future_index].observation
    prompt = (
        f"{prefix_context(trace, index)}\n\n"
        f"THINK_{index}:\n{transition.thinking.strip() or 'No additional reasoning.'}\n\n"
        f"FUTURE_OBSERVATION_{future_index}:\n"
    )
    return prompt, truncate_words(future, max_words)


def compile_aux_loss_examples(
    traces: list[EpisodeTrace],
    *,
    echo_weight: float,
    score_weight: float,
    action_good_weight: float,
    thought_weight: float,
    future_weight: float,
    future_horizon: int,
    echo_max_words: int,
) -> list[AuxLossExample]:
    examples: list[AuxLossExample] = []
    for trace in traces:
        for index, transition in enumerate(trace.transitions):
            if echo_weight > 0 and transition.next_observation:
                prompt, completion = observation_prediction_example(
                    transition,
                    max_words=echo_max_words,
                )
                examples.append(
                    AuxLossExample("echo", prompt, completion, echo_weight)
                )
            if score_weight > 0:
                prompt, completion = score_prediction_example(
                    transition,
                    max_observation_words=echo_max_words,
                )
                examples.append(
                    AuxLossExample("score", prompt, completion, score_weight)
                )

            weight = consequence_weight(trace, transition)
            if action_good_weight > 0 and weight > 0:
                prompt, completion = action_good_example(trace, index, transition)
                examples.append(
                    AuxLossExample(
                        "action_good",
                        prompt,
                        completion,
                        action_good_weight * weight,
                    )
                )
            if thought_weight > 0 and weight > 0 and transition.thinking.strip():
                prompt, completion = thought_example(trace, index, transition)
                examples.append(
                    AuxLossExample(
                        "thought",
                        prompt,
                        completion,
                        thought_weight * weight,
                    )
                )
            if future_weight > 0 and weight > 0:
                future = future_consistency_example(
                    trace,
                    index,
                    transition,
                    horizon=future_horizon,
                    max_words=echo_max_words,
                )
                if future is not None:
                    prompt, completion = future
                    examples.append(
                        AuxLossExample(
                            "future",
                            prompt,
                            completion,
                            future_weight * weight,
                        )
                    )
    return examples


def node_outcome_key(node: dict[str, Any]) -> tuple[int, float, int]:
    solved = int(bool(node.get("done", False)))
    score = float(node.get("score", 0.0) or 0.0)
    path_len = len(node.get("path_actions", []) or [])
    return solved, score, -path_len


def parse_node_step_thinking(step: dict[str, Any]) -> str:
    raw = str(step.get("raw_response", "") or "")
    parsed = parse_think_action_response(raw)
    return parsed.thinking


def node_continuation(node: dict[str, Any]) -> str:
    parts: list[str] = []
    for step in node.get("steps", []) or []:
        thinking = parse_node_step_thinking(step)
        action = str(step.get("action", "") or "").strip()
        if thinking:
            parts.append(f"THINK: {thinking}")
        parts.append(f"ACTION: {action}")
    return "\n".join(parts).strip()


def branch_pair_prompt(node: dict[str, Any]) -> str:
    steps = node.get("steps", []) or []
    first = steps[0] if steps else {}
    memory = str(first.get("memory_before", "") or "").strip() or "empty"
    observation = str(first.get("observation", "") or "").strip()
    return (
        "BRANCH PREFIX\n"
        f"MEMORY:\n{memory}\n\n"
        f"OBSERVATION:\n{observation}\n\n"
        "CONTINUATION:\n"
    )


def load_branch_contrast_examples(
    path: Path,
    *,
    weight: float,
    max_pairs: int,
) -> list[BranchPreferenceExample]:
    if not path.exists():
        return []
    pairs: list[BranchPreferenceExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_parent: dict[str | None, list[dict[str, Any]]] = {}
            for node in row.get("nodes", []) or []:
                by_parent.setdefault(node.get("parent_id"), []).append(node)
            for siblings in by_parent.values():
                if len(siblings) < 2:
                    continue
                ordered = sorted(siblings, key=node_outcome_key)
                rejected = ordered[0]
                preferred = ordered[-1]
                if node_outcome_key(preferred) <= node_outcome_key(rejected):
                    continue
                preferred_completion = node_continuation(preferred)
                rejected_completion = node_continuation(rejected)
                if not preferred_completion or not rejected_completion:
                    continue
                pairs.append(
                    BranchPreferenceExample(
                        prompt=branch_pair_prompt(preferred),
                        preferred=preferred_completion,
                        rejected=rejected_completion,
                        weight=weight,
                    )
                )
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    if max_pairs == 1:
        return [pairs[-1]]
    last = len(pairs) - 1
    indexes = sorted({round(i * last / (max_pairs - 1)) for i in range(max_pairs)})
    return [pairs[index] for index in indexes]


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
    history_limit: int | None,
    structured_think_action: bool,
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
    memory_writes: list[MemoryWrite] = []
    memory = ""
    score = 0.0
    solved = False
    total_reward = 0.0

    for t in range(max_steps):
        valid_actions = env.valid_actions()
        memory_enabled = mode == "blimp"
        history = block_history if memory_enabled else full_history
        branch_hint = "Training rollout: choose the action most likely to complete the task."
        if structured_think_action:
            prompt = build_structured_action_prompt(
                observation=observation,
                valid_actions=valid_actions,
                memory=memory,
                history=history,
                branch_hint=branch_hint,
                memory_enabled=memory_enabled,
                history_limit=history_limit,
            )
            parsed = policy.generate_think_action(
                prompt,
                valid_actions,
                temperature=temperature,
                epsilon=epsilon,
                greedy=greedy,
                rng=rng,
                max_new_tokens=memory_max_tokens,
            )
            resolved_action = resolve_action(parsed.action, valid_actions)
            parsed_valid = resolved_action is not None
            action = resolved_action or (valid_actions[0] if valid_actions else "look")
            thinking = parsed.thinking
            raw_completion = parsed.raw_text
            policy_completion = raw_completion or structured_completion(thinking, action)
        else:
            prompt = build_action_prompt(
                observation=observation,
                valid_actions=valid_actions,
                memory=memory,
                history=history,
                branch_hint=branch_hint,
                memory_enabled=memory_enabled,
                history_limit=history_limit,
            )
            action, _ = policy.choose_action(
                prompt,
                valid_actions,
                temperature=temperature,
                epsilon=epsilon,
                greedy=greedy,
                rng=rng,
            )
            parsed_valid = True
            thinking = ""
            raw_completion = action_completion(action)
            policy_completion = action_completion(action)
        result = env.step(action)
        score = float(result.info.get("score", result.reward))
        reward = float(result.reward)
        step_success = transition_success(result.done, score, result.info)
        if step_success:
            reward += 1.0
        total_reward += reward
        transitions.append(
            Transition(
                prompt=prompt,
                action=action,
                reward=reward,
                done=result.done,
                score=score,
                observation=observation,
                thinking=thinking,
                raw_completion=raw_completion,
                policy_completion=policy_completion,
                next_observation=result.observation,
                score_delta=float(result.info.get("score_delta", result.reward)),
                won=step_success,
                valid=parsed_valid and bool(result.info.get("valid", True)),
            )
        )
        actions.append(action)
        history_line = format_history_line(observation, thinking, action)
        full_history.append(history_line)
        block_history.append(history_line)
        observation = result.observation
        solved = step_success
        if result.done or solved:
            break

        if memory_enabled and (t + 1) % block_len == 0 and (t + 1) < max_steps:
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
                memory_writes.append(
                    MemoryWrite(
                        prompt=note_prompt,
                        completion=memory,
                        after_step=len(transitions),
                    )
                )
            block_history = []

    return EpisodeTrace(
        episode_id=episode_id,
        mode=mode,
        game_file=game_file,
        solved=solved,
        score=score,
        total_reward=total_reward,
        actions=actions,
        memories=memories,
        transitions=transitions,
        memory_writes=memory_writes,
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


def find_game_files(game_file: str | None, game_dir: str | None) -> list[str]:
    if game_file:
        return [game_file]
    if not game_dir:
        return []
    root = Path(game_dir)
    suffixes = {".ulx", ".z8"}
    return sorted(str(path) for path in root.rglob("*") if path.suffix.lower() in suffixes)


def parse_scienceworld_tasks(raw: str) -> list[str]:
    return [task.strip() for task in raw.split(",") if task.strip()]


def build_scienceworld_specs(
    *,
    tasks: list[str],
    simplification: str,
    step_limit: int,
    train_examples: int,
    eval_examples: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    all_specs: list[str] = []
    for task in tasks:
        if task not in SCIENCEWORLD_TASK_VARIATIONS:
            raise ValueError(
                f"Unknown ScienceWorld task `{task}`. Add its variation count to "
                "SCIENCEWORLD_TASK_VARIATIONS before using it."
            )
        count = SCIENCEWORLD_TASK_VARIATIONS[task]
        all_specs.extend(
            f"{task}:{variation}:{simplification}:{step_limit}"
            for variation in range(count)
        )
    rng = random.Random(seed)
    rng.shuffle(all_specs)
    needed = train_examples + eval_examples
    if needed > len(all_specs):
        raise ValueError(
            f"Requested {needed} ScienceWorld specs but selected tasks provide "
            f"only {len(all_specs)} variations."
        )
    return all_specs[:train_examples], all_specs[train_examples:needed]


def select_game_file(game_files: list[str], index: int) -> str | None:
    if not game_files:
        return None
    return game_files[index % len(game_files)]


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging requested but wandb is not installed. "
            "Install `wandb` or omit --wandb-project."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
    )


def transition_success(done: bool, score: float, info: dict[str, Any]) -> bool:
    if "won" in info:
        return bool(info["won"])
    max_score = float(info.get("max_score", 0.0) or 0.0)
    if max_score > 0:
        return score >= max_score
    return done and score >= 1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online REINFORCE over valid text-environment actions."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--env",
        choices=["tiny", "hard", "recall", "textworld", "scienceworld"],
        default="hard",
    )
    parser.add_argument("--game-file", default=None)
    parser.add_argument("--game-dir", default=None)
    parser.add_argument("--eval-game-file", default=None)
    parser.add_argument("--eval-game-dir", default=None)
    parser.add_argument("--scienceworld-tasks", default=DEFAULT_SCIENCEWORLD_TASKS)
    parser.add_argument("--scienceworld-simplification", default="easy")
    parser.add_argument("--scienceworld-step-limit", type=int, default=100)
    parser.add_argument("--scienceworld-train-examples", type=int, default=2048)
    parser.add_argument("--scienceworld-eval-examples", type=int, default=256)
    parser.add_argument("--mode", choices=["standard", "blimp"], default="blimp")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="Optional smaller step cap for evaluation episodes.",
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--block-len", type=int, default=5)
    parser.add_argument("--memory-words", type=int, default=240)
    parser.add_argument("--memory-max-tokens", type=int, default=160)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--history-limit",
        type=int,
        default=16,
        help="Number of recent observation/action pairs in the action prompt. Use 0 for all.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--structured-think-action",
        action="store_true",
        help="Sample explicit THINK/ACTION completions instead of scoring only valid ACTION completions.",
    )
    parser.add_argument("--echo-weight", type=float, default=0.0)
    parser.add_argument("--score-weight", type=float, default=0.0)
    parser.add_argument("--action-good-weight", type=float, default=0.0)
    parser.add_argument("--thought-weight", type=float, default=0.0)
    parser.add_argument("--future-weight", type=float, default=0.0)
    parser.add_argument("--future-horizon", type=int, default=2)
    parser.add_argument("--branch-contrast-weight", type=float, default=0.0)
    parser.add_argument("--branch-contrast-jsonl", default=None)
    parser.add_argument(
        "--memory-policy-weight",
        type=float,
        default=0.0,
        help="Policy-gradient weight for block-boundary memory writes.",
    )
    parser.add_argument(
        "--aux-max-items",
        type=int,
        default=0,
        help="Maximum auxiliary examples or branch pairs per update. Use 0 for all.",
    )
    parser.add_argument(
        "--echo-max-words",
        type=int,
        default=220,
        help="Word cap for next-observation targets in auxiliary losses.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-episodes", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=0,
        help="Save full model checkpoints every N training updates. Use 0 to disable.",
    )
    parser.add_argument("--out", default="runs/reinforce-latest")
    args = parser.parse_args()

    train_game_files = find_game_files(args.game_file, args.game_dir)
    eval_game_files = find_game_files(args.eval_game_file, args.eval_game_dir) or train_game_files
    if args.env == "scienceworld":
        if args.game_file or args.game_dir:
            train_game_files = find_game_files(args.game_file, args.game_dir)
            eval_game_files = (
                find_game_files(args.eval_game_file, args.eval_game_dir)
                or train_game_files
            )
        else:
            train_game_files, eval_game_files = build_scienceworld_specs(
                tasks=parse_scienceworld_tasks(args.scienceworld_tasks),
                simplification=args.scienceworld_simplification,
                step_limit=args.scienceworld_step_limit,
                train_examples=args.scienceworld_train_examples,
                eval_examples=args.scienceworld_eval_examples,
                seed=args.seed,
            )
    if args.env == "textworld" and not train_game_files:
        raise ValueError("TextWorld training requires --game-file or --game-dir")
    if args.env == "textworld" and not eval_game_files:
        raise ValueError("TextWorld evaluation requires --eval-game-file/--eval-game-dir or train games")
    if args.env == "scienceworld" and not train_game_files:
        raise ValueError("ScienceWorld training requires generated or file-based specs")
    if args.env == "scienceworld" and not eval_game_files:
        raise ValueError("ScienceWorld evaluation requires generated or file-based specs")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["train_game_files"] = train_game_files
    config["eval_game_files"] = eval_game_files
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    wandb_run = maybe_init_wandb(args, config)

    rng = random.Random(args.seed)
    policy = ValidActionPolicy(
        args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        score_batch_size=args.score_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    started = time.time()
    global_episode = 0
    for update in range(args.updates + 1):
        should_eval = update % args.eval_every == 0 and not (
            update == 0 and args.skip_initial_eval
        )
        if should_eval:
            eval_traces = []
            for i in range(args.eval_episodes):
                trace = run_episode(
                    policy=policy,
                    episode_id=10_000 + update * 100 + i,
                    env_name=args.env,
                    game_file=select_game_file(
                        eval_game_files,
                        update * max(args.eval_episodes, 1) + i,
                    ),
                    seed=args.seed,
                    mode=args.mode,
                    max_steps=args.eval_max_steps or args.max_steps,
                    block_len=args.block_len,
                    memory_words=args.memory_words,
                    memory_max_tokens=args.memory_max_tokens,
                    history_limit=args.history_limit,
                    structured_think_action=args.structured_think_action,
                    temperature=1.0,
                    epsilon=0.0,
                    greedy=True,
                    rng=rng,
                )
                eval_traces.append(trace)
                if args.log_episodes:
                    print(
                        json.dumps(
                            {
                                "phase": "eval_episode",
                                "update": update,
                                "episode": i,
                                "solved": trace.solved,
                                "score": trace.score,
                                "total_reward": trace.total_reward,
                                "steps": len(trace.actions),
                                "last_action": trace.actions[-1]
                                if trace.actions
                                else None,
                            }
                        ),
                        flush=True,
                    )
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
            if wandb_run is not None:
                wandb_run.log(row, step=update)
            print(json.dumps(row), flush=True)

        if update == args.updates:
            break

        train_traces = []
        for i in range(args.episodes_per_update):
            trace = run_episode(
                policy=policy,
                episode_id=global_episode + i,
                env_name=args.env,
                game_file=select_game_file(train_game_files, global_episode + i),
                seed=args.seed,
                mode=args.mode,
                max_steps=args.max_steps,
                block_len=args.block_len,
                memory_words=args.memory_words,
                memory_max_tokens=args.memory_max_tokens,
                history_limit=args.history_limit,
                structured_think_action=args.structured_think_action,
                temperature=args.temperature,
                epsilon=args.epsilon,
                greedy=False,
                rng=rng,
            )
            train_traces.append(trace)
            if args.log_episodes:
                print(
                    json.dumps(
                        {
                            "phase": "train_episode",
                            "update": update + 1,
                            "episode": global_episode + i,
                            "solved": trace.solved,
                            "score": trace.score,
                            "total_reward": trace.total_reward,
                            "steps": len(trace.actions),
                            "last_action": trace.actions[-1]
                            if trace.actions
                            else None,
                        }
                    ),
                    flush=True,
                )
        global_episode += args.episodes_per_update
        update_stats = policy.update(
            train_traces,
            gamma=args.gamma,
            normalize_advantages=not args.no_normalize_advantages,
            grad_clip=args.grad_clip,
            echo_weight=args.echo_weight,
            score_weight=args.score_weight,
            action_good_weight=args.action_good_weight,
            thought_weight=args.thought_weight,
            future_weight=args.future_weight,
            future_horizon=args.future_horizon,
            branch_contrast_weight=args.branch_contrast_weight,
            branch_contrast_jsonl=args.branch_contrast_jsonl,
            memory_policy_weight=args.memory_policy_weight,
            aux_max_items=args.aux_max_items,
            echo_max_words=args.echo_max_words,
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
        if wandb_run is not None:
            wandb_run.log(row, step=update + 1)
        print(json.dumps(row), flush=True)
        if (
            not args.no_save_model
            and args.save_every_updates > 0
            and (update + 1) % args.save_every_updates == 0
        ):
            checkpoint_dir = out_dir / "checkpoints" / f"update_{update + 1:04d}"
            policy.save(checkpoint_dir)

    if not args.no_save_model:
        policy.save(out_dir / "adapter")
    summary_rows = [
        json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()
    ]
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows[-10:], handle, indent=2)
        handle.write("\n")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
