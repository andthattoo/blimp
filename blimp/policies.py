from __future__ import annotations

import random
import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Protocol


MEMORY_FIELDS = [
    "GOAL",
    "LOCATION",
    "INVENTORY",
    "MAP",
    "OBJECTS",
    "LOCKS",
    "TODO",
    "FAILED",
]

PROMPT_BOUNDARY_FIELDS = {
    "ACTION",
    "OBSERVATION",
    "VALID ACTIONS",
    "MEMORY_PATCH",
    "BRANCH HINT",
    "CURRENT MEMORY",
    "RECENT ACTIONS",
    "OUTPUT EXACTLY",
}

DEFAULT_MEMORY = {
    "GOAL": "complete the environment objective",
    "LOCATION": "unknown",
    "INVENTORY": "empty",
    "MAP": "unknown",
    "OBJECTS": "unknown",
    "LOCKS": "unknown",
    "TODO": "explore, find useful items, unlock blocked paths, complete the objective",
    "FAILED": "none",
}


@dataclass
class PolicyOutput:
    action: str
    memory: str
    raw_text: str = ""


class Policy(Protocol):
    def act(
        self,
        observation: str,
        memory: str,
        valid_actions: list[str],
        *,
        history: list[str],
        branch_hint: str,
        rng: random.Random,
    ) -> PolicyOutput:
        ...


class RandomValidPolicy:
    def act(
        self,
        observation: str,
        memory: str,
        valid_actions: list[str],
        *,
        history: list[str],
        branch_hint: str,
        rng: random.Random,
    ) -> PolicyOutput:
        del observation, history, branch_hint
        action = rng.choice(valid_actions) if valid_actions else "look"
        return PolicyOutput(action=action, memory=normalize_memory(memory), raw_text="random-valid")


class ScriptedTinyPolicy:
    """Deterministic policy for validating rollout mechanics on TinyQuestEnv."""

    def act(
        self,
        observation: str,
        memory: str,
        valid_actions: list[str],
        *,
        history: list[str],
        branch_hint: str,
        rng: random.Random,
    ) -> PolicyOutput:
        del history, branch_hint, rng
        text = observation.lower()
        action = "look"

        if "blue gem rests" in text or "visible items: blue gem" in text:
            action = "take blue gem"
        elif "north door is unlocked" in text and "go north" in valid_actions:
            action = "go north"
        elif "north door is locked" in text and "brass key" in text:
            action = "unlock door with brass key"
        elif "you are in a long hall" in text:
            action = "go west" if "brass key" not in text else "unlock door with brass key"
        elif "you are in a cold cellar" in text and "visible items: brass key" in text:
            action = "take brass key"
        elif "you are in a cold cellar" in text:
            action = "go north"
        elif "you are in the foyer" in text and "brass key" in text:
            action = "go east"
        elif "you are in the foyer" in text:
            action = "go south"

        action = resolve_action(action, valid_actions) or action
        new_memory = summarize_tiny_memory(observation, memory)
        return PolicyOutput(action=action, memory=new_memory, raw_text="scripted-tiny")


class ScriptedHardPolicy:
    """Deterministic policy for validating HardQuestEnv mechanics."""

    def act(
        self,
        observation: str,
        memory: str,
        valid_actions: list[str],
        *,
        history: list[str],
        branch_hint: str,
        rng: random.Random,
    ) -> PolicyOutput:
        del history, branch_hint, rng
        obs = observation.lower()
        text = (memory + "\n" + observation).lower()
        inventory = _inventory_items(text)
        action = "look"

        priority_actions = [
            "take amber seal",
            "say ember",
            "unlock vault with moon key",
            "take moon key",
            "light storm lantern",
            "unlock iron gate with iron key",
            "take iron key",
            "take storm lantern",
        ]
        if "passphrase ember" not in text:
            priority_actions.append("read brass plaque")
        for candidate in priority_actions:
            if candidate in valid_actions:
                action = candidate
                break
        else:
            if "you are in the archive" in obs:
                action = "go south"
            elif "you are in the pantry" in obs:
                action = "go east"
            elif "you are in the armory" in obs:
                action = "go north"
            elif "you stand in the vault antechamber" in obs:
                action = "go east" if "go east" in valid_actions else "go west"
            elif "you are in the final vault" in obs:
                action = "go west"
            elif "you are in the crypt" in obs:
                if "moon key" in inventory and "go east" in valid_actions:
                    action = "go east"
                elif "go west" in valid_actions:
                    action = "go west"
            elif "you are in the gallery" in obs:
                if "iron key" in inventory and "go east" in valid_actions:
                    action = "go east"
                elif "iron key" in inventory and "unlock iron gate" in valid_actions:
                    action = "unlock iron gate"
                else:
                    action = "go south"
            elif "you are in the atrium" in obs:
                if "passphrase ember" not in text:
                    action = "go north"
                elif "storm lantern" not in inventory:
                    action = "go west"
                else:
                    action = "go east"

        action = resolve_action(action, valid_actions) or action
        new_memory = summarize_hard_memory(observation, memory)
        return PolicyOutput(action=action, memory=new_memory, raw_text="scripted-hard")


class HFPolicy:
    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.7,
        max_new_tokens: int = 96,
        trust_remote_code: bool = False,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFPolicy requires transformers and torch. Install with "
                "`pip install -r requirements-gpu.txt` on the GPU host."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def act(
        self,
        observation: str,
        memory: str,
        valid_actions: list[str],
        *,
        history: list[str],
        branch_hint: str,
        rng: random.Random,
    ) -> PolicyOutput:
        del rng
        prompt = build_prompt(observation, memory, valid_actions, history, branch_hint)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            inputs = inputs.to(self.model.device)
        except AttributeError:
            pass

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        parsed_action, parsed_memory = parse_action_memory(generated)
        action = resolve_action(parsed_action, valid_actions)
        if action is None:
            action = valid_actions[0] if valid_actions else parsed_action or "look"
        merged_memory = merge_memory_patch(memory, parsed_memory)
        return PolicyOutput(
            action=action,
            memory=merged_memory,
            raw_text=generated.strip(),
        )


def build_prompt(
    observation: str,
    memory: str,
    valid_actions: list[str],
    history: list[str],
    branch_hint: str,
) -> str:
    actions = "\n".join(f"- {action}" for action in valid_actions) or "- look"
    recent = "\n".join(history[-8:]) if history else "None."
    memory_text = normalize_memory(memory)
    return f"""You are an agent in a text environment.
Complete the environment objective as directly as possible.
Choose exactly one valid action and update structured memory for future short blocks.
If a goal-relevant item is visible and a matching take action is valid, take it.
Do not invent rooms, objects, exits, or actions that are not in the observation.
Use memory for durable offscreen facts only. If a fact is unknown, write unknown.

Output exactly:
ACTION: <one valid action>
MEMORY_PATCH:
GOAL: <current objective>
LOCATION: <current known location>
INVENTORY: <items held, or empty>
MAP: <known room/exit edges>
OBJECTS: <known object locations>
LOCKS: <locked/unlocked doors and required keys>
TODO: <next useful subgoals>
FAILED: <failed actions or loops to avoid>

Branch hint: {branch_hint}

Current memory:
{memory_text}

Recent actions:
{recent}

Observation:
{observation}

Valid actions:
{actions}
"""


def parse_action_memory(text: str) -> tuple[str, str]:
    action = ""
    memory = ""

    action_match = re.search(r"(?im)^\s*ACTION\s*:\s*(.+)$", text)
    if action_match:
        action = action_match.group(1).strip()
    else:
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        action = first.removeprefix("-").strip()

    memory_match = re.search(r"(?ims)^\s*MEMORY(?:_PATCH)?\s*:\s*(.+)$", text)
    if memory_match:
        memory = memory_match.group(1).strip()
        memory = re.split(
            r"(?im)^\s*(ACTION|OBSERVATION|VALID ACTIONS|BRANCH HINT|CURRENT MEMORY|RECENT ACTIONS|OUTPUT EXACTLY)\s*:",
            memory,
        )[0]
        memory = memory.strip()

    return action, memory


def resolve_action(action: str, valid_actions: list[str]) -> str | None:
    if not valid_actions:
        return action.strip() if action.strip() else None
    candidate = " ".join(action.lower().strip().split())
    normalized = {" ".join(valid.lower().split()): valid for valid in valid_actions}
    if candidate in normalized:
        return normalized[candidate]

    for norm, valid in normalized.items():
        if norm in candidate or candidate in norm:
            return valid

    matches = get_close_matches(candidate, list(normalized), n=1, cutoff=0.55)
    if matches:
        return normalized[matches[0]]
    return None


def truncate_memory(memory: str, max_words: int) -> str:
    words = memory.split()
    if max_words <= 0 or len(words) <= max_words:
        return memory.strip()
    return " ".join(words[-max_words:]).strip()


def corrupt_memory(memory: str, rng: random.Random) -> str:
    parsed = parse_memory(memory)
    distractor_locations = ["attic", "garden", "library", "river"]
    parsed["LOCATION"] = rng.choice(distractor_locations)
    parsed["INVENTORY"] = rng.choice(["empty", "silver coin", "red key"])
    parsed["MAP"] = rng.choice(
        [
            "attic -> garden; garden -> river",
            "library -> tower; tower -> cellar",
            "unknown",
        ]
    )
    parsed["OBJECTS"] = rng.choice(
        [
            "blue gem in garden",
            "brass key in attic",
            "amber seal in observatory; moon key in pantry",
            "blue gem unknown; brass key unknown",
        ]
    )
    parsed["LOCKS"] = rng.choice(
        [
            "south door locked by red key",
            "north door already open",
            "final vault passphrase mirror; iron gate already open",
            "unknown",
        ]
    )
    parsed["TODO"] = rng.choice(
        [
            "avoid the cellar; go east, unlock the north door, then take the blue gem",
            "go east repeatedly until a garden or vault appears",
            "use the red key on the south door, then search the garden",
            "skip the archive; say mirror in the final vault",
        ]
    )
    parsed["FAILED"] = rng.choice(
        [
            "going south wastes time",
            "the brass key was already collected",
            "the north door does not need the brass key",
        ]
    )
    return format_memory(parsed)


def summarize_tiny_memory(observation: str, memory: str) -> str:
    parsed = parse_memory(memory)
    text = (memory + "\n" + observation).lower()
    parsed["GOAL"] = "recover the blue gem"
    if "you are in the foyer" in text:
        parsed["LOCATION"] = "foyer"
    elif "you are in a cold cellar" in text:
        parsed["LOCATION"] = "cellar"
    elif "you are in a long hall" in text:
        parsed["LOCATION"] = "hall"
    elif "you are in the vault" in text:
        parsed["LOCATION"] = "vault"
    if "brass key" in text:
        if "inventory: brass key" in text or "you take the brass key" in text:
            parsed["INVENTORY"] = "brass key"
        else:
            parsed["OBJECTS"] = merge_fact(parsed["OBJECTS"], "brass key in cellar")
    if "north door is unlocked" in text or "you unlock the north door" in text:
        parsed["LOCKS"] = merge_fact(parsed["LOCKS"], "hall north door unlocked")
    elif "north door is locked" in text:
        parsed["LOCKS"] = merge_fact(parsed["LOCKS"], "hall north door locked; needs brass key")
    if "blue gem" in text:
        parsed["OBJECTS"] = merge_fact(parsed["OBJECTS"], "blue gem in vault")
    parsed["MAP"] = merge_fact(parsed["MAP"], "foyer east hall; foyer south cellar; hall north vault")
    parsed["TODO"] = "get brass key, unlock hall north door, go north to vault, take blue gem"
    return format_memory(parsed)


def summarize_hard_memory(observation: str, memory: str) -> str:
    parsed = parse_memory(memory)
    text = (memory + "\n" + observation).lower()
    parsed["GOAL"] = "recover the amber seal"

    room_markers = [
        ("you are in the atrium", "atrium"),
        ("you are in the archive", "archive"),
        ("you are in the pantry", "pantry"),
        ("you are in the gallery", "gallery"),
        ("you are in the armory", "armory"),
        ("you are in the crypt", "crypt"),
        ("you are in the observatory", "observatory"),
        ("you stand in the vault antechamber", "vault antechamber"),
        ("you are in the final vault", "final vault"),
    ]
    for marker, location in room_markers:
        if marker in observation.lower():
            parsed["LOCATION"] = location
            break

    inventory = _inventory_items(observation.lower())
    if inventory:
        parsed["INVENTORY"] = ", ".join(inventory)

    object_facts = [
        ("storm lantern", "storm lantern in pantry"),
        ("iron key", "iron key in armory"),
        ("moon key", "moon key on crypt altar"),
        ("amber seal", "amber seal in final vault"),
    ]
    for marker, fact in object_facts:
        if marker in text:
            parsed["OBJECTS"] = merge_fact(parsed["OBJECTS"], fact)

    parsed["MAP"] = (
        "atrium north archive; atrium west pantry; atrium east gallery; "
        "gallery south armory; gallery east crypt; crypt east vault antechamber; "
        "vault antechamber east final vault"
    )
    if "passphrase ember" in text or "ember opens" in text:
        parsed["LOCKS"] = merge_fact(parsed["LOCKS"], "final vault passphrase ember")
    if "iron gate is locked" in text:
        parsed["LOCKS"] = merge_fact(parsed["LOCKS"], "iron gate needs iron key")
    if "iron gate is unlocked" in text or "unlock the iron gate" in text:
        parsed["LOCKS"] = merge_fact(parsed["LOCKS"], "iron gate unlocked")
    if "moon lock" in text:
        parsed["LOCKS"] = merge_fact(parsed["LOCKS"], "moon lock needs moon key")
    if "unlock the vault door" in text or "moon lock is open" in text:
        parsed["LOCKS"] = merge_fact(parsed["LOCKS"], "moon lock open")

    inv_text = parsed["INVENTORY"].lower()
    if "passphrase ember" not in parsed["LOCKS"].lower():
        parsed["TODO"] = "go to archive and read brass plaque"
    elif "storm lantern" not in inv_text:
        parsed["TODO"] = "go to pantry and take storm lantern"
    elif "iron key" not in inv_text:
        parsed["TODO"] = "go to armory and take iron key"
    elif "iron gate unlocked" not in parsed["LOCKS"].lower():
        parsed["TODO"] = "go to gallery and unlock iron gate with iron key"
    elif "moon key" not in inv_text:
        parsed["TODO"] = "go to crypt, light storm lantern, take moon key"
    elif "moon lock open" not in parsed["LOCKS"].lower():
        parsed["TODO"] = "go to vault antechamber and unlock vault with moon key"
    elif "amber seal" not in inv_text:
        parsed["TODO"] = "go east to final vault, say ember, take amber seal"
    else:
        parsed["TODO"] = "objective complete"

    return format_memory(parsed)


def _inventory_items(text: str) -> list[str]:
    matches = re.findall(r"inventory:\s*([^.\n]+)", text)
    if not matches:
        return []
    last = matches[-1].strip()
    if not last or last == "empty":
        return []
    return [item.strip() for item in last.split(",") if item.strip()]


def parse_memory(memory: str) -> dict[str, str]:
    parsed = dict(DEFAULT_MEMORY)
    if not memory.strip():
        return parsed

    current_field: str | None = None
    current_lines: list[str] = []
    for raw_line in memory.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", line)
        if match and match.group(1).upper() in MEMORY_FIELDS:
            if current_field is not None:
                parsed[current_field] = clean_memory_value(" ".join(current_lines))
            current_field = match.group(1).upper()
            current_lines = [match.group(2).strip()]
        elif _is_prompt_boundary(line):
            break
        elif current_field is not None:
            current_lines.append(line)

    if current_field is not None:
        parsed[current_field] = clean_memory_value(" ".join(current_lines))
    elif memory.strip():
        parsed["TODO"] = clean_memory_value(memory.strip())

    return parsed


def format_memory(memory: dict[str, str]) -> str:
    lines = []
    for field in MEMORY_FIELDS:
        value = clean_memory_value(memory.get(field, DEFAULT_MEMORY[field]))
        lines.append(f"{field}: {value}")
    return "\n".join(lines)


def normalize_memory(memory: str) -> str:
    return format_memory(parse_memory(memory))


def merge_memory_patch(base: str, patch: str) -> str:
    parsed = parse_memory(base)
    if not patch.strip():
        return format_memory(parsed)
    for field, value in parse_memory_patch_fields(patch).items():
        if value and value.lower() not in {"unknown", "none", "n/a"}:
            parsed[field] = clean_memory_value(value)
    return format_memory(parsed)


def parse_memory_patch_fields(patch: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    current_field: str | None = None
    current_lines: list[str] = []
    for raw_line in patch.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", line)
        if match and match.group(1).upper() in MEMORY_FIELDS:
            if current_field is not None:
                parsed[current_field] = clean_memory_value(" ".join(current_lines))
            current_field = match.group(1).upper()
            current_lines = [match.group(2).strip()]
        elif _is_prompt_boundary(line):
            break
        elif current_field is not None:
            current_lines.append(line)
    if current_field is not None:
        parsed[current_field] = clean_memory_value(" ".join(current_lines))
    return parsed


def merge_fact(existing: str, fact: str) -> str:
    if not existing or existing.lower() in {"unknown", "none"}:
        return fact
    parts = [part.strip() for part in re.split(r"[;\n]", existing) if part.strip()]
    if fact not in parts:
        parts.append(fact)
    return "; ".join(parts)


def clean_memory_value(value: str) -> str:
    value = " ".join(value.split())
    value = re.sub(
        r"(?i)\b(ACTION|OBSERVATION|VALID ACTIONS|MEMORY_PATCH|BRANCH HINT|CURRENT MEMORY|RECENT ACTIONS|OUTPUT EXACTLY)\s*:.*$",
        "",
        value,
    )
    return value.strip(" -") or "unknown"


def _is_prompt_boundary(line: str) -> bool:
    match = re.match(r"^([A-Za-z_ ]+)\s*:", line)
    if not match:
        return False
    return match.group(1).replace("_", " ").upper() in PROMPT_BOUNDARY_FIELDS


def make_policy(
    policy_name: str,
    *,
    model_name: str | None = None,
    temperature: float = 0.7,
    max_new_tokens: int = 96,
    trust_remote_code: bool = False,
    device_map: str = "auto",
) -> Policy:
    if policy_name == "random":
        return RandomValidPolicy()
    if policy_name == "scripted-tiny":
        return ScriptedTinyPolicy()
    if policy_name == "scripted-hard":
        return ScriptedHardPolicy()
    if policy_name == "hf":
        if not model_name:
            raise ValueError("--model is required with --policy hf")
        return HFPolicy(
            model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    raise ValueError(f"Unknown policy: {policy_name}")
