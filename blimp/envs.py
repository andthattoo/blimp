from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class TextEnv(Protocol):
    """Minimal text environment interface used by the rollout code."""

    def reset(self, seed: int | None = None) -> str:
        ...

    def step(self, action: str) -> StepResult:
        ...

    def valid_actions(self) -> list[str]:
        ...


class TinyQuestEnv:
    """Small deterministic text-world used for smoke tests.

    The task is intentionally simple but memory-sensitive under weak policies:
    find the brass key in the cellar, unlock the north door in the hall, and
    take the gem from the vault.
    """

    def __init__(self) -> None:
        self.rooms = {
            "foyer": {
                "desc": "You are in the foyer. Passages lead east and south.",
                "exits": {"east": "hall", "south": "cellar"},
                "items": [],
            },
            "cellar": {
                "desc": "You are in a cold cellar. The brass key is on a hook.",
                "exits": {"north": "foyer"},
                "items": ["brass key"],
            },
            "hall": {
                "desc": "You are in a long hall. A locked door blocks the north exit.",
                "exits": {"west": "foyer", "north": "vault"},
                "items": [],
            },
            "vault": {
                "desc": "You are in the vault. A blue gem rests on a pedestal.",
                "exits": {"south": "hall"},
                "items": ["blue gem"],
            },
        }
        self.reset()

    def reset(self, seed: int | None = None) -> str:
        del seed
        self.location = "foyer"
        self.inventory: list[str] = []
        self.door_unlocked = False
        self.done = False
        self.steps = 0
        self.score = 0.0
        self._room_items = {
            room: list(data["items"]) for room, data in self.rooms.items()
        }
        return self._observation("New episode.")

    def clone(self) -> "TinyQuestEnv":
        return deepcopy(self)

    def valid_actions(self) -> list[str]:
        if self.done:
            return []

        actions = ["look", "inventory"]
        exits = self.rooms[self.location]["exits"]
        for direction in exits:
            if self.location == "hall" and direction == "north" and not self.door_unlocked:
                actions.append("unlock door")
            else:
                actions.append(f"go {direction}")

        for item in self._room_items[self.location]:
            actions.append(f"take {item}")

        if "brass key" in self.inventory and self.location == "hall" and not self.door_unlocked:
            actions.append("unlock door with brass key")

        return actions

    def step(self, action: str) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._observation("The task is already complete."),
                reward=0.0,
                done=True,
                info={"score": self.score, "valid": False},
            )

        self.steps += 1
        normalized = " ".join(action.lower().strip().split())
        valid = normalized in self.valid_actions()
        reward = -0.01

        if normalized in {"look", ""}:
            message = "You look around."
        elif normalized == "inventory":
            if self.inventory:
                message = "Inventory: " + ", ".join(self.inventory) + "."
            else:
                message = "Inventory: empty."
        elif normalized.startswith("go "):
            message, reward = self._go(normalized[3:])
        elif normalized.startswith("take "):
            message, reward = self._take(normalized[5:])
        elif normalized in {"unlock door", "unlock door with brass key"}:
            message, reward = self._unlock()
        else:
            message = f"'{action}' is not a useful action here."
            reward = -0.05

        if "blue gem" in self.inventory:
            self.done = True
            self.score = 1.0
            reward = 1.0
            message += " You have recovered the blue gem."

        return StepResult(
            observation=self._observation(message),
            reward=reward,
            done=self.done,
            info={
                "score": self.score,
                "valid": valid,
                "steps": self.steps,
                "location": self.location,
                "inventory": list(self.inventory),
            },
        )

    def _go(self, direction: str) -> tuple[str, float]:
        exits = self.rooms[self.location]["exits"]
        if direction not in exits:
            return f"You cannot go {direction} from here.", -0.05
        if self.location == "hall" and direction == "north" and not self.door_unlocked:
            return "The north door is locked.", -0.05
        self.location = exits[direction]
        return f"You go {direction}.", 0.0

    def _take(self, item: str) -> tuple[str, float]:
        if item not in self._room_items[self.location]:
            return f"There is no {item} here.", -0.05
        self._room_items[self.location].remove(item)
        self.inventory.append(item)
        return f"You take the {item}.", 0.1

    def _unlock(self) -> tuple[str, float]:
        if self.location != "hall":
            return "There is no locked door here.", -0.05
        if "brass key" not in self.inventory:
            return "You need a key for the locked door.", -0.05
        self.door_unlocked = True
        return "You unlock the north door.", 0.2

    def _observation(self, message: str) -> str:
        room = self.rooms[self.location]
        items = self._room_items[self.location]
        item_text = " Visible items: " + ", ".join(items) + "." if items else ""
        inventory = " Inventory: " + ", ".join(self.inventory) + "." if self.inventory else ""
        door = ""
        if self.location == "hall":
            door = " The north door is unlocked." if self.door_unlocked else " The north door is locked."
        return (
            "Goal: recover the blue gem.\n"
            f"{message}\n{room['desc']}{door}{item_text}{inventory}"
        )


class HardQuestEnv:
    """Longer deterministic quest with durable clue and key dependencies.

    The optimal route crosses several five-step block boundaries. The agent
    must remember the passphrase learned in the archive, carry two tools, and
    avoid treating local observations as the whole state.
    """

    PASS_OPTIONS = ["ember", "mirror", "tide", "brass"]

    def __init__(self) -> None:
        self.rooms = {
            "atrium": {
                "desc": "You are in the atrium. Exits lead north, east, and west.",
                "exits": {"north": "archive", "east": "gallery", "west": "pantry"},
            },
            "archive": {
                "desc": (
                    "You are in the archive. A brass plaque is fixed to the wall."
                ),
                "exits": {"south": "atrium"},
            },
            "pantry": {
                "desc": "You are in the pantry. A storm lantern hangs from a peg.",
                "exits": {"east": "atrium"},
            },
            "gallery": {
                "desc": (
                    "You are in the gallery. Corridors lead west and south. "
                    "An iron gate blocks the east passage."
                ),
                "exits": {"west": "atrium", "south": "armory", "east": "crypt"},
            },
            "armory": {
                "desc": "You are in the armory. An iron key lies in an open drawer.",
                "exits": {"north": "gallery"},
            },
            "crypt": {
                "desc": (
                    "You are in the crypt beyond the iron gate. The altar is hard "
                    "to inspect without a lit lantern."
                ),
                "exits": {"west": "gallery", "north": "observatory", "east": "vault_ante"},
            },
            "observatory": {
                "desc": (
                    "You are in the observatory. A star map says the final vault "
                    "lies east of the crypt."
                ),
                "exits": {"south": "crypt"},
            },
            "vault_ante": {
                "desc": "You stand in the vault antechamber. A moon lock guards the east door.",
                "exits": {"west": "crypt", "east": "vault"},
            },
            "vault": {
                "desc": (
                    "You are in the final vault. A speaking lock protects a glass case."
                ),
                "exits": {"west": "vault_ante"},
            },
        }
        self.reset()

    def reset(self, seed: int | None = None) -> str:
        del seed
        self.location = "atrium"
        self.inventory: list[str] = []
        self.iron_gate_unlocked = False
        self.lantern_lit = False
        self.vault_unlocked = False
        self.passphrase_spoken = False
        self.plaque_read = False
        self.done = False
        self.steps = 0
        self.score = 0.0
        self._room_items = {
            "pantry": ["storm lantern"],
            "armory": ["iron key"],
            "crypt": ["moon key"],
            "vault": ["amber seal"],
        }
        return self._observation("New episode.")

    def valid_actions(self) -> list[str]:
        if self.done:
            return []

        actions = ["look", "inventory"]
        for direction in self.rooms[self.location]["exits"]:
            if self.location == "gallery" and direction == "east" and not self.iron_gate_unlocked:
                actions.append("unlock iron gate")
            elif self.location == "vault_ante" and direction == "east" and not self.vault_unlocked:
                actions.append("inspect moon lock")
            else:
                actions.append(f"go {direction}")

        if self.location == "archive":
            actions.append("read brass plaque")
        if (
            self.location == "gallery"
            and "iron key" in self.inventory
            and not self.iron_gate_unlocked
        ):
            actions.append("unlock iron gate with iron key")
        if "storm lantern" in self.inventory and not self.lantern_lit:
            actions.append("light storm lantern")
        if (
            self.location == "vault_ante"
            and "moon key" in self.inventory
            and not self.vault_unlocked
        ):
            actions.append("unlock vault with moon key")
        if self.location == "vault" and not self.passphrase_spoken:
            actions.extend(f"say {word}" for word in self.PASS_OPTIONS)

        for item in self._visible_items():
            actions.append(f"take {item}")

        return actions

    def step(self, action: str) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._observation("The task is already complete."),
                reward=0.0,
                done=True,
                info={"score": self.score, "valid": False},
            )

        self.steps += 1
        normalized = " ".join(action.lower().strip().split())
        valid = normalized in self.valid_actions()
        reward = -0.01

        if normalized in {"look", ""}:
            message = "You look around."
        elif normalized == "inventory":
            message = "Inventory: " + (", ".join(self.inventory) if self.inventory else "empty") + "."
        elif normalized == "read brass plaque":
            reward = 0.05 if not self.plaque_read else -0.01
            self.plaque_read = True
            message = "The brass plaque says: passphrase EMBER opens the final vault."
        elif normalized == "light storm lantern":
            if "storm lantern" not in self.inventory:
                message = "You need the storm lantern first."
                reward = -0.05
            else:
                self.lantern_lit = True
                message = "You light the storm lantern."
                reward = 0.05
        elif normalized.startswith("go "):
            message, reward = self._go(normalized[3:])
        elif normalized.startswith("take "):
            message, reward = self._take(normalized[5:])
        elif normalized in {"unlock iron gate", "unlock iron gate with iron key"}:
            message, reward = self._unlock_iron_gate()
        elif normalized in {"inspect moon lock", "unlock vault with moon key"}:
            message, reward = self._unlock_vault(normalized)
        elif normalized.startswith("say "):
            message, reward = self._say(normalized[4:])
        else:
            message = f"'{action}' is not a useful action here."
            reward = -0.05

        if "amber seal" in self.inventory:
            self.done = True
            self.score = 1.0
            reward = 1.0
            message += " You have recovered the amber seal."

        return StepResult(
            observation=self._observation(message),
            reward=reward,
            done=self.done,
            info={
                "score": self.score,
                "valid": valid,
                "steps": self.steps,
                "location": self.location,
                "inventory": list(self.inventory),
            },
        )

    def _go(self, direction: str) -> tuple[str, float]:
        exits = self.rooms[self.location]["exits"]
        if direction not in exits:
            return f"You cannot go {direction} from here.", -0.05
        if self.location == "gallery" and direction == "east" and not self.iron_gate_unlocked:
            return "The iron gate is locked.", -0.05
        if self.location == "vault_ante" and direction == "east" and not self.vault_unlocked:
            return "The moon lock holds the east door shut.", -0.05
        self.location = exits[direction]
        return f"You go {direction}.", 0.0

    def _take(self, item: str) -> tuple[str, float]:
        if item not in self._visible_items():
            return f"There is no reachable {item} here.", -0.05
        self._room_items[self.location].remove(item)
        self.inventory.append(item)
        return f"You take the {item}.", 0.1

    def _unlock_iron_gate(self) -> tuple[str, float]:
        if self.location != "gallery":
            return "There is no iron gate here.", -0.05
        if "iron key" not in self.inventory:
            return "The iron gate needs an iron key.", -0.05
        self.iron_gate_unlocked = True
        return "You unlock the iron gate with the iron key.", 0.15

    def _unlock_vault(self, action: str) -> tuple[str, float]:
        if self.location != "vault_ante":
            return "There is no moon lock here.", -0.05
        if action != "unlock vault with moon key":
            return "The moon lock is shaped for a moon key.", -0.02
        if "moon key" not in self.inventory:
            return "You need the moon key for the vault.", -0.05
        self.vault_unlocked = True
        return "You unlock the vault door with the moon key.", 0.2

    def _say(self, word: str) -> tuple[str, float]:
        if self.location != "vault":
            return "No lock is listening here.", -0.05
        if word == "ember":
            self.passphrase_spoken = True
            return "The speaking lock accepts EMBER and opens the glass case.", 0.2
        return "The speaking lock rejects the passphrase.", -0.05

    def _visible_items(self) -> list[str]:
        items = list(self._room_items.get(self.location, []))
        if self.location == "crypt" and not self.lantern_lit:
            items = [item for item in items if item != "moon key"]
        if self.location == "vault" and not self.passphrase_spoken:
            items = [item for item in items if item != "amber seal"]
        return items

    def _observation(self, message: str) -> str:
        room = self.rooms[self.location]
        item_text = (
            " Visible items: " + ", ".join(self._visible_items()) + "."
            if self._visible_items()
            else ""
        )
        inventory = " Inventory: " + ", ".join(self.inventory) + "." if self.inventory else ""
        status = []
        if self.location == "gallery":
            status.append("The iron gate is unlocked." if self.iron_gate_unlocked else "The iron gate is locked.")
        if self.location == "crypt" and not self.lantern_lit:
            status.append("The altar is too dark to read.")
        if self.location == "vault_ante":
            status.append("The moon lock is open." if self.vault_unlocked else "The moon lock is closed.")
        if self.location == "vault" and self.passphrase_spoken:
            status.append("The glass case is open.")
        status_text = " " + " ".join(status) if status else ""
        return (
            "Goal: recover the amber seal.\n"
            f"{message}\n{room['desc']}{status_text}{item_text}{inventory}"
        )


class TextWorldEnv:
    """Thin adapter for a TextWorld game file.

    BLiMP branch expansion replays root-to-node actions from a fresh reset, so
    this adapter does not require access to TextWorld's internal clone state.
    """

    def __init__(self, game_file: str | Path) -> None:
        self.game_file = str(game_file)
        self._env = None
        self._state = None
        self._last_score = 0.0

    def reset(self, seed: int | None = None) -> str:
        del seed
        try:
            import textworld
            from textworld import EnvInfos
        except ImportError as exc:
            raise RuntimeError(
                "TextWorld is not installed. Install with `pip install textworld` "
                "or use `--env tiny` for a smoke test."
            ) from exc

        infos = EnvInfos(
            admissible_commands=True,
            description=True,
            inventory=True,
            location=True,
            objective=True,
            score=True,
            max_score=True,
            won=True,
            lost=True,
        )
        try:
            self._env = textworld.start(self.game_file, request_infos=infos)
        except TypeError:
            self._env = textworld.start(self.game_file, infos=infos)
        self._state = self._env.reset()
        self._last_score = self._score_from_state(self._state, fallback=0.0)
        return self._feedback(self._state)

    def valid_actions(self) -> list[str]:
        if self._state is None:
            return []
        return list(getattr(self._state, "admissible_commands", None) or [])

    def step(self, action: str) -> StepResult:
        if self._env is None:
            raise RuntimeError("TextWorld environment must be reset before stepping.")
        was_valid = action in self.valid_actions()
        self._state, raw_reward, done = self._env.step(action)
        score = self._score_from_state(self._state, fallback=float(raw_reward))
        score_delta = score - self._last_score
        self._last_score = score
        return StepResult(
            observation=self._feedback(self._state),
            reward=score_delta,
            done=bool(done),
            info={
                "score": score,
                "max_score": self._max_score_from_state(self._state),
                "score_delta": score_delta,
                "raw_reward": float(raw_reward),
                "valid": was_valid,
                "admissible_commands": self.valid_actions(),
                "won": bool(getattr(self._state, "won", False)),
                "lost": bool(getattr(self._state, "lost", False)),
            },
        )

    @staticmethod
    def _score_from_state(state: Any, *, fallback: float) -> float:
        score = getattr(state, "score", None)
        if score is None:
            return fallback
        return float(score)

    @staticmethod
    def _max_score_from_state(state: Any) -> float:
        max_score = getattr(state, "max_score", None)
        if max_score is None:
            return 0.0
        return float(max_score)

    @staticmethod
    def _feedback(state: Any) -> str:
        parts = []
        objective = getattr(state, "objective", None)
        if objective:
            parts.append(f"Objective: {objective}")
        feedback = getattr(state, "feedback", None)
        if feedback:
            parts.append(f"Feedback: {feedback}")
        description = getattr(state, "description", None)
        if description:
            parts.append(f"Room: {description}")
        inventory = getattr(state, "inventory", None)
        if inventory:
            parts.append(f"Inventory: {inventory}")
        score = getattr(state, "score", None)
        max_score = getattr(state, "max_score", None)
        if score is not None:
            if max_score is not None:
                parts.append(f"Score: {score}/{max_score}")
            else:
                parts.append(f"Score: {score}")
        return "\n".join(str(part) for part in parts if part).strip()


class ScienceWorldEnv:
    """Thin adapter for a single ScienceWorld task variation spec.

    The spec format is `task:variation:simplification[:step_limit]`, for example
    `use-thermometer:17:easy:100`. ScienceWorld uses a Java backend; this adapter
    keeps one backend process per Python process and reloads task variations.
    """

    _backend: Any = None
    _step_limit: int | None = None

    def __init__(self, spec: str | Path) -> None:
        task_name, variation, simplification, step_limit = self._parse_spec(str(spec))
        self.task_name = task_name
        self.variation = variation
        self.simplification = simplification
        self.step_limit = step_limit
        self._env = None
        self._score = 0.0
        self._done = False
        self._task_description = ""

    @staticmethod
    def _parse_spec(spec: str) -> tuple[str, int, str, int]:
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(
                "ScienceWorld spec must be task:variation:simplification[:step_limit]"
            )
        task_name = parts[0]
        variation = int(parts[1])
        simplification = parts[2] if len(parts) >= 3 else "easy"
        step_limit = int(parts[3]) if len(parts) >= 4 else 100
        return task_name, variation, simplification, step_limit

    @classmethod
    def _get_backend(cls, step_limit: int) -> Any:
        if cls._backend is None or cls._step_limit != step_limit:
            try:
                from scienceworld import ScienceWorldEnv as BackendEnv
            except ImportError as exc:
                raise RuntimeError(
                    "ScienceWorld is not installed. Install with `pip install scienceworld`."
                ) from exc
            cls._backend = BackendEnv("", None, envStepLimit=step_limit)
            cls._step_limit = step_limit
        return cls._backend

    def reset(self, seed: int | None = None) -> str:
        del seed
        self._env = self._get_backend(self.step_limit)
        self._env.load(self.task_name, self.variation, self.simplification)
        observation, info = self._env.reset()
        self._task_description = str(self._env.get_task_description())
        self._score = float(info.get("score", 0.0) or 0.0)
        self._done = False
        return self._feedback(observation)

    def valid_actions(self) -> list[str]:
        if self._env is None or self._done:
            return []
        try:
            rows = self._env.get_valid_action_object_combinations_with_templates()
            actions = [str(row["action"]).lower().strip() for row in rows if row.get("action")]
        except Exception:
            actions = []
        actions.extend(["look around", "inventory"])
        return sorted(set(action for action in actions if self._keep_action(action)))

    def _keep_action(self, action: str) -> bool:
        action = " ".join(action.lower().strip().split())
        if not action:
            return False

        # ScienceWorld exposes many valid-but-immediately-failing focus actions
        # such as `focus on air` for phase-change tasks. Keep focus only when it
        # plausibly targets the substance named by the task.
        if action.startswith("focus on "):
            target = action.removeprefix("focus on ").strip()
            if target in {
                "agent",
                "air",
                "inventory",
                "hallway",
                "picture",
            }:
                return False
            desired = self._phase_target()
            if desired:
                # `focus on orange` is terminally wrong for `boil orange juice`;
                # allow exact target hits or fuller object descriptions only.
                return target == desired or desired in target

        # These combinatorial templates massively inflate the action space and
        # are rarely useful as first-pass LM actions without tool-specific
        # structure. Re-enable later if a task family needs them.
        if action.startswith(("mix ", "eat ")):
            return False
        return True

    def _phase_target(self) -> str | None:
        phase_match = re.search(
            r"\b(?:boil|melt|freeze)\s+([a-z][a-z0-9 -]*?)(?:\.|,|$)",
            self._task_description.lower(),
        )
        if not phase_match:
            return None
        return " ".join(phase_match.group(1).strip().split())

    def step(self, action: str) -> StepResult:
        if self._env is None:
            raise RuntimeError("ScienceWorld environment must be reset before stepping.")
        normalized = " ".join(action.lower().strip().split())
        was_valid = normalized in self.valid_actions()
        observation, reward, done, info = self._env.step(normalized)
        self._score = float(info.get("score", self._score) or self._score)
        won = self._score >= 100.0
        self._done = bool(done)
        return StepResult(
            observation=self._feedback(observation),
            reward=float(reward),
            done=bool(done),
            info={
                "score": self._score,
                "max_score": 100.0,
                "valid": was_valid,
                "won": won,
                "task_name": self.task_name,
                "variation": self.variation,
                "simplification": self.simplification,
            },
        )

    def _feedback(self, observation: str) -> str:
        assert self._env is not None
        parts = [
            f"Task: {self.task_name}",
            f"Variation: {self.variation}",
            f"Goal: {self._task_description}",
            f"Observation: {observation}",
            f"Look: {self._env.look()}",
            f"Inventory: {self._env.inventory()}",
            f"Score: {self._score}/100",
        ]
        return "\n".join(str(part) for part in parts if part).strip()


def make_env(env_name: str, game_file: str | None = None) -> TextEnv:
    if env_name == "tiny":
        return TinyQuestEnv()
    if env_name == "hard":
        return HardQuestEnv()
    if env_name == "textworld":
        if not game_file:
            raise ValueError("--game-file is required when --env textworld")
        return TextWorldEnv(game_file)
    if env_name == "scienceworld":
        if not game_file:
            raise ValueError("ScienceWorld requires a task variation spec")
        return ScienceWorldEnv(game_file)
    raise ValueError(f"Unknown environment: {env_name}")


def replay_actions(
    env_name: str,
    actions: list[str],
    game_file: str | None = None,
    seed: int | None = None,
) -> tuple[TextEnv, str, bool, float]:
    env = make_env(env_name, game_file)
    observation = env.reset(seed=seed)
    done = False
    score = 0.0
    for action in actions:
        result = env.step(action)
        observation = result.observation
        done = result.done
        score = float(result.info.get("score", result.reward))
        if done:
            break
    return env, observation, done, score
