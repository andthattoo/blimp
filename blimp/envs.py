from __future__ import annotations

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


class TextWorldEnv:
    """Thin adapter for a TextWorld game file.

    BLiMP branch expansion replays root-to-node actions from a fresh reset, so
    this adapter does not require access to TextWorld's internal clone state.
    """

    def __init__(self, game_file: str | Path) -> None:
        self.game_file = str(game_file)
        self._env = None
        self._state = None

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
            score=True,
        )
        self._env = textworld.start(self.game_file, infos=infos)
        self._state = self._env.reset()
        return self._feedback(self._state)

    def valid_actions(self) -> list[str]:
        if self._state is None:
            return []
        return list(getattr(self._state, "admissible_commands", None) or [])

    def step(self, action: str) -> StepResult:
        if self._env is None:
            raise RuntimeError("TextWorld environment must be reset before stepping.")
        self._state, score, done = self._env.step(action)
        return StepResult(
            observation=self._feedback(self._state),
            reward=float(score),
            done=bool(done),
            info={
                "score": float(getattr(self._state, "score", score) or score),
                "valid": action in self.valid_actions(),
                "admissible_commands": self.valid_actions(),
            },
        )

    @staticmethod
    def _feedback(state: Any) -> str:
        feedback = getattr(state, "feedback", None)
        if feedback:
            return str(feedback)
        parts = [
            getattr(state, "description", ""),
            getattr(state, "inventory", ""),
        ]
        return "\n".join(str(part) for part in parts if part).strip()


def make_env(env_name: str, game_file: str | None = None) -> TextEnv:
    if env_name == "tiny":
        return TinyQuestEnv()
    if env_name == "textworld":
        if not game_file:
            raise ValueError("--game-file is required when --env textworld")
        return TextWorldEnv(game_file)
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
