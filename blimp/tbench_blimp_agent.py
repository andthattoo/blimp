from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from blimp.sglang_rollout import strip_reasoning

try:  # Terminal-Bench is optional for the rest of this repo.
    from terminal_bench.agents import AgentResult, BaseAgent
except Exception:  # pragma: no cover - exercised when terminal-bench is absent.
    try:
        from terminal_bench.agents.base_agent import AgentResult, BaseAgent
    except Exception:  # pragma: no cover

        @dataclass
        class AgentResult:  # type: ignore[no-redef]
            total_input_tokens: int = 0
            total_output_tokens: int = 0
            failure_mode: Any = None
            timestamped_markers: list[tuple[float, str]] = field(default_factory=list)

        class BaseAgent:  # type: ignore[no-redef]
            def __init__(self, **_: Any) -> None:
                pass

            @staticmethod
            def name() -> str:
                return "base-agent-stub"


try:
    from terminal_bench.agents.failure_mode import FailureMode
except Exception:  # pragma: no cover - terminal-bench optional in local tests.
    FailureMode = None  # type: ignore[assignment]


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandSpec:
    keystrokes: str
    is_blocking: bool = False
    timeout_sec: float = 5.0
    min_wait_sec: float = 0.7


@dataclass
class AgentDecision:
    memory: str
    commands: list[CommandSpec]
    is_task_complete: bool = False
    notes: str = ""


@dataclass
class MazeSolveResult:
    completed: bool
    steps: int
    map_text: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class LLMCallError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_base: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.2,
        max_tokens: int = 900,
        timeout: float = 120.0,
        disable_thinking: bool = True,
    ) -> None:
        self.chat_url = normalize_chat_url(api_base)
        self.model_name = normalize_model_name(model_name)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.disable_thinking = disable_thinking

    def chat(self, prompt: str) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMCallError(f"LLM HTTP {exc.code}: {detail}") from exc

        message = raw["choices"][0]["message"]
        usage = raw.get("usage") or {}
        prompt_tokens = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("total_input_tokens")
            or 0
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("total_output_tokens")
            or 0
        )
        return ChatResult(
            content=message.get("content") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=raw,
        )


Coord = tuple[int, int]


DIR_DELTA: dict[str, Coord] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
OPPOSITE_DIR: dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}
DIR_ORDER = ("N", "E", "S", "W")


class BlindMazeController:
    """Deterministic graph explorer for Terminal-Bench blind-maze tasks."""

    def __init__(self, agent: "BLiMPTerminusAgent") -> None:
        self.agent = agent
        self.current: Coord = (0, 0)
        self.start: Coord = (0, 0)
        self.exit: Coord | None = None
        self.open_cells: set[Coord] = {self.start}
        self.wall_cells: set[Coord] = set()
        self.edges: dict[Coord, dict[str, Coord]] = {self.start: {}}
        self.tried: dict[Coord, set[str]] = {self.start: set()}
        self.events: list[dict[str, Any]] = []
        self.step_count = 0

    def solve(self, *, session: Any, logging_dir: Path | None) -> MazeSolveResult:
        start_output = self.agent._send_command(
            session,
            CommandSpec("./maze_game.sh\n", is_blocking=False, timeout_sec=3.0),
        )
        self._record("start", "./maze_game.sh", [], [], start_output)

        while self.step_count < self.agent.max_episodes * self.agent.max_commands_per_episode:
            target = self._nearest_frontier_cell()
            if target is None:
                break
            if target != self.current:
                path = self._path_between(self.current, target)
                if not path:
                    break
                self._send_moves(session, path)
                continue

            direction = self._next_untried_direction(self.current)
            if direction is None:
                continue
            self._send_moves(session, [direction])

        map_text = self._render_map()
        exit_output = self.agent._send_command(
            session,
            CommandSpec("exit\n", is_blocking=False, timeout_sec=2.0),
        )
        self._record("exit", "exit", [], [], exit_output)
        write_output = self.agent._send_command(
            session,
            CommandSpec(write_map_command(map_text), is_blocking=False, timeout_sec=3.0),
        )
        self._record("write_map", "cat > /app/maze_map.txt", [], [], write_output)
        check_output = self.agent._send_command(
            session,
            CommandSpec(
                "python3 - <<'PY'\n"
                "from pathlib import Path\n"
                "p=Path('/app/maze_map.txt')\n"
                "print(p.exists(), len(p.read_text().splitlines()))\n"
                "print(p.read_text())\n"
                "PY\n",
                is_blocking=False,
                timeout_sec=3.0,
            ),
        )
        self._record("check_map", "python3 check", [], [], check_output)
        self._write_debug(logging_dir, map_text)
        return MazeSolveResult(
            completed=True,
            steps=self.step_count,
            map_text=map_text,
        )

    def _send_moves(self, session: Any, directions: list[str]) -> None:
        if not directions:
            return
        command = "move " + " & ".join(directions) + "\n"
        output = self.agent._send_command(
            session,
            CommandSpec(command, is_blocking=False, timeout_sec=2.0),
        )
        responses = parse_maze_responses(output)
        self._apply_responses(directions, responses)
        self._record("move", command.strip(), directions, responses, output)
        self.step_count += len(directions)

    def _apply_responses(self, directions: list[str], responses: list[str]) -> None:
        for direction, response in zip(directions, responses):
            self.tried.setdefault(self.current, set()).add(direction)
            nxt = add_coords(self.current, DIR_DELTA[direction])
            if response == "hit wall":
                if nxt not in self.open_cells:
                    self.wall_cells.add(nxt)
                continue
            if response not in {"moved", "reached exit"}:
                continue
            src = self.current
            self.open_cells.add(nxt)
            self.wall_cells.discard(nxt)
            self.edges.setdefault(src, {})[direction] = nxt
            self.edges.setdefault(nxt, {})[OPPOSITE_DIR[direction]] = src
            self.tried.setdefault(nxt, set()).add(OPPOSITE_DIR[direction])
            if response == "reached exit":
                self.exit = nxt
            self.current = nxt

    def _nearest_frontier_cell(self) -> Coord | None:
        queue: deque[Coord] = deque([self.current])
        seen = {self.current}
        while queue:
            cell = queue.popleft()
            if self._next_untried_direction(cell) is not None:
                return cell
            for nxt in self.edges.get(cell, {}).values():
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return None

    def _next_untried_direction(self, cell: Coord) -> str | None:
        tried = self.tried.setdefault(cell, set())
        for direction in DIR_ORDER:
            if direction not in tried:
                return direction
        return None

    def _path_between(self, start: Coord, target: Coord) -> list[str]:
        queue: deque[Coord] = deque([start])
        parent: dict[Coord, tuple[Coord, str] | None] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == target:
                break
            for direction, nxt in self.edges.get(cell, {}).items():
                if nxt not in parent:
                    parent[nxt] = (cell, direction)
                    queue.append(nxt)
        if target not in parent:
            return []

        path: list[str] = []
        cell = target
        while cell != start:
            prev = parent[cell]
            if prev is None:
                break
            cell, direction = prev
            path.append(direction)
        path.reverse()
        return path

    def _render_map(self) -> str:
        known = self.open_cells | self.wall_cells
        min_x = min(x for x, _ in known)
        max_x = max(x for x, _ in known)
        min_y = min(y for _, y in known)
        max_y = max(y for _, y in known)
        rows: list[str] = []
        for y in range(min_y, max_y + 1):
            chars: list[str] = []
            for x in range(min_x, max_x + 1):
                cell = (x, y)
                if cell == self.start:
                    chars.append("S")
                elif cell == self.exit:
                    chars.append("E")
                elif cell in self.open_cells:
                    chars.append(" ")
                else:
                    chars.append("#")
            rows.append("".join(chars))
        return "\n".join(rows) + "\n"

    def _record(
        self,
        kind: str,
        command: str,
        directions: list[str],
        responses: list[str],
        output: str,
    ) -> None:
        self.events.append(
            {
                "kind": kind,
                "command": command,
                "directions": directions,
                "responses": responses,
                "current": self.current,
                "open_cells": sorted(self.open_cells),
                "wall_cells": sorted(self.wall_cells),
                "exit": self.exit,
                "output": output,
            }
        )

    def _write_debug(self, logging_dir: Path | None, map_text: str) -> None:
        if logging_dir is None:
            return
        debug_dir = logging_dir / "maze-controller"
        debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "current": self.current,
            "start": self.start,
            "exit": self.exit,
            "open_cells": sorted(self.open_cells),
            "wall_cells": sorted(self.wall_cells),
            "events": self.events,
            "map_text": map_text,
            "steps": self.step_count,
        }
        with (debug_dir / "debug.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


class BLiMPTerminusAgent(BaseAgent):
    """Terminal-Bench custom agent with short transcript plus durable memory.

    This is intentionally small and inspectable. It is a memory/control
    experiment for tasks like blind-maze, not a leaderboard-optimized agent.
    """

    @staticmethod
    def name() -> str:
        return "blimp-terminus"

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        max_episodes: int | str = 48,
        max_commands_per_episode: int | str = 3,
        recent_events: int | str = 6,
        memory_words: int | str = 420,
        max_prompt_chars: int | str = 24000,
        temperature: float | str = 0.2,
        max_tokens: int | str = 900,
        request_timeout: float | str = 120.0,
        command_wait_sec: float | str = 0.7,
        clear_tmux_history: bool | str = False,
        enable_thinking: bool | str = False,
        maze_controller: bool | str = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name or os.environ.get("MODEL") or "Qwen/Qwen3.5-4B"
        self.api_base = (
            api_base
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://127.0.0.1:30000/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.max_episodes = int(max_episodes)
        self.max_commands_per_episode = int(max_commands_per_episode)
        self.recent_events = int(recent_events)
        self.memory_words = int(memory_words)
        self.max_prompt_chars = int(max_prompt_chars)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.request_timeout = float(request_timeout)
        self.command_wait_sec = float(command_wait_sec)
        self.clear_tmux_history = parse_bool(clear_tmux_history)
        self.disable_thinking = not parse_bool(enable_thinking)
        self.maze_controller = parse_bool(maze_controller)
        self._timestamped_markers: list[tuple[float, str]] = []

    def perform_task(
        self,
        instruction: str,
        session: Any,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        logging_dir = Path(logging_dir) if logging_dir is not None else None
        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
        if self.maze_controller and "blind maze" in instruction.lower():
            return self._perform_blind_maze_task(instruction, session, logging_dir)

        client = OpenAICompatibleChatClient(
            api_base=self.api_base,
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
            disable_thinking=self.disable_thinking,
        )

        memory = initial_memory()
        recent: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            current_terminal = session.get_incremental_output()
        except Exception as exc:
            current_terminal = f"Unable to read terminal state yet: {exc}"

        for episode in range(self.max_episodes):
            prompt = build_blimp_prompt(
                instruction=instruction,
                memory=memory,
                recent_events=recent,
                terminal_state=current_terminal,
                max_prompt_chars=self.max_prompt_chars,
            )
            try:
                response = client.chat(prompt)
            except LLMCallError as exc:
                self._write_debug(
                    logging_dir,
                    episode,
                    {
                        "prompt": prompt,
                        "error": str(exc),
                        "memory": memory,
                        "recent": recent,
                    },
                )
                return make_agent_result(
                    total_input_tokens,
                    total_output_tokens,
                    failure_name="LLM_ERROR",
                    markers=self._timestamped_markers,
                )

            total_input_tokens += response.prompt_tokens
            total_output_tokens += response.completion_tokens
            try:
                decision = parse_agent_decision(response.content)
            except ValueError as exc:
                recent.append(
                    format_event(
                        f"LLM_PARSE_ERROR episode={episode}",
                        str(exc),
                    )
                )
                recent = trim_recent_events(recent, self.recent_events)
                self._write_debug(
                    logging_dir,
                    episode,
                    {
                        "prompt": prompt,
                        "response": response.content,
                        "error": str(exc),
                        "usage": response.raw.get("usage"),
                    },
                )
                continue

            if decision.memory.strip():
                memory = truncate_words(decision.memory, self.memory_words)
            commands = decision.commands[: self.max_commands_per_episode]

            command_records = []
            for command in commands:
                output = self._send_command(session, command)
                current_terminal = output
                record = {
                    "keystrokes": command.keystrokes,
                    "is_blocking": command.is_blocking,
                    "effective_blocking": command.is_blocking
                    and not is_multiline_keystrokes(command.keystrokes),
                    "timeout_sec": command.timeout_sec,
                    "output": output,
                }
                command_records.append(record)
                recent.append(format_event(command.keystrokes, output))
                recent = trim_recent_events(recent, self.recent_events)
                if self.clear_tmux_history:
                    try:
                        session.clear_history()
                    except Exception:
                        pass

            self._mark(session, f"episode-{episode}")
            self._write_debug(
                logging_dir,
                episode,
                {
                    "prompt": prompt,
                    "response": response.content,
                    "parsed": {
                        "memory": memory,
                        "is_task_complete": decision.is_task_complete,
                        "notes": decision.notes,
                        "commands": [asdict(command) for command in commands],
                    },
                    "command_records": command_records,
                    "usage": response.raw.get("usage"),
                    "approx_prompt_chars": len(prompt),
                },
            )

            if decision.is_task_complete:
                break
            if not commands:
                recent.append(
                    format_event(
                        f"NO_COMMAND episode={episode}",
                        "The model returned no commands; ask it to continue.",
                    )
                )
                recent = trim_recent_events(recent, self.recent_events)

        return make_agent_result(
            total_input_tokens,
            total_output_tokens,
            failure_name="NONE",
            markers=self._timestamped_markers,
        )

    def _perform_blind_maze_task(
        self,
        instruction: str,
        session: Any,
        logging_dir: Path | None,
    ) -> AgentResult:
        controller = BlindMazeController(self)
        result = controller.solve(session=session, logging_dir=logging_dir)
        return make_agent_result(
            result.total_input_tokens,
            result.total_output_tokens,
            failure_name="NONE" if result.completed else "UNKNOWN",
            markers=self._timestamped_markers,
        )

    def _send_command(self, session: Any, command: CommandSpec) -> str:
        keys = tmux_keys_from_keystrokes(command.keystrokes)
        # Terminal-Bench implements blocking by appending a tmux wait command to
        # the final Enter. That is fine for one-line shell commands, but it can
        # corrupt heredocs or other multiline input, so multiline sends use a
        # timed non-blocking wait.
        block = command.is_blocking and not is_multiline_keystrokes(command.keystrokes)
        try:
            session.send_keys(
                keys,
                block=block,
                min_timeout_sec=max(command.min_wait_sec, self.command_wait_sec),
                max_timeout_sec=command.timeout_sec,
            )
            return session.get_incremental_output()
        except TimeoutError as exc:
            try:
                output = session.get_incremental_output()
            except Exception:
                output = ""
            return f"COMMAND_TIMEOUT: {exc}\n{output}".strip()
        except Exception as exc:
            try:
                output = session.get_incremental_output()
            except Exception:
                output = ""
            return f"COMMAND_ERROR: {exc}\n{output}".strip()

    def _mark(self, session: Any, marker: str) -> None:
        try:
            timestamp = float(session.get_asciinema_timestamp())
        except Exception:
            timestamp = 0.0
        self._timestamped_markers.append((timestamp, marker))

    @staticmethod
    def _write_debug(
        logging_dir: Path | None,
        episode: int,
        payload: dict[str, Any],
    ) -> None:
        if logging_dir is None:
            return
        episode_dir = logging_dir / f"episode-{episode}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        with (episode_dir / "debug.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


def build_blimp_prompt(
    *,
    instruction: str,
    memory: str,
    recent_events: list[str],
    terminal_state: str,
    max_prompt_chars: int,
) -> str:
    recent_text = "\n\n".join(recent_events) if recent_events else "No commands yet."
    prompt = f"""You are BLiMP-Terminus, a Terminal-Bench agent.

Complete the task by sending terminal keystrokes. You have a compact durable
memory block and a short recent transcript. Older terminal history is not shown,
so all long-lived state must be maintained in MEMORY.

Critical rules:
- Return exactly one JSON object. Do not wrap it in Markdown.
- Keep MEMORY concise and factual; do not store chain-of-thought.
- For blind-maze tasks, MEMORY should track current coordinate, known open
  cells, known wall edges, exit coordinate if found, frontier/untried exits,
  path back to start/current location, and the next exploration plan.
- Batch maze commands are sequential. Update position one response at a time.
- Reaching the exit is not enough; create /app/maze_map.txt only after the maze
  is fully explored. Exit the game before writing the map file.
- Use is_blocking=false for interactive maze inputs. Use is_blocking=true only
  for shell commands expected to finish.
- After creating the final file, run a quick shell check, then set
  is_task_complete=true.

JSON schema:
{{
  "memory": "compact durable state for the next turn",
  "commands": [
    {{"keystrokes": "./maze_game.sh\\n", "is_blocking": false, "timeout_sec": 3}}
  ],
  "is_task_complete": false,
  "notes": "brief non-durable note, optional"
}}

TASK INSTRUCTION:
{instruction}

MEMORY:
{memory}

RECENT TERMINAL EVENTS:
{recent_text}

CURRENT TERMINAL STATE:
{terminal_state}
"""
    if len(prompt) <= max_prompt_chars:
        return prompt
    overflow = len(prompt) - max_prompt_chars
    clipped_recent = recent_text[overflow + 100 :] if overflow + 100 < len(recent_text) else ""
    return f"""You are BLiMP-Terminus, a Terminal-Bench agent.

Return exactly one JSON object with keys memory, commands, is_task_complete,
and optional notes. Maintain all durable state in MEMORY. Use only a few
terminal commands per turn.

TASK INSTRUCTION:
{instruction}

MEMORY:
{memory}

RECENT TERMINAL EVENTS:
{clipped_recent or "Recent transcript clipped; rely on MEMORY."}

CURRENT TERMINAL STATE:
{terminal_state[-6000:]}
"""


def parse_agent_decision(text: str) -> AgentDecision:
    cleaned = strip_reasoning(text)
    data = json.loads(extract_json_object(cleaned))
    if not isinstance(data, dict):
        raise ValueError("model response JSON must be an object")

    commands = [
        normalize_command_spec(raw)
        for raw in data.get("commands", [])
        if raw is not None
    ]
    return AgentDecision(
        memory=str(data.get("memory", "") or ""),
        commands=commands,
        is_task_complete=parse_bool(data.get("is_task_complete", False)),
        notes=str(data.get("notes", "") or data.get("explanation", "") or ""),
    )


def normalize_command_spec(raw: Any) -> CommandSpec:
    if isinstance(raw, str):
        return CommandSpec(keystrokes=clean_keystrokes(raw))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid command entry: {raw!r}")
    keystrokes = clean_keystrokes(str(raw.get("keystrokes", "") or ""))
    if not keystrokes:
        raise ValueError("command keystrokes cannot be empty")
    return CommandSpec(
        keystrokes=keystrokes,
        is_blocking=parse_bool(raw.get("is_blocking", False)),
        timeout_sec=float(raw.get("timeout_sec", 5.0) or 5.0),
        min_wait_sec=float(raw.get("min_wait_sec", 0.7) or 0.7),
    )


def extract_json_object(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated JSON object")


def normalize_chat_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def parse_maze_responses(output: str) -> list[str]:
    response_line = re.compile(
        r"^\s*(?:hit wall|reached exit|moved)(?:\s*&\s*(?:hit wall|reached exit|moved))*\s*$"
    )
    for line in reversed(output.splitlines()):
        if response_line.match(line):
            return re.findall(r"hit wall|reached exit|moved", line)
    return re.findall(r"hit wall|reached exit|moved", output)


def add_coords(left: Coord, right: Coord) -> Coord:
    return (left[0] + right[0], left[1] + right[1])


def write_map_command(map_text: str) -> str:
    return "cat > /app/maze_map.txt <<'EOF'\n" + map_text + "EOF\n"


def normalize_model_name(model_name: str) -> str:
    return model_name.removeprefix("openai/")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def clean_keystrokes(text: str) -> str:
    text = text.replace("\x00", "")
    if len(text) > 4000:
        text = text[:4000]
    return text


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return (" ".join(words[:max_words]) + " [truncated]").strip()


def tmux_keys_from_keystrokes(keystrokes: str) -> str | list[str]:
    if "\n" not in keystrokes and "\r" not in keystrokes:
        return keystrokes

    keys: list[str] = []
    for line in keystrokes.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        if text:
            keys.append(text)
        if line.endswith(("\n", "\r")):
            keys.append("Enter")
    if not keys and keystrokes:
        return keystrokes
    return keys


def is_multiline_keystrokes(keystrokes: str) -> bool:
    stripped = keystrokes.rstrip("\r\n")
    return "\n" in stripped or "\r" in stripped


def trim_recent_events(events: list[str], max_events: int) -> list[str]:
    if max_events <= 0:
        return []
    return events[-max_events:]


def format_event(command: str, output: str) -> str:
    return f"COMMAND:\n{command.rstrip()}\nOUTPUT:\n{output.strip()}"


def initial_memory() -> str:
    return (
        "No durable facts yet. Start the task, then maintain a compact map/state "
        "summary here. For blind maze: current=(0,0)=S, known cells/walls, "
        "frontier, route to current, and next plan."
    )


def make_agent_result(
    total_input_tokens: int,
    total_output_tokens: int,
    *,
    failure_name: str,
    markers: list[tuple[float, str]],
) -> AgentResult:
    kwargs: dict[str, Any] = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "timestamped_markers": markers,
    }
    if FailureMode is not None:
        fallback = getattr(FailureMode, "NONE", None) or getattr(FailureMode, "UNSET", None)
        kwargs["failure_mode"] = getattr(FailureMode, failure_name, fallback)
    return AgentResult(**kwargs)


__all__ = [
    "AgentDecision",
    "BLiMPTerminusAgent",
    "CommandSpec",
    "OpenAICompatibleChatClient",
    "build_blimp_prompt",
    "extract_json_object",
    "is_multiline_keystrokes",
    "normalize_chat_url",
    "normalize_model_name",
    "parse_maze_responses",
    "parse_agent_decision",
    "tmux_keys_from_keystrokes",
]
