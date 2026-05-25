from __future__ import annotations

import unittest

from blimp.tbench_blimp_agent import (
    CommandSpec,
    build_blimp_prompt,
    extract_json_object,
    is_multiline_keystrokes,
    normalize_chat_url,
    normalize_model_name,
    parse_agent_decision,
    tmux_keys_from_keystrokes,
)


class TBenchBLiMPAgentTest(unittest.TestCase):
    def test_extract_json_object_from_fenced_response(self) -> None:
        text = """Some preface.
```json
{"memory": "m", "commands": [{"keystrokes": "move S\\n"}]}
```
Trailing text.
"""
        self.assertEqual(
            extract_json_object(text),
            '{"memory": "m", "commands": [{"keystrokes": "move S\\n"}]}',
        )

    def test_parse_agent_decision_normalizes_command(self) -> None:
        decision = parse_agent_decision(
            """
{
  "memory": "current=(0,0)",
  "commands": [
    {"keystrokes": "move S\\n", "is_blocking": false, "timeout_sec": 2}
  ],
  "is_task_complete": "false"
}
"""
        )

        self.assertEqual(decision.memory, "current=(0,0)")
        self.assertEqual(
            decision.commands,
            [CommandSpec(keystrokes="move S\n", is_blocking=False, timeout_sec=2.0)],
        )
        self.assertFalse(decision.is_task_complete)

    def test_tmux_keys_from_keystrokes_preserves_enter(self) -> None:
        self.assertEqual(tmux_keys_from_keystrokes("move S\n"), ["move S", "Enter"])
        self.assertFalse(is_multiline_keystrokes("move S\n"))
        self.assertEqual(
            tmux_keys_from_keystrokes("cat > x <<'EOF'\nhi\nEOF\n"),
            ["cat > x <<'EOF'", "Enter", "hi", "Enter", "EOF", "Enter"],
        )
        self.assertTrue(is_multiline_keystrokes("cat > x <<'EOF'\nhi\nEOF\n"))

    def test_openai_compat_normalization(self) -> None:
        self.assertEqual(
            normalize_chat_url("http://127.0.0.1:30000"),
            "http://127.0.0.1:30000/v1/chat/completions",
        )
        self.assertEqual(
            normalize_chat_url("http://127.0.0.1:30000/v1"),
            "http://127.0.0.1:30000/v1/chat/completions",
        )
        self.assertEqual(
            normalize_model_name("openai/Qwen/Qwen3.5-4B"),
            "Qwen/Qwen3.5-4B",
        )

    def test_prompt_is_bounded_when_recent_events_are_long(self) -> None:
        prompt = build_blimp_prompt(
            instruction="Create /app/maze_map.txt",
            memory="current=(0,0)",
            recent_events=["x" * 1000 for _ in range(20)],
            terminal_state="screen" * 1000,
            max_prompt_chars=6000,
        )

        self.assertLessEqual(len(prompt), 9000)
        self.assertIn("MEMORY:", prompt)
        self.assertIn("current=(0,0)", prompt)


if __name__ == "__main__":
    unittest.main()
