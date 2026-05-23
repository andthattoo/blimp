from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from blimp.train_reinforce import (
    EpisodeTrace,
    MemoryWrite,
    Transition,
    compile_aux_loss_examples,
    discounted_returns,
    load_branch_contrast_examples,
    memory_policy_items,
    parse_think_action_response,
)


def make_transition(
    index: int,
    *,
    reward: float = 0.0,
    score_delta: float = 0.0,
    valid: bool = True,
    won: bool = False,
) -> Transition:
    return Transition(
        prompt=f"prompt {index}",
        observation=f"obs {index}",
        thinking=f"think {index}",
        action=f"action {index}",
        raw_completion=f"THINK: think {index}\nACTION: action {index}",
        policy_completion=f"THINK: think {index}\nACTION: action {index}",
        reward=reward,
        done=won,
        score=float(index),
        next_observation=f"obs {index + 1}",
        score_delta=score_delta,
        won=won,
        valid=valid,
    )


class AuxiliaryLossCompilerTest(unittest.TestCase):
    def test_parse_think_action_response(self) -> None:
        parsed = parse_think_action_response(
            "THINK: remember the key\nACTION: unlock door with brass key"
        )

        self.assertEqual(parsed.thinking, "remember the key")
        self.assertEqual(parsed.action, "unlock door with brass key")

    def test_compiler_trains_prediction_always_and_behavior_selectively(self) -> None:
        solved_trace = EpisodeTrace(
            episode_id=0,
            mode="blimp",
            game_file=None,
            solved=True,
            score=1.0,
            total_reward=1.0,
            actions=["action 0", "action 1", "action 2"],
            memories=[],
            transitions=[
                make_transition(0, reward=0.1, score_delta=0.1),
                make_transition(1, reward=-0.1, valid=False),
                make_transition(2, reward=1.0, score_delta=1.0, won=True),
            ],
        )
        failed_trace = EpisodeTrace(
            episode_id=1,
            mode="blimp",
            game_file=None,
            solved=False,
            score=0.0,
            total_reward=-0.2,
            actions=["action 0", "action 1"],
            memories=[],
            transitions=[
                make_transition(0, reward=-0.1),
                make_transition(1, reward=-0.1),
            ],
        )

        examples = compile_aux_loss_examples(
            [solved_trace, failed_trace],
            echo_weight=1.0,
            score_weight=0.0,
            action_good_weight=1.0,
            thought_weight=1.0,
            future_weight=1.0,
            future_horizon=2,
            echo_max_words=200,
        )
        counts = {}
        for example in examples:
            counts[example.loss_type] = counts.get(example.loss_type, 0) + 1

        self.assertEqual(counts["echo"], 5)
        self.assertEqual(counts["action_good"], 2)
        self.assertEqual(counts["thought"], 2)
        self.assertEqual(counts["future"], 1)

    def test_branch_contrast_prefers_better_sibling(self) -> None:
        row = {
            "nodes": [
                {
                    "node_id": "good",
                    "parent_id": "root",
                    "done": True,
                    "score": 1.0,
                    "path_actions": ["take gem"],
                    "steps": [
                        {
                            "observation": "start",
                            "memory_before": "",
                            "raw_response": "THINK: finish now\nACTION: take gem",
                            "action": "take gem",
                        }
                    ],
                },
                {
                    "node_id": "bad",
                    "parent_id": "root",
                    "done": False,
                    "score": 0.0,
                    "path_actions": ["look"],
                    "steps": [
                        {
                            "observation": "start",
                            "memory_before": "",
                            "raw_response": "THINK: stall\nACTION: look",
                            "action": "look",
                        }
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            pairs = load_branch_contrast_examples(path, weight=0.5, max_pairs=0)

        self.assertEqual(len(pairs), 1)
        self.assertIn("take gem", pairs[0].preferred)
        self.assertIn("look", pairs[0].rejected)
        self.assertEqual(pairs[0].weight, 0.5)

    def test_memory_policy_items_use_future_return_after_boundary(self) -> None:
        trace = EpisodeTrace(
            episode_id=0,
            mode="blimp",
            game_file=None,
            solved=True,
            score=1.0,
            total_reward=6.0,
            actions=["a0", "a1", "a2"],
            memories=["remember next"],
            transitions=[
                make_transition(0, reward=1.0),
                make_transition(1, reward=2.0),
                make_transition(2, reward=3.0, won=True),
            ],
            memory_writes=[
                MemoryWrite(
                    prompt="memory prompt",
                    completion="remember next",
                    after_step=1,
                ),
                MemoryWrite(
                    prompt="terminal memory prompt",
                    completion="unused",
                    after_step=3,
                ),
            ],
        )

        returns = discounted_returns(trace.transitions, gamma=1.0)
        items = memory_policy_items(trace, returns)

        self.assertEqual(returns, [6.0, 5.0, 3.0])
        self.assertEqual(items[0], ("memory prompt", "remember next", 5.0))
        self.assertEqual(items[1], ("terminal memory prompt", "unused", 0.0))


if __name__ == "__main__":
    unittest.main()
