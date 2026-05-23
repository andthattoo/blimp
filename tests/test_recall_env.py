from __future__ import annotations

import unittest

from blimp.envs import RecallPassphraseEnv


class RecallPassphraseEnvTest(unittest.TestCase):
    def test_passphrase_is_only_needed_after_delay(self) -> None:
        env = RecallPassphraseEnv()
        env.reset(seed=7)
        passphrase = env.passphrase

        for _ in range(env.CORRIDOR_STEPS):
            result = env.step("go forward")
            self.assertFalse(result.done)

        self.assertIn(f"say {passphrase}", env.valid_actions())
        result = env.step(f"say {passphrase}")

        self.assertTrue(result.done)
        self.assertEqual(result.info["score"], 1.0)
        self.assertTrue(result.info["won"])

    def test_wrong_passphrase_fails(self) -> None:
        env = RecallPassphraseEnv()
        env.reset(seed=3)
        wrong = next(word for word in env.PASS_OPTIONS if word != env.passphrase)

        for _ in range(env.CORRIDOR_STEPS):
            env.step("go forward")
        result = env.step(f"say {wrong}")

        self.assertTrue(result.done)
        self.assertEqual(result.info["score"], 0.0)
        self.assertFalse(result.info["won"])


if __name__ == "__main__":
    unittest.main()
