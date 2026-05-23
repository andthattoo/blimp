from __future__ import annotations

import unittest

from blimp.envs import make_env


class MiniGridTextEnvTest(unittest.TestCase):
    def test_minigrid_wrapper_smoke(self) -> None:
        try:
            env = make_env("minigrid", "MiniGrid-MemoryS7-v0")
            obs = env.reset(seed=0)
        except RuntimeError as exc:
            if "MiniGrid is not installed" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertIn("Mission:", obs)
        self.assertIn("Local egocentric view", obs)
        self.assertIn("move forward", env.valid_actions())

        result = env.step("turn left")
        self.assertTrue(result.info["valid"])
        self.assertIn("Facing:", result.observation)


if __name__ == "__main__":
    unittest.main()
