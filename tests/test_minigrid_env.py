from __future__ import annotations

from types import SimpleNamespace
import unittest

from blimp.envs import MiniGridTextEnv, make_env


class MiniGridTextEnvTest(unittest.TestCase):
    def test_constant_mapping_falls_back_to_forward_map(self) -> None:
        module = SimpleNamespace(STATE_TO_IDX={"open": 0, "closed": 1, "locked": 2})

        self.assertEqual(
            MiniGridTextEnv._idx_mapping(
                module, idx_name="IDX_TO_STATE", forward_name="STATE_TO_IDX"
            ),
            {0: "open", 1: "closed", 2: "locked"},
        )

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
        self.assertNotIn("done", env.valid_actions())

        result = env.step("turn left")
        self.assertTrue(result.info["valid"])
        self.assertIn("Facing:", result.observation)


if __name__ == "__main__":
    unittest.main()
