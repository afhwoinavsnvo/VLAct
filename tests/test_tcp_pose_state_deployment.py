"""Pose-state deployment must preserve coordinates and keep joint actions."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.state_normalization import RoboTwinStateNormalizer
from examples.Robotwin.eval_files import model2robotwin_interface as interface


def config():
    return {
        "framework": {"name": "QwenOFT", "action_model": {"future_action_window_size": 31},
                      "conditioning": {"enabled": True, "include_state": True, "include_tcp": True,
                                       "state_dim": 14, "tcp_dim": 14, "state_input_space": "raw",
                                       "state_representation": "tcp_pose"}},
        "datasets": {"vla_data": {"data_mix": "robotwin_tcp_state_wrap_32", "state_representation": "tcp_pose"}},
    }


def pose():
    return np.array([-.3, -.2, .9, 1., 0., 0., 0., .3, -.2, .9, 1., 0., 0., 0.], dtype=np.float32)


def statistics():
    state = {name: [value] * 14 for name, value in {
        "min": -10., "max": 10., "mean": 0., "std": 2., "q01": -9., "q99": 9.,
    }.items()}
    return {"new_embodiment": {"state": state,
            "action": {"min": [-10.] * 14, "max": [10.] * 14, "mask": [True] * 12 + [False, False]}}}


class TCPPoseStateDeploymentTests(unittest.TestCase):
    def test_pose_transform_preserves_all_14_values_without_mutating_input(self):
        raw = pose()
        before = raw.copy()
        normalizer = RoboTwinStateNormalizer(config()["datasets"]["vla_data"], statistics()["new_embodiment"])
        actual = normalizer.apply_state(raw)
        np.testing.assert_array_equal(actual, raw[None])
        np.testing.assert_array_equal(raw, before)
        self.assertEqual(actual.dtype, np.float32)

    def test_invalid_quaternions_cannot_silently_enter_pose_state(self):
        normalizer = RoboTwinStateNormalizer(config()["datasets"]["vla_data"], statistics()["new_embodiment"])
        with self.assertRaisesRegex(ValueError, "unit wxyz"):
            normalizer.apply_state(np.zeros(14, dtype=np.float32))

    def test_server_advertises_raw_pose_state_and_preserves_future_target(self):
        policy = SimpleNamespace(config=config(), norm_stats=statistics(), chunk_len=32, predict_action=Mock())
        wrapper = PolicyServerWrapper(policy)
        current = pose()
        future = pose()
        future[[0, 7]] += .2
        wrapper.predict_action([{"state": current, "target_tcp_pose": future, "image": [], "lang": "Move"}])
        supplied = policy.predict_action.call_args.kwargs["examples"][0]
        np.testing.assert_array_equal(supplied["state"], current[None])
        self.assertIs(supplied["target_tcp_pose"], future)
        self.assertEqual(wrapper.metadata["state_representation"], "tcp_pose")
        self.assertEqual(wrapper.metadata["model_state_space"], "raw_tcp_pose")
        self.assertEqual(wrapper.metadata["state_normalization"], "none")

    def test_model_and_dataset_state_contracts_cannot_disagree(self):
        model_config = config()
        model_config["datasets"]["vla_data"]["state_representation"] = "joint"
        with self.assertRaisesRegex(ValueError, "state_representation disagree"):
            PolicyServerWrapper(SimpleNamespace(config=model_config, norm_stats=statistics()))

    def test_new_data_mix_still_decodes_joint_actions_with_original_wrap(self):
        websocket = SimpleNamespace(predict_action=Mock(return_value={
            "data": {"normalized_actions": np.full((1, 32, 14), 2.4, dtype=np.float32)},
        }))
        with patch.object(interface, "read_mode_config", return_value=(config(), statistics())), \
                patch.object(interface, "WebsocketClientPolicy", return_value=websocket):
            client = interface.ModelClient("/unused/checkpoint.pt")
        sample = {"lang": "Move", "image": [np.zeros((2, 2, 3), dtype=np.uint8)],
                  "state": pose(), "target_tcp_pose": pose()}
        action = client.step(sample)
        self.assertEqual(client.joint_action_unnorm_mode, "wrap")
        self.assertEqual(client.state_representation, "tcp_pose")
        np.testing.assert_allclose(action[:6], 2.4, atol=2e-6, rtol=0)

    @staticmethod
    def observation():
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return {"observation": {name: {"rgb": frame} for name in ("head_camera", "left_camera", "right_camera")},
                "target_tcp_pose": pose()}

    def test_eval_uses_explicit_current_pose_without_requiring_joint_state(self):
        observation = self.observation()
        observation["current_tcp_pose"] = pose()
        model = SimpleNamespace(conditioning_enabled=True, include_state=True, state_representation="tcp_pose",
                                step=Mock(return_value=np.zeros(14)))
        task = SimpleNamespace(get_instruction=lambda: "Move", take_action_cnt=0, take_action=Mock())
        interface.eval(task, model, observation)
        supplied = model.step.call_args.args[0]
        np.testing.assert_array_equal(supplied["state"], observation["current_tcp_pose"])
        self.assertIs(supplied["target_tcp_pose"], observation["target_tcp_pose"])

    def test_eval_converts_current_ee_to_tcp_with_the_original_tool_offset(self):
        observation = self.observation()
        q = np.sqrt(.5)
        observation["endpose"] = {"left_endpose": [-.3, -.2, .9, 1., 0., 0., 0.],
                                  "right_endpose": [.3, -.2, .9, q, 0., 0., q]}
        actual = interface._current_tcp_pose(observation)
        np.testing.assert_allclose(actual[:3], [-.18, -.2, .9], atol=1e-7, rtol=0)
        np.testing.assert_allclose(actual[7:10], [.3, -.08, .9], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(actual[3:7], [1., 0., 0., 0.])
        self.assertEqual(actual.dtype, np.float32)

    def test_future_tcp_target_is_never_used_as_the_current_state(self):
        with self.assertRaisesRegex(ValueError, "current_tcp_pose or current"):
            interface._current_tcp_pose(self.observation())


if __name__ == "__main__":
    unittest.main()
