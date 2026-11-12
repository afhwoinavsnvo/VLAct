"""Mocked RoboTwin client checks; no websocket connection or simulator."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from examples.Robotwin.eval_files import model2robotwin_interface as interface


def config(state=True, tcp=True, *, enabled=True):
    return {
        "framework": {"name": "QwenOFT", "action_model": {"future_action_window_size": 31},
                      "conditioning": {"enabled": enabled, "include_state": state, "include_tcp": tcp}},
        "datasets": {"vla_data": {"data_mix": "robotwin_wrap_32"}},
    }


def stats():
    state = {key: [value] * 12 + [0.0, 0.0] for key, value in {
        "min": -2.0, "max": 6.0, "mean": 2.0, "std": 1.0, "q01": -50.0, "q99": 50.0,
    }.items()}
    state["max"][-2:] = [1.0, 1.0]
    state["q99"][-2:] = [1.0, 1.0]
    return {"new_embodiment": {"state": state, "action": {
        "min": [-10.0] * 14, "max": [10.0] * 14, "mask": [True] * 12 + [False, False],
    }}}


class FakeWebsocket:
    def predict_action(self, request):
        self.request = request
        return {"data": {"normalized_actions": np.full((1, 32, 14), 2.4, dtype=np.float32)}}


class RobotwinConditioningClientTests(unittest.TestCase):
    def make_client(self, model_config):
        fake = FakeWebsocket()
        with (
            mock.patch.object(interface, "read_mode_config", return_value=(model_config, stats())),
            mock.patch.object(interface, "WebsocketClientPolicy", return_value=fake),
        ):
            client = interface.ModelClient("/unused/checkpoint.pt")
        return client, fake

    @staticmethod
    def example():
        return {"lang": "Move the bottle", "image": [np.zeros((3, 4, 3), dtype=np.uint8)] * 3,
                "state": np.arange(14, dtype=np.float32)[None],
                "target_tcp_pose": np.linspace(-0.5, 0.8, 14, dtype=np.float32)}

    def test_new_state_and_goal_are_forwarded_in_original_units_with_linear_resize(self):
        client, websocket = self.make_client(config())
        sample = self.example()
        with mock.patch.object(interface.cv, "resize", side_effect=lambda image, size, interpolation: image) as resize:
            action = client.step(sample)
        forwarded = websocket.request["examples"][0]
        self.assertIs(forwarded["state"], sample["state"])
        self.assertIs(forwarded["target_tcp_pose"], sample["target_tcp_pose"])
        self.assertTrue(all(call.kwargs["interpolation"] == interface.cv.INTER_LINEAR for call in resize.call_args_list))
        self.assertEqual(websocket.request["unnorm_key"], "new_embodiment")
        self.assertEqual(client.joint_action_unnorm_mode, "wrap")
        np.testing.assert_allclose(action[:6], 2.4, atol=2e-6, rtol=0)

    def test_new_switches_are_independent_and_missing_state_is_safe_when_disabled(self):
        for state in (False, True):
            for tcp in (False, True):
                with self.subTest(state=state, tcp=tcp):
                    client, websocket = self.make_client(config(state, tcp))
                    sample = self.example()
                    if not state:
                        sample.pop("state")
                    if not tcp:
                        sample.pop("target_tcp_pose")
                    client.step(sample)
                    forwarded = websocket.request["examples"][0]
                    self.assertEqual("state" in forwarded, state)
                    self.assertEqual("target_tcp_pose" in forwarded, tcp)

    def test_legacy_client_still_drops_state_and_uses_area_resize(self):
        client, websocket = self.make_client(config(enabled=False))
        sample = self.example()
        with mock.patch.object(interface.cv, "resize", side_effect=lambda image, size, interpolation: image) as resize:
            client.step(sample)
        self.assertNotIn("state", websocket.request["examples"][0])
        self.assertTrue(all(call.kwargs["interpolation"] == interface.cv.INTER_AREA for call in resize.call_args_list))
        self.assertNotIn("unnorm_key", websocket.request)
        self.assertEqual(client.joint_action_unnorm_mode, "stats")

    def test_missing_external_goal_fails_without_substituting_current_tcp(self):
        client, websocket = self.make_client(config())
        sample = self.example()
        sample.pop("target_tcp_pose")
        sample["current_tcp_pose"] = np.zeros(14, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "externally supplied future target_tcp_pose"):
            client.step(sample)
        self.assertFalse(hasattr(websocket, "request"))

    def test_pose_aliases_are_forwarded_unchanged_for_server_canonicalization(self):
        for key in ("tcp_pose", "target_pose"):
            with self.subTest(key=key):
                client, websocket = self.make_client(config(False, True))
                sample = self.example()
                goal = sample.pop("target_tcp_pose")
                sample[key] = goal
                client.step(sample)
                self.assertIs(websocket.request["examples"][0][key], goal)

    def test_eval_reorders_state_only_for_the_new_state_channel(self):
        for enabled, include_state in ((False, False), (True, False), (True, True)):
            with self.subTest(enabled=enabled, include_state=include_state):
                frame = np.zeros((2, 2, 3), dtype=np.uint8)
                native = np.arange(14, dtype=np.float32)
                goal = np.full(14, 0.25, dtype=np.float32)
                observation = {"observation": {key: {"rgb": frame} for key in ("head_camera", "left_camera", "right_camera")},
                               "joint_action": {"vector": native}, "target_tcp_pose": goal}
                model = SimpleNamespace(conditioning_enabled=enabled, include_state=include_state, step=mock.Mock(return_value=np.zeros(14)))
                task = SimpleNamespace(get_instruction=lambda: "Move", take_action_cnt=0, take_action=mock.Mock())
                interface.eval(task, model, observation)
                forwarded = model.step.call_args.args[0]
                expected = native[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]] if enabled and include_state else native
                np.testing.assert_array_equal(forwarded["state"], expected)
                self.assertIs(forwarded["target_tcp_pose"], goal)

    def test_normalized_server_option_uses_the_exact_training_state_adapter(self):
        model_config = config()
        model_config["framework"]["conditioning"]["state_input_space"] = "normalized"
        client, websocket = self.make_client(model_config)
        sample = self.example()
        sample["state"] = np.full((1, 14), 2.0, dtype=np.float32)
        original = sample["state"].copy()
        client.step(sample)
        forwarded = websocket.request["examples"][0]
        np.testing.assert_array_equal(forwarded["state"][:, :12], np.zeros((1, 12)))
        np.testing.assert_array_equal(sample["state"], original)


if __name__ == "__main__":
    unittest.main()
