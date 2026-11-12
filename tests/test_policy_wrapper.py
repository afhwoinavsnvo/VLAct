"""CPU deployment-contract tests; no model weights, server, or simulator."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.state_normalization import RoboTwinStateNormalizer


def model_config(state=True, tcp=True):
    return {
        "framework": {
            "name": "QwenOFT",
            "action_model": {"past_action_window_size": 0, "future_action_window_size": 31},
            "conditioning": {
                "enabled": True, "include_state": state, "include_tcp": tcp,
                "state_dim": 14, "tcp_dim": 14, "state_input_space": "raw",
            },
        },
        "datasets": {"vla_data": {"data_mix": "robotwin_wrap_32"}},
    }


def statistics():
    state = {key: [value] * 12 + [0.0, 0.0] for key, value in {
        "min": -2.0, "max": 6.0, "mean": 2.0, "std": 1.0, "q01": -100.0, "q99": 100.0,
    }.items()}
    for key in ("max", "q99"):
        state[key][-2:] = [1.0, 1.0]
    return {"new_embodiment": {"state": state}}


class CapturePolicy:
    def __init__(self, config):
        self.config = config
        self.norm_stats = statistics()
        self.chunk_len = 32
        self.result = {"normalized_actions": np.full((1, 32, 14), 2.4, dtype=np.float32)}

    def predict_action(self, *, examples, **kwargs):
        self.examples, self.options = examples, kwargs
        return self.result

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class PolicyWrapperTests(unittest.TestCase):
    @staticmethod
    def example():
        state = np.full((1, 14), 2.0, dtype=np.float32)
        state[:, -2:] = [0.48, 0.5]
        return {"lang": "Adjust the bottle", "image": [object()], "state": state,
                "target_tcp_pose": np.linspace(-0.2, 0.8, 14, dtype=np.float32)}

    def test_four_independent_switches_preserve_vlact_action_wire_format(self):
        for state in (False, True):
            for tcp in (False, True):
                with self.subTest(state=state, tcp=tcp):
                    policy = CapturePolicy(model_config(state, tcp))
                    wrapper = PolicyServerWrapper(policy)
                    original = self.example()
                    before = original["state"].copy()
                    result = wrapper.predict_action([original], do_sample=False)
                    self.assertIs(result, policy.result)
                    self.assertEqual(set(result), {"normalized_actions"})
                    self.assertEqual(float(result["normalized_actions"][0, 0, 0]), np.float32(2.4))
                    forwarded = policy.examples[0]
                    self.assertEqual("state" in forwarded, state)
                    self.assertEqual("target_tcp_pose" in forwarded, tcp)
                    self.assertEqual(forwarded["lang"], original["lang"])
                    self.assertIs(forwarded["image"], original["image"])
                    if state:
                        np.testing.assert_array_equal(forwarded["state"][:, :12], np.zeros((1, 12)))
                        np.testing.assert_array_equal(forwarded["state"][:, 12:], [[0.0, 1.0]])
                    if tcp:
                        self.assertIs(forwarded["target_tcp_pose"], original["target_tcp_pose"])
                    np.testing.assert_array_equal(original["state"], before)

    def test_both_target_aliases_are_canonicalized_without_normalizing_tcp(self):
        for alias in ("tcp_pose", "target_pose"):
            with self.subTest(alias=alias):
                policy = CapturePolicy(model_config(False, True))
                sample = self.example()
                goal = sample.pop("target_tcp_pose")
                sample[alias] = goal
                PolicyServerWrapper(policy).predict_action([sample])
                self.assertIs(policy.examples[0]["target_tcp_pose"], goal)
                self.assertNotIn(alias, policy.examples[0])
                self.assertIn(alias, sample)

    def test_conflicting_tcp_aliases_fail_before_inference(self):
        sample = self.example()
        sample["tcp_pose"] = sample["target_tcp_pose"] + 0.01
        policy = CapturePolicy(model_config(False, True))
        with self.assertRaisesRegex(ValueError, "conflicting TCP pose aliases"):
            PolicyServerWrapper(policy).predict_action([sample])
        self.assertFalse(hasattr(policy, "examples"))

    def test_disabled_inputs_are_not_required_or_interpreted(self):
        policy = CapturePolicy(model_config(False, False))
        wrapper = PolicyServerWrapper(policy)
        for sample in ({"lang": "Move", "image": []}, {
            "lang": "Move", "image": [], "state": object(), "target_tcp_pose": object(),
            "tcp_pose": object(), "target_pose": object(),
        }):
            wrapper.predict_action([sample])
            self.assertEqual(set(policy.examples[0]), {"lang", "image"})
        self.assertFalse(wrapper._normalizers)

    def test_normalized_state_option_bypasses_raw_state_normalization(self):
        config = model_config()
        config["framework"]["conditioning"]["state_input_space"] = "normalized"
        policy = CapturePolicy(config)
        sample = self.example()
        wrapper = PolicyServerWrapper(policy)
        wrapper.predict_action([sample])
        self.assertIs(policy.examples[0]["state"], sample["state"])
        self.assertFalse(wrapper._normalizers)
        self.assertEqual(wrapper.metadata["state_input_space"], "normalized")
        self.assertEqual(wrapper.metadata["model_state_space"], "normalized")

    def test_legacy_oft_and_other_frameworks_preserve_exact_request_objects(self):
        for mode in ("old_oft", "disabled", "pi"):
            with self.subTest(mode=mode):
                config = model_config(False, False)
                if mode == "old_oft":
                    del config["framework"]["conditioning"]
                elif mode == "disabled":
                    config["framework"]["conditioning"]["enabled"] = False
                else:
                    config["framework"]["name"] = "QwenPI"
                policy = CapturePolicy(config)
                examples = [self.example()]
                wrapper = PolicyServerWrapper(policy)
                self.assertIs(wrapper.predict_action(examples), policy.result)
                self.assertIs(policy.examples, examples)
                self.assertFalse(wrapper.metadata["conditioning_enabled"])

    def test_switches_require_real_booleans(self):
        for flag in ("enabled", "include_state", "include_tcp"):
            with self.subTest(flag=flag):
                config = model_config()
                config["framework"]["conditioning"][flag] = "False"
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    PolicyServerWrapper(CapturePolicy(config))

    def test_metadata_uses_saved_tcp_coordinate_contract_without_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            contract = {
                "coordinate_frame": "world", "position_unit": "meters", "quaternion_order": "wxyz",
                "layout": [f"pose_{i}" for i in range(14)],
                "tcp_definition": "p_TCP=p_EE+R(q_wxyz)@[0.12,0,0]; q_TCP=q_EE",
                "raw_source": "/private/source/hdf5",
            }
            (run / "target_tcp_contract.json").write_text(json.dumps({"pose_contract": contract, "data_root": "/private/data"}))
            policy = CapturePolicy(model_config())
            wrapper = PolicyServerWrapper(policy, checkpoint_path=run / "checkpoints/model.pt")
            metadata = wrapper.metadata
            self.assertEqual(metadata["tcp_pose_key"], "target_tcp_pose")
            self.assertEqual(metadata["tcp_pose_aliases"], ["tcp_pose", "target_pose"])
            self.assertEqual(metadata["tcp_pose_contract"]["coordinate_frame"], "world")
            self.assertIn("0.12", metadata["tcp_pose_contract"]["tcp_definition"])
            self.assertEqual(metadata["action_decoding"], "client")
            self.assertEqual(metadata["action_chunk_size"], 32)
            self.assertNotIn("/private", json.dumps(metadata))
            self.assertNotIn(directory, json.dumps(metadata))

    def test_dataset_manifest_provides_tcp_contract_before_a_run_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "adjust_bottle/meta"
            meta.mkdir(parents=True)
            (root / "preparation_manifest.json").write_text(json.dumps({"status": "passed", "tasks": [{"task": "adjust_bottle"}]}))
            (meta / "tcp_pose.json").write_text(json.dumps({"coordinate_frame": "world", "position_unit": "meters", "quaternion_order": "wxyz"}))
            config = model_config(False, True)
            config["datasets"]["vla_data"]["data_root_dir"] = directory
            wrapper = PolicyServerWrapper(CapturePolicy(config))
            self.assertEqual(wrapper.metadata["tcp_pose_contract"]["quaternion_order"], "wxyz")
            self.assertNotIn(directory, json.dumps(wrapper.metadata))

    def test_state_normalizer_matches_training_transform_and_does_not_mutate_statistics(self):
        stats = statistics()["new_embodiment"]
        before = deepcopy(stats)
        normalizer = RoboTwinStateNormalizer(model_config()["datasets"]["vla_data"], stats)
        raw = np.asarray([-2, 0, 2, 4, 6, -1] * 2 + [0.48, 0.5], dtype=np.float32)
        actual = normalizer.apply_state(raw)
        expected = np.concatenate(((raw[:12] + 2) / 4 - 1, [0.0, 1.0]))[None]
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
        self.assertEqual(stats, before)

    def test_server_main_installs_wrapper_metadata_without_starting_real_transport(self):
        from deployment.model_server import server_policy

        policy = CapturePolicy(model_config(False, False))
        args = SimpleNamespace(ckpt_path="/unused/model.pt", use_bf16=False, port=5694, idle_timeout=10)
        with (
            mock.patch.object(server_policy, "PartialState"),
            mock.patch.object(server_policy.baseframework, "from_pretrained", return_value=policy),
            mock.patch.object(server_policy.socket, "gethostname", return_value="test"),
            mock.patch.object(server_policy.socket, "gethostbyname", return_value="127.0.0.1"),
            mock.patch.object(server_policy, "WebsocketPolicyServer") as server,
        ):
            server_policy.main(args)
        self.assertIsInstance(server.call_args.kwargs["policy"], PolicyServerWrapper)
        self.assertEqual(server.call_args.kwargs["metadata"]["action_output_key"], "normalized_actions")
        server.return_value.serve_forever.assert_called_once()


if __name__ == "__main__":
    unittest.main()
