"""Non-symmetric quaternions expose missed or repeated order conversions."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from examples.Robotwin.eval_files import model2robotwin_interface as interface


P = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 11, 12, 13, 10]


def wxyz_pose():
    a = np.array([1., 2., 3., 4.]) / np.sqrt(30.)
    b = np.array([-.5, .5, .5, .5])
    return np.concatenate(([-.3, -.2, .9], a, [.3, -.2, .9], b)).astype(np.float32)


def config():
    return {
        'framework': {'name': 'QwenOFT', 'action_model': {'future_action_window_size': 31},
                      'conditioning': {'enabled': True, 'include_state': True, 'include_tcp': True,
                                       'state_representation': 'tcp_pose', 'state_input_space': 'raw',
                                       'tcp_quaternion_order': 'xyzw'}},
        'datasets': {'vla_data': {'data_mix': 'robotwin_tcp_state_wrap_32',
                                  'state_representation': 'tcp_pose', 'tcp_quaternion_order': 'xyzw'}},
    }


def statistics():
    values = {key: [value] * 14 for key, value in {
        'min': -10., 'max': 10., 'mean': 0., 'std': 2., 'q01': -9., 'q99': 9.,
    }.items()}
    return {'new_embodiment': {'state': values,
            'action': {'min': [-10.] * 14, 'max': [10.] * 14, 'mask': [True] * 12 + [False, False]}}}


class TCPXYZWDeploymentTests(unittest.TestCase):
    def test_server_keeps_state_and_goal_in_declared_xyzw_order(self):
        model = SimpleNamespace(config=config(), norm_stats=statistics(), chunk_len=32, predict_action=Mock())
        wrapper = PolicyServerWrapper(model)
        state = wxyz_pose()[P]
        goal = state.copy()
        goal[[0, 7]] += .2
        wrapper.predict_action([{'state': state, 'target_tcp_pose': goal, 'tcp_quaternion_order': 'xyzw'}])
        forwarded = model.predict_action.call_args.kwargs['examples'][0]
        np.testing.assert_array_equal(forwarded['state'], state[None])
        self.assertIs(forwarded['target_tcp_pose'], goal)
        self.assertEqual(wrapper.metadata['state_quaternion_order'], 'xyzw')
        self.assertEqual(wrapper.metadata['tcp_quaternion_order'], 'xyzw')

    def test_request_declaring_wrong_order_is_rejected_before_model(self):
        model = SimpleNamespace(config=config(), norm_stats=statistics(), predict_action=Mock())
        with self.assertRaisesRegex(ValueError, 'requires TCP quaternion order xyzw'):
            PolicyServerWrapper(model).predict_action([{'state': wxyz_pose(), 'target_tcp_pose': wxyz_pose(),
                                                        'tcp_quaternion_order': 'wxyz'}])
        model.predict_action.assert_not_called()

    def test_configured_model_order_must_match_dataset_and_saved_contract(self):
        cfg = config()
        cfg['datasets']['vla_data']['tcp_quaternion_order'] = 'wxyz'
        with self.assertRaisesRegex(ValueError, 'quaternion orders disagree'):
            PolicyServerWrapper(SimpleNamespace(config=cfg, norm_stats=statistics()))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'target_tcp_contract.json').write_text(json.dumps({'pose_contract': {'quaternion_order': 'wxyz'}}))
            with self.assertRaisesRegex(ValueError, 'Saved TCP contract'):
                PolicyServerWrapper(SimpleNamespace(config=config(), norm_stats=statistics()),
                                    checkpoint_path=root / 'final_model/model.pt')

    def test_explicit_current_pose_converts_only_when_source_order_is_declared(self):
        source = wxyz_pose()
        observation = {'current_tcp_pose': source, 'current_tcp_quaternion_order': 'wxyz'}
        actual = interface._current_tcp_pose(observation, 'xyzw')
        np.testing.assert_array_equal(actual, source[P])
        np.testing.assert_array_equal(source, wxyz_pose())
        np.testing.assert_array_equal(interface._current_tcp_pose({'current_tcp_pose': actual}, 'xyzw'), actual)

    def test_robotwin_raw_ee_is_interpreted_as_wxyz_then_sent_as_xyzw(self):
        source = wxyz_pose()
        observation = {'endpose': {'left_endpose': source[:7], 'right_endpose': source[7:]}}
        original = interface._current_tcp_pose(observation, 'wxyz')
        converted = interface._current_tcp_pose(observation, 'xyzw')
        np.testing.assert_array_equal(converted, original[P])
        np.testing.assert_array_equal(converted[[0, 1, 2, 7, 8, 9]], original[[0, 1, 2, 7, 8, 9]])

    def test_client_forwards_xyzw_inputs_with_marker_and_preserves_joint_actions(self):
        websocket = SimpleNamespace(predict_action=Mock(return_value={
            'data': {'normalized_actions': np.full((1, 32, 14), 2.4, dtype=np.float32)}}))
        with patch.object(interface, 'read_mode_config', return_value=(config(), statistics())), \
                patch.object(interface, 'WebsocketClientPolicy', return_value=websocket):
            client = interface.ModelClient('/unused/checkpoint.pt')
        state, goal = wxyz_pose()[P], wxyz_pose()[P]
        sample = {'lang': 'Move', 'image': [np.zeros((2, 2, 3), dtype=np.uint8)],
                  'state': state, 'target_tcp_pose': goal}
        action = client.step(sample)
        forwarded = websocket.predict_action.call_args.args[0]['examples'][0]
        np.testing.assert_array_equal(forwarded['state'], state)
        self.assertIs(forwarded['target_tcp_pose'], goal)
        self.assertEqual(forwarded['tcp_quaternion_order'], 'xyzw')
        np.testing.assert_allclose(action[:6], 2.4, rtol=0, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
