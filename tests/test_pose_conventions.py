import unittest

import numpy as np

from deployment.model_server.pose_conventions import pose_layout, reorder_tcp_pose, tcp_array_name


class PoseConventionTests(unittest.TestCase):
    def test_non_symmetric_pose_has_exact_requested_component_order(self):
        value = np.arange(14, dtype=np.float32)
        result = reorder_tcp_pose(value, 'wxyz', 'xyzw')
        np.testing.assert_array_equal(result, value[[0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 11, 12, 13, 10]])
        np.testing.assert_array_equal(reorder_tcp_pose(result, 'xyzw', 'wxyz'), value)
        self.assertEqual(result.dtype, value.dtype)

    def test_batched_conversion_is_reversible_and_never_mutates_input(self):
        value = np.arange(4 * 32 * 14, dtype=np.float32).reshape(4, 32, 14)
        before = value.copy()
        converted = reorder_tcp_pose(value, 'wxyz', 'xyzw')
        np.testing.assert_array_equal(reorder_tcp_pose(converted, 'xyzw', 'wxyz'), value)
        np.testing.assert_array_equal(value, before)
        self.assertFalse(np.shares_memory(value, converted))

    def test_layout_and_sidecar_key_are_explicit(self):
        self.assertEqual(pose_layout('xyzw')[:7], ['left_x', 'left_y', 'left_z', 'left_qx', 'left_qy', 'left_qz', 'left_qw'])
        self.assertEqual(tcp_array_name('xyzw'), 'tcp_pose_world_xyzw')
        self.assertEqual(tcp_array_name('wxyz'), 'tcp_pose_world_wxyz')

    def test_invalid_shapes_values_and_undeclared_orders_are_rejected(self):
        for order in ('zwxy', '', None, True):
            with self.assertRaises(ValueError):
                pose_layout(order)
        for value in (np.zeros(7), np.zeros((2, 13)), np.full(14, np.nan)):
            with self.assertRaises(ValueError):
                reorder_tcp_pose(value, 'wxyz', 'xyzw')


if __name__ == '__main__':
    unittest.main()
