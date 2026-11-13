"""Small NumPy-only helpers for explicitly declared dual-arm TCP layouts."""

import numpy as np


def validate_quaternion_order(order):
    if type(order) is not str or order not in ("wxyz", "xyzw"):
        raise ValueError("TCP quaternion order must be 'wxyz' or 'xyzw'")
    return order


def pose_layout(order):
    order = validate_quaternion_order(order)
    components = ("x", "y", "z", *("q" + axis for axis in order))
    return [f"{arm}_{component}" for arm in ("left", "right") for component in components]


def tcp_array_name(order):
    return "tcp_pose_world_" + validate_quaternion_order(order)


def reorder_tcp_pose(value, source_order, target_order):
    """Permute the two quaternion blocks without geometry or numeric changes."""
    source_order = validate_quaternion_order(source_order)
    target_order = validate_quaternion_order(target_order)
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[-1] != 14:
        raise ValueError("Dual-arm TCP poses must have a final dimension of 14")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("TCP poses must contain finite numeric values")
    indices = []
    for offset in (0, 7):
        indices.extend((offset, offset + 1, offset + 2))
        indices.extend(offset + 3 + source_order.index(axis) for axis in target_order)
    return np.take(array, indices, axis=-1)
