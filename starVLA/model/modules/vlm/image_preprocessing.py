"""Shared image sizes for numeric loaders and policy inference.

Bucket selection and OpenCV interpolation match VLAct commit
4c41b666228b70d8ae166bb65ac8f6b392ea7919, ``VideoResizeBucketedSizes``.
This module imports neither torch nor the VLM; OpenCV/NumPy load only when a
bucket resize is needed. All size pairs use PIL's (width, height) order.
"""

from __future__ import annotations

import ast
from numbers import Integral

from PIL import Image


def _size_pair(size, *, name: str) -> tuple[int, int]:
    try:
        values = tuple(size)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two positive integers (width, height)") from exc
    if len(values) != 2 or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain two positive integers (width, height)")
    return int(values[0]), int(values[1])


def _resolve_buckets(raw_buckets):
    if raw_buckets is None:
        return None
    if isinstance(raw_buckets, str):
        try:
            raw_buckets = ast.literal_eval(raw_buckets)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("image_size_buckets must be a non-empty sequence of size pairs") from exc
    try:
        buckets = tuple(_size_pair(size, name="image_size_buckets entry") for size in raw_buckets)
    except TypeError as exc:
        raise ValueError("image_size_buckets must be a non-empty sequence of size pairs") from exc
    if not buckets:
        raise ValueError("image_size_buckets must be a non-empty sequence of size pairs")
    return buckets


def resize_image_for_training(image, *, image_size_buckets=None, image_size=224) -> Image.Image:
    """Apply the training size contract to one PIL image or RGB array.

    Buckets choose the nearest aspect ratio, retaining list order for ties,
    then resize directly with OpenCV INTER_LINEAR (without crop or padding).
    Without buckets the existing PIL default resize is retained. An explicit
    ``image_size=None`` leaves the original dimensions unchanged.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buckets = _resolve_buckets(image_size_buckets)
    if buckets is not None:
        width, height = image.size
        aspect_ratio = width / float(height)
        target = min(buckets, key=lambda size: abs(size[0] / float(size[1]) - aspect_ratio))
        if image.size == target:
            return image
        import cv2
        import numpy as np

        resized = cv2.resize(np.asarray(image), target, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(resized)
    if image_size is None:
        return image
    if isinstance(image_size, Integral) and not isinstance(image_size, bool):
        image_size = (image_size, image_size)
    target = _size_pair(image_size, name="image_size")
    return image.resize(target)


def preprocess_image_batch(images, *, image_size_buckets=None, image_size=224):
    """Preserve batch/view nesting while applying the same per-image resize."""
    if isinstance(images, list):
        return [preprocess_image_batch(image, image_size_buckets=image_size_buckets, image_size=image_size)
                for image in images]
    if isinstance(images, tuple):
        return tuple(preprocess_image_batch(image, image_size_buckets=image_size_buckets, image_size=image_size)
                     for image in images)
    return resize_image_for_training(images, image_size_buckets=image_size_buckets, image_size=image_size)
