"""Decoder threading changes must preserve selected frames and RGB pixels."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader import lerobot_datasets
from starVLA.dataloader.gr00t_lerobot import video
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class PyAVDecoderOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "frames.mp4"
        rng = np.random.default_rng(42)
        with av.open(str(cls.path), "w") as container:
            stream = container.add_stream("mpeg4", rate=15)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            stream.codec_context.gop_size = 3
            for _ in range(12):
                frame = av.VideoFrame.from_ndarray(
                    rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), format="rgb24",
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_default_does_not_change_existing_codec_configuration(self):
        stream = SimpleNamespace(codec_context=SimpleNamespace(thread_count=7))
        video._configure_pyav_decoder(stream, {})
        self.assertEqual(stream.codec_context.thread_count, 7)
        for count in (0, 1, 4):
            video._configure_pyav_decoder(stream, {"thread_count": count})
            self.assertEqual(stream.codec_context.thread_count, count)

    def test_invalid_thread_options_are_rejected(self):
        stream = SimpleNamespace(codec_context=SimpleNamespace(thread_count=0))
        for value in (True, False, "1", 1.5, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                video._configure_pyav_decoder(stream, {"thread_count": value})
        with self.assertRaises(ValueError):
            video._configure_pyav_decoder(stream, {"thread_count": -1})
        with self.assertRaises(TypeError):
            video._configure_pyav_decoder(stream, [])

    def test_all_three_apis_preserve_frame_selection_and_pixels(self):
        indices = [11, 0, 5, 5, 1, 8]
        timestamps = [index / 15 for index in indices]
        # Include a timestamp between frames to exercise nearest-frame matching.
        timestamps.append(2.4 / 15)
        queries = [
            (video.get_frames_by_indices, {"indices": indices}),
            (video.get_frames_by_timestamps, {"timestamps": timestamps}),
            (video.get_all_frames, {}),
            (video.get_all_frames, {"resize_size": (32, 24)}),
        ]
        for function, kwargs in queries:
            with self.subTest(api=function.__name__, resize=kwargs.get("resize_size")):
                baseline = function(str(self.path), video_backend="pyav", **kwargs)
                candidate = function(
                    str(self.path), video_backend="pyav", video_backend_kwargs={"thread_count": 1}, **kwargs,
                )
                self.assertEqual(candidate.dtype, np.uint8)
                self.assertEqual(candidate.shape, baseline.shape)
                self.assertEqual(candidate.tobytes(), baseline.tobytes())

    def test_actual_av1_training_video_preserves_pixels(self):
        root = Path("/scratch/jliu3375/datasets/RoboTwin-Clean/click_bell/videos")
        path = next(root.rglob("*.mp4"), None) if root.exists() else None
        if path is None:
            self.skipTest("Local RoboTwin AV1 video is unavailable")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps, length = float(stream.average_rate), stream.frames
        timestamps = np.asarray([0, 1, length // 2, length - 2], dtype=np.float64) / fps
        baseline = video.get_frames_by_timestamps(str(path), timestamps, video_backend="pyav")
        candidate = video.get_frames_by_timestamps(
            str(path), timestamps, video_backend="pyav", video_backend_kwargs={"thread_count": 1},
        )
        self.assertEqual(candidate.shape, baseline.shape)
        self.assertEqual(candidate.tobytes(), baseline.tobytes())


class VideoBackendFactoryTests(unittest.TestCase):
    def make_dataset(self, cfg):
        data_config = SimpleNamespace(modality_config=lambda: {}, transform=lambda data_cfg=None: None)
        with patch.dict(lerobot_datasets.ROBOT_TYPE_CONFIG_MAP, {"decoder_test": data_config}), \
                patch.dict(lerobot_datasets.ROBOT_TYPE_TO_EMBODIMENT_TAG,
                           {"decoder_test": EmbodimentTag.NEW_EMBODIMENT}), \
                patch.object(lerobot_datasets, "LeRobotSingleDataset") as constructor:
            lerobot_datasets.make_LeRobotSingleDataset(Path("/dataset"), "task", "decoder_test", data_cfg=cfg)
        return constructor.call_args.kwargs

    def test_configured_kwargs_reach_dataset_as_plain_independent_dict(self):
        cfg = OmegaConf.create({"video_backend": "pyav", "video_backend_kwargs": {"thread_count": 1}})
        kwargs = self.make_dataset(cfg)
        self.assertEqual(kwargs["video_backend"], "pyav")
        self.assertIs(type(kwargs["video_backend_kwargs"]), dict)
        self.assertEqual(kwargs["video_backend_kwargs"], {"thread_count": 1})
        kwargs["video_backend_kwargs"]["thread_count"] = 8
        self.assertEqual(cfg.video_backend_kwargs.thread_count, 1)

    def test_absent_kwargs_keep_default(self):
        self.assertEqual(self.make_dataset({"video_backend": "pyav"})["video_backend_kwargs"], {})

    def test_non_mapping_kwargs_fail_before_dataset_construction(self):
        for value in (None, "threads=1", [1]):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "must be a mapping"):
                self.make_dataset({"video_backend": "pyav", "video_backend_kwargs": value})


if __name__ == "__main__":
    unittest.main()
