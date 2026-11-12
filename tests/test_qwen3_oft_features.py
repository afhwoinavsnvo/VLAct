"""Check the no-logits Qwen3 feature path against an actual tiny transformer."""

import contextlib
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface


class Qwen3OFTFeaturesTests(unittest.TestCase):
    def test_feature_path_matches_full_model_without_materializing_vocabulary_logits(self):
        config = Qwen3VLConfig(
            text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 8,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                           "num_heads": 2, "out_hidden_size": 32, "deepstack_visual_indexes": []},
            image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
        )
        model = Qwen3VLForConditionalGeneration(config).eval()
        interface = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
        nn.Module.__init__(interface)
        interface.model = model
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
                  "attention_mask": torch.ones(2, 4, dtype=torch.long)}
        with torch.no_grad(), patch.object(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()):
            expected = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states[-1]
            original_head = model.lm_head.forward
            def empty_projection(features):
                self.assertEqual(features.shape[1], 0, "OFT must not project observed tokens into vocabulary logits")
                result = original_head(features)
                self.assertEqual(result.numel(), 0)
                return result
            with patch.object(model.lm_head, "forward", side_effect=empty_projection):
                actual = interface.encode_last_hidden_state(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (2, 4, 32))


if __name__ == "__main__":
    unittest.main()
