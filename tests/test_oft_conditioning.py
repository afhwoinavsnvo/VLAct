"""Pure-Torch checks for OFT's projection and separate numeric conditions."""

import itertools
import unittest

import torch

from starVLA.model.modules.action_model.oft_conditioning import OFTConditioning


class OFTConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(11)
        self.tokens = torch.randn(2, 4, 8)
        self.state = torch.randn(2, 1, 6)
        self.tcp = torch.randn(2, 1, 7)

    def adapter(self, **values):
        kwargs = dict(
            hidden_dim=8, encoder_hidden_dim=16, state_dim=6, tcp_dim=7,
            include_state=True, include_tcp=True, enabled=True,
        )
        kwargs.update(values)
        return OFTConditioning(**kwargs)

    def test_disabled_adapter_preserves_original_object_weights_and_rng(self):
        before = torch.get_rng_state().clone()
        adapter = self.adapter(enabled=False)
        self.assertEqual(list(adapter.parameters()), [])
        self.assertEqual(dict(adapter.state_dict()), {})
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertIs(adapter(self.tokens, state=object(), tcp=object()), self.tokens)

    def test_projection_only_is_initially_identity_and_still_trainable(self):
        adapter = self.adapter(include_state=False, include_tcp=False)
        result = adapter(self.tokens, state=object(), tcp=object())
        torch.testing.assert_close(result, self.tokens, rtol=0, atol=0)
        torch.testing.assert_close(adapter.output_projection.weight, torch.eye(8), rtol=0, atol=0)
        torch.testing.assert_close(adapter.output_projection.bias, torch.zeros(8), rtol=0, atol=0)
        result.square().mean().backward()
        self.assertGreater(float(adapter.output_projection.weight.grad.abs().sum()), 0)
        self.assertIsNone(adapter.state_encoder)
        self.assertIsNone(adapter.tcp_encoder)

    def test_all_switch_combinations_add_one_condition_token_over_every_action(self):
        for use_state, use_tcp in itertools.product((False, True), repeat=2):
            with self.subTest(state=use_state, tcp=use_tcp):
                adapter = self.adapter(include_state=use_state, include_tcp=use_tcp)
                expected = adapter.output_projection(self.tokens)
                if use_state:
                    encoded = adapter.state_encoder(self.state)
                    self.assertEqual(encoded.shape, (2, 1, 8))
                    expected = expected + encoded
                if use_tcp:
                    encoded = adapter.tcp_encoder(self.tcp)
                    self.assertEqual(encoded.shape, (2, 1, 8))
                    expected = expected + encoded
                result = adapter(
                    self.tokens,
                    state=self.state if use_state else object(),
                    tcp=self.tcp if use_tcp else object(),
                )
                self.assertEqual(result.shape, self.tokens.shape)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                residual = result - adapter.output_projection(self.tokens)
                torch.testing.assert_close(residual, residual[:, :1].expand_as(residual))

    def test_two_dimensional_conditions_match_single_token_conditions(self):
        adapter = self.adapter()
        flat = adapter(self.tokens, state=self.state[:, 0], tcp=self.tcp[:, 0])
        tokens = adapter(self.tokens, state=self.state, tcp=self.tcp)
        torch.testing.assert_close(flat, tokens, rtol=0, atol=0)

    def test_projection_is_applied_before_adding_condition_features(self):
        adapter = self.adapter()
        with torch.no_grad():
            adapter.output_projection.weight.copy_(2 * torch.eye(8))
            adapter.output_projection.bias.fill_(0.25)
        state = adapter.state_encoder(self.state)
        tcp = adapter.tcp_encoder(self.tcp)
        actual = adapter(self.tokens, state=self.state, tcp=self.tcp)
        expected = 2 * self.tokens + 0.25 + state + tcp
        wrong_order = adapter.output_projection(self.tokens + state + tcp)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, wrong_order))

    def test_frozen_vlm_tokens_allow_gradients_in_projection_encoders_and_head(self):
        adapter = self.adapter()
        head = torch.nn.Linear(8, 3)
        self.assertFalse(self.tokens.requires_grad)
        actions = head(adapter(self.tokens, state=self.state, tcp=self.tcp))
        actions.square().mean().backward()
        for name, module in (
            ("projection", adapter.output_projection), ("state", adapter.state_encoder),
            ("tcp", adapter.tcp_encoder), ("head", head),
        ):
            gradients = [parameter.grad for parameter in module.parameters()]
            with self.subTest(module=name):
                self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
                self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0)
        self.assertIsNone(self.tokens.grad)

    def test_independent_conditions_can_each_change_actions(self):
        adapter = self.adapter()
        head = torch.nn.Linear(8, 3)
        original = head(adapter(self.tokens, state=self.state, tcp=self.tcp))
        for state, tcp in ((self.state + 1, self.tcp), (self.state, self.tcp + 1)):
            changed = head(adapter(self.tokens, state=state, tcp=tcp))
            self.assertGreater(float((original - changed).abs().max()), 1e-7)

    def test_missing_enabled_conditions_raise_without_reading_disabled_branches(self):
        for name in ("state", "tcp"):
            adapter = self.adapter(include_state=name == "state", include_tcp=name == "tcp")
            inputs = {"state": object(), "tcp": object()}
            inputs[name] = None
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name} is required"):
                adapter(self.tokens, **inputs)

    def test_condition_shape_batch_type_and_finiteness_validation(self):
        for name, feature_dim in (("state", 6), ("tcp", 7)):
            adapter = self.adapter(include_state=name == "state", include_tcp=name == "tcp")
            bad_shapes = (
                torch.ones(()), torch.ones(feature_dim), torch.ones(1, feature_dim),
                torch.ones(2, feature_dim + 1), torch.ones(2, 2, feature_dim),
                torch.ones(2, 1, feature_dim + 1),
            )
            for value in bad_shapes:
                with self.subTest(name=name, shape=value.shape), self.assertRaises(ValueError):
                    adapter(self.tokens, **{name: value})
            with self.subTest(name=name, kind="type"), self.assertRaises(TypeError):
                adapter(self.tokens, **{name: [[1] * feature_dim] * 2})
            with self.subTest(name=name, kind="dtype"), self.assertRaisesRegex(ValueError, "floating-point"):
                adapter(self.tokens, **{name: torch.ones(2, feature_dim, dtype=torch.int64)})
            for value in (float("nan"), float("inf"), float("-inf")):
                invalid = torch.zeros(2, feature_dim)
                invalid[0, 0] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    adapter(self.tokens, **{name: invalid})

    def test_action_token_shape_type_and_finiteness_validation(self):
        adapter = self.adapter(include_state=False, include_tcp=False)
        for shape in ((), (8,), (2, 8), (2, 4, 9), (0, 4, 8), (2, 0, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                adapter(torch.ones(shape))
        with self.assertRaises(TypeError):
            adapter([[[1.] * 8] * 4] * 2)
        with self.assertRaisesRegex(ValueError, "floating-point"):
            adapter(torch.ones(2, 4, 8, dtype=torch.int64))
        for value in (float("nan"), float("inf"), float("-inf")):
            tokens = self.tokens.clone()
            tokens[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                adapter(tokens)

    def test_constructor_rejects_invalid_dimensions_and_nonboolean_flags(self):
        for field in ("hidden_dim", "encoder_hidden_dim", "state_dim", "tcp_dim"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.adapter(**{field: value})
        for field in ("enabled", "include_state", "include_tcp"):
            for value in ("true", "false", 0, 1, None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "boolean"):
                    self.adapter(**{field: value})

    def test_disabled_zero_dimension_conditions_and_strict_checkpoint_roundtrip(self):
        adapter = self.adapter(include_state=False, include_tcp=False, state_dim=0, tcp_dim=0)
        clone = self.adapter(include_state=False, include_tcp=False, state_dim=0, tcp_dim=0)
        status = clone.load_state_dict(adapter.state_dict(), strict=True)
        self.assertFalse(status.missing_keys or status.unexpected_keys)
        torch.testing.assert_close(adapter(self.tokens), clone(self.tokens), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
