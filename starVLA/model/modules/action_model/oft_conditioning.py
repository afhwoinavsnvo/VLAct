"""Numeric state/TCP conditioning after OFT's action-token extraction.

This module has no VLM or configuration dependencies. Callers provide tensors
in the module's device/dtype. State uses the training state transform; target
TCP retains the dataset's world-frame meters and wxyz quaternion convention.
"""

import torch
from torch import nn


def _dimension(value: int, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _floating_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must contain floating-point values")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


class _ConditionEncoder(nn.Module):
    """Encode one current-state or target-TCP vector into one OFT-width token."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, name: str):
        super().__init__()
        self.input_dim = input_dim
        self.name = name
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _floating_tensor(values, self.name)
        if values.ndim == 2 and values.shape[1] == self.input_dim:
            values = values.unsqueeze(1)
        elif values.ndim != 3 or values.shape[1:] != (1, self.input_dim):
            raise ValueError(
                f"{self.name} must have shape (B, {self.input_dim}) or (B, 1, {self.input_dim}); "
                f"got {tuple(values.shape)}"
            )
        if values.shape[0] == 0:
            raise ValueError(f"{self.name} batch must be nonempty")
        return self.layers(values)


class OFTConditioning(nn.Module):
    """Project OFT action tokens, then add independently encoded conditions.

    ``action_tokens`` has shape ``(B, H, hidden_dim)``. Each enabled condition
    accepts ``(B, input_dim)`` or ``(B, 1, input_dim)`` and produces one token
    ``(B, 1, hidden_dim)``, broadcast over the original H action positions.

    The output projection starts as the identity. It is applied before adding
    condition features and always remains trainable when this adapter is enabled.
    Both condition switches may be off, leaving a projection-only adapter.

    A disabled adapter creates no parameters, consumes no initialization RNG,
    and returns its input object directly. Disabled condition branches never
    inspect their supplied values. Flags define a saved model architecture;
    enable or disable them when constructing the model, before optimizer setup.
    """

    def __init__(
        self,
        hidden_dim: int,
        encoder_hidden_dim: int = 1024,
        state_dim: int = 14,
        tcp_dim: int = 14,
        include_state: bool = False,
        include_tcp: bool = False,
        enabled: bool = False,
    ):
        super().__init__()
        self.hidden_dim = _dimension(hidden_dim, "hidden_dim")
        _dimension(encoder_hidden_dim, "encoder_hidden_dim")
        self.state_dim = _dimension(state_dim, "state_dim", minimum=0)
        self.tcp_dim = _dimension(tcp_dim, "tcp_dim", minimum=0)
        for name, value in (("enabled", enabled), ("include_state", include_state), ("include_tcp", include_tcp)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        self.enabled = enabled
        self.include_state = include_state
        self.include_tcp = include_tcp
        self.output_projection = None
        self.state_encoder = None
        self.tcp_encoder = None
        if not enabled:
            return

        if include_state:
            _dimension(state_dim, "state_dim")
        if include_tcp:
            _dimension(tcp_dim, "tcp_dim")
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.eye_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        if include_state:
            self.state_encoder = _ConditionEncoder(state_dim, encoder_hidden_dim, hidden_dim, "state")
        if include_tcp:
            self.tcp_encoder = _ConditionEncoder(tcp_dim, encoder_hidden_dim, hidden_dim, "tcp")

    def forward(
        self,
        action_tokens: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        tcp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return action_tokens

        _floating_tensor(action_tokens, "action_tokens")
        if action_tokens.ndim != 3 or action_tokens.shape[2] != self.hidden_dim:
            raise ValueError(
                f"action_tokens must have shape (B, H, {self.hidden_dim}); got {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[0] == 0 or action_tokens.shape[1] == 0:
            raise ValueError("action_tokens batch and horizon must be nonempty")

        conditions = (
            ("state", self.state_encoder, state),
            ("tcp", self.tcp_encoder, tcp),
        )
        for name, encoder, values in conditions:
            if encoder is None:
                continue
            if values is None:
                raise ValueError(f"{name} is required when include_{name}=True")
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if values.ndim == 0 or values.shape[0] != action_tokens.shape[0]:
                raise ValueError(f"{name} and action_tokens must have the same batch size")

        projected = self.output_projection(action_tokens)
        for _, encoder, values in conditions:
            if encoder is not None:
                projected = projected + encoder(values)
        return projected
