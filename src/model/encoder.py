from __future__ import annotations
from typing import Union
import torch
import torch.nn as nn

RnnState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

# =============================== RNN Encoder ===============================
class RnnEncoder(nn.Module):
    """Unified RNN encoder supporting both GRU and LSTM cell types."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        cell_type: str = "GRU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cell_type = cell_type
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, state: RnnState | None = None,
    ) -> tuple[torch.Tensor, RnnState]:
        """
        x     : (batch, seq_len, input_dim)
        state : previous hidden state (GRU: h_n tensor, LSTM: (h_n, c_n) tuple)

        Returns:
            outputs : (batch, seq_len, hidden_dim)
            state   : updated hidden state (same format as input)
        """
        outputs, state = self.rnn(x, state)
        return outputs, state


# =============================== Identity Encoder ===============================
class IdentityEncoder(nn.Module):
    """No-op encoder: per-position output equals the input, no temporal mixing.

    Used to ablate the recurrent encoder. Mimics ``RnnEncoder``'s
    ``forward(x, state) -> (output, state)`` interface so ``_TimeGradBase``
    works unchanged. Hidden state is the last observation reshaped to
    ``(1, B, D)`` (GRU convention), so ``_last_hidden`` returns ``(B, D)``.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.cell_type = "identity"
        self.input_dim = input_dim

    def forward(
        self, x: torch.Tensor, state: RnnState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last = x[:, -1:, :].transpose(0, 1).contiguous()   # (1, B, D)
        return x, last


# =============================== Flat Window Encoder ===============================
class FlatWindowEncoder(nn.Module):
    """Encoderless conditioning that flattens a fixed-size history window.

    For each output position, returns the concatenation of the most recent
    ``window_size`` input observations (zero-padded for early positions).
    Mimics the (output, state) interface of ``RnnEncoder`` so ``_TimeGradBase``
    works unchanged: state has shape ``(1, B, window_size * input_dim)`` so
    ``_last_hidden(state) = state[-1]`` returns the conditioning vector.
    """

    def __init__(self, input_dim: int, window_size: int):
        super().__init__()
        self.cell_type   = "flat_window"
        self.input_dim   = input_dim
        self.window_size = window_size
        self.output_dim  = input_dim * window_size

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        s = self.window_size
        if state is None:
            queue = torch.zeros(B, s - 1, D, device=x.device, dtype=x.dtype)
        else:
            # state: (1, B, s*D) — recover last window, drop oldest to form the queue
            window = state[0].reshape(B, s, D)
            queue  = window[:, 1:, :]
        padded  = torch.cat([queue, x], dim=1)                              # (B, s-1+L, D)
        windows = padded.unfold(1, s, 1).permute(0, 1, 3, 2).contiguous()   # (B, L, s, D)
        out     = windows.reshape(B, L, s * D)                              # (B, L, s*D)
        final_window = padded[:, -s:, :].reshape(B, 1, s * D).transpose(0, 1).contiguous()
        return out, final_window


def build_encoder(
    input_dim:   int,
    hidden_dim:  int,
    num_layers:  int,
    cell_type:   str,
    dropout:     float,
    window_size: int = 1,
) -> nn.Module:
    """Dispatch to the requested encoder. ``cell_type='identity'`` returns the
    no-op encoder; ``cell_type='flat_window'`` returns the rolling-window
    flatten encoder of width ``window_size``. For RNN cells, ``window_size``
    is ignored and ``hidden_dim``/``num_layers``/``dropout`` are used."""
    if cell_type == "identity":
        return IdentityEncoder(input_dim)
    if cell_type == "flat_window":
        return FlatWindowEncoder(input_dim, window_size)
    return RnnEncoder(input_dim, hidden_dim, num_layers, cell_type=cell_type, dropout=dropout)
