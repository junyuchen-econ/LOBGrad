from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================== Sinusoidal Embedding ===============================
class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, N_max: int = 500):
        super().__init__()
        self.emb_dim = emb_dim
        self.N_max   = N_max
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half  = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(self.N_max)
            * torch.arange(half, device=k.device).float()
            / (half - 1)
        )                                                       # (half,)
        args = k.float().unsqueeze(-1) * freqs.unsqueeze(0)    # (batch, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)     # (batch, emb_dim)
        return self.proj(emb)


# =============================== Noise Predictor (MLP) ===============================
class NoisePredictor(nn.Module):
    """
    MLP-based noise predictor (epsilon_theta).
    Takes noisy target, diffusion step embedding, and projected RNN conditioning
    vector, and predicts the noise that was added.
    """

    def __init__(
        self,
        output_dim: int,
        cond_dim: int,
        hidden_dim: int,
        emb_dim: int,
        time_embedding: bool = True,
    ):
        super().__init__()
        self.time_embedding = time_embedding
        if time_embedding:
            self.embed = SinusoidalEmbedding(emb_dim)
            k_feature_dim = emb_dim
        else:
            self.embed = None
            k_feature_dim = 1

        self.net = nn.Sequential(
            nn.Linear(output_dim + cond_dim + k_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, y_noisy: torch.Tensor, k: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        y_noisy : (batch, output_dim) — noisy target
        k       : (batch,)   — diffusion step index
        cond    : (batch, cond_dim) — projected conditioning vector

        Returns:
            predicted_noise : (batch, output_dim)
        """
        if self.time_embedding:
            k_feat = self.embed(k)
        else:
            k_feat = k.float().unsqueeze(-1)
        x = torch.cat([y_noisy, cond, k_feat], dim=-1)
        return self.net(x)


# =============================== WaveNet Components ===============================

class DiffusionEmbedding(nn.Module):
    """Precomputed sinusoidal lookup table with two Linear+SiLU projections."""

    def __init__(self, dim: int, proj_dim: int, max_steps: int = 500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False,
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    @staticmethod
    def _build_embedding(dim: int, max_steps: int) -> torch.Tensor:
        steps = torch.arange(max_steps).unsqueeze(1)          # (T, 1)
        dims = torch.arange(dim).unsqueeze(0)                 # (1, dim)
        table = steps * 10.0 ** (dims * 4.0 / dim)           # (T, dim)
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T, 2*dim)


class _ResidualBlock(nn.Module):
    """Dilated Conv1d residual block with gated activation and skip connection."""

    def __init__(self, hidden_size: int, residual_channels: int, dilation: int):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels, 3,
            padding=dilation, dilation=dilation, padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular",
        )
        self.output_projection = nn.Conv1d(
            residual_channels, 2 * residual_channels, 1,
        )

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(
        self, x: torch.Tensor, conditioner: torch.Tensor, diffusion_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class _CondUpsampler(nn.Module):
    """Projects the conditioning vector to target_dim via two Linear+LeakyReLU layers."""

    def __init__(self, cond_length: int, target_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


# =============================== Noise Predictor (WaveNet) ===============================

class WaveNetNoisePredictor(nn.Module):
    """
    WaveNet-style noise predictor (epsilon_theta) from the pytorch-ts reference.
    Uses dilated 1D convolutions with gated activations, residual connections,
    and skip connections. Shares the same forward(y_noisy, k, cond) interface
    as NoisePredictor for drop-in substitution.
    """

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        time_emb_dim: int = 16,
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        residual_hidden: int = 64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular",
        )
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden,
        )
        self.cond_upsampler = _CondUpsampler(
            target_dim=target_dim, cond_length=cond_dim,
        )
        self.residual_layers = nn.ModuleList([
            _ResidualBlock(
                residual_channels=residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
                hidden_size=residual_hidden,
            )
            for i in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, y_noisy: torch.Tensor, k: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        y_noisy : (batch, output_dim) — noisy target
        k       : (batch,)            — diffusion step index
        cond    : (batch, cond_dim)    — projected conditioning vector

        Returns:
            predicted_noise : (batch, output_dim)
        """
        x = y_noisy.unsqueeze(1)                    # (B, 1, D)
        x = self.input_projection(x)                # (B, residual_channels, D+4)
        x = F.leaky_relu(x, 0.4)

        diff_step = self.diffusion_embedding(k)     # (B, residual_hidden)
        cond_up = self.cond_upsampler(cond)          # (B, D)
        cond_up = cond_up.unsqueeze(1)               # (B, 1, D) for Conv1d

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diff_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)               # (B, 1, D)
        return x.squeeze(1)                          # (B, D)
