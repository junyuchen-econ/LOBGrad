from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoder import RnnEncoder, build_encoder
from .noise_predictor import NoisePredictor, WaveNetNoisePredictor


# =============================== Diffusion Schedule ===============================
class DiffusionSchedule(nn.Module):
    """
    Manages the noise schedule for a Gaussian diffusion process (DDPM).

    Precomputes all quantities needed for both the forward process q(x_t | x_0)
    and the posterior-based reverse process q(x_{t-1} | x_t, x_0), following
    the formulation in Ho et al. (2020) and the pytorch-ts reference.
    """

    def __init__(
        self,
        K: int,
        beta_start: float,
        beta_end: float,
        schedule: str = "linear",
    ):
        super().__init__()
        self.K = K

        # --- Build beta schedule ---
        if schedule == "linear":
            betas = np.linspace(beta_start, beta_end, K)
        elif schedule == "quad":
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, K) ** 2
        elif schedule == "const":
            betas = beta_end * np.ones(K)
        elif schedule == "jsd":
            # Jensen-Shannon divergence inspired: betas = 1/T, 1/(T-1), ..., 1
            betas = 1.0 / np.linspace(K, 1, K)
        elif schedule == "sigmoid":
            x = np.linspace(-6, 6, K)
            betas = (beta_end - beta_start) / (np.exp(-x) + 1) + beta_start
        elif schedule == "cosine":
            # Nichol & Dhariwal (2021) cosine schedule
            s = 0.008
            steps = np.arange(K + 1, dtype=np.float64)
            f = np.cos(((steps / K + s) / (1 + s)) * np.pi / 2) ** 2
            alpha_bar_cos = f / f[0]
            betas = np.clip(1 - alpha_bar_cos[1:] / alpha_bar_cos[:-1], 0.0, 0.999)
        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                "Choose from 'linear', 'quad', 'const', 'jsd', 'sigmoid', 'cosine'."
            )

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)          # \bar{alpha}_t
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # \bar{alpha}_{t-1}

        to_buf = lambda x: torch.tensor(x, dtype=torch.float32)

        # --- Core schedule buffers ---
        self.register_buffer("beta", to_buf(betas))
        self.register_buffer("alpha", to_buf(alphas))
        self.register_buffer("alpha_bar", to_buf(alphas_cumprod))
        self.register_buffer("alpha_bar_prev", to_buf(alphas_cumprod_prev))

        # --- Forward process q(x_t | x_0) coefficients ---
        self.register_buffer("sqrt_alpha_bar", to_buf(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alpha_bar", to_buf(np.sqrt(1.0 - alphas_cumprod)))

        # --- Reverse process: recovering x_0 from noise prediction ---
        # Clamp to avoid division by zero for extreme schedules (e.g. jsd)
        alphas_cumprod_safe = np.maximum(alphas_cumprod, 1e-20)
        self.register_buffer("sqrt_recip_alpha_bar", to_buf(np.sqrt(1.0 / alphas_cumprod_safe)))
        self.register_buffer("sqrt_recipm1_alpha_bar", to_buf(np.sqrt(1.0 / alphas_cumprod_safe - 1)))

        # --- Posterior q(x_{t-1} | x_t, x_0) coefficients ---
        # Tighter variance than using beta_t directly (Ho et al. Eq. 7)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_buf(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_buf(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        # Coefficients for computing the posterior mean (Ho et al. Eq. 7)
        self.register_buffer(
            "posterior_mean_coef1",
            to_buf(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_buf((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    def _extract(self, buf: torch.Tensor, k: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Index into a 1-D buffer by diffusion step k and reshape for broadcasting."""
        out = buf.gather(-1, k)
        return out.reshape(k.shape[0], *((1,) * (len(shape) - 1)))

    # ----- forward process q(x_t | x_0) -----
    def q_sample(self, y0: torch.Tensor, k: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample from q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
        return (
            self._extract(self.sqrt_alpha_bar, k, y0.shape) * y0
            + self._extract(self.sqrt_one_minus_alpha_bar, k, y0.shape) * noise
        )

    # ----- reverse process helpers -----
    def predict_start_from_noise(self, x_t: torch.Tensor, k: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Recover x_0 estimate from the predicted noise: x_0 = (x_t - sqrt(1/alpha_bar - 1) * eps) / sqrt(1/alpha_bar)."""
        return (
            self._extract(self.sqrt_recip_alpha_bar, k, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alpha_bar, k, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, k: torch.Tensor):
        """Compute mean and log-variance of the posterior q(x_{t-1} | x_t, x_0)."""
        mean = (
            self._extract(self.posterior_mean_coef1, k, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, k, x_t.shape) * x_t
        )
        log_var = self._extract(self.posterior_log_variance_clipped, k, x_t.shape)
        return mean, log_var

    def p_sample(self, x: torch.Tensor, k: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        """One reverse step: predict x_0, compute posterior, then sample (no noise at k=0)."""
        x_start = self.predict_start_from_noise(x, k, predicted_noise)
        mean, log_var = self.q_posterior(x_start, x, k)
        noise = torch.randn_like(x)
        # Mask noise at k=0 so the final step is deterministic
        nonzero_mask = (k != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise

    def log_prob(
        self,
        y: torch.Tensor,
        cond: torch.Tensor,
        noise_net: nn.Module,
        loss_type: str = "l2",
        loss_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Per-sample log-probability estimated via noise prediction.

        y           : (N, dim) — clean targets
        cond        : (N, cond_dim) — conditioning vectors
        loss_weights: optional (dim,) tensor of non-negative per-dim weights.
                      If provided, the per-dim loss is reduced as
                      ``(per_dim_loss * loss_weights).sum(-1)`` instead of
                      a uniform mean. Pass weights that sum to 1.0 to keep
                      the overall loss magnitude comparable to the unweighted
                      mean. ``None`` preserves the legacy uniform-mean behavior.

        Returns (N,) approximate log-prob (negative loss; higher = better fit).
        """

        # sample a random diffusion step k per position, corrupt, predict noise
        k = torch.randint(0, self.K, (y.shape[0],), device=y.device, dtype=torch.long)
        noise = torch.randn_like(y)

        # diffusion
        y_noisy = self.q_sample(y, k, noise)

        # denoising
        predicted_noise = noise_net(y_noisy, k, cond)

        # choose loss function
        loss_fns = {"l1": F.l1_loss, "l2": F.mse_loss, "huber": F.smooth_l1_loss}
        if loss_type not in loss_fns:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Choose 'l1', 'l2', or 'huber'.")

        # compute loss per dimension: (N, dim)
        per_dim_loss = loss_fns[loss_type](predicted_noise, noise, reduction="none")

        # Reduce: weighted sum if loss_weights given, else uniform mean.
        if loss_weights is None:
            reduced = per_dim_loss.mean(dim=-1)
        else:
            reduced = (per_dim_loss * loss_weights).sum(dim=-1)

        # Negate so higher = better fit (log-prob convention from pytorch-ts reference)
        return -reduced


# =============================== TimeGrad Base ===============================
class _TimeGradBase(nn.Module):
    """Shared encoder + diffusion logic. Subclasses set ``self.noise_net``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_encoder_dim: int,
        num_encoder_layers: int,
        K: int,
        beta_start: float,
        beta_end: float,
        schedule: str = "linear",
        conditioning_length: int | None = None,
        loss_type: str = "l2",
        dropout_rate: float = 0.0,
        scaling: bool = False,
        cell_type: str = "GRU",
        use_cond_projection: bool = True,
        flat_window_size: int = 1,
        loss_weights: torch.Tensor | list | tuple | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.scaling = scaling
        self.loss_type = loss_type
        self.cell_type = cell_type
        self.use_cond_projection = use_cond_projection
        self.flat_window_size = flat_window_size

        # Optional per-dim loss weighting. Registered as a buffer so it moves
        # to the right device with .to() and is saved in state_dict() — the
        # weights are part of the trained model's loss spec.
        if loss_weights is None:
            self.loss_weights = None
        else:
            w = torch.as_tensor(loss_weights, dtype=torch.float32)
            if w.numel() != output_dim:
                raise ValueError(
                    f"loss_weights must have {output_dim} elements, got {w.numel()}"
                )
            self.register_buffer("loss_weights", w)

        # Encoder output dim depends on cell_type:
        #   identity     → input_dim (passthrough)
        #   flat_window  → input_dim * flat_window_size (rolling-window flatten)
        #   GRU/LSTM     → hidden_encoder_dim
        if cell_type == "identity":
            effective_hidden = input_dim
        elif cell_type == "flat_window":
            effective_hidden = input_dim * flat_window_size
        else:
            effective_hidden = hidden_encoder_dim

        if use_cond_projection:
            self._cond_dim = conditioning_length if conditioning_length is not None else effective_hidden
        else:
            self._cond_dim = effective_hidden

        self.encoder = build_encoder(
            input_dim, hidden_encoder_dim, num_encoder_layers, cell_type, dropout_rate,
            window_size=flat_window_size,
        )
        if use_cond_projection:
            self.cond_projection = nn.Linear(effective_hidden, self._cond_dim)
        else:
            self.cond_projection = nn.Identity()
        self.diffusion = DiffusionSchedule(K, beta_start, beta_end, schedule)

        # self.noise_net is expected to be set by subclasses

    def _last_hidden(self, state) -> torch.Tensor:
        """Extract last-layer hidden vector from either GRU or LSTM state."""
        if self.cell_type == "LSTM":
            return state[0][-1]   # h_n component, last layer
        return state[-1]          # GRU: h_n tensor, last layer

    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Mean absolute value per sample, clamped away from zero.  (batch, 1, 1)"""
        return x.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-5)

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------
    def _encode_and_project(
        self, x_hist: torch.Tensor, y_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """
        Teacher-forced RNN encoding with causal shift, shared by loss() and log_prob().

        Returns:
            y_flat : (batch * length, dim) — scaled targets
            h_flat : (batch * length, cond_dim) — projected conditioning vectors
            shape  : (batch, length, dim) for reshaping back
        """
        device = y_target.device
        # Teacher forcing: RNN sees the true sequence [context | target]
        full_input = torch.cat([x_hist, y_target], dim=1)

        if self.scaling:
            # Scale from context only — avoids leaking future info during training
            scale = self._compute_scale(x_hist)
            full_input = full_input / scale

        rnn_out, _ = self.encoder(full_input)

        # Shift right by 1: conditioning at time t uses only observations up to t-1
        zeros = torch.zeros(rnn_out.size(0), 1, rnn_out.size(2), device=device)
        h_shifted = torch.cat([zeros, rnn_out[:, :-1, :]], dim=1)
        h_cond = self.cond_projection(h_shifted)

        # Flatten time into batch dimension so diffusion operates on all positions in parallel
        batch_size, length, dim = full_input.shape
        y_flat = full_input.reshape(batch_size * length, dim)
        h_flat = h_cond.reshape(batch_size * length, -1)

        return y_flat, h_flat, full_input.shape

    def loss(
        self, x_hist: torch.Tensor, y_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_hist   : (batch, context_length, input_dim)
        y_target : (batch, prediction_length, input_dim)

        Returns:
            loss : scalar — mean of per-position noise-prediction losses
        """
        y_flat, h_flat, (batch_size, length, _) = self._encode_and_project(x_hist, y_target)

        # Per-position log-prob via simplified DDPM objective (no per-step weighting)
        per_sample_logp = self.diffusion.log_prob(
            y_flat, h_flat, self.noise_net, self.loss_type,
            loss_weights=self.loss_weights,
        )
        # Negate log-prob to get a loss (lower = better)
        return -per_sample_logp.mean()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_prob(
        self, x_hist: torch.Tensor, y_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-position log-probability for evaluation / monitoring.

        x_hist   : (batch, context_length, input_dim)
        y_target : (batch, prediction_length, input_dim)

        Returns:
            (batch, context_length + prediction_length) log-prob per position
        """

        y_flat, h_flat, (batch_size, length, _) = self._encode_and_project(x_hist, y_target)
        per_sample_logp = self.diffusion.log_prob(
            y_flat, h_flat, self.noise_net, self.loss_type,
            loss_weights=self.loss_weights,
        )
        return per_sample_logp.reshape(batch_size, length)

    # ------------------------------------------------------------------
    # Inference / sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, x_hist: torch.Tensor, prediction_steps: int = 1) -> torch.Tensor:
        """
        Autoregressively generate prediction_steps future samples (Algorithm 2).

        For each future step:
          1. Run DDPM reverse process conditioned on h_{t-1} to draw y_t
          2. Feed y_t back into the RNN to update h_t

        Arguments:
            x_hist           : (batch, context_length, input_dim)
            prediction_steps : number of future steps to generate

        Returns:
            (batch, prediction_steps, output_dim)
        """
        batch_size = x_hist.size(0)
        device = x_hist.device

        scale = self._compute_scale(x_hist) if self.scaling else None
        encoder_input = x_hist / scale if self.scaling else x_hist

        # Encode context to obtain initial hidden state for autoregressive rollout
        _, h_n = self.encoder(encoder_input)

        generated = []

        for _ in range(prediction_steps):
            # Project last-layer hidden state to conditioning vector for the denoiser
            h = self._last_hidden(h_n)
            cond = self.cond_projection(h)

            # Start from pure Gaussian noise and run the full K-step reverse process
            y = torch.randn(batch_size, self.output_dim, device=device)
            for k in reversed(range(self.diffusion.K)):
                k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.long)
                eps = self.noise_net(y, k_tensor, cond)
                y = self.diffusion.p_sample(y, k_tensor, eps)

            if self.scaling:
                y = y * scale.squeeze(1)

            generated.append(y)

            # Feed generated sample back into RNN (in scaled space) to update h_t
            rnn_input = y / scale.squeeze(1) if self.scaling else y
            _, h_n = self.encoder(rnn_input.unsqueeze(1), h_n)

        return torch.stack(generated, dim=1)


# =============================== TimeGradMLP ===============================
class TimeGradMLP(_TimeGradBase):
    """TimeGrad with the lightweight MLP noise predictor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_encoder_dim: int,
        hidden_noise_dim: int,
        num_encoder_layers: int,
        emb_dim: int,
        K: int,
        beta_start: float,
        beta_end: float,
        schedule: str = "linear",
        conditioning_length: int | None = None,
        loss_type: str = "l2",
        dropout_rate: float = 0.0,
        scaling: bool = False,
        cell_type: str = "GRU",
        time_embedding: bool = True,
        use_cond_projection: bool = True,
        flat_window_size: int = 1,
        loss_weights: torch.Tensor | list | tuple | None = None,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_encoder_dim=hidden_encoder_dim,
            num_encoder_layers=num_encoder_layers,
            K=K,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=schedule,
            conditioning_length=conditioning_length,
            loss_type=loss_type,
            dropout_rate=dropout_rate,
            scaling=scaling,
            cell_type=cell_type,
            use_cond_projection=use_cond_projection,
            flat_window_size=flat_window_size,
            loss_weights=loss_weights,
        )
        self.noise_net = NoisePredictor(
            output_dim,
            self._cond_dim,
            hidden_noise_dim,
            emb_dim,
            time_embedding=time_embedding,
        )


# =============================== TimeGradWaveNet ===============================
class TimeGradWaveNet(_TimeGradBase):
    """TimeGrad with the WaveNet-style convolutional noise predictor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_encoder_dim: int,
        num_encoder_layers: int,
        emb_dim: int,
        K: int,
        beta_start: float,
        beta_end: float,
        schedule: str = "linear",
        conditioning_length: int | None = None,
        loss_type: str = "l2",
        dropout_rate: float = 0.0,
        scaling: bool = False,
        cell_type: str = "GRU",
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        flat_window_size: int = 1,
        loss_weights: torch.Tensor | list | tuple | None = None,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_encoder_dim=hidden_encoder_dim,
            num_encoder_layers=num_encoder_layers,
            K=K,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=schedule,
            conditioning_length=conditioning_length,
            loss_type=loss_type,
            dropout_rate=dropout_rate,
            scaling=scaling,
            cell_type=cell_type,
            flat_window_size=flat_window_size,
            loss_weights=loss_weights,
        )
        self.noise_net = WaveNetNoisePredictor(
            target_dim=output_dim,
            cond_dim=self._cond_dim,
            time_emb_dim=emb_dim,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )
