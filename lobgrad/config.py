from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "raw")


@dataclass
class BenchmarkConfig:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir: str = _DEFAULT_DATA_DIR
    symbol: str = "BTCUSDT"
    levels: int = 10           # k: levels per side → 2k-dim queue representation
    price_level: float = 0.1   # tick size δ in USDT  (max price move per step = k*δ)
    queue_repr: str = "signed" # "signed": bids<0/asks>0, sign(Q)*log(|Q|+1)
                               # "unsigned": all vols ≥ 0, log(Q+1)
    queue_mode: str = "absolute" # "absolute": model raw queue sizes directly
                                 # "delta": model step-wise changes ΔQ_t = Q_t − Q_{t-1}
    time_diff_ms: int = 5_000  # snapshot interval in ms (5 s)
    depth: int = 100           # raw order-book depth to read from JSON

    # ------------------------------------------------------------------
    # Model selection  (subset of: "TimeGrad", "CGAN", "Hawkes", "Poisson")
    # ------------------------------------------------------------------
    models: List[str] = field(default_factory=lambda: ["TimeGrad", "CGAN", "Hawkes", "Poisson"])

    # ------------------------------------------------------------------
    # Train / val / test splits  (inclusive on both ends)
    # ------------------------------------------------------------------
    train_start: str = "2023-10-01"
    train_end:   str = "2023-10-15"
    val_start:   str = "2023-10-16"
    val_end:     str = "2023-10-17"
    test_start:  str = "2023-10-18"
    test_end:    str = "2023-10-21"

    # ------------------------------------------------------------------
    # Benchmark protocol (Track B: conditional)
    # ------------------------------------------------------------------
    context_len:    int = 100   # N : conditioning window length (snapshots)
    horizon:        int = 100   # M : steps to generate per window
    n_paths:        int = 50    # generated paths per test window
    n_eval_windows: int = 100   # test windows to evaluate

    # ------------------------------------------------------------------
    # cGAN hyperparameters
    # ------------------------------------------------------------------
    cgan_latent_dim:       int   = 64
    cgan_hidden_dim:       int   = 128
    cgan_n_encoder_layers: int   = 2
    cgan_n_decoder_layers: int   = 1
    cgan_lr_g:             float = 1e-4
    cgan_lr_d:             float = 1e-4
    cgan_n_critic:         int   = 3      # discriminator steps per generator step
    cgan_lambda_gp:        float = 10.0   # gradient penalty weight
    cgan_batch_size:       int   = 64
    cgan_epochs:           int   = 25
    cgan_stride:           int   = 10   # stride for sliding windows (1 = 223K windows, 10 = 22K)
    cgan_max_batches:      int   = 50  # max batches per epoch (None = unlimited)

    # ------------------------------------------------------------------
    # TimeGrad  (points to the existing trained checkpoint)
    # ------------------------------------------------------------------
    timegrad_checkpoint: str = "checkpoints/final/joint_timegrad/best_model.pt"
    timegrad_k:               int   = 80
    timegrad_beta_start:      float = 0.0001
    timegrad_beta_end:        float = 0.1
    timegrad_schedule:        str   = "linear"
    timegrad_hidden_enc:      int   = 32
    timegrad_hidden_noise:    int   = 64
    timegrad_num_enc_layers:  int   = 2
    timegrad_emb_dim:         int   = 32
    timegrad_cell_type:       str   = "GRU"   # "GRU", "LSTM", "identity" (no encoder; cond = previous obs), or "flat_window" (no encoder; cond = flattened past `flat_window_size` obs)
    timegrad_flat_window_size: int   = 1      # only used when cell_type == "flat_window"

    # Ablation toggles (defaults preserve existing behavior)
    timegrad_time_embedding:      bool = True   # False: concat raw k scalar instead of sinusoidal emb
    timegrad_use_cond_projection: bool = True   # False: feed RNN hidden state directly as cond

    # Phase 3 HPO additions (defaults reproduce existing behavior)
    timegrad_loss_type:               str   = "l2"      # "l1", "l2", "huber"
    timegrad_dropout:                 float = 0.0       # encoder RNN dropout
    timegrad_noise_type:              str   = "mlp"     # "mlp" or "wavenet"
    timegrad_wavenet_res_layers:      int   = 8
    timegrad_wavenet_res_channels:    int   = 8
    timegrad_wavenet_dilation_cycle:  int   = 2

    # ------------------------------------------------------------------
    # TimeGrad training  (defaults mirror config.yaml)
    # ------------------------------------------------------------------
    timegrad_train:             bool  = False   # False = load checkpoint only
    timegrad_history_length:    int   = 100   # match benchmark context_len
    timegrad_prediction_length: int   = 100
    timegrad_batch_size:        int   = 64
    timegrad_lr:                float = 0.0015
    timegrad_epochs:            int   = 30
    timegrad_max_batches:       int   = 500   # use full dataset each epoch
    timegrad_save_dir:          str   = "checkpoints/final/joint_timegrad"

    # Δmid clip used by the joint (price + queue) generators (in raw USDT).
    # Order of magnitude: a few standard deviations of the per-step price
    # change at the configured price_level. Override per dataset.
    delta_mid_clip:             float = 24.0

    # ------------------------------------------------------------------
    # Returns-conditioning ablation (joint model only)
    # ------------------------------------------------------------------
    # Used by JointReturnsCondTimeGradGenerator. Ignored by all other paths.
    # The returns-MLP takes the last `timegrad_return_window` Δmid values and
    # projects them to a `timegrad_return_proj_dim` feature concat'd to the
    # encoder cond. The MLP's final layer is zero-initialised so the model
    # starts identical to the joint baseline; per-sample modality dropout
    # zeroes the path during training to prevent encoder atrophy.
    timegrad_return_window:           int   = 100  # past returns fed to the path
    timegrad_return_proj_dim:         int   = 32   # MLP output dim (when use_return_mlp=True)
    timegrad_return_mlp_hidden:       int   = 64   # MLP hidden width  (when use_return_mlp=True)
    timegrad_return_index:            int   = 0    # feature index of Δmid in joint vec
    timegrad_return_modality_dropout: float = 0.25 # per-sample mask probability
    timegrad_use_return_mlp:          bool  = True # True: MLP projection (zero-init); False: raw passthrough

    # ------------------------------------------------------------------
    # Hawkes process
    # ------------------------------------------------------------------
    hawkes_beta:  float = 1.0    # exponential kernel decay
    hawkes_max_iter: int = 200   # MLE optimisation iterations

    # ------------------------------------------------------------------
    # Derived (read-only helpers)
    # ------------------------------------------------------------------
    @property
    def input_dim(self) -> int:
        """2k-dim: k signed bid queues (≤0) + k signed ask queues (≥0). Anchor is separate metadata."""
        return 2 * self.levels

    @property
    def timegrad_input_dim(self) -> int:
        """2k-dim: signed queue features only (anchor not modelled)."""
        return 2 * self.levels

    @property
    def joint_input_dim(self) -> int:
        """(2k+1)-dim: Δmid feature prepended to queue features."""
        return 2 * self.levels + 1


# GRU -> RNN
# Layer Multiple
# Prediction length