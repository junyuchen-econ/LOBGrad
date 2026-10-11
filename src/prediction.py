from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import time

# =============================== Autoregressive Prediction ===============================
def generate_autoregressive_prediction(
    model: nn.Module,
    X_test: np.ndarray,           # (num_timesteps, full_dim) where full_dim = 2*levels+1, col 0 = midprice
    start_idx: int,
    context_length: int,
    prediction_steps: int,
    levels: int,
    num_paths: int = 50,
    bid_clip: tuple = (-8, -1),
    ask_clip: tuple = (1, 8),
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully autoregressive simulation: the model generates `prediction_steps`
    future steps in one shot (its own internal rollout), conditioned on the
    true history window.  This is called once with all paths batched together
    to obtain a set of stochastic trajectories (same seed, different noise).

    The model's input and output dimension is full_dim = 2*levels + 1,
    laid out as [midprice, bid_1, ..., bid_L, ask_1, ..., ask_L].

    Returns
    -------
    price_paths      : (num_paths, prediction_steps)  predicted midprice (normalized)
    quantity_paths    : (num_paths, prediction_steps, 2*levels)  log-normalized quantities
    """
    model.eval()
    model.to(device)

    # Tile context num_paths times so each copy gets independent stochastic noise
    # during the reverse diffusion process, producing diverse trajectories
    seed = X_test[start_idx - context_length : start_idx]  # (context_length, full_dim)
    targets_context = torch.tensor(
        np.tile(seed, (num_paths, 1, 1)), dtype=torch.float32, device=device
    )  # (num_paths, context_length, full_dim)

    if verbose:
        print("Generating autoregressive paths …")
        t0 = time.time()

    with torch.no_grad():
        samples = model.sample(targets_context, prediction_steps)  # (num_paths, prediction_steps, full_dim)

    samples = samples.cpu().numpy()

    # Clip quantities — bids at indices 1:levels+1, asks at indices levels+1:
    samples[:, :, 1:levels + 1] = np.clip(samples[:, :, 1:levels + 1], *bid_clip)
    samples[:, :, levels + 1:]  = np.clip(samples[:, :, levels + 1:],  *ask_clip)

    price_paths    = samples[:, :, 0]   # (num_paths, prediction_steps)
    quantity_paths = samples[:, :, 1:]  # (num_paths, prediction_steps, 2*levels)

    if verbose:
        print(f"Done in {time.time() - t0:.2f}s")

    return price_paths, quantity_paths


# =============================== Teacher-Forced Prediction ===============================
def generate_teacher_forced_prediction(
    model: nn.Module,
    X_test: np.ndarray,           # (num_timesteps, 2*levels+1)
    start_idx: int,
    context_length: int,
    prediction_steps: int,
    levels: int,
    num_paths: int = 50,
    bid_clip: tuple = (-8, -1),
    ask_clip: tuple = (1, 8),
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling one-step prediction using true history (no recursive feedback).
    At each step the true history window from X_test is used as input,
    producing `num_paths` independent stochastic samples.

    The model's input and output dimension is full_dim = 2*levels + 1,
    laid out as [midprice, bid_1, ..., bid_L, ask_1, ..., ask_L].

    Returns
    -------
    price_paths      : (num_paths, prediction_steps)  predicted midprice (normalized)
    queue_paths    : (num_paths, prediction_steps, 2*levels)  log-normalized quantities
    """
    model.eval()
    model.to(device)

    price_paths = np.zeros((num_paths, prediction_steps))
    queue_paths = np.zeros((num_paths, prediction_steps, 2 * levels))

    for step in range(prediction_steps):
        if verbose and step % 10 == 0:
            start_time = time.time()

        # True history window (teacher-forced), expanded to num_paths for stochastic sampling
        x_window = X_test[start_idx - context_length + step : start_idx + step]          # (context_length, input_dim)
        x = torch.tensor(x_window, dtype=torch.float32, device=device).unsqueeze(0)     # (1, context_length, input_dim)
        x = x.expand(num_paths, -1, -1)                                                 # (num_paths, context_length, input_dim)

        with torch.no_grad():
            predicted = model.sample(x, prediction_steps=1).cpu().numpy()               # (num_paths, 1, output_dim)
            predicted = predicted[:, 0, :]                                              # (num_paths, output_dim)
    
        # Clip quantities — bids at indices 1:levels+1, asks at indices levels+1:
        predicted[:, 1:levels + 1]  = np.clip(predicted[:, 1:levels + 1], *bid_clip)
        predicted[:, levels + 1:]   = np.clip(predicted[:, levels + 1:], *ask_clip)

        # Store results — midprice is index 0, quantities are indices 1:
        price_paths[:, step] = predicted[:, 0]
        queue_paths[:, step] = predicted[:, 1:]

        if verbose and step % 10 == 0:
            print(f"Step {step}/{prediction_steps} | time: {time.time() - start_time:.3f}s")

    return price_paths, queue_paths


if __name__ == "__main__":

    import random
    from model import TimeGradMLP, TimeGradWaveNet
    from dataloader import LOB_Dataset
    from utils import normalize_midprice, normalize_quantities, add_midprice
    from plot import (
        plot_fan_chart,
        plot_queue_size_bands,
        plot_queue_size_distributions,
    )
    from config import config as c

    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    torch.cuda.manual_seed_all(c.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    dataset_test = LOB_Dataset(
        SYMBOL=c.symbol,
        START_DATE=c.start_date_test,
        END_DATE=c.end_date_test,
        LEVELS=c.level,
        PRICE_LEVEL=c.price_level,
        TIME_DIFF=c.time_diff,
        DEPTH=c.depth,
    )
    queue_test = dataset_test.centered_quantities 
    price_test = dataset_test.mid_prices

    # Normalize separately then concatenate
    queue_test = normalize_quantities(queue_test, c.level, norm_method="log")
    price_test = normalize_midprice(price_test, norm_method="Z_Score")
    X_test = add_midprice(queue_test, price_test).T # (num_timesteps, 2*levels+1)
    print(f"X_test shape: {X_test.shape}")

    shared_kwargs = dict(
        input_dim=c.input_dim,
        output_dim=c.input_dim,
        hidden_encoder_dim=c.hidden_encoder_dim,
        num_encoder_layers=c.num_encoder_layers,
        emb_dim=c.emb_dim,
        K=c.k,
        beta_start=c.beta_start,
        beta_end=c.beta_end,
        schedule=c.schedule,
        conditioning_length=c.conditioning_length,
        loss_type=c.loss_type,
        dropout_rate=c.dropout_rate,
        scaling=c.scaling,
        cell_type=c.cell_type,
    )

    if c.noise_predictor_type == "mlp":
        model = TimeGradMLP(**shared_kwargs, hidden_noise_dim=c.hidden_noise_dim)
    else:
        model = TimeGradWaveNet(
            **shared_kwargs,
            residual_layers=c.residual_layers,
            residual_channels=c.residual_channels,
            dilation_cycle_length=c.dilation_cycle_length,
        )
        
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params}")
    
    model.load_state_dict(torch.load('checkpoints/final_model.pt', weights_only=True)['model_state_dict'])
    model.eval()
    model.to(device)

    prediction_steps = c.prediction_steps
    start_idx = c.start_index
    y_true_prices = price_test[start_idx : start_idx + prediction_steps]
    y_true_queue = queue_test[:, start_idx : start_idx + prediction_steps].T   # (prediction_steps, 2*levels)

    # =========================================================================
    # Teacher-Forced Prediction
    # =========================================================================
    print("\n=== Teacher-Forced Prediction ===")
    one_step_price_paths, one_step_queue_paths = generate_teacher_forced_prediction(
        model=model,
        X_test=X_test,
        start_idx=start_idx,
        context_length=c.context_length,
        prediction_steps=prediction_steps,
        levels=c.level,
        num_paths=c.num_generated_paths,
        device=device,
        verbose=True,
    )

    plot_queue_size_bands(y_true_queue, one_step_queue_paths, levels=c.level, save_path="figures/one_step_queue_size_bands.png")
    plot_queue_size_distributions(y_true_queue, one_step_queue_paths, levels=c.level, save_path="figures/one_step_queue_size_distributions.png")
    
    one_step_mean_price = one_step_price_paths.mean(axis=0)  # (prediction_steps,)
    plot_fan_chart(y_true=y_true_prices, y_pred=one_step_mean_price, price_samples=one_step_price_paths.T, save_path="figures/autoregressive_price_fan_chart.png")

    # =========================================================================
    # Autoregressive Prediction
    # =========================================================================
    print("\n=== Autoregressive Prediction ===")
    autoregressive_price_paths, autoregressive_queue_paths = generate_autoregressive_prediction(
        model=model,
        X_test=X_test,
        start_idx=start_idx,
        context_length=c.context_length,
        prediction_steps=prediction_steps,
        levels=c.level,
        num_paths=c.num_generated_paths,
        device=device,
        verbose=True,
    )

    ar_mean_price = autoregressive_price_paths.mean(axis=0)  # (prediction_steps,)
    plot_fan_chart(y_true=y_true_prices, y_pred=ar_mean_price, price_samples=autoregressive_price_paths.T, save_path="figures/autoregressive_price_fan_chart.png")