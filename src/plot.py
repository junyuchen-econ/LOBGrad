import numpy as np
import matplotlib.pyplot as plt

# =============================== plot_true_vs_predicted_panels ===============================
#
def plot_true_vs_predicted_panels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = None,
    save_path: str = "prediction_panels.png",
) -> None:
    """True vs Predicted in separate panels."""
    y_t = y_true[:max_points] if max_points else y_true
    y_p = y_pred[:max_points] if max_points else y_pred

    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    ax[0].plot(y_t, label="True", color="black")
    ax[0].legend(loc="upper right")
    ax[1].plot(y_p, label="Predicted", color="red")
    ax[1].legend(loc="upper right")
    plt.tight_layout()
    plt.xlabel("Time")
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =============================== plot_true_vs_predicted_overlay ===============================
def plot_true_vs_predicted_overlay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = None,
    save_path: str = "prediction_overlay.png",
) -> None:
    """True vs Predicted overlaid on the same axis."""
    y_t = y_true[:max_points] if max_points else y_true
    y_p = y_pred[:max_points] if max_points else y_pred

    plt.figure(figsize=(12, 5))
    plt.plot(y_t, label="True", color="black")
    plt.plot(y_p, label="Predicted", color="red")
    plt.legend(loc="upper right")
    plt.xlabel("Time")
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =============================== plot_fan_chart ===============================
def plot_fan_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_samples: np.ndarray,
    save_path: str = "prediction_fan_chart.png",
) -> None:
    """
    Fan chart with nested prediction-interval bands.

    Parameters
    ----------
    y_true :        (T,)
    y_pred :        (T,)  mean prediction
    price_samples : (T, num_samples)
    """
    t = np.arange(len(y_pred))

    q05 = np.percentile(price_samples, 5, axis=1)
    q10 = np.percentile(price_samples, 10, axis=1)
    q25 = np.percentile(price_samples, 25, axis=1)
    q50 = np.median(price_samples, axis=1)
    q75 = np.percentile(price_samples, 75, axis=1)
    q90 = np.percentile(price_samples, 90, axis=1)
    q95 = np.percentile(price_samples, 95, axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(t, q05, q95, alpha=0.15, color="tab:blue", label="5–95% interval")
    ax.fill_between(t, q10, q90, alpha=0.25, color="tab:blue", label="10–90% interval")
    ax.fill_between(t, q25, q75, alpha=0.40, color="tab:blue", label="25–75% interval")
    ax.plot(t, q50, color="tab:blue", linewidth=1.2, label="Median")
    ax.plot(t, y_pred, color="tab:red", linewidth=1.2, linestyle="--", label="Mean")
    ax.plot(t, y_true, color="black", linewidth=1.2, label="True")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mid Price (Z-Score)")
    ax.set_title("Prediction Intervals (Fan Chart)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =============================== plot_sample_paths ===============================
def plot_sample_paths(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_samples: np.ndarray,
    max_paths: int,
    save_path: str = "prediction_sample_paths.png",
) -> None:
    """
    Individual diffusion sample paths.

    Parameters
    ----------
    y_true :        (T,)
    y_pred :        (T,)  mean prediction
    price_samples : (T, num_samples)
    max_paths :     how many paths to draw
    """
    t = np.arange(len(y_pred))
    n_paths = min(price_samples.shape[1], max_paths)

    fig, ax = plt.subplots(figsize=(14, 6))
    for s in range(n_paths):
        ax.plot(t, price_samples[:, s], color="tab:blue", alpha=0.08, linewidth=0.5)
    ax.plot(t, y_pred, color="tab:red", linewidth=1.5, label="Mean")
    ax.plot(t, y_true, color="black", linewidth=1.5, label="True")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mid Price (Z-Score)")
    ax.set_title(f"Sample Paths ({n_paths} draws)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =============================== plot_predictive_distributions ===============================
def plot_predictive_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_samples: np.ndarray,
    save_path: str = "prediction_predictive_distributions.png",
) -> None:
    """
    Histograms of the predictive distribution at evenly-spaced timesteps.

    Parameters
    ----------
    y_true :        (T,)
    y_pred :        (T,)  mean prediction
    price_samples : (T, num_samples)
    """
    T = len(y_pred)
    timesteps = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    fig, axes = plt.subplots(
        1, len(timesteps), figsize=(4 * len(timesteps), 4), sharey=True
    )
    for ax, ts in zip(axes, timesteps):
        ax.hist(
            price_samples[ts], bins=20, density=True,
            color="tab:blue", alpha=0.6, edgecolor="white",
        )
        ax.axvline(y_true[ts], color="black", linewidth=1.5, label="True")
        ax.axvline(y_pred[ts], color="tab:red", linewidth=1.5, linestyle="--", label="Mean")
        ax.set_title(f"t = {ts}")
        ax.set_xlabel("Mid Price")
    axes[0].set_ylabel("Density")
    axes[-1].legend(loc="upper right")
    plt.suptitle("Predictive Distribution at Selected Timesteps", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================== plot_queue_size_bands ===============================
def plot_queue_size_bands(
    real: np.ndarray,
    gen_samples: np.ndarray,
    levels: int,
    save_path: str = "queue_size_bands.png",
) -> None:
    """
    Prediction-interval bands for bid/ask queue sizes at each level.

    Parameters
    ----------
    real        : (T, 2*levels)  true queue sizes
    gen_samples : (num_paths, T, 2*levels)  generated queue-size samples
    levels      : number of LOB levels per side
    """
    T = real.shape[0]
    q10 = np.quantile(gen_samples, 0.10, axis=0)
    q25 = np.quantile(gen_samples, 0.25, axis=0)
    q50 = np.median(gen_samples, axis=0)
    q75 = np.quantile(gen_samples, 0.75, axis=0)
    q90 = np.quantile(gen_samples, 0.90, axis=0)

    fig, axes = plt.subplots(2, levels, figsize=(18, 6), sharex=True)
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14

    for i in range(levels):
        bid_col = levels - 1 - i
        for row, col in [(0, bid_col), (1, levels + i)]:
            axes[row, i].fill_between(range(T), q10[:, col], q90[:, col], alpha=0.2, label="10%–90%")
            axes[row, i].fill_between(range(T), q25[:, col], q75[:, col], alpha=0.4, label="25%–75%")
            axes[row, i].plot(q50[:, col], label="Median", linewidth=1, color="red")
            axes[row, i].plot(real[:, col], label="Real", linewidth=1, color="black")
            #axes[row, i].set_ylabel('Queue Size', fontsize=label_fontsize)
            axes[row, i].tick_params(axis='both', labelsize=tick_fontsize)

        axes[0, i].set_title(f"Bid Queue {i + 1}", fontsize=title_fontsize)
        axes[1, i].set_title(f"Ask Queue {i + 1}", fontsize=title_fontsize)

    # Move legend to lower middle of the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=label_fontsize, ncol=2, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# =============================== plot_queue_size_distributions ===============================
def plot_queue_size_distributions(
    real: np.ndarray,
    gen_samples: np.ndarray,
    levels: int,
    bins: int = 50,
    save_path: str = "queue_size_distributions.png",
) -> None:
    """
    Overlaid histograms comparing real vs generated queue-size distributions
    for each bid/ask level.

    Parameters
    ----------
    real        : (T, 2*levels)  true queue sizes over time
    gen_samples : (num_paths, T, 2*levels)  generated queue-size samples
    levels      : number of LOB levels per side
    bins        : histogram bin count
    """
    real_flat = real.reshape(-1, 2 * levels)
    gen_flat = gen_samples.reshape(-1, 2 * levels)

    fig, axes = plt.subplots(2, levels, figsize=(15, 6))

    for i in range(levels):
        bid_col = levels - 1 - i
        axes[0, i].hist(real_flat[:, bid_col], bins=bins, alpha=0.5, label="Real", density=True)
        axes[0, i].hist(gen_flat[:, bid_col], bins=bins, alpha=0.5, label="Generated", density=True)
        axes[0, i].set_title(f"Bid level {i + 1}")

        axes[1, i].hist(real_flat[:, levels + i], bins=bins, alpha=0.5, label="Real", density=True)
        axes[1, i].hist(gen_flat[:, levels + i], bins=bins, alpha=0.5, label="Generated", density=True)
        axes[1, i].set_title(f"Ask level {i + 1}")

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
