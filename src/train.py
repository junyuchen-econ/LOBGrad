from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================== TimeSeriesDataset ===============================
class TimeSeriesDataset(Dataset):
    """
    Creates sequences from time series data for training.
    Each sample consists of a history window and a target value.
    """
    def __init__(self, targets: np.ndarray, context_length: int, prediction_length: int) -> None:
        """
        Parameters:
        -----------
        targets: np.ndarray, shape (num_timesteps, output_dim)
            Target values to predict (scalar or multivariate)
        context_length: int
            Length of history window
        prediction_length: int
            Length of prediction window
        """

        self.targets = targets # (num_timesteps, output_dim)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = self.targets.shape[0]

        assert self.total_length >= context_length + prediction_length, "Not enough data."
        
    @property
    def valid_indices(self) -> list[int]:
        """Valid prediction start positions for use with torch.utils.data.Subset."""
        return list(range(self.context_length, self.total_length - self.prediction_length + 1))

    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        ----------
        index: int
            Prediction start position in the time series (absolute index).
            history = targets[index - context_length : index]
            prediction = targets[index : index + prediction_length]

        Returns:
        -------
        history: torch.Tensor, shape (context_length, output_dim)
        prediction: torch.Tensor, shape (prediction_length, output_dim)
        """
        history = self.targets[index - self.context_length : index].astype(np.float32)
        prediction = self.targets[index : index + self.prediction_length].astype(np.float32)
        
        return torch.from_numpy(history), torch.from_numpy(prediction)

# =============================== train_epoch ===============================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int = None,
    max_grad_norm: float | None = None,
) -> float:
    """Train for one epoch, sampling at most `max_batches` random batches."""

    model.train()
    total_loss = 0.0
    num_samples = 0

    total = max_batches if max_batches is not None else len(dataloader)
    pbar = tqdm(dataloader, total=total, leave=True, desc="  train")
    for x, y in pbar:
        if max_batches is not None and num_samples >= max_batches:
            break

        # (batch, history_length, output_dim)
        x = x.to(device)
        y = y.to(device)

        loss = model.loss(x, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to stabilize diffusion training (as in pytorch-ts reference)
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_samples += 1
        avg_so_far = total_loss / num_samples
        pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{avg_so_far:.5f}")

    avg_loss = total_loss / num_samples

    return avg_loss

# =============================== validate ===============================
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, leave=True, desc="    val"):
            # (batch, context_length, output_dim)
            x = x.to(device)
            # (batch, prediction_length, output_dim)
            y = y.to(device)
            
            loss = model.loss(x, y)
            total_loss += loss.item()

        # compute average loss
        avg_loss = total_loss / len(dataloader)

    return avg_loss

# =============================== train_model ===============================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    verbose: bool = True,
    save_dir: str = 'checkpoints',
    max_batches: int = None,
    max_grad_norm: float | None = None,
    scheduler=None,
) -> None:

    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # train for num_epochs
    epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch", disable=not verbose)
    for epoch in epoch_bar:

        train_loss = train_epoch(model, train_loader, optimizer, device, max_batches, max_grad_norm)
        val_loss = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step()

        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}", best=f"{min(val_losses):.5f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
    
    # Save fully trained model (final epoch)
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(save_dir, 'final_model.pt'))
    
    # Save training loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150)
    plt.close()
    
    print(f'\nTraining complete! Best validation loss: {best_val_loss:.6f}')


