"""
LSTMTrainer — supervised training loop for the LSTM transition model.

Trains via MSE on consecutive (window → next_state) pairs from the dataset.
Saves weights to results/lstm_weights.pt after training.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..shared.config import ConfigManager
from ..shared.seeding import set_global_seed
from .lstm_model import LSTMTransitionModel


class LSTMTrainer:
    """
    Build, train, and persist the LSTMTransitionModel.

    Input:  data dict from DataService.run_pipeline()
    Output: dict with keys: model, train_losses, val_losses
    Setup:  ConfigManager with lstm/* and paths/* keys
    """

    def __init__(self, cfg: ConfigManager) -> None:
        """Initialise with shared config; no model is built until train() is called."""
        self._cfg = cfg

    def train(self, data: dict) -> dict:
        """
        Full supervised training loop.

        Steps:
            1. Build LSTMTransitionModel from config hyperparams.
            2. Wrap train/val tensors in DataLoaders.
            3. Run Adam optimisation for the configured number of epochs.
            4. Record train and val MSE loss after each epoch.
            5. Save model weights to config paths.lstm_weights.
            6. Return model and loss histories.

        Args:
            data: Output of DataService.run_pipeline() — must contain keys
                  X_train, X_actions_train, y_train, X_val, X_actions_val, y_val.

        Returns:
            dict with keys: model (LSTMTransitionModel), train_losses (list[float]),
            val_losses (list[float]).
        """
        set_global_seed(self._cfg.get("seed"))  # reproducible LSTM weights
        model = self._build_model()
        lstm_cfg = self._cfg.get("lstm") or {}
        lr: float = lstm_cfg.get("learning_rate", 1e-3)
        epochs: int = lstm_cfg.get("epochs", 50)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_loader = self._make_loader(
            data["X_train"], data["X_actions_train"], data["y_train"], shuffle=True
        )
        val_loader = self._make_loader(
            data["X_val"], data["X_actions_val"], data["y_val"], shuffle=False
        )

        train_losses: list[float] = []
        val_losses: list[float] = []
        for _ in range(epochs):
            train_losses.append(self._epoch(train_loader, model, optimizer, criterion, True))
            val_losses.append(self._epoch(val_loader, model, optimizer, criterion, False))

        self._save(model)
        return {"model": model, "train_losses": train_losses, "val_losses": val_losses}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> LSTMTransitionModel:
        """Instantiate LSTMTransitionModel from config values."""
        lstm_cfg = self._cfg.get("lstm") or {}
        data_cfg = self._cfg.get("data") or {}
        return LSTMTransitionModel(
            state_dim=data_cfg.get("state_dim", 5),
            n_actions=data_cfg.get("n_actions", 6),
            action_embed_dim=lstm_cfg.get("action_embed_dim", 8),
            hidden_size=lstm_cfg.get("hidden_size", 64),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.2),
        )

    def _make_loader(self, x_s, x_a, y, *, shuffle: bool) -> DataLoader:
        """Wrap tensors in a TensorDataset and return a DataLoader."""
        batch_size: int = (self._cfg.get("lstm") or {}).get("batch_size", 64)
        return DataLoader(TensorDataset(x_s, x_a, y), batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def _epoch(
        loader: DataLoader,
        model: LSTMTransitionModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        training: bool,
    ) -> float:
        """Run one epoch; return mean loss. Handles both train and eval modes."""
        model.train(training)
        total = 0.0
        with torch.set_grad_enabled(training):
            for x_s, x_a, y_batch in loader:
                pred = model(x_s, x_a)
                loss = criterion(pred, y_batch)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item()
        return total / max(len(loader), 1)

    def _save(self, model: LSTMTransitionModel) -> None:
        """Save model state_dict to the path defined in config."""
        save_path = (self._cfg.get("paths") or {}).get("lstm_weights", "results/lstm_weights.pt")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
