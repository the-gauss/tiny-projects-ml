from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _move_to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors in a structure to the given device."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    return x


def _unpack_batch(
    batch: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Unpack common batch shapes to (inputs, lengths, targets).

    Supported forms:
    - (inputs, targets)
    - (inputs, lengths, targets)
    - {"x"/"inputs": ..., "lengths"?: ..., "y"/"targets": ...}
    """
    lengths = None
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, lengths, y = batch
        else:
            raise ValueError(f"Unsupported batch tuple size: {len(batch)}")
        return x, lengths, y

    if isinstance(batch, dict):
        x = batch.get("x", batch.get("inputs"))
        y = batch.get("y", batch.get("targets"))
        lengths = batch.get("lengths")
        if x is None or y is None:
            raise ValueError("Batch dict must contain 'x'/'inputs' and 'y'/'targets'.")
        return x, lengths, y

    raise ValueError("Unsupported batch type; expected tuple/list or dict.")


@torch.no_grad()
def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct, total


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: Optional[float] = 1.0,
    log_interval: int = 100,
) -> Tuple[float, float]:
    """Train for a single epoch and return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, batch in enumerate(dataloader, start=1):
        x, lengths, y = _unpack_batch(batch)
        x, y = _move_to_device(x, device), _move_to_device(y, device)
        lengths = _move_to_device(lengths, device) if lengths is not None else None

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, lengths) if lengths is not None else model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, lengths) if lengths is not None else model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * y.size(0)
        c, n = _compute_metrics(logits, y)
        correct += c
        total += n

        if log_interval and step % log_interval == 0:
            avg_loss = running_loss / total
            acc = correct / total
            print(f"  step {step:5d} | loss {avg_loss:.4f} | acc {acc:.4f}")

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate on a dataloader and return (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        x, lengths, y = _unpack_batch(batch)
        x, y = _move_to_device(x, device), _move_to_device(y, device)
        lengths = _move_to_device(lengths, device) if lengths is not None else None

        logits = model(x, lengths) if lengths is not None else model(x)
        loss = loss_fn(logits, y)

        running_loss += loss.item() * y.size(0)
        c, n = _compute_metrics(logits, y)
        correct += c
        total += n

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    *,
    scheduler: Optional[Any] = None,
    grad_clip: Optional[float] = 1.0,
    mixed_precision: bool = True,
    early_stopping_patience: Optional[int] = 5,
    checkpoint_path: Optional[str] = None,
    log_interval: int = 100,
) -> List[TrainResult]:
    """Full training loop with validation, LR scheduling and early stopping.

    The model's forward can be either model(x) or model(x, lengths), matching the
    collate_fn used to create batches. Batches can be tuples/lists or dicts.
    """
    device = torch.device(device)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device.type == "cuda") else None

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0
    history: List[TrainResult] = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler=scaler,
            grad_clip=grad_clip,
            log_interval=log_interval,
        )

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        # Scheduler handling: ReduceLROnPlateau vs others
        if scheduler is not None:
            if hasattr(scheduler, "step"):
                try:
                    # Try ReduceLROnPlateau signature
                    scheduler.step(val_loss)
                except TypeError:
                    scheduler.step()

        result = TrainResult(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history.append(result)

        print(
            f"  train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        # Early stopping + checkpointing
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if checkpoint_path:
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    return history


__all__ = [
    "TrainResult",
    "set_seed",
    "train_one_epoch",
    "evaluate",
    "fit",
]

