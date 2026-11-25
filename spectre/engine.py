"""Training accepts only train/validation loaders. Test inference is a separate operation."""
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def set_seed(seed):
    # Must be set before any CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def loader(dataset, batch_size, workers, seed, training=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      drop_last=False, num_workers=workers, generator=generator,
                      worker_init_fn=seed_worker)


def epoch_loss(model, batches, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, samples, heads_total = 0.0, 0, None
    criterion = torch.nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for images, targets in batches:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if len(outputs) != targets.shape[1]:
                raise ValueError("Number of model heads differs from target count")
            losses = torch.stack([criterion(output, targets[:, i]) for i, output in enumerate(outputs)])
            loss = losses.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; run stopped without reporting test metrics")
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite gradients")
                optimizer.step()
            count = len(images)
            samples += count
            total += loss.detach().item() * count
            values = losses.detach().cpu().numpy().astype(float) * count
            heads_total = values if heads_total is None else heads_total + values
    if samples == 0:
        raise ValueError("Empty loader")
    return total / samples, (heads_total / samples).tolist()


def fit(model, train_loader, validation_loader, device, epochs, patience, lr, directory):
    """Select one checkpoint using internal validation loss only; no test argument exists."""
    if epochs < 1 or patience < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs/patience/lr must be positive and finite")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / "best.pt"
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    best, best_epoch, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_heads = epoch_loss(model, train_loader, device, optimizer)
        val_loss, val_heads = epoch_loss(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "train_head_losses": train_heads, "validation_head_losses": val_heads})
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            tmp = checkpoint.with_suffix(".tmp")
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, tmp)
            tmp.replace(checkpoint)
        else:
            stale += 1
        write_json(directory / "history.json", history)
        print(f"Epoch {epoch}: train loss={train_loss:.6f}, inner validation loss={val_loss:.6f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    selection = {"criterion": "minimum_internal_validation_loss", "best_epoch": best_epoch,
                 "best_validation_loss": best, "epochs_run": len(history)}
    write_json(directory / "selection.json", selection)
    return selection


def predict(model, batches, device):
    model.eval()
    labels, predictions = [], []
    with torch.inference_mode():
        for images, targets in batches:
            outputs = model(images.to(device))
            if len(outputs) != targets.shape[1] or any(not torch.isfinite(x).all() for x in outputs):
                raise ValueError("Invalid model predictions")
            predictions.append(torch.stack([output.argmax(dim=1) for output in outputs], dim=1).cpu().numpy())
            labels.append(targets.numpy())
    if not labels:
        raise ValueError("Empty evaluation loader")
    return np.concatenate(labels), np.concatenate(predictions)
