"""
pareto.py
=========
Visualisation tool for the fairness-accuracy Pareto frontier.

Trains a fresh model for each value of the fairness regularization
hyperparameter ``eta``, evaluates accuracy and demographic parity gap on a
validation set, then generates and saves a Pareto frontier plot.

Usage
-----
    from training.pareto import plot_pareto_frontier

    results = plot_pareto_frontier(
        model_fn=lambda: MyNet(),
        train_loader=train_dl,
        val_loader=val_dl,
        eta_values=[0.0, 0.05, 0.1, 0.5, 1.0, 2.0],
        epochs=15,
        save_path='reports/pareto.png',
    )
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from fairness_regularizer import FairnessRegularizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model: nn.Module,
    val_loader,
    device: str,
    sensitive_key: str,
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.

    Returns
    -------
    accuracy : float
    demographic_parity_gap : float
        Max pairwise difference in mean predicted probabilities across groups.
        Lower = fairer.
    """
    model.eval()
    all_probs, all_y, all_sensitive = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            logits = model(X).squeeze(-1)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_y.append(batch["y"])
            all_sensitive.append(batch[sensitive_key])

    probs = torch.cat(all_probs)
    y = torch.cat(all_y).float()
    sensitive = torch.cat(all_sensitive)

    preds = (probs >= 0.5).float()
    accuracy = (preds == y).float().mean().item()

    groups = sensitive.unique()
    group_means = [probs[sensitive == g].mean().item() for g in groups]
    dp_gap = max(group_means) - min(group_means) if len(group_means) >= 2 else 0.0

    return accuracy, dp_gap


# ---------------------------------------------------------------------------
# Single-eta training run
# ---------------------------------------------------------------------------

def _train_single(
    model_fn: Callable[[], nn.Module],
    train_loader,
    val_loader,
    eta: float,
    device: str,
    epochs: int,
    lr: float,
    sensitive_key: str,
    sensitive_dict_key: Optional[List[str]] = None,  
) -> Tuple[float, float]:
    """Train one model with the given eta and return (accuracy, dp_gap)."""
    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FairnessRegularizer(eta=eta)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(X).squeeze(-1)

            if sensitive_dict_key is not None:
                s_dict = {
                    k: batch[k].to(device)
                    for k in sensitive_dict_key
                }
                loss = loss_fn(logits, y, sensitive_dict=s_dict)
            else:
                sensitive = batch[sensitive_key].to(device)
                loss = loss_fn(logits, y, sensitive=sensitive)

            loss.backward()
            optimizer.step()

    return _evaluate(model, val_loader, device, sensitive_key)


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def _is_pareto_efficient(accuracies: List[float], fairness_gaps: List[float]) -> List[bool]:
    """
    Return a boolean mask indicating which points lie on the Pareto frontier.

    A point is Pareto-efficient if no other point is simultaneously more
    accurate AND fairer (lower gap).
    """
    n = len(accuracies)
    on_frontier = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if it's at least as accurate and strictly fairer,
            # or strictly more accurate and at least as fair
            if (
                accuracies[j] >= accuracies[i]
                and fairness_gaps[j] <= fairness_gaps[i]
                and (accuracies[j] > accuracies[i] or fairness_gaps[j] < fairness_gaps[i])
            ):
                on_frontier[i] = False
                break
    return on_frontier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_pareto_frontier(
    model_fn: Callable[[], nn.Module],
    train_loader,
    val_loader,
    eta_values: Optional[List[float]] = None,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
    sensitive_key: str = "sensitive",
    sensitive_dict_key: Optional[List[str]] = None,
    save_path: str = "pareto_frontier.png",
    title: str = "Pareto Frontier: Fairness vs Accuracy",
) -> List[Tuple[float, float, float]]:
    """
    Sweep eta values, train a fresh model per value, and plot the Pareto frontier.

    Parameters
    ----------
    model_fn : callable() -> nn.Module
        Factory that returns a freshly initialised model.
    train_loader : DataLoader
        Batches are dicts with ``'X'``, ``'y'``, and ``sensitive_key``.
    val_loader : DataLoader
        Same structure as ``train_loader``.
    eta_values : list of float or None
        Fairness regularization strengths to sweep.
        Defaults to ``[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]``.
    device : str
    epochs : int
        Training epochs per eta value.
    lr : float
        Adam learning rate.
    sensitive_key : str
        Key in each batch dict holding the sensitive attribute tensor.
    sensitive_dict_key : list of str or None
        If provided, these keys are collected from the batch into a
        ``sensitive_dict`` for intersectional regularization.
    save_path : str
        File path for the saved figure.
    title : str
        Plot title.

    Returns
    -------
    list of (eta, accuracy, dp_gap) tuples
    """
    if eta_values is None:
        eta_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    results: List[Tuple[float, float, float]] = []

    for eta in eta_values:
        print(f"  Training η={eta} ...", end=" ", flush=True)
        acc, gap = _train_single(
            model_fn=model_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            eta=eta,
            device=device,
            epochs=epochs,
            lr=lr,
            sensitive_key=sensitive_key,
            sensitive_dict_key=sensitive_dict_key,
        )
        results.append((eta, acc, gap))
        print(f"Acc={acc:.4f}  DP-Gap={gap:.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    etas       = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    dp_gaps    = [r[2] for r in results]
    on_frontier = _is_pareto_efficient(accuracies, dp_gaps)

    fig, ax = plt.subplots(figsize=(9, 6))

    # All points, coloured by eta
    scatter = ax.scatter(
        dp_gaps, accuracies,
        c=etas, cmap="viridis", s=90, zorder=4, alpha=0.85,
        label="All models",
    )

    # Pareto-efficient points highlighted
    pf_gaps = [dp_gaps[i] for i, ok in enumerate(on_frontier) if ok]
    pf_accs = [accuracies[i] for i, ok in enumerate(on_frontier) if ok]
    # Sort by gap for the connecting line
    pf_sorted = sorted(zip(pf_gaps, pf_accs))
    if pf_sorted:
        pf_x, pf_y = zip(*pf_sorted)
        ax.plot(pf_x, pf_y, "r--", linewidth=1.5, zorder=3, label="Pareto frontier")
        ax.scatter(pf_x, pf_y, color="red", s=120, zorder=5, marker="*")

    # Annotations
    for eta, acc, gap in results:
        ax.annotate(
            f"η={eta}",
            (gap, acc),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("η (fairness regularization strength)")

    ax.set_xlabel("Demographic Parity Gap  (↓ fairer)", fontsize=11)
    ax.set_ylabel("Accuracy  (↑ better)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved → {save_path}")

    return results