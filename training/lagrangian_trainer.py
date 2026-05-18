"""
lagrangian_trainer.py
=====================
Trainer class implementing Lagrangian optimization for hard fairness
constraints on PyTorch neural networks.

The trainer maintains two sets of parameters:
  - Model weights      → gradient *descent*  (minimise the Lagrangian)
  - Lagrange multipliers → gradient *ascent*  (enforce the constraints)

The resulting saddle-point solution satisfies the fairness constraints while
minimising the primary loss (BCE).

Usage
-----
    from training.lagrangian_trainer import LagrangianFairnessTrainer, demographic_parity_constraint

    trainer = LagrangianFairnessTrainer(
        model=my_model,
        constraint_fns=[demographic_parity_constraint],
        model_lr=1e-3,
        multiplier_lr=1e-2,
    )
    trainer.fit(train_loader, epochs=20)
    preds = trainer.predict(X_test_tensor)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Built-in constraint functions
# ---------------------------------------------------------------------------

def demographic_parity_constraint(
    logits: torch.Tensor,
    sensitive: torch.Tensor,
    slack: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Demographic parity: |P(ŷ=1|A=0) - P(ŷ=1|A=1)| - slack ≤ 0.

    Returns the constraint *violation* (positive = violated).
    """
    probs = torch.sigmoid(logits)
    groups = sensitive.unique()
    if len(groups) < 2:
        return torch.tensor(0.0, device=logits.device)

    means = [probs[sensitive == g].mean() for g in groups]
    # Use max pairwise difference
    violations = torch.stack([
        (means[i] - means[j]).abs() - slack
        for i in range(len(means))
        for j in range(i + 1, len(means))
    ])
    return violations.max()


def equalized_odds_constraint(
    logits: torch.Tensor,
    sensitive: torch.Tensor,
    targets: torch.Tensor,
    slack: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Equalized odds: equalise TPR and FPR across groups.

    Returns the max constraint violation.
    """
    probs = torch.sigmoid(logits)
    groups = sensitive.unique()
    if len(groups) < 2:
        return torch.tensor(0.0, device=logits.device)

    violations = []
    for y_val in [0, 1]:
        mask_y = targets == y_val
        if mask_y.sum() == 0:
            continue
        rates = []
        for g in groups:
            mask = mask_y & (sensitive == g)
            if mask.sum() == 0:
                continue
            rates.append(probs[mask].mean())
        if len(rates) >= 2:
            for i in range(len(rates)):
                for j in range(i + 1, len(rates)):
                    violations.append((rates[i] - rates[j]).abs() - slack)

    if not violations:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(violations).max()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LagrangianFairnessTrainer:
    """
    Implements Lagrangian (min-max) optimization for fairness-constrained
    neural network training.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    constraint_fns : list of callables
        Each callable has signature ``(logits, sensitive, **kwargs) -> Tensor``
        and should return a scalar representing the constraint *violation*
        (positive = violated, ≤ 0 = satisfied).
    constraint_kwargs : list of dicts or None
        Extra kwargs forwarded to each constraint function (e.g. ``targets``
        for equalized odds). Length must match ``constraint_fns``.
    model_lr : float, default=1e-3
        Learning rate for model weights.
    multiplier_lr : float, default=1e-2
        Learning rate for Lagrange multipliers.
    constraint_slack : float, default=0.0
        Allowed tolerance on each constraint (subtracted from violation).
    device : str, default='cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        constraint_fns: List[Callable],
        constraint_kwargs: Optional[List[Dict]] = None,
        model_lr: float = 1e-3,
        multiplier_lr: float = 1e-2,
        constraint_slack: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.constraint_fns = constraint_fns
        self.constraint_kwargs = constraint_kwargs or [{} for _ in constraint_fns]
        self.constraint_slack = constraint_slack
        self.device = device

        # One non-negative multiplier per constraint
        self._multipliers = nn.Parameter(
            torch.zeros(len(constraint_fns), device=device)
        )

        self._bce = nn.BCEWithLogitsLoss()
        self._model_opt = torch.optim.Adam(model.parameters(), lr=model_lr)
        self._mult_opt = torch.optim.Adam([self._multipliers], lr=multiplier_lr)

        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def multipliers(self) -> torch.Tensor:
        """Current Lagrange multipliers (clamped to ≥ 0)."""
        return torch.clamp(self._multipliers, min=0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lagrangian(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sensitive: torch.Tensor,
    ) -> tuple:
        """
        L(θ, λ) = BCE(θ) + Σ λ_i · g_i(θ)

        Returns (L, bce_value, [g_i values]).
        """
        bce = self._bce(logits, targets.float())
        violations = []
        for fn, kw in zip(self.constraint_fns, self.constraint_kwargs):
            v = fn(logits, sensitive, targets=targets, **kw) - self.constraint_slack
            violations.append(v)

        lagrangian = bce + torch.stack([
            self.multipliers.detach()[i] * violations[i]
            for i in range(len(violations))
        ]).sum()
        return lagrangian, bce.item(), [v.item() for v in violations]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader,
        sensitive_key: str = "sensitive",
    ) -> Dict:
        self.model.train()
        sum_loss = sum_bce = 0.0
        sum_constraints = [0.0] * len(self.constraint_fns)
        n = 0

        for batch in dataloader:
            X = batch["X"].to(self.device)
            y = batch["y"].to(self.device)
            sensitive = batch[sensitive_key].to(self.device)

            logits = self.model(X).squeeze(-1)

            lagrangian, bce_val, violations = self._lagrangian(logits, y, sensitive)

            # --- Descent on model weights ---
            self._model_opt.zero_grad()
            lagrangian.backward()
            self._model_opt.step()

            # --- Ascent on multipliers ---
            logits_dual = self.model(X).squeeze(-1)   # fresh forward pass
            dual_violations = [
                fn(logits_dual, sensitive, targets=y, **kw) - self.constraint_slack
                for fn, kw in zip(self.constraint_fns, self.constraint_kwargs)
            ]
            # Compute violations gradient manually
            dual_loss = torch.stack([
                self.multipliers[i] * dual_violations[i]
                for i in range(len(dual_violations))
            ]).sum()

            self._mult_opt.zero_grad()
            (-dual_loss).backward()
            self._mult_opt.step()

            # Project multipliers onto non-negative orthant
            with torch.no_grad():
                self._multipliers.clamp_(min=0.0)

            sum_loss += lagrangian.item()
            sum_bce += bce_val
            for i, v in enumerate(violations):
                sum_constraints[i] += v
            n += 1

        stats = {
            "lagrangian":  sum_loss / max(n, 1),
            "bce":         sum_bce  / max(n, 1),
            "constraints": [c / max(n, 1) for c in sum_constraints],
            "multipliers": self.multipliers.detach().cpu().tolist(),
        }
        self.history.append(stats)
        return stats

    def fit(
        self,
        dataloader,
        epochs: int = 10,
        sensitive_key: str = "sensitive",
        verbose: bool = True,
    ) -> "LagrangianFairnessTrainer":
        """
        Train the model for ``epochs`` epochs.

        Parameters
        ----------
        dataloader : DataLoader
            Batches must be dicts with keys ``'X'``, ``'y'``, and
            ``sensitive_key``.
        epochs : int
        sensitive_key : str
        verbose : bool

        Returns
        -------
        self
        """
        for epoch in range(1, epochs + 1):
            stats = self.train_epoch(dataloader, sensitive_key)
            if verbose:
                c_str = ", ".join(f"{c:.4f}" for c in stats["constraints"])
                m_str = ", ".join(f"{m:.4f}" for m in stats["multipliers"])
                print(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Lagrangian: {stats['lagrangian']:.4f} | "
                    f"BCE: {stats['bce']:.4f} | "
                    f"Violations: [{c_str}] | "
                    f"λ: [{m_str}]"
                )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device)).squeeze(-1)
            return (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device)).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()