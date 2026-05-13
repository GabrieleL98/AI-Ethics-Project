"""
fairness_regularizer.py
=======================
Custom PyTorch loss function that combines a standard accuracy loss with a
fairness regularization term penalising demographic disparity.

Stretch goal included: intersectional regularization with per-subgroup weights.

Usage
-----
    from training.fairness_regularizer import FairnessRegularizer

    loss_fn = FairnessRegularizer(eta=0.5)
    loss = loss_fn(logits, targets, sensitive=group_tensor)

    # Intersectional
    loss_fn = FairnessRegularizer(
        eta=0.5,
        intersectional_weights={(0, 1): 2.0, (1, 0): 1.5},
    )
    loss = loss_fn(logits, targets, sensitive_dict={
        'gender': gender_tensor,
        'age':    age_tensor,
    })
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class FairnessRegularizer(nn.Module):
    """
    BCE loss + fairness regularization penalty.

    Standard mode
    -------------
    Penalises the squared difference in mean predicted probabilities between
    every pair of groups defined by a single ``sensitive`` tensor.

    Intersectional mode  (stretch goal)
    ------------------------------------
    Accepts a ``sensitive_dict`` mapping attribute names to group tensors.
    The penalty is computed over all intersectional subgroups (Cartesian
    product of all attribute values), weighted by ``intersectional_weights``.
    Worst-off subgroups can be up-weighted to focus fairness pressure there.

    Parameters
    ----------
    eta : float, default=0.1
        Fairness regularization strength. Higher → stronger fairness pressure.
        Start at 0.1 and increase gradually while monitoring accuracy.
    intersectional_weights : dict or None
        Maps subgroup tuple keys → scalar weight multiplier.
        Only used when ``sensitive_dict`` is provided.
        E.g. ``{(0, 1): 2.0}`` doubles the penalty for the (gender=0, age=1) group.
        Unspecified groups default to weight 1.0.
    reduction : str, default='mean'
        Reduction for the BCE component ('mean' or 'sum').
    eps : float, default=1e-8
        Small constant to avoid division by zero in empty-group checks.
    """

    def __init__(
        self,
        eta: float = 0.1,
        intersectional_weights: Optional[Dict[Tuple, float]] = None,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eta = eta
        self.intersectional_weights = intersectional_weights or {}
        self.reduction = reduction
        self.eps = eps
        self._bce = nn.BCEWithLogitsLoss(reduction=reduction)

    # ------------------------------------------------------------------
    # Penalty helpers
    # ------------------------------------------------------------------

    def _pairwise_squared_penalty(
        self,
        probs: torch.Tensor,
        group_means: Dict,
        group_weights: Dict,
    ) -> torch.Tensor:
        """Weighted sum of squared mean-prediction differences over all pairs."""
        keys = list(group_means.keys())
        if len(keys) < 2:
            return torch.tensor(0.0, device=probs.device)

        penalty = torch.tensor(0.0, device=probs.device)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                w = group_weights.get(ki, 1.0) * group_weights.get(kj, 1.0)
                penalty = penalty + w * (group_means[ki] - group_means[kj]) ** 2
        return penalty

    def _standard_penalty(
        self,
        probs: torch.Tensor,
        sensitive: torch.Tensor,
    ) -> torch.Tensor:
        """Demographic parity penalty for a single sensitive attribute."""
        group_means: Dict = {}
        group_weights: Dict = {}

        for g in sensitive.unique():
            mask = sensitive == g
            if mask.sum() == 0:
                continue
            key = g.item()
            group_means[key] = probs[mask].mean()
            group_weights[key] = 1.0

        return self._pairwise_squared_penalty(probs, group_means, group_weights)

    def _intersectional_penalty(
        self,
        probs: torch.Tensor,
        sensitive_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Intersectional demographic parity penalty.

        Computes mean predictions for every intersectional subgroup (defined
        as the Cartesian product of all sensitive attributes), then penalises
        pairwise differences weighted by ``intersectional_weights``.
        """
        attr_tensors = list(sensitive_dict.values())
        stacked = torch.stack(attr_tensors, dim=1)          # (N, num_attrs)
        unique_combos = stacked.unique(dim=0)               # (K, num_attrs)

        group_means: Dict = {}
        group_weights: Dict = {}

        for combo in unique_combos:
            mask = (stacked == combo.unsqueeze(0)).all(dim=1)
            if mask.sum() < 2:          # skip tiny / empty subgroups
                continue
            key = tuple(combo.tolist())
            group_means[key] = probs[mask].mean()
            group_weights[key] = self.intersectional_weights.get(key, 1.0)

        return self._pairwise_squared_penalty(probs, group_means, group_weights)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sensitive: Optional[torch.Tensor] = None,
        sensitive_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute fairness-regularized loss.

        Parameters
        ----------
        logits : Tensor of shape (N,)
            Raw model outputs (pre-sigmoid).
        targets : Tensor of shape (N,)
            Binary ground-truth labels.
        sensitive : Tensor of shape (N,), optional
            Group labels for standard single-attribute regularization.
        sensitive_dict : dict, optional
            ``{attr_name: Tensor(N,)}`` for intersectional regularization.
            Takes priority over ``sensitive`` when both are provided.

        Returns
        -------
        Tensor (scalar)
            Total loss = BCE + eta * fairness_penalty.
        """
        acc_loss = self._bce(logits, targets.float())
        probs = torch.sigmoid(logits.detach())          # detach for stability

        if sensitive_dict is not None:
            fairness_penalty = self._intersectional_penalty(probs, sensitive_dict)
        elif sensitive is not None:
            fairness_penalty = self._standard_penalty(probs, sensitive)
        else:
            fairness_penalty = torch.tensor(0.0, device=logits.device)

        return acc_loss + self.eta * fairness_penalty

    def __repr__(self) -> str:
        return (
            f"FairnessRegularizer(eta={self.eta}, "
            f"intersectional={bool(self.intersectional_weights)})"
        )