"""
fairness_regularizer.py
=======================
A PyTorch loss module that combines binary cross-entropy with a
soft demographic-parity penalty.

Single-attribute mode
---------------------
    L = BCE(logits, y) + eta * (E[p|s=1] - E[p|s=0])²

Intersectional mode
-------------------
When ``sensitive_dict`` is passed to forward(), the penalty is computed
over every (attribute, group-pair) combination and optionally weighted
per subgroup via ``intersectional_weights``:

    L = BCE(logits, y)
        + eta * Σ_attr  Σ_{v0,v1} w(v0,v1) * (E[p|a=v1] - E[p|a=v0])²

Usage
-----
    # Single attribute
    loss_fn = FairnessRegularizer(eta=0.5)
    loss = loss_fn(logits, y, sensitive=s)

    # Intersectional
    loss_fn = FairnessRegularizer(
        eta=0.5,
        intersectional_weights={(0.0, 0.0): 2.0, (1.0, 1.0): 1.5},
    )
    loss = loss_fn(logits, y, sensitive_dict={'gender': s, 'age': a})
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class FairnessRegularizer(nn.Module):
    """
    Binary cross-entropy loss with a soft demographic-parity penalty.

    Parameters
    ----------
    eta : float, default=1.0
        Regularization strength. eta=0 reduces to plain BCE.
    reduction : str, default='mean'
        Passed through to BCEWithLogitsLoss ('mean' or 'sum').
    intersectional_weights : dict or None
        Optional mapping of ``(group_value_0, group_value_1) -> float``
        used to up- or down-weight specific subgroup pairs in the penalty.
        Keys are 2-tuples of floats matching the values present in the
        sensitive tensors. Missing pairs default to weight 1.0.
        Only used when ``sensitive_dict`` is passed to ``forward()``.

    Forward inputs
    --------------
    logits : torch.Tensor, shape (n,)
        Raw (pre-sigmoid) model outputs.
    targets : torch.Tensor, shape (n,)
        Binary labels in {0, 1}.
    sensitive : torch.Tensor, shape (n,), optional
        Binary group indicator for single-attribute mode.
    sensitive_dict : dict[str, torch.Tensor], optional
        Mapping of attribute name -> group tensor for intersectional mode.
        Exactly one of ``sensitive`` or ``sensitive_dict`` must be provided.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """

    def __init__(
        self,
        eta: float = 1.0,
        reduction: str = "mean",
        intersectional_weights: Optional[Dict[Tuple[float, float], float]] = None,
    ) -> None:
        super().__init__()
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}")
        self.eta = eta
        self.intersectional_weights = intersectional_weights or {}
        self._bce = nn.BCEWithLogitsLoss(reduction=reduction)

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
        Compute BCE + eta * fairness penalty.

        Exactly one of ``sensitive`` or ``sensitive_dict`` must be provided.
        """
        if sensitive is None and sensitive_dict is None:
            raise ValueError("Provide either `sensitive` or `sensitive_dict`.")
        if sensitive is not None and sensitive_dict is not None:
            raise ValueError("Provide only one of `sensitive` or `sensitive_dict`, not both.")

        bce_loss = self._bce(logits, targets.to(logits.dtype))

        if sensitive is not None:
            dp_penalty = self._dp_penalty(logits, sensitive)
        else:
            assert sensitive_dict is not None  # narrows type for the type checker
            dp_penalty = self._intersectional_penalty(logits, sensitive_dict)


        return bce_loss + self.eta * dp_penalty

    # ------------------------------------------------------------------
    # Single-attribute penalty
    # ------------------------------------------------------------------

    def _dp_penalty(
        self,
        logits: torch.Tensor,
        sensitive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Squared difference in mean predicted probability between two groups.

            penalty = (E[p | s=1] - E[p | s=0])²

        Returns zero (on the computation graph) if a group is absent from
        the batch.
        """
        proba = torch.sigmoid(logits)
        values = sensitive.unique()

        if values.numel() < 2:
            return torch.zeros(1, dtype=logits.dtype, device=logits.device).squeeze()

        v0, v1 = values[0], values[1]
        mean0 = proba[sensitive == v0].mean()
        mean1 = proba[sensitive == v1].mean()
        return (mean1 - mean0) ** 2

    # ------------------------------------------------------------------
    # Intersectional penalty
    # ------------------------------------------------------------------

    def _intersectional_penalty(
        self,
        logits: torch.Tensor,
        sensitive_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        For each sensitive attribute, sum weighted squared gaps over all
        pairs of observed group values.

            penalty = Σ_attr  Σ_{v0 < v1}  w(v0,v1) * (E[p|a=v1] - E[p|a=v0])²
        """
        proba = torch.sigmoid(logits)
        total = torch.zeros(1, dtype=logits.dtype, device=logits.device).squeeze()

        for attr_name, sensitive in sensitive_dict.items():
            values = sensitive.unique()
            if values.numel() < 2:
                continue  # no contrast possible for this attribute in this batch

            for v0, v1 in combinations(values, 2):
                mask0 = sensitive == v0
                mask1 = sensitive == v1
                if mask0.sum() == 0 or mask1.sum() == 0:
                    continue

                mean0 = proba[mask0].mean()
                mean1 = proba[mask1].mean()
                gap_sq = (mean1 - mean0) ** 2

                # Look up weight; key uses plain Python floats.
                # Also try the reversed pair as a symmetric fallback.
                weight = self.intersectional_weights.get(
                    (v0.item(), v1.item()),
                    self.intersectional_weights.get(
                        (v1.item(), v0.item()), 1.0
                    ),
                )
                total = total + weight * gap_sq

        return total

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        parts = [f"eta={self.eta}"]
        if self.intersectional_weights:
            parts.append(f"intersectional_weights={self.intersectional_weights}")
        return ", ".join(parts)