"""
adversarial.py
==============
Adversarial Debiasing trainer for PyTorch.  (Stretch goal)

Architecture
------------
  Encoder  →  Predictor head  →  target prediction
       ↓
  GradientReversalLayer
       ↓
  Adversary head  →  sensitive-attribute prediction

The GradientReversalLayer negates gradients during back-propagation so the
encoder learns representations that are simultaneously predictive of the
target label and *uninformative* about the sensitive attribute — framing
fairness as a minimax game.

Usage
-----
    from training.adversarial import (
        AdversarialDebiasingTrainer,
        GradientReversalLayer,
    )

    trainer = AdversarialDebiasingTrainer(
        encoder=MyEncoder(),
        predictor_head=MyPredictor(),
        adversary_head=MyAdversary(),
        alpha=1.0,
        model_lr=1e-3,
        adversary_lr=1e-3,
    )
    trainer.fit(train_loader, epochs=20)
    proba = trainer.predict_proba(X_tensor)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GRLFunction(torch.autograd.Function):
    """
    Forward  : identity
    Backward : multiply gradient by -alpha

    This is the standard "gradient reversal" trick from
    Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse gradient flowing into the encoder; no gradient for alpha
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).

    During forward passes it is the identity function.
    During back-propagation it multiplies the incoming gradient by ``-alpha``,
    forcing the upstream encoder to minimise the adversary's loss.

    Parameters
    ----------
    alpha : float, default=1.0
        Reversal strength. Increase gradually during training for stability.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AdversarialDebiasingTrainer:
    """
    Trains a predictor network and an adversary jointly to produce
    representations that are uninformative about sensitive attributes.

    Architecture
    ------------
    ``encoder`` produces a shared representation h.
    ``predictor_head(h)`` → target logits.
    ``adversary_head(GRL(h))`` → sensitive-attribute logits.

    The GRL reverses gradients so the encoder *maximises* the adversary's
    loss (making representations debiased) while the adversary *minimises*
    its own classification loss.

    Parameters
    ----------
    encoder : nn.Module
        Shared feature extractor.
    predictor_head : nn.Module
        Maps representation → target logits.
    adversary_head : nn.Module
        Maps (reversed) representation → sensitive attribute logits.
    adversary_input : str, default='representation'
        Whether to feed the adversary the encoder ``'representation'`` or
        the ``'prediction'`` logits.
    alpha : float, default=1.0
        GRL reversal strength.
    model_lr : float, default=1e-3
    adversary_lr : float, default=1e-3
    device : str, default='cpu'
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor_head: nn.Module,
        adversary_head: nn.Module,
        adversary_input: str = "representation",
        alpha: float = 1.0,
        model_lr: float = 1e-3,
        adversary_lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        if adversary_input not in ("representation", "prediction"):
            raise ValueError("adversary_input must be 'representation' or 'prediction'")

        self.encoder = encoder.to(device)
        self.predictor_head = predictor_head.to(device)
        self.adversary_head = adversary_head.to(device)
        self.adversary_input = adversary_input
        self.device = device
        self.grl = GradientReversalLayer(alpha=alpha)

        # Predictor optimises encoder + predictor_head
        predictor_params = (
            list(encoder.parameters()) + list(predictor_head.parameters())
        )
        self._pred_opt = torch.optim.Adam(predictor_params, lr=model_lr)
        self._adv_opt  = torch.optim.Adam(adversary_head.parameters(), lr=adversary_lr)

        self._pred_loss_fn = nn.BCEWithLogitsLoss()
        self._adv_loss_fn  = nn.BCEWithLogitsLoss()

        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, X: torch.Tensor):
        """
        Returns (representation, pred_logits, adv_logits_with_grl).

        The adversary logits here are computed with GRL active, so the
        gradient of adv_loss w.r.t. encoder parameters is *reversed*.
        """
        h = self.encoder(X)
        pred_logits = self.predictor_head(h)

        if self.adversary_input == "representation":
            adv_input = self.grl(h)
        else:
            adv_input = self.grl(pred_logits)

        adv_logits = self.adversary_head(adv_input)
        return h, pred_logits, adv_logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader,
        sensitive_key: str = "sensitive",
    ) -> Dict:
        self.encoder.train()
        self.predictor_head.train()
        self.adversary_head.train()

        sum_pred, sum_adv = 0.0, 0.0
        n = 0

        for batch in dataloader:
            X         = batch["X"].to(self.device)
            y         = batch["y"].to(self.device).float()
            sensitive = batch[sensitive_key].to(self.device).float()

            # ---- Step 1: update predictor + encoder (with GRL) ----
            self._pred_opt.zero_grad()
            _, pred_logits, adv_logits_grl = self._forward(X)
            pred_loss = self._pred_loss_fn(pred_logits.squeeze(-1), y)
            # The adversary loss here flows reversed gradients into the encoder
            adv_loss_grl = self._adv_loss_fn(adv_logits_grl.squeeze(-1), sensitive)
            (pred_loss + adv_loss_grl).backward()
            self._pred_opt.step()

            # ---- Step 2: update adversary alone (no GRL) ----
            self._adv_opt.zero_grad()
            with torch.no_grad():
                h = self.encoder(X)
                if self.adversary_input == "representation":
                    adv_input_pure = h
                else:
                    adv_input_pure = self.predictor_head(h)
            adv_logits_pure = self.adversary_head(adv_input_pure)
            adv_loss = self._adv_loss_fn(adv_logits_pure.squeeze(-1), sensitive)
            adv_loss.backward()
            self._adv_opt.step()

            sum_pred += pred_loss.item()
            sum_adv  += adv_loss.item()
            n += 1

        stats = {
            "pred_loss": sum_pred / max(n, 1),
            "adv_loss":  sum_adv  / max(n, 1),
        }
        self.history.append(stats)
        return stats

    def fit(
        self,
        dataloader,
        epochs: int = 10,
        sensitive_key: str = "sensitive",
        alpha_schedule: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> "AdversarialDebiasingTrainer":
        """
        Train for ``epochs`` epochs.

        Parameters
        ----------
        dataloader : DataLoader
            Batches must be dicts with ``'X'``, ``'y'``, and ``sensitive_key``.
        epochs : int
        sensitive_key : str
        alpha_schedule : list of float or None
            Per-epoch GRL alpha values. Length must equal ``epochs``.
            If None, alpha stays constant. Increasing alpha gradually
            improves training stability.
        verbose : bool

        Returns
        -------
        self
        """
        if alpha_schedule is not None and len(alpha_schedule) != epochs:
            raise ValueError("alpha_schedule length must equal epochs")

        for epoch in range(1, epochs + 1):
            if alpha_schedule is not None:
                self.grl.alpha = alpha_schedule[epoch - 1]

            stats = self.train_epoch(dataloader, sensitive_key)

            if verbose:
                print(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Pred Loss: {stats['pred_loss']:.4f} | "
                    f"Adv Loss: {stats['adv_loss']:.4f} | "
                    f"α={self.grl.alpha:.3f}"
                )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        self.encoder.eval()
        self.predictor_head.eval()
        with torch.no_grad():
            h = self.encoder(X.to(self.device))
            logits = self.predictor_head(h).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_representation(self, X: torch.Tensor) -> np.ndarray:
        """Return encoder representations (useful for inspection)."""
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(X.to(self.device)).cpu().numpy()