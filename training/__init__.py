
Copia

"""
training
========
Fairness-aware training components for scikit-learn and PyTorch.
 
Components
----------
ReductionsWrapper
    Scikit-learn wrapper using fairlearn's ExponentiatedGradient.
 
FairnessRegularizer
    PyTorch loss function with soft fairness penalty (standard + intersectional).
 
LagrangianFairnessTrainer
    PyTorch trainer enforcing hard fairness constraints via Lagrangian optimization.
 
GroupFairnessCalibrator
    Post-training group-specific calibration (Platt, Isotonic, Temperature).
 
plot_pareto_frontier
    Sweep eta values and plot the fairness-accuracy Pareto frontier.
 
AdversarialDebiasingTrainer
    Minimax adversarial training for debiased representations. (stretch goal)
 
GradientReversalLayer
    GRL building block for adversarial debiasing. (stretch goal)
"""
 
from .reductions_wrapper import ReductionsWrapper
from .fairness_regularizer import FairnessRegularizer
from .lagrangian_trainer import (
    LagrangianFairnessTrainer,
    demographic_parity_constraint,
    equalized_odds_constraint,
)
from .calibrator import (
    GroupFairnessCalibrator,
    PlattScaler,
    IsotonicScaler,
    TemperatureScaler,
)
from .pareto import plot_pareto_frontier
from .adversarial import AdversarialDebiasingTrainer, GradientReversalLayer
 
__all__ = [
    # Core
    "ReductionsWrapper",
    "FairnessRegularizer",
    "LagrangianFairnessTrainer",
    "demographic_parity_constraint",
    "equalized_odds_constraint",
    "GroupFairnessCalibrator",
    "PlattScaler",
    "IsotonicScaler",
    "TemperatureScaler",
    "plot_pareto_frontier",
    # Stretch goals
    "AdversarialDebiasingTrainer",
    "GradientReversalLayer",
]
 