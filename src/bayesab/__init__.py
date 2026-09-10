"""bayesab: a from-scratch Bayesian A/B testing toolkit.

Beta-Binomial conjugate posteriors for conversion-rate experiments, credible
intervals (equal-tailed and highest-density), probability-to-beat-control,
and expected-loss based stopping rules.
"""

from bayesab.betadist import BetaPosterior
from bayesab.comparison import prob_b_beats_a, prob_each_is_best
from bayesab.experiment import Experiment, Variant, VariantReport
from bayesab.intervals import equal_tailed_interval, hdi
from bayesab.loss import expected_loss, stopping_decision

__all__ = [
    "BetaPosterior",
    "prob_b_beats_a",
    "prob_each_is_best",
    "Experiment",
    "Variant",
    "VariantReport",
    "equal_tailed_interval",
    "hdi",
    "expected_loss",
    "stopping_decision",
]

__version__ = "0.1.0"
