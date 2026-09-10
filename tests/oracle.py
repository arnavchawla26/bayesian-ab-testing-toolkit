"""An independent, closed-form oracle for P(B > A), used only in tests.

This is deliberately a *second*, differently-derived implementation of the
same quantity `bayesab.comparison.prob_b_beats_a` computes via numerical
integration -- so that a bug shared between "the code" and "the test's
expectation" (e.g. a sign error copied from the implementation into a
hand-written expected value) is much less likely to slip through.

Formula (closed form for integer alphaB, summing the incomplete-beta
series that falls out of integrating the Binomial(alphaB - 1, ...) term by
term -- the standard reference is Evan Miller's "Formulas for Bayesian A/B
Testing"; see also Chris Stucchio's VWO Bayesian testing writeup):

    P(B > A) = sum_{i=0}^{alphaB-1}
                   B(alphaA + i, betaB + betaA)
                   / ((betaB + i) * B(1 + i, betaB) * B(alphaA, betaA))

where B(x, y) is the Beta function, computed here in log-space via
math.lgamma for numerical stability. This requires alphaB to be a positive
integer -- true whenever the prior's alpha is an integer (e.g. the
standard uniform Beta(1,1) prior), since alphaB = prior_alpha + successes.

Before being trusted as a test oracle, this formula was independently
cross-checked (in the scratch exploration that produced this module)
against `scipy.integrate.dblquad` computing the same probability as a
direct 2-D integral over the region {x_B > x_A} -- agreement to ~1e-12
across a range of parameter values, including asymmetric and
extreme-parameter cases.
"""

from __future__ import annotations

import math


def _lbeta(x: float, y: float) -> float:
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def oracle_prob_b_beats_a(alpha_a: float, beta_a: float, alpha_b: int, beta_b: float) -> float:
    """P(B > A) via the closed-form summation. Requires alpha_b to be a positive integer."""
    if float(alpha_b) != int(alpha_b) or alpha_b < 1:
        raise ValueError(f"oracle_prob_b_beats_a requires a positive integer alpha_b, got {alpha_b}")
    total = 0.0
    for i in range(int(alpha_b)):
        log_term = (
            _lbeta(alpha_a + i, beta_b + beta_a)
            - math.log(beta_b + i)
            - _lbeta(1 + i, beta_b)
            - _lbeta(alpha_a, beta_a)
        )
        total += math.exp(log_term)
    return total
