"""Probability comparisons between variant posteriors.

Two independent methods are provided for the two-variant case
(`prob_b_beats_a` and, as an oracle, `prob_b_beats_a_monte_carlo`) so that
the exact numerical-integration path can be cross-checked against
simulation rather than trusted blindly -- see tests/test_comparison.py,
which additionally cross-checks both against a closed-form combinatorial
summation formula for the common integer-parameter case.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from bayesab.betadist import BetaPosterior


def prob_b_beats_a(a: BetaPosterior, b: BetaPosterior) -> float:
    """P(p_B > p_A) for two independent Beta-distributed rates, via quadrature.

    Derivation: P(B > A) = integral over x of f_B(x) * P(A < x) dx
                          = integral over x of f_B(x) * F_A(x) dx.
    scipy.integrate.quad evaluates this to near machine precision (the
    reported absolute error is checked and raised on if it looks large,
    since a silently-inaccurate quadrature would be worse than an
    exception here).
    """
    def integrand(x: float) -> float:
        return b.pdf(x) * a.cdf(x)

    lo_a, hi_a = a.effective_support()
    lo_b, hi_b = b.effective_support()
    lo, hi = min(lo_a, lo_b), max(hi_a, hi_b)
    value, abserr = integrate.quad(integrand, lo, hi, limit=200)
    if abserr > 1e-6:
        raise RuntimeError(
            f"prob_b_beats_a: quadrature error estimate too large ({abserr!r}) "
            f"for a={a}, b={b}"
        )
    return float(np.clip(value, 0.0, 1.0))


def prob_b_beats_a_monte_carlo(
    a: BetaPosterior, b: BetaPosterior, n_samples: int = 200_000, seed: int | None = None
) -> float:
    """Monte Carlo estimate of P(p_B > p_A), used as an independent cross-check."""
    rng = np.random.default_rng(seed)
    samples_a = a.rvs(size=n_samples, random_state=rng)
    samples_b = b.rvs(size=n_samples, random_state=rng)
    return float(np.mean(samples_b > samples_a))


def prob_each_is_best(
    variants: dict[str, BetaPosterior], n_samples: int = 200_000, seed: int | None = None
) -> dict[str, float]:
    """P(variant is the single best of all given variants), for two or more variants.

    There is no simple closed form once there are more than two variants
    (it is a probability over the max of several dependent order
    statistics), so this draws joint Monte Carlo samples -- one sampled
    rate per variant per draw -- and counts how often each variant's
    sampled rate is the maximum. Ties (possible only with pathological
    floating point coincidence) are broken by NumPy's `argmax`, which
    always picks the first maximal index; this has negligible effect at
    any realistic sample size since Beta draws are continuous.
    """
    if len(variants) < 2:
        raise ValueError("prob_each_is_best requires at least two variants")
    names = list(variants.keys())
    rng = np.random.default_rng(seed)
    samples = np.column_stack(
        [variants[name].rvs(size=n_samples, random_state=rng) for name in names]
    )
    winners = np.argmax(samples, axis=1)
    counts = np.bincount(winners, minlength=len(names))
    return {name: float(counts[i]) / n_samples for i, name in enumerate(names)}
