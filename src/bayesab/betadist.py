"""Beta-Binomial conjugate posterior for a single variant's conversion rate.

The model: a variant's true conversion rate p has a Beta(alpha, beta) prior.
Observing `successes` conversions out of `trials` impressions with a
Binomial likelihood gives, by conjugacy, a Beta(alpha + successes,
beta + trials - successes) posterior. No sampling or approximation is
needed for this step -- it is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats


@dataclass(frozen=True)
class BetaPosterior:
    """A Beta(alpha, beta) distribution over a conversion rate.

    Used both as the prior a caller supplies and as the posterior produced
    by :meth:`update`. alpha and beta must be strictly positive.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"alpha and beta must be strictly positive, got alpha={self.alpha}, "
                f"beta={self.beta}"
            )

    @classmethod
    def uniform_prior(cls) -> "BetaPosterior":
        """Beta(1, 1): the uniform, maximally-uninformative prior on [0, 1]."""
        return cls(1.0, 1.0)

    @classmethod
    def jeffreys_prior(cls) -> "BetaPosterior":
        """Beta(0.5, 0.5): the Jeffreys prior, invariant under reparameterization."""
        return cls(0.5, 0.5)

    def update(self, successes: int, trials: int) -> "BetaPosterior":
        """Return the posterior after observing `successes` out of `trials`.

        This is the conjugate update: alpha' = alpha + successes,
        beta' = beta + (trials - successes). No approximation.
        """
        if trials < 0:
            raise ValueError(f"trials must be >= 0, got {trials}")
        if successes < 0 or successes > trials:
            raise ValueError(
                f"successes must be in [0, trials], got successes={successes}, "
                f"trials={trials}"
            )
        failures = trials - successes
        return BetaPosterior(self.alpha + successes, self.beta + failures)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return self.variance**0.5

    @property
    def mode(self) -> float | None:
        """The distribution's mode, or None when it has no unique interior mode.

        A unique interior mode exists only for alpha > 1 and beta > 1; the
        Beta(1,1) uniform case and U-shaped/monotonic cases (alpha <= 1 or
        beta <= 1) have no single well-defined interior mode.
        """
        a, b = self.alpha, self.beta
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        return None

    def pdf(self, x):
        """Density at x, via scipy.special.betaln directly.

        Equivalent to scipy.stats.beta.pdf(x, alpha, beta) (checked against
        it throughout the test suite), but calling scipy.special's ufuncs
        directly skips scipy.stats's generic rv_continuous argument-
        broadcasting machinery, which dominates runtime when this is
        called thousands of times inside a quadrature loop (as
        comparison.prob_b_beats_a and loss.expected_loss do).
        """
        a, b = self.alpha, self.beta
        x = np.asarray(x, dtype=float)
        log_norm = special.betaln(a, b)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_pdf = (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - log_norm
            result = np.exp(log_pdf)
            result = np.where((x < 0.0) | (x > 1.0), 0.0, result)
        return float(result) if result.ndim == 0 else result

    def cdf(self, x):
        """P(X <= x) = the regularized incomplete beta function I_x(alpha, beta)."""
        x = np.asarray(x, dtype=float)
        clipped = np.clip(x, 0.0, 1.0)
        result = special.betainc(self.alpha, self.beta, clipped)
        return float(result) if result.ndim == 0 else result

    def sf(self, x):
        """Survival function, 1 - cdf(x), computed directly for numerical accuracy."""
        x = np.asarray(x, dtype=float)
        clipped = np.clip(x, 0.0, 1.0)
        result = special.betaincc(self.alpha, self.beta, clipped)
        return float(result) if result.ndim == 0 else result

    def ppf(self, q):
        """Quantile function (inverse CDF), via the inverse regularized incomplete beta."""
        q = np.asarray(q, dtype=float)
        result = special.betaincinv(self.alpha, self.beta, q)
        return float(result) if result.ndim == 0 else result

    def rvs(self, size: int = 1, random_state=None):
        """Draw Monte Carlo samples from this distribution."""
        return stats.beta.rvs(self.alpha, self.beta, size=size, random_state=random_state)

    def as_scipy(self):
        """The underlying frozen scipy.stats distribution, for advanced use."""
        return stats.beta(self.alpha, self.beta)

    def effective_support(self, widths: float = 12.0) -> tuple[float, float]:
        """A [lo, hi] subinterval of [0, 1] containing effectively all of this
        distribution's mass, for restricting numerical integration bounds.

        For a large-sample posterior the density concentrates in a narrow
        band around the mean; integrating the full unit interval then
        wastes the vast majority of quadrature evaluations on regions with
        negligible density. `mean +/- widths * std` clipped to [0, 1] is a
        generous (12 standard deviations by default) window: for any
        unimodal Beta the density at that distance from the mean is many
        orders of magnitude below its peak, so the mass excluded is far
        below the 1e-6 absolute-error tolerance callers check for. This is
        a performance optimization only -- callers combine two
        distributions' windows (e.g. via min/max of both bounds) so that
        it never narrows the domain below what genuinely matters for
        *either* distribution being integrated against.
        """
        lo = max(0.0, self.mean - widths * self.std)
        hi = min(1.0, self.mean + widths * self.std)
        if lo >= hi:  # degenerate/near-zero-variance edge case: keep a tiny valid window
            return (max(0.0, self.mean - 1e-9), min(1.0, self.mean + 1e-9))
        return (lo, hi)
