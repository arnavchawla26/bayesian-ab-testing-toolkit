"""Credible intervals for a Beta posterior.

Two kinds are provided:

- Equal-tailed interval: [ppf((1-mass)/2), ppf((1+mass)/2)]. Simple,
  standard, but for a skewed posterior it is not the shortest interval
  containing the given probability mass.
- Highest density interval (HDI): the shortest interval containing the
  given probability mass. For a unimodal density it is characterized by
  the two endpoints having equal density (the horizontal line at that
  density level cuts off exactly `mass` probability in between).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import optimize

from bayesab.betadist import BetaPosterior


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __iter__(self):
        yield self.lower
        yield self.upper


def equal_tailed_interval(dist: BetaPosterior, mass: float = 0.95) -> Interval:
    """The equal-tailed `mass`-credible interval: mass/2 probability in each tail."""
    _validate_mass(mass)
    tail = (1.0 - mass) / 2.0
    lower = dist.ppf(tail)
    upper = dist.ppf(1.0 - tail)
    return Interval(float(lower), float(upper))


def hdi(dist: BetaPosterior, mass: float = 0.95) -> Interval:
    """The highest-density `mass`-credible interval (shortest such interval).

    Beta(alpha, beta) is unimodal whenever alpha > 1 and beta > 1, and the
    HDI is found by minimizing interval width over the family of intervals
    [lo, ppf(cdf(lo) + mass)] parameterized by the left tail probability
    cdf(lo) in [0, 1 - mass]. At the optimum, pdf(lo) == pdf(hi), which is
    exactly the defining property of the HDI for a unimodal density.

    For alpha <= 1 or beta <= 1 the density is monotonic or U-shaped and
    has no interior mode; in the monotonic cases (exactly one of alpha,
    beta <= 1) the HDI is one of the two equal-tailed extremes (all mass
    pushed to one side), which this function detects and returns directly.
    The U-shaped case (both alpha <= 1 and beta <= 1) has a
    disconnected HDI in general; this function falls back to the
    equal-tailed interval there with a clear caveat in the docstring
    rather than returning a misleading single interval.
    """
    _validate_mass(mass)
    a, b = dist.alpha, dist.beta

    if a <= 1 and b <= 1:
        # U-shaped (or uniform, a == b == 1): true HDI can be disconnected
        # (two intervals near 0 and near 1). A single contiguous interval
        # cannot represent that faithfully, so fall back to equal-tailed.
        return equal_tailed_interval(dist, mass)

    if a <= 1:
        # Monotonically decreasing density: the HDI is the leftmost interval.
        upper = dist.ppf(mass)
        return Interval(0.0, float(upper))

    if b <= 1:
        # Monotonically increasing density: the HDI is the rightmost interval.
        lower = dist.ppf(1.0 - mass)
        return Interval(float(lower), 1.0)

    # Unimodal interior case: minimize width over the left-tail probability.
    def width_given_left_tail(left_tail: float) -> float:
        lo = dist.ppf(left_tail)
        hi = dist.ppf(left_tail + mass)
        return hi - lo

    result = optimize.minimize_scalar(
        width_given_left_tail,
        bounds=(1e-12, 1.0 - mass - 1e-12),
        method="bounded",
        options={"xatol": 1e-10},
    )
    left_tail = result.x
    lower = dist.ppf(left_tail)
    upper = dist.ppf(left_tail + mass)
    return Interval(float(lower), float(upper))


def _validate_mass(mass: float) -> None:
    if not (0.0 < mass < 1.0):
        raise ValueError(f"mass must be in (0, 1), got {mass}")
