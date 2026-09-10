"""Expected loss ("risk") and the expected-loss stopping rule.

Point estimates and credible intervals answer "which variant looks
better," but not "how much do we stand to lose if we pick wrong." The
expected loss of choosing variant X, when the true rates are unknown
draws from their posteriors, is:

    loss(choose X over Y) = E[max(p_Y - p_X, 0)]

i.e. the average amount by which Y would have beaten X, counting zero
whenever X would in fact have been the better (or equal) choice. This is
the standard Bayesian A/B testing risk measure (Chris Stucchio, "Bayesian
A/B Testing at VWO"; Evan Miller's Bayesian formulas write-up). Stopping
once the expected loss of the apparent winner drops below a small
practical threshold (e.g. 0.0001 = one basis point of conversion rate)
bounds the expected cost of stopping "too early" by construction, unlike
a fixed-sample-size or naive first-significant-p-value rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import integrate

from bayesab.betadist import BetaPosterior


@dataclass(frozen=True)
class LossResult:
    """Expected loss of choosing each of two variants, in absolute rate units."""

    loss_choosing_a: float
    loss_choosing_b: float

    @property
    def better(self) -> str:
        """Which variant has the lower (safer) expected loss: 'a' or 'b'.

        Ties resolve to 'a' (the control), matching :func:`stopping_decision`.
        """
        return "a" if self.loss_choosing_a <= self.loss_choosing_b else "b"


def expected_loss(a: BetaPosterior, b: BetaPosterior) -> LossResult:
    """Expected loss of choosing A vs. choosing B, given their posteriors.

    Uses the identity E[max(Y-X,0)] = E[Y*1{Y>X}] - E[X*1{Y>X}]
                                     = integral(y * f_Y(y) * F_X(y) dy)
                                       - integral(x * f_X(x) * (1-F_Y(x)) dx),
    which reduces a 2-D expectation to two 1-D numerical integrals -- exact
    up to quadrature tolerance, no simulation needed. See
    tests/test_loss.py for a Monte Carlo cross-check of this identity.
    """
    return LossResult(
        loss_choosing_a=_expected_positive_part(x=a, y=b),
        loss_choosing_b=_expected_positive_part(x=b, y=a),
    )


def _expected_positive_part(x: BetaPosterior, y: BetaPosterior) -> float:
    """E[max(y_rv - x_rv, 0)] for independent x_rv ~ x, y_rv ~ y."""

    def term1_integrand(t: float) -> float:
        return t * y.pdf(t) * x.cdf(t)

    def term2_integrand(t: float) -> float:
        return t * x.pdf(t) * (1.0 - y.cdf(t))

    lo_x, hi_x = x.effective_support()
    lo_y, hi_y = y.effective_support()
    lo, hi = min(lo_x, lo_y), max(hi_x, hi_y)

    term1, err1 = integrate.quad(term1_integrand, lo, hi, limit=200)
    term2, err2 = integrate.quad(term2_integrand, lo, hi, limit=200)
    if err1 > 1e-6 or err2 > 1e-6:
        raise RuntimeError(
            f"_expected_positive_part: quadrature error estimate too large "
            f"(err1={err1!r}, err2={err2!r}) for x={x}, y={y}"
        )
    # Clamp tiny negative results from quadrature noise (the true value is >= 0).
    return max(0.0, term1 - term2)


def stopping_decision(loss: LossResult, threshold: float) -> str:
    """Decide whether to stop and, if so, which variant to ship.

    Returns "stop_choose_a", "stop_choose_b", or "continue". Stops on
    whichever variant has expected loss at or below `threshold` and is no
    worse than the other's expected loss; if neither variant's expected
    loss is below the threshold, the test should continue collecting data.
    Ties (equal loss, both under threshold) resolve to A (the control).
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    a_ok = loss.loss_choosing_a <= threshold
    b_ok = loss.loss_choosing_b <= threshold
    if a_ok and loss.loss_choosing_a <= loss.loss_choosing_b:
        return "stop_choose_a"
    if b_ok and loss.loss_choosing_b <= loss.loss_choosing_a:
        return "stop_choose_b"
    return "continue"
