import pytest
from scipy import stats

from bayesab.betadist import BetaPosterior
from bayesab.intervals import equal_tailed_interval, hdi


def test_equal_tailed_matches_scipy_interval():
    dist = BetaPosterior(12.0, 30.0)
    lo, hi_ = equal_tailed_interval(dist, mass=0.95)
    scipy_lo, scipy_hi = stats.beta.interval(0.95, 12.0, 30.0)
    assert lo == pytest.approx(scipy_lo)
    assert hi_ == pytest.approx(scipy_hi)


def test_equal_tailed_contains_correct_mass():
    dist = BetaPosterior(8.0, 15.0)
    interval = equal_tailed_interval(dist, mass=0.9)
    mass_inside = dist.cdf(interval.upper) - dist.cdf(interval.lower)
    assert mass_inside == pytest.approx(0.9, abs=1e-9)


def test_equal_tailed_rejects_invalid_mass():
    dist = BetaPosterior(2.0, 2.0)
    with pytest.raises(ValueError):
        equal_tailed_interval(dist, mass=0.0)
    with pytest.raises(ValueError):
        equal_tailed_interval(dist, mass=1.0)
    with pytest.raises(ValueError):
        equal_tailed_interval(dist, mass=-0.1)


def test_hdi_contains_correct_mass():
    dist = BetaPosterior(30.0, 8.0)  # skewed
    interval = hdi(dist, mass=0.9)
    mass_inside = dist.cdf(interval.upper) - dist.cdf(interval.lower)
    assert mass_inside == pytest.approx(0.9, abs=1e-6)


def test_hdi_is_narrower_than_equal_tailed_for_skewed_posterior():
    # For a skewed unimodal density, the HDI is strictly the shortest
    # interval with the given mass, so it must not be wider than the
    # equal-tailed interval (and for real skew, strictly narrower).
    dist = BetaPosterior(3.0, 40.0)  # strongly right-skewed
    equal_tailed = equal_tailed_interval(dist, mass=0.9)
    hdi_interval = hdi(dist, mass=0.9)
    assert hdi_interval.width < equal_tailed.width


def test_hdi_endpoints_have_equal_density_for_unimodal_case():
    # The defining property of the HDI for a unimodal density: the two
    # endpoints sit at the same density level.
    dist = BetaPosterior(6.0, 20.0)
    interval = hdi(dist, mass=0.9)
    assert dist.pdf(interval.lower) == pytest.approx(dist.pdf(interval.upper), rel=1e-3)


def test_hdi_matches_equal_tailed_for_symmetric_posterior():
    # A symmetric Beta (alpha == beta) has a symmetric density about 0.5,
    # so its HDI and equal-tailed interval coincide.
    dist = BetaPosterior(20.0, 20.0)
    equal_tailed = equal_tailed_interval(dist, mass=0.95)
    hdi_interval = hdi(dist, mass=0.95)
    assert hdi_interval.lower == pytest.approx(equal_tailed.lower, abs=1e-6)
    assert hdi_interval.upper == pytest.approx(equal_tailed.upper, abs=1e-6)


def test_hdi_monotonic_decreasing_density_is_leftmost_interval():
    # alpha <= 1, beta > 1: density is monotonically decreasing on (0, 1),
    # so all mass concentrates near 0 and the HDI starts at 0.
    dist = BetaPosterior(1.0, 5.0)
    interval = hdi(dist, mass=0.9)
    assert interval.lower == 0.0
    assert interval.upper == pytest.approx(dist.ppf(0.9))


def test_hdi_monotonic_increasing_density_is_rightmost_interval():
    dist = BetaPosterior(5.0, 1.0)
    interval = hdi(dist, mass=0.9)
    assert interval.upper == 1.0
    assert interval.lower == pytest.approx(dist.ppf(0.1))


def test_hdi_u_shaped_falls_back_to_equal_tailed():
    dist = BetaPosterior(0.5, 0.5)
    interval = hdi(dist, mass=0.9)
    equal_tailed = equal_tailed_interval(dist, mass=0.9)
    assert interval.lower == pytest.approx(equal_tailed.lower)
    assert interval.upper == pytest.approx(equal_tailed.upper)


def test_interval_width_property():
    dist = BetaPosterior(10.0, 10.0)
    interval = equal_tailed_interval(dist, mass=0.5)
    assert interval.width == pytest.approx(interval.upper - interval.lower)


def test_interval_is_iterable_as_tuple():
    dist = BetaPosterior(10.0, 10.0)
    interval = equal_tailed_interval(dist, mass=0.5)
    lo, hi_ = interval
    assert lo == interval.lower
    assert hi_ == interval.upper
