import math

import pytest
from scipy import stats

from bayesab.betadist import BetaPosterior


def test_uniform_prior_is_beta_1_1():
    p = BetaPosterior.uniform_prior()
    assert p.alpha == 1.0
    assert p.beta == 1.0


def test_jeffreys_prior_is_beta_half_half():
    p = BetaPosterior.jeffreys_prior()
    assert p.alpha == 0.5
    assert p.beta == 0.5


def test_rejects_nonpositive_parameters():
    with pytest.raises(ValueError):
        BetaPosterior(0.0, 1.0)
    with pytest.raises(ValueError):
        BetaPosterior(1.0, -1.0)


def test_update_is_conjugate_addition():
    prior = BetaPosterior(2.0, 3.0)
    post = prior.update(successes=7, trials=10)
    assert post.alpha == pytest.approx(2.0 + 7)
    assert post.beta == pytest.approx(3.0 + 3)  # 3 failures


def test_update_rejects_invalid_counts():
    prior = BetaPosterior.uniform_prior()
    with pytest.raises(ValueError):
        prior.update(successes=-1, trials=10)
    with pytest.raises(ValueError):
        prior.update(successes=11, trials=10)
    with pytest.raises(ValueError):
        prior.update(successes=0, trials=-1)


def test_update_is_sequential_and_order_independent():
    # Updating with two batches in either order (or as one combined batch)
    # must land on the same posterior -- conjugacy makes the Beta-Binomial
    # update commutative and associative in the sufficient statistics.
    prior = BetaPosterior(1.0, 1.0)
    combined = prior.update(successes=12, trials=20)

    batch_order_1 = prior.update(successes=5, trials=8).update(successes=7, trials=12)
    batch_order_2 = prior.update(successes=7, trials=12).update(successes=5, trials=8)

    assert batch_order_1.alpha == pytest.approx(combined.alpha)
    assert batch_order_1.beta == pytest.approx(combined.beta)
    assert batch_order_2.alpha == pytest.approx(combined.alpha)
    assert batch_order_2.beta == pytest.approx(combined.beta)


@pytest.mark.parametrize(
    "alpha,beta",
    [(2.0, 3.0), (1.0, 1.0), (10.0, 20.0), (0.5, 0.5), (100.0, 5.0)],
)
def test_mean_and_variance_match_scipy(alpha, beta):
    dist = BetaPosterior(alpha, beta)
    scipy_dist = stats.beta(alpha, beta)
    assert dist.mean == pytest.approx(scipy_dist.mean())
    assert dist.variance == pytest.approx(scipy_dist.var())
    assert dist.std == pytest.approx(scipy_dist.std())


def test_mode_matches_known_formula_when_defined():
    dist = BetaPosterior(5.0, 3.0)
    assert dist.mode == pytest.approx((5 - 1) / (5 + 3 - 2))


def test_mode_is_none_for_uniform_and_u_shaped():
    assert BetaPosterior(1.0, 1.0).mode is None
    assert BetaPosterior(0.5, 0.5).mode is None
    assert BetaPosterior(0.5, 3.0).mode is None  # alpha <= 1: monotonic, no interior mode


def test_pdf_cdf_ppf_roundtrip_matches_scipy():
    dist = BetaPosterior(4.0, 6.0)
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x in xs:
        assert dist.pdf(x) == pytest.approx(stats.beta.pdf(x, 4.0, 6.0))
        assert dist.cdf(x) == pytest.approx(stats.beta.cdf(x, 4.0, 6.0))
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        assert dist.ppf(q) == pytest.approx(stats.beta.ppf(q, 4.0, 6.0))
        # ppf and cdf must be inverses of each other
        assert dist.cdf(dist.ppf(q)) == pytest.approx(q, abs=1e-9)


def test_sf_is_one_minus_cdf():
    dist = BetaPosterior(3.0, 8.0)
    for x in [0.05, 0.4, 0.8]:
        assert dist.sf(x) == pytest.approx(1.0 - dist.cdf(x))


def test_rvs_is_reproducible_with_seed():
    dist = BetaPosterior(2.0, 5.0)
    a = dist.rvs(size=1000, random_state=42)
    b = dist.rvs(size=1000, random_state=42)
    assert (a == b).all()


def test_rvs_mean_converges_to_true_mean():
    dist = BetaPosterior(4.0, 4.0)
    samples = dist.rvs(size=200_000, random_state=0)
    assert math.isclose(samples.mean(), dist.mean, abs_tol=0.01)
