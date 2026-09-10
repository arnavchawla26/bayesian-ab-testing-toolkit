import pytest

from bayesab.betadist import BetaPosterior
from bayesab.comparison import prob_b_beats_a, prob_b_beats_a_monte_carlo, prob_each_is_best
from oracle import oracle_prob_b_beats_a


def test_identical_distributions_give_fifty_fifty():
    a = BetaPosterior(10.0, 10.0)
    b = BetaPosterior(10.0, 10.0)
    assert prob_b_beats_a(a, b) == pytest.approx(0.5, abs=1e-9)


def test_strictly_better_posterior_gives_high_probability():
    a = BetaPosterior(10.0, 90.0)  # ~10% rate
    b = BetaPosterior(90.0, 10.0)  # ~90% rate
    assert prob_b_beats_a(a, b) > 0.999


def test_is_antisymmetric():
    a = BetaPosterior(15.0, 35.0)
    b = BetaPosterior(22.0, 30.0)
    assert prob_b_beats_a(a, b) == pytest.approx(1.0 - prob_b_beats_a(b, a), abs=1e-9)


@pytest.mark.parametrize(
    "alpha_a,beta_a,alpha_b,beta_b",
    [
        (5.0, 15.0, 7.0, 13.0),
        (1.0, 1.0, 1.0, 1.0),
        (10.0, 20.0, 15.0, 15.0),
        (121.0, 1879.0, 151.0, 1849.0),  # ~120/2000 vs ~150/2000, uniform prior
        (1.0, 999.0, 1.0, 1.0),
        (50.0, 50.0, 1.0, 1.0),
        (3.0, 3.0, 1.0, 500.0),
    ],
)
def test_matches_independent_closed_form_oracle(alpha_a, beta_a, alpha_b, beta_b):
    # alpha_b must be an integer for the oracle's summation formula; every
    # case here uses an integer alpha_b by construction.
    a = BetaPosterior(alpha_a, beta_a)
    b = BetaPosterior(alpha_b, beta_b)
    expected = oracle_prob_b_beats_a(alpha_a, beta_a, alpha_b, beta_b)
    assert prob_b_beats_a(a, b) == pytest.approx(expected, abs=1e-8)


def test_matches_monte_carlo_within_tolerance():
    a = BetaPosterior(20.0, 80.0)
    b = BetaPosterior(28.0, 72.0)
    exact = prob_b_beats_a(a, b)
    mc = prob_b_beats_a_monte_carlo(a, b, n_samples=300_000, seed=1234)
    assert mc == pytest.approx(exact, abs=0.01)


def test_monte_carlo_is_reproducible_with_seed():
    a = BetaPosterior(5.0, 5.0)
    b = BetaPosterior(6.0, 5.0)
    v1 = prob_b_beats_a_monte_carlo(a, b, n_samples=10_000, seed=7)
    v2 = prob_b_beats_a_monte_carlo(a, b, n_samples=10_000, seed=7)
    assert v1 == v2


def test_prob_each_is_best_sums_to_one():
    variants = {
        "control": BetaPosterior(120.0, 1880.0),
        "b": BetaPosterior(150.0, 1850.0),
        "c": BetaPosterior(90.0, 1910.0),
    }
    result = prob_each_is_best(variants, n_samples=100_000, seed=1)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(result.keys()) == set(variants.keys())


def test_prob_each_is_best_favors_the_clearly_better_variant():
    variants = {
        "control": BetaPosterior(20.0, 180.0),  # ~10%
        "treatment": BetaPosterior(70.0, 130.0),  # ~35%
    }
    result = prob_each_is_best(variants, n_samples=100_000, seed=2)
    assert result["treatment"] > 0.99
    assert result["control"] < 0.01


def test_prob_each_is_best_with_two_variants_matches_pairwise():
    # With exactly two variants, prob_each_is_best must agree with the
    # exact pairwise prob_b_beats_a computation (both are estimating the
    # same probability, one exactly and one via Monte Carlo).
    a = BetaPosterior(40.0, 60.0)
    b = BetaPosterior(55.0, 45.0)
    exact_b_wins = prob_b_beats_a(a, b)
    result = prob_each_is_best({"a": a, "b": b}, n_samples=300_000, seed=99)
    assert result["b"] == pytest.approx(exact_b_wins, abs=0.01)
    assert result["a"] == pytest.approx(1.0 - exact_b_wins, abs=0.01)


def test_prob_each_is_best_requires_at_least_two_variants():
    with pytest.raises(ValueError):
        prob_each_is_best({"only": BetaPosterior(1.0, 1.0)})


def test_prob_b_beats_a_result_is_clamped_to_unit_interval():
    a = BetaPosterior(0.001, 0.001)
    b = BetaPosterior(0.001, 0.001)
    result = prob_b_beats_a(a, b)
    assert 0.0 <= result <= 1.0
