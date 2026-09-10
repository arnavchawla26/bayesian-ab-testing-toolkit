import numpy as np
import pytest

from bayesab.betadist import BetaPosterior
from bayesab.loss import LossResult, expected_loss, stopping_decision


def _monte_carlo_expected_loss(a: BetaPosterior, b: BetaPosterior, n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    samples_a = a.rvs(size=n, random_state=rng)
    samples_b = b.rvs(size=n, random_state=rng)
    loss_a = np.mean(np.maximum(samples_b - samples_a, 0.0))
    loss_b = np.mean(np.maximum(samples_a - samples_b, 0.0))
    return loss_a, loss_b


def test_identical_distributions_have_equal_nonzero_loss():
    # Even when A and B have the same distribution, independent samples
    # differ, so the expected loss of "guessing wrong" is not exactly
    # zero -- it reflects the posterior's residual uncertainty.
    a = BetaPosterior(20.0, 20.0)
    b = BetaPosterior(20.0, 20.0)
    result = expected_loss(a, b)
    assert result.loss_choosing_a == pytest.approx(result.loss_choosing_b, abs=1e-9)
    assert result.loss_choosing_a > 0.0


def test_clearly_better_variant_has_near_zero_loss_when_chosen():
    a = BetaPosterior(20.0, 180.0)  # ~10%, narrow after many trials
    b = BetaPosterior(700.0, 300.0)  # ~70%, narrow after many trials
    result = expected_loss(a, b)
    # Choosing B (the clear winner) risks almost nothing; choosing A risks a lot.
    assert result.loss_choosing_b < 0.001
    assert result.loss_choosing_a > 0.4


def test_matches_monte_carlo_cross_check():
    a = BetaPosterior(30.0, 70.0)
    b = BetaPosterior(38.0, 62.0)
    exact = expected_loss(a, b)
    mc_loss_a, mc_loss_b = _monte_carlo_expected_loss(a, b, n=400_000, seed=42)
    assert exact.loss_choosing_a == pytest.approx(mc_loss_a, abs=0.001)
    assert exact.loss_choosing_b == pytest.approx(mc_loss_b, abs=0.001)


def test_more_uncertain_priors_have_more_loss_at_stake():
    # Two variants tied at the same mean, but one pair has seen far less
    # data (wider posteriors): the wider pair should have strictly higher
    # expected loss for either choice, since there's more residual
    # uncertainty about which is actually better.
    a_narrow = BetaPosterior(500.0, 500.0)
    b_narrow = BetaPosterior(500.0, 500.0)
    a_wide = BetaPosterior(5.0, 5.0)
    b_wide = BetaPosterior(5.0, 5.0)
    narrow = expected_loss(a_narrow, b_narrow)
    wide = expected_loss(a_wide, b_wide)
    assert wide.loss_choosing_a > narrow.loss_choosing_a
    assert wide.loss_choosing_b > narrow.loss_choosing_b


def test_better_property_picks_lower_loss_variant():
    a = BetaPosterior(20.0, 180.0)
    b = BetaPosterior(700.0, 300.0)
    result = expected_loss(a, b)
    assert result.better == "b"


def test_better_property_ties_to_a():
    result = LossResult(loss_choosing_a=0.001, loss_choosing_b=0.001)
    assert result.better == "a"


def test_stopping_decision_continues_when_both_losses_exceed_threshold():
    loss = LossResult(loss_choosing_a=0.01, loss_choosing_b=0.02)
    assert stopping_decision(loss, threshold=0.0001) == "continue"


def test_stopping_decision_stops_on_the_safer_variant():
    loss = LossResult(loss_choosing_a=0.0002, loss_choosing_b=0.00002)
    assert stopping_decision(loss, threshold=0.0001) == "stop_choose_b"


def test_stopping_decision_stops_on_a_when_only_a_is_under_threshold():
    loss = LossResult(loss_choosing_a=0.00005, loss_choosing_b=0.0002)
    assert stopping_decision(loss, threshold=0.0001) == "stop_choose_a"


def test_stopping_decision_ties_resolve_to_a():
    loss = LossResult(loss_choosing_a=0.00005, loss_choosing_b=0.00005)
    assert stopping_decision(loss, threshold=0.0001) == "stop_choose_a"


def test_stopping_decision_picks_the_lower_loss_even_if_both_under_threshold():
    # Both under threshold, but B is markedly safer -- should still pick B,
    # not just "whichever is under threshold first."
    loss = LossResult(loss_choosing_a=0.00009, loss_choosing_b=0.00001)
    assert stopping_decision(loss, threshold=0.0001) == "stop_choose_b"


def test_stopping_decision_rejects_negative_threshold():
    loss = LossResult(loss_choosing_a=0.01, loss_choosing_b=0.01)
    with pytest.raises(ValueError):
        stopping_decision(loss, threshold=-0.001)


def test_loss_result_is_never_negative():
    # Quadrature noise near-zero true loss could in principle nudge the
    # raw integral below zero; the implementation must clamp this.
    a = BetaPosterior(5000.0, 5000.0)
    b = BetaPosterior(5000.0, 5000.0)
    result = expected_loss(a, b)
    assert result.loss_choosing_a >= 0.0
    assert result.loss_choosing_b >= 0.0
