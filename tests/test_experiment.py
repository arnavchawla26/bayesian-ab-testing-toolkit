import pytest

from bayesab.betadist import BetaPosterior
from bayesab.experiment import Experiment, Variant


def test_variant_rejects_invalid_counts():
    with pytest.raises(ValueError):
        Variant("x", successes=5, trials=3)
    with pytest.raises(ValueError):
        Variant("x", successes=1, trials=-1)


def test_variant_observed_rate():
    v = Variant("x", successes=25, trials=100)
    assert v.observed_rate == pytest.approx(0.25)


def test_variant_observed_rate_with_zero_trials_is_zero():
    v = Variant("x", successes=0, trials=0)
    assert v.observed_rate == 0.0


def test_variant_posterior_uses_its_prior():
    prior = BetaPosterior(2.0, 2.0)
    v = Variant("x", successes=10, trials=20, prior=prior)
    post = v.posterior()
    assert post.alpha == pytest.approx(12.0)
    assert post.beta == pytest.approx(12.0)


def test_experiment_rejects_no_treatments():
    control = Variant("control", 100, 1000)
    with pytest.raises(ValueError):
        Experiment(control, [])


def test_experiment_rejects_duplicate_names():
    control = Variant("control", 100, 1000)
    dup = Variant("control", 120, 1000)
    with pytest.raises(ValueError):
        Experiment(control, [dup])


def test_two_variant_analysis_clear_winner():
    control = Variant("control", 120, 2000)  # 6%
    treatment = Variant("treatment", 220, 2000)  # 11%
    experiment = Experiment(control, [treatment], loss_threshold=0.0001)
    report = experiment.analyze(seed=1, mc_samples=100_000)

    assert report.leader == "treatment"
    assert report.runner_up == "control"
    assert report.variants["treatment"].prob_beats_control > 0.999
    assert report.variants["control"].prob_beats_control is None
    assert report.decision == "stop_choose_treatment"
    assert report.expected_loss_leader < report.loss_threshold


def test_two_variant_analysis_inconclusive_stays_continue():
    control = Variant("control", 100, 2000)
    treatment = Variant("treatment", 103, 2000)  # nearly identical, little data
    experiment = Experiment(control, [treatment], loss_threshold=0.0001)
    report = experiment.analyze(seed=2, mc_samples=100_000)
    assert report.decision == "continue"


def test_report_credible_intervals_contain_posterior_mean_roughly():
    control = Variant("control", 500, 1000)
    treatment = Variant("treatment", 520, 1000)
    experiment = Experiment(control, [treatment])
    report = experiment.analyze(seed=3, mc_samples=50_000)
    for v in report.variants.values():
        assert v.credible_interval.lower <= v.posterior_mean <= v.credible_interval.upper


def test_report_uses_hdi_when_requested():
    control = Variant("control", 30, 1000)  # skewed low-rate posterior
    treatment = Variant("treatment", 35, 1000)
    equal_tailed_experiment = Experiment(control, [treatment], use_hdi=False)
    hdi_experiment = Experiment(control, [treatment], use_hdi=True)
    equal_tailed_report = equal_tailed_experiment.analyze(seed=4, mc_samples=20_000)
    hdi_report = hdi_experiment.analyze(seed=4, mc_samples=20_000)
    control_equal_tailed = equal_tailed_report.variants["control"].credible_interval
    control_hdi = hdi_report.variants["control"].credible_interval
    # For a skewed posterior these should genuinely differ.
    assert control_hdi.width <= control_equal_tailed.width


def test_three_variant_experiment_prob_best_sums_to_one():
    control = Variant("control", 100, 2000)
    b = Variant("b", 130, 2000)
    c = Variant("c", 90, 2000)
    experiment = Experiment(control, [b, c])
    report = experiment.analyze(seed=5, mc_samples=100_000)
    assert sum(report.prob_best.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(report.prob_best.keys()) == {"control", "b", "c"}


def test_three_variant_experiment_leader_is_highest_mean():
    control = Variant("control", 100, 2000)
    b = Variant("b", 130, 2000)
    c = Variant("c", 90, 2000)
    experiment = Experiment(control, [b, c])
    report = experiment.analyze(seed=6, mc_samples=50_000)
    means = {name: v.posterior_mean for name, v in report.variants.items()}
    assert report.leader == max(means, key=means.get)


def test_analyze_is_reproducible_with_seed():
    control = Variant("control", 100, 2000)
    treatment = Variant("treatment", 130, 2000)
    experiment = Experiment(control, [treatment])
    report1 = experiment.analyze(seed=123, mc_samples=20_000)
    report2 = experiment.analyze(seed=123, mc_samples=20_000)
    assert report1.prob_best == report2.prob_best
