import pytest

from bayesab.betadist import BetaPosterior
from bayesab.simulate import simulate_sequential_test


def test_rejects_invalid_true_rates():
    with pytest.raises(ValueError):
        simulate_sequential_test(true_rate_control=-0.1, true_rate_treatment=0.1)
    with pytest.raises(ValueError):
        simulate_sequential_test(true_rate_control=0.1, true_rate_treatment=1.5)


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        simulate_sequential_test(true_rate_control=0.1, true_rate_treatment=0.2, batch_size=0)


def test_stops_and_picks_correct_winner_with_large_true_gap():
    # A large, obvious gap in true rates should be detected quickly and
    # correctly by the stopping rule.
    result = simulate_sequential_test(
        true_rate_control=0.05,
        true_rate_treatment=0.15,
        batch_size=500,
        max_trials_per_arm=200_000,
        loss_threshold=0.0005,
        seed=1,
    )
    assert result.stopped
    assert result.chosen_variant == "treatment"
    assert result.true_best_variant == "treatment"
    assert result.correct is True


def test_never_stops_reports_none_correct():
    # With truly identical rates and a very tight threshold, the test
    # should not have stopped within a small trial budget.
    result = simulate_sequential_test(
        true_rate_control=0.10,
        true_rate_treatment=0.10,
        batch_size=50,
        max_trials_per_arm=200,
        loss_threshold=1e-9,
        seed=2,
    )
    assert result.stopped is False
    assert result.chosen_variant is None
    assert result.correct is None
    assert result.trials_per_arm_at_stop == 200


def test_history_trials_increase_monotonically_by_batch_size():
    result = simulate_sequential_test(
        true_rate_control=0.10,
        true_rate_treatment=0.10,
        batch_size=100,
        max_trials_per_arm=500,
        loss_threshold=1e-9,
        seed=3,
    )
    trials_seen = [snap.trials_per_arm for snap in result.history]
    assert trials_seen == sorted(trials_seen)
    assert trials_seen[0] == 100
    assert all(b - a == 100 for a, b in zip(trials_seen, trials_seen[1:]))


def test_reproducible_with_seed():
    kwargs = dict(
        true_rate_control=0.08,
        true_rate_treatment=0.12,
        batch_size=200,
        max_trials_per_arm=20_000,
        loss_threshold=0.0002,
        seed=99,
    )
    r1 = simulate_sequential_test(**kwargs)
    r2 = simulate_sequential_test(**kwargs)
    assert r1.trials_per_arm_at_stop == r2.trials_per_arm_at_stop
    assert r1.decision == r2.decision
    assert len(r1.history) == len(r2.history)


def test_custom_prior_pulls_early_posterior_mean_toward_itself():
    # A strong prior (large alpha+beta, i.e. a large "pseudo sample size")
    # centered well below the true rates should visibly pull the very
    # first batch's posterior mean down, compared to a weak/uniform prior
    # seeing the identical simulated data (same seed => identical draws).
    # This is the one directionally-guaranteed effect of a strong low
    # prior: it must dominate a single small batch of real data.
    #
    # NOTE: an earlier version of this test instead asserted that a
    # "wrong" prior would slow down *time-to-stop* relative to a uniform
    # prior. That assumption turned out to be false when actually run:
    # because the same prior is applied identically to *both* arms, a
    # strong prior shrinks each posterior's variance for a given amount of
    # real data, which can shrink the expected-loss gap (and hence trigger
    # the stopping rule) *faster* than a weak prior, even while biasing
    # each arm's mean. See benchmarks/compare_prior_strength.py for a
    # measured (not assumed) account of that effect across many seeds.
    common_kwargs = dict(
        true_rate_control=0.10,
        true_rate_treatment=0.20,
        batch_size=100,
        max_trials_per_arm=100,  # exactly one batch
        loss_threshold=1e-9,  # effectively never stop; we just want history[0]
        seed=5,
    )
    uniform = simulate_sequential_test(prior=BetaPosterior.uniform_prior(), **common_kwargs)
    strong_low_prior = simulate_sequential_test(prior=BetaPosterior(1.0, 200.0), **common_kwargs)

    first_uniform = uniform.history[0]
    first_strong = strong_low_prior.history[0]
    assert first_strong.posterior_mean_control < first_uniform.posterior_mean_control
    assert first_strong.posterior_mean_treatment < first_uniform.posterior_mean_treatment


def test_aggregate_accuracy_across_many_seeds_with_clear_gap():
    # This is the empirical check that the stopping rule "actually works":
    # across many independent simulated runs with a real, sizeable gap
    # between true rates, it should pick the true winner in the large
    # majority of runs that do stop.
    n_runs = 40
    correct = 0
    stopped_count = 0
    for seed in range(n_runs):
        result = simulate_sequential_test(
            true_rate_control=0.06,
            true_rate_treatment=0.10,
            batch_size=300,
            max_trials_per_arm=100_000,
            loss_threshold=0.0005,
            seed=seed,
        )
        if result.stopped:
            stopped_count += 1
            if result.correct:
                correct += 1
    assert stopped_count >= n_runs * 0.8
    assert correct >= stopped_count * 0.9
