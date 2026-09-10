#!/usr/bin/env python3
"""Compare naive continuous-peeking frequentist significance testing against
bayesab's expected-loss stopping rule, under the null hypothesis (true rates
equal).

This is the classic "optional stopping" / "peeking problem" demonstration:
if you repeatedly run a two-proportion z-test as data trickles in and stop
the instant p < 0.05, your actual false-positive rate is far higher than
5%, because you're taking the minimum p-value over many correlated looks,
not a single look. This script *measures* that inflation.

IMPORTANT interpretation note -- this is the actual result of running this
script, not an assumption: bayesab's expected-loss rule *also* stops and
recommends a variant very often under the null (much more often than 5%).
The first version of this script mislabeled that the same way as the
frequentist false-positive rate. That framing is wrong, and the mistake
is left visible here deliberately (see the printed explanation below)
rather than quietly fixed, because it is the kind of thing worth tracing
rather than assuming away: a frequentist p < 0.05 asserts "a difference
exists" -- under the null that assertion is simply false, a real error.
The Bayesian rule instead asserts "the expected cost of picking either
variant is below `threshold`" -- under the null (or anywhere the true
rates are close), that assertion is TRUE almost by definition, because
there genuinely isn't a meaningful difference to lose by picking either
one. Stopping there isn't a mistake to be caught later; it's the correct
recognition that further data collection has no practical payoff. What
should actually be compared is not "how often does each method stop" but
"how much does the decision each method makes actually cost, and does
either method *bound* that cost." This script measures both.

Usage: python benchmarks/compare_frequentist_peeking.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from bayesab.betadist import BetaPosterior
from bayesab.loss import expected_loss, stopping_decision

TRUE_RATE = 0.10
N_RUNS = 300
BATCH_SIZE = 250
MAX_TRIALS_PER_ARM = 10_000
FREQUENTIST_ALPHA = 0.05
BAYES_LOSS_THRESHOLD = 0.0005


def two_proportion_p_value(successes_a: int, trials_a: int, successes_b: int, trials_b: int) -> float:
    """Two-sided two-proportion z-test p-value (pooled variance, standard formula)."""
    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    p_pool = (successes_a + successes_b) / (trials_a + trials_b)
    se = (p_pool * (1 - p_pool) * (1 / trials_a + 1 / trials_b)) ** 0.5
    if se == 0:
        return 1.0
    z = (p_b - p_a) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def simulate_one_run(seed: int):
    """Returns (freq_stopped, freq_trials, bayes_stopped, bayes_trials, bayes_realized_loss)."""
    rng = np.random.default_rng(seed)
    prior = BetaPosterior.uniform_prior()

    successes_a = successes_b = trials = 0
    freq_stopped = False
    freq_trials_at_stop = MAX_TRIALS_PER_ARM
    bayes_stopped = False
    bayes_trials_at_stop = MAX_TRIALS_PER_ARM
    bayes_realized_loss = None

    while trials < MAX_TRIALS_PER_ARM:
        successes_a += int(rng.binomial(1, TRUE_RATE, size=BATCH_SIZE).sum())
        successes_b += int(rng.binomial(1, TRUE_RATE, size=BATCH_SIZE).sum())
        trials += BATCH_SIZE

        if not freq_stopped:
            p = two_proportion_p_value(successes_a, trials, successes_b, trials)
            if p < FREQUENTIST_ALPHA:
                freq_stopped = True
                freq_trials_at_stop = trials

        if not bayes_stopped:
            post_a = prior.update(successes_a, trials)
            post_b = prior.update(successes_b, trials)
            loss = expected_loss(post_a, post_b)
            decision = stopping_decision(loss, BAYES_LOSS_THRESHOLD)
            if decision != "continue":
                bayes_stopped = True
                bayes_trials_at_stop = trials
                bayes_realized_loss = (
                    loss.loss_choosing_a if decision == "stop_choose_a" else loss.loss_choosing_b
                )

        if freq_stopped and bayes_stopped:
            break

    return freq_stopped, freq_trials_at_stop, bayes_stopped, bayes_trials_at_stop, bayes_realized_loss


def main() -> None:
    freq_stops = 0
    freq_trials_at_stop = []
    bayes_stops = 0
    bayes_trials_at_stop = []
    bayes_realized_losses = []

    for seed in range(N_RUNS):
        freq_stopped, freq_t, bayes_stopped, bayes_t, bayes_loss = simulate_one_run(seed)
        if freq_stopped:
            freq_stops += 1
            freq_trials_at_stop.append(freq_t)
        if bayes_stopped:
            bayes_stops += 1
            bayes_trials_at_stop.append(bayes_t)
            bayes_realized_losses.append(bayes_loss)

    print(f"Simulated {N_RUNS} runs under the null (true_rate_control == true_rate_treatment == {TRUE_RATE})")
    print(f"Each run peeks every {BATCH_SIZE} trials/arm, up to {MAX_TRIALS_PER_ARM} trials/arm.\n")

    print("Naive frequentist (two-proportion z-test, stop at first p < 0.05 on ANY peek):")
    print(f"  stops and declares 'significant difference': {freq_stops / N_RUNS:.1%} of runs")
    print(f"  (nominal alpha was {FREQUENTIST_ALPHA:.0%} -- this IS a real error rate: the null is true,")
    print("   so every one of these runs asserts a difference that does not exist)")
    if freq_trials_at_stop:
        print(f"  median trials/arm at (false) stop: {int(np.median(freq_trials_at_stop))}")
    print()

    print(f"bayesab expected-loss stopping rule (threshold = {BAYES_LOSS_THRESHOLD}):")
    print(f"  stops and recommends a variant: {bayes_stops / N_RUNS:.1%} of runs")
    print("  (NOT a comparable 'error rate' -- see the module docstring. What matters is")
    print("   whether the recommendation's *realized cost* is actually bounded by threshold:)")
    if bayes_realized_losses:
        print(f"  mean realized loss of the chosen decision:   {np.mean(bayes_realized_losses):.6f}")
        print(f"  max realized loss of the chosen decision:    {np.max(bayes_realized_losses):.6f}")
        print(f"  threshold:                                    {BAYES_LOSS_THRESHOLD:.6f}")
        all_bounded = all(loss <= BAYES_LOSS_THRESHOLD + 1e-9 for loss in bayes_realized_losses)
        print(f"  every stop's realized loss <= threshold: {all_bounded}")
    if bayes_trials_at_stop:
        print(f"  median trials/arm at stop: {int(np.median(bayes_trials_at_stop))}")


if __name__ == "__main__":
    main()
