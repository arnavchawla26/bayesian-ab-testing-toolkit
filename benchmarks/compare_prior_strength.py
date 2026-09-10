#!/usr/bin/env python3
"""Measure how prior strength affects time-to-stop, across many seeds.

This script exists because of a real mistake caught while building this
project's test suite: a test assumed that giving the sequential simulator
a "wrong" prior (concentrated well below the true rates) would slow down
the expected-loss stopping rule relative to a weak/uniform prior. That
assumption was false on the seed the test happened to use. This script
replaces the assumption with a measurement across many seeds: since the
same prior is applied identically to both arms, a *strong* prior (large
alpha+beta) shrinks each arm's posterior variance for a given amount of
real data, which can shrink the estimated gap's uncertainty -- and
therefore the expected loss -- faster than a weak prior, even though it
also biases each arm's mean away from the truth.

Usage: python benchmarks/compare_prior_strength.py
"""

from __future__ import annotations

import numpy as np

from bayesab.betadist import BetaPosterior
from bayesab.simulate import simulate_sequential_test

TRUE_RATE_CONTROL = 0.10
TRUE_RATE_TREATMENT = 0.20
BATCH_SIZE = 100
MAX_TRIALS_PER_ARM = 50_000
LOSS_THRESHOLD = 0.0005
N_SEEDS = 200

PRIORS = {
    "uniform Beta(1,1)": BetaPosterior.uniform_prior(),
    "weak-and-wrong Beta(1,10)": BetaPosterior(1.0, 10.0),
    "strong-and-wrong Beta(1,200)": BetaPosterior(1.0, 200.0),
    "strong-and-roughly-right Beta(15,85)": BetaPosterior(15.0, 85.0),
}


def main() -> None:
    print(
        f"true rates: control={TRUE_RATE_CONTROL}, treatment={TRUE_RATE_TREATMENT}; "
        f"{N_SEEDS} seeds per prior\n"
    )
    header = f"{'prior':<38}{'mean trials/arm':<18}{'median trials/arm':<20}{'accuracy':<10}"
    print(header)
    print("-" * len(header))
    for label, prior in PRIORS.items():
        trials_at_stop = []
        correct = 0
        stopped = 0
        for seed in range(N_SEEDS):
            result = simulate_sequential_test(
                true_rate_control=TRUE_RATE_CONTROL,
                true_rate_treatment=TRUE_RATE_TREATMENT,
                batch_size=BATCH_SIZE,
                max_trials_per_arm=MAX_TRIALS_PER_ARM,
                loss_threshold=LOSS_THRESHOLD,
                prior=prior,
                seed=seed,
            )
            if result.stopped:
                stopped += 1
                trials_at_stop.append(result.trials_per_arm_at_stop)
                if result.correct:
                    correct += 1
        mean_trials = np.mean(trials_at_stop) if trials_at_stop else float("nan")
        median_trials = np.median(trials_at_stop) if trials_at_stop else float("nan")
        accuracy = correct / stopped if stopped else float("nan")
        print(f"{label:<38}{mean_trials:<18.0f}{median_trials:<20.0f}{accuracy:<10.1%}")


if __name__ == "__main__":
    main()
