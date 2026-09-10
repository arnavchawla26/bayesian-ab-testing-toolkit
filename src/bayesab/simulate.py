"""Sequential simulation: does the expected-loss stopping rule actually work?

Given assumed *true* conversion rates for a control and a treatment (which
a real experimenter never knows), this simulates traffic arriving in
batches, applies the expected-loss stopping rule after each batch, and
reports when -- if ever -- it would have stopped, and whether the variant
it picked was in fact the better one. Running this across many seeds is
how the stopping rule's practical behavior (how much traffic it typically
needs, how often it picks the true loser) can be checked empirically
rather than assumed from theory -- see tests/test_simulate.py, which runs
many seeds and checks aggregate error rates rather than any single run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bayesab.betadist import BetaPosterior
from bayesab.loss import expected_loss, stopping_decision


@dataclass(frozen=True)
class SequentialSnapshot:
    trials_per_arm: int
    posterior_mean_control: float
    posterior_mean_treatment: float
    loss_choosing_control: float
    loss_choosing_treatment: float
    decision: str


@dataclass(frozen=True)
class SequentialSimulationResult:
    stopped: bool
    trials_per_arm_at_stop: int
    decision: str
    chosen_variant: str | None  # "control", "treatment", or None if never stopped
    true_best_variant: str
    correct: bool | None  # None if never stopped
    history: list[SequentialSnapshot] = field(default_factory=list)


def simulate_sequential_test(
    true_rate_control: float,
    true_rate_treatment: float,
    batch_size: int = 200,
    max_trials_per_arm: int = 50_000,
    loss_threshold: float = 0.0001,
    prior: BetaPosterior | None = None,
    seed: int | None = None,
) -> SequentialSimulationResult:
    """Simulate one sequential A/B test run against known true rates.

    Traffic is split evenly and generated in batches of `batch_size` per
    arm via independent Binomial draws at the given true rates. After each
    batch, the posterior is updated (conjugate, exact) and the
    expected-loss stopping rule is evaluated. Stops as soon as the rule
    says to stop, or after `max_trials_per_arm` trials per arm if it never
    does.
    """
    if not (0.0 <= true_rate_control <= 1.0 and 0.0 <= true_rate_treatment <= 1.0):
        raise ValueError("true rates must be in [0, 1]")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    prior = prior if prior is not None else BetaPosterior.uniform_prior()
    rng = np.random.default_rng(seed)

    successes_c = successes_t = trials = 0
    history: list[SequentialSnapshot] = []

    while trials < max_trials_per_arm:
        batch_c = int(rng.binomial(1, true_rate_control, size=batch_size).sum())
        batch_t = int(rng.binomial(1, true_rate_treatment, size=batch_size).sum())
        successes_c += batch_c
        successes_t += batch_t
        trials += batch_size

        post_c = prior.update(successes_c, trials)
        post_t = prior.update(successes_t, trials)
        loss_pair = expected_loss(post_c, post_t)  # a=control, b=treatment
        decision = stopping_decision(loss_pair, loss_threshold)

        history.append(
            SequentialSnapshot(
                trials_per_arm=trials,
                posterior_mean_control=post_c.mean,
                posterior_mean_treatment=post_t.mean,
                loss_choosing_control=loss_pair.loss_choosing_a,
                loss_choosing_treatment=loss_pair.loss_choosing_b,
                decision=decision,
            )
        )
        if decision != "continue":
            break

    final = history[-1]
    stopped = final.decision != "continue"
    chosen_variant = None
    if stopped:
        chosen_variant = "control" if final.decision == "stop_choose_a" else "treatment"
    true_best_variant = "control" if true_rate_control >= true_rate_treatment else "treatment"
    correct = (chosen_variant == true_best_variant) if stopped else None

    return SequentialSimulationResult(
        stopped=stopped,
        trials_per_arm_at_stop=final.trials_per_arm,
        decision=final.decision,
        chosen_variant=chosen_variant,
        true_best_variant=true_best_variant,
        correct=correct,
        history=history,
    )
