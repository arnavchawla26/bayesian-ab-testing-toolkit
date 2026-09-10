"""Ties the primitives together into a runnable A/B(/N) test analysis.

A :class:`Variant` is raw observed data (successes out of trials) plus a
prior. An :class:`Experiment` is one control variant plus one or more
treatment variants; :meth:`Experiment.analyze` produces a full report:
each variant's posterior, its credible interval, its probability of
beating the control, and -- for the overall test -- which variant
currently leads, the expected loss of shipping it instead of the
runner-up, and whether the expected-loss stopping rule says to stop.

N-arm caveat: the expected-loss stopping decision compares only the
current leader (highest posterior mean) against the runner-up. This is a
standard practical simplification (the alternative -- a full joint
multi-way loss -- requires either a much larger joint quadrature or
accepting the same Monte Carlo error `prob_each_is_best` already carries).
`prob_each_is_best` is reported for all variants via Monte Carlo so the
full picture is visible even when more than two variants are in play.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bayesab.betadist import BetaPosterior
from bayesab.comparison import prob_b_beats_a, prob_each_is_best
from bayesab.intervals import Interval, equal_tailed_interval, hdi
from bayesab.loss import expected_loss, stopping_decision


@dataclass(frozen=True)
class Variant:
    """Raw observed data for one arm of the experiment."""

    name: str
    successes: int
    trials: int
    prior: BetaPosterior = field(default_factory=BetaPosterior.uniform_prior)

    def __post_init__(self) -> None:
        if self.trials < 0:
            raise ValueError(f"trials must be >= 0, got {self.trials}")
        if self.successes < 0 or self.successes > self.trials:
            raise ValueError(
                f"successes must be in [0, trials], got successes={self.successes}, "
                f"trials={self.trials}"
            )

    def posterior(self) -> BetaPosterior:
        return self.prior.update(self.successes, self.trials)

    @property
    def observed_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


@dataclass(frozen=True)
class VariantReport:
    name: str
    successes: int
    trials: int
    observed_rate: float
    prior_alpha: float
    prior_beta: float
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_std: float
    credible_interval: Interval
    prob_beats_control: float | None  # None for the control's own row


@dataclass(frozen=True)
class ExperimentReport:
    variants: dict[str, VariantReport]
    prob_best: dict[str, float]
    leader: str
    runner_up: str
    expected_loss_leader: float
    expected_loss_runner_up: float
    decision: str  # "stop_choose_<name>" or "continue"
    credible_mass: float
    loss_threshold: float


class Experiment:
    """A control variant plus one or more treatment variants."""

    def __init__(
        self,
        control: Variant,
        treatments: list[Variant],
        credible_mass: float = 0.95,
        loss_threshold: float = 0.0001,
        use_hdi: bool = False,
    ) -> None:
        if not treatments:
            raise ValueError("an experiment needs at least one treatment variant")
        names = [control.name] + [t.name for t in treatments]
        if len(set(names)) != len(names):
            raise ValueError(f"variant names must be unique, got {names}")
        self.control = control
        self.treatments = treatments
        self.credible_mass = credible_mass
        self.loss_threshold = loss_threshold
        self.use_hdi = use_hdi

    def analyze(self, seed: int | None = None, mc_samples: int = 200_000) -> ExperimentReport:
        all_variants = [self.control] + list(self.treatments)
        posteriors: dict[str, BetaPosterior] = {v.name: v.posterior() for v in all_variants}
        control_posterior = posteriors[self.control.name]
        interval_fn = hdi if self.use_hdi else equal_tailed_interval

        reports: dict[str, VariantReport] = {}
        for v in all_variants:
            post = posteriors[v.name]
            interval = interval_fn(post, self.credible_mass)
            is_control = v.name == self.control.name
            prob_beats = None if is_control else prob_b_beats_a(control_posterior, post)
            reports[v.name] = VariantReport(
                name=v.name,
                successes=v.successes,
                trials=v.trials,
                observed_rate=v.observed_rate,
                prior_alpha=v.prior.alpha,
                prior_beta=v.prior.beta,
                posterior_alpha=post.alpha,
                posterior_beta=post.beta,
                posterior_mean=post.mean,
                posterior_std=post.std,
                credible_interval=interval,
                prob_beats_control=prob_beats,
            )

        prob_best = prob_each_is_best(posteriors, n_samples=mc_samples, seed=seed)

        ranked = sorted(posteriors.items(), key=lambda kv: kv[1].mean, reverse=True)
        leader_name, leader_post = ranked[0]
        runner_name, runner_post = ranked[1]

        loss_pair = expected_loss(runner_post, leader_post)  # a=runner_up, b=leader
        decision_ab = stopping_decision(loss_pair, self.loss_threshold)
        if decision_ab == "stop_choose_a":
            decision = f"stop_choose_{runner_name}"
        elif decision_ab == "stop_choose_b":
            decision = f"stop_choose_{leader_name}"
        else:
            decision = "continue"

        return ExperimentReport(
            variants=reports,
            prob_best=prob_best,
            leader=leader_name,
            runner_up=runner_name,
            expected_loss_leader=loss_pair.loss_choosing_b,
            expected_loss_runner_up=loss_pair.loss_choosing_a,
            decision=decision,
            credible_mass=self.credible_mass,
            loss_threshold=self.loss_threshold,
        )
