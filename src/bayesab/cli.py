"""Command-line interface for bayesab.

Two subcommands:

  bayesab analyze    -- analyze one control vs. one or more treatments from
                         observed successes/trials counts.
  bayesab simulate   -- simulate a sequential test against assumed true
                         rates, to see how the stopping rule behaves.
"""

from __future__ import annotations

import argparse
import sys

from bayesab.betadist import BetaPosterior
from bayesab.experiment import Experiment, ExperimentReport, Variant
from bayesab.simulate import simulate_sequential_test


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesab",
        description="From-scratch Bayesian A/B testing toolkit (Beta-Binomial).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="analyze a control vs. one or more treatments"
    )
    analyze.add_argument(
        "--control",
        nargs=2,
        metavar=("SUCCESSES", "TRIALS"),
        required=True,
        type=int,
        help="control conversions and trials, e.g. --control 120 2000",
    )
    analyze.add_argument(
        "--control-name", default="control", help="name for the control variant"
    )
    analyze.add_argument(
        "--treatment",
        nargs=3,
        metavar=("NAME", "SUCCESSES", "TRIALS"),
        action="append",
        required=True,
        help="a treatment: name, conversions, trials. Repeatable, e.g. "
        "--treatment variant_b 150 2000 --treatment variant_c 140 2000",
    )
    analyze.add_argument("--prior-alpha", type=float, default=1.0)
    analyze.add_argument("--prior-beta", type=float, default=1.0)
    analyze.add_argument("--credible-mass", type=float, default=0.95)
    analyze.add_argument("--loss-threshold", type=float, default=0.0001)
    analyze.add_argument(
        "--hdi",
        action="store_true",
        help="use the highest-density interval instead of the equal-tailed one",
    )
    analyze.add_argument(
        "--mc-samples", type=int, default=200_000, help="Monte Carlo draws for prob-best"
    )
    analyze.add_argument("--seed", type=int, default=None)

    simulate = sub.add_parser(
        "simulate", help="simulate a sequential test against assumed true rates"
    )
    simulate.add_argument("--true-control-rate", type=float, required=True)
    simulate.add_argument("--true-treatment-rate", type=float, required=True)
    simulate.add_argument("--batch-size", type=int, default=200)
    simulate.add_argument("--max-trials", type=int, default=50_000)
    simulate.add_argument("--loss-threshold", type=float, default=0.0001)
    simulate.add_argument("--prior-alpha", type=float, default=1.0)
    simulate.add_argument("--prior-beta", type=float, default=1.0)
    simulate.add_argument("--seed", type=int, default=None)

    return parser


def _run_analyze(args: argparse.Namespace) -> None:
    prior = BetaPosterior(args.prior_alpha, args.prior_beta)
    control = Variant(args.control_name, args.control[0], args.control[1], prior=prior)
    treatments = [
        Variant(name, int(successes), int(trials), prior=prior)
        for name, successes, trials in args.treatment
    ]
    experiment = Experiment(
        control,
        treatments,
        credible_mass=args.credible_mass,
        loss_threshold=args.loss_threshold,
        use_hdi=args.hdi,
    )
    report = experiment.analyze(seed=args.seed, mc_samples=args.mc_samples)
    _print_report(report)


def _print_report(report: ExperimentReport) -> None:
    interval_label = "credible interval"
    header = (
        f"{'variant':<14}{'successes/trials':<18}{'obs. rate':<12}"
        f"{'post. mean':<12}{f'{report.credible_mass:.0%} ' + interval_label:<26}"
        f"{'P(beats control)':<18}{'P(best)':<10}"
    )
    print(header)
    print("-" * len(header))
    for name, v in report.variants.items():
        beats = "--" if v.prob_beats_control is None else f"{v.prob_beats_control:.4f}"
        interval = f"[{v.credible_interval.lower:.4f}, {v.credible_interval.upper:.4f}]"
        marker = " *" if name == report.leader else ""
        print(
            f"{name + marker:<14}{f'{v.successes}/{v.trials}':<18}{v.observed_rate:<12.4f}"
            f"{v.posterior_mean:<12.4f}{interval:<26}{beats:<18}"
            f"{report.prob_best.get(name, 0.0):<10.4f}"
        )
    print()
    print(f"leader: {report.leader}  (runner-up: {report.runner_up})")
    print(f"expected loss of choosing {report.leader}: {report.expected_loss_leader:.6f}")
    print(f"expected loss of choosing {report.runner_up}: {report.expected_loss_runner_up:.6f}")
    print(f"loss threshold: {report.loss_threshold}")
    print(f"decision: {report.decision}")


def _run_simulate(args: argparse.Namespace) -> None:
    prior = BetaPosterior(args.prior_alpha, args.prior_beta)
    result = simulate_sequential_test(
        true_rate_control=args.true_control_rate,
        true_rate_treatment=args.true_treatment_rate,
        batch_size=args.batch_size,
        max_trials_per_arm=args.max_trials,
        loss_threshold=args.loss_threshold,
        prior=prior,
        seed=args.seed,
    )
    print(f"true control rate:   {args.true_control_rate}")
    print(f"true treatment rate: {args.true_treatment_rate}")
    print(f"stopped: {result.stopped}")
    print(f"trials per arm at stop: {result.trials_per_arm_at_stop}")
    print(f"decision: {result.decision}")
    print(f"chosen variant: {result.chosen_variant}")
    print(f"true best variant: {result.true_best_variant}")
    print(f"correct: {result.correct}")
    print(f"batches simulated: {len(result.history)}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        _run_analyze(args)
    elif args.command == "simulate":
        _run_simulate(args)
    else:  # pragma: no cover - argparse enforces a valid subcommand
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
