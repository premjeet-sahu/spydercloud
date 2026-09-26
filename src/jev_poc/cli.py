from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import load_cases, markdown_report, prediction_dict, summarize
from .providers import get_provider, load_dotenv


def evaluate(args: argparse.Namespace) -> int:
    load_dotenv(args.env)
    cases = load_cases(Path(args.dataset))
    providers = []
    for name in args.providers.split(","):
        try:
            providers.append(get_provider(name))
        except Exception as exc:
            print(f"Cannot initialize {name}: {exc}", file=sys.stderr)
            return 2

    records = []
    for provider in providers:
        print(f"Running {provider.name} on {len(cases)} cases x {args.repeats} repeat(s)...")
        for run in range(1, args.repeats + 1):
            for index, case in enumerate(cases, 1):
                expected = {
                    "route": case["route"],
                    "urgent": case["urgent"],
                    "refund_requested": case["refund_requested"],
                }
                record = {
                    "id": case["id"],
                    "provider": provider.name,
                    "run": run,
                    "expected": expected,
                }
                try:
                    record["prediction"] = prediction_dict(provider.predict(case["text"]))
                    status = "ok"
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    status = "ERROR"
                records.append(record)
                print(f"  run {run} [{index:02}/{len(cases):02}] {case['id']}: {status}")

    threshold = args.threshold
    summary = summarize(records, threshold)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8"
    )
    report = markdown_report(summary)
    (output / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Wrote {output / 'results.json'} and {output / 'report.md'}")
    return 0


def smoke(args: argparse.Namespace) -> int:
    load_dotenv(args.env)
    try:
        provider = get_provider(args.provider)
        prediction = prediction_dict(provider.predict(args.text))
    except Exception as exc:
        print(f"Smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(prediction, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Jev against Bedrock and rules.")
    parser.add_argument("--env", default=".env", help="Path to KEY=VALUE environment file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Classify one ticket")
    smoke_parser.add_argument("--provider", choices=["rules", "jev", "bedrock"], required=True)
    smoke_parser.add_argument(
        "--text",
        default="Production checkout has failed for an hour. Please fix this ASAP.",
    )
    smoke_parser.set_defaults(func=smoke)

    eval_parser = subparsers.add_parser("evaluate", help="Run a labeled benchmark")
    eval_parser.add_argument("--providers", default="rules,jev,bedrock")
    eval_parser.add_argument("--dataset", default="data/tickets.jsonl")
    eval_parser.add_argument("--output", default="results/latest")
    eval_parser.add_argument("--threshold", type=float, default=0.8)
    eval_parser.add_argument("--repeats", type=int, default=1)
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
