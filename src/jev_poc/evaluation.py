from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .providers import Prediction


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                case = json.loads(line)
                required = {"id", "text", "route", "urgent", "refund_requested"}
                missing = required - case.keys()
                if missing:
                    raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
                cases.append(case)
    return cases


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(records: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["provider"]].append(record)

    summaries = {}
    for provider, rows in grouped.items():
        successful = [row for row in rows if row.get("prediction")]
        failures = len(rows) - len(successful)
        exact = []
        route_correct = []
        urgent_correct = []
        refund_correct = []
        automated = []
        costs = []
        urgent_brier = []
        refund_brier = []
        signatures: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in successful:
            pred = row["prediction"]
            expected = row["expected"]
            flags = (
                pred["route"] == expected["route"],
                pred["urgent"] == expected["urgent"],
                pred["refund_requested"] == expected["refund_requested"],
            )
            route_correct.append(flags[0])
            urgent_correct.append(flags[1])
            refund_correct.append(flags[2])
            exact.append(all(flags))
            urgent_brier.append((pred["urgent_probability"] - float(expected["urgent"])) ** 2)
            refund_brier.append((pred["refund_probability"] - float(expected["refund_requested"])) ** 2)
            signatures[row["id"]].append(
                (pred["route"], pred["urgent"], pred["refund_requested"])
            )
            if pred["decision_confidence"] >= threshold:
                automated.append(all(flags))
            if pred.get("estimated_cost_usd") is not None:
                costs.append(pred["estimated_cost_usd"])

        def ratio(items: list[bool]) -> float | None:
            return sum(items) / len(items) if items else None

        latencies = [row["prediction"]["latency_ms"] for row in successful]
        repeated_ids = [values for values in signatures.values() if len(values) > 1]
        repeat_agreement = (
            sum(len(set(values)) == 1 for values in repeated_ids) / len(repeated_ids)
            if repeated_ids
            else None
        )
        summaries[provider] = {
            "calls": len(rows),
            "unique_cases": len({row["id"] for row in rows}),
            "parse_or_call_failures": failures,
            "route_accuracy": ratio(route_correct),
            "urgent_accuracy": ratio(urgent_correct),
            "refund_accuracy": ratio(refund_correct),
            "exact_match_accuracy": ratio(exact),
            "mean_latency_ms": statistics.mean(latencies) if latencies else None,
            "p95_latency_ms": percentile(latencies, 0.95) if latencies else None,
            "automation_threshold": threshold,
            "automation_coverage": len(automated) / len(rows) if rows else None,
            "automated_exact_accuracy": ratio(automated),
            "urgent_brier_score": statistics.mean(urgent_brier) if urgent_brier else None,
            "refund_brier_score": statistics.mean(refund_brier) if refund_brier else None,
            "repeat_agreement": repeat_agreement,
            "estimated_total_cost_usd": sum(costs) if costs else None,
        }
    return summaries


def estimated_cost(prediction: Prediction) -> float | None:
    if prediction.provider == "jev" and prediction.input_tokens is not None:
        input_price = float(os.environ.get("JEV_INPUT_USD_PER_MILLION", "0.042"))
        return prediction.input_tokens / 1_000_000 * input_price
    if prediction.provider == "bedrock" and prediction.input_tokens is not None:
        input_price = float(os.environ.get("BEDROCK_INPUT_USD_PER_MILLION", "0"))
        output_price = float(os.environ.get("BEDROCK_OUTPUT_USD_PER_MILLION", "0"))
        if input_price == 0 and output_price == 0:
            return None
        return (
            prediction.input_tokens / 1_000_000 * input_price
            + (prediction.output_tokens or 0) / 1_000_000 * output_price
        )
    return 0.0 if prediction.provider == "rules" else None


def prediction_dict(prediction: Prediction) -> dict[str, Any]:
    data = asdict(prediction)
    data.pop("raw", None)
    data.update(
        urgent=prediction.urgent,
        refund_requested=prediction.refund_requested,
        decision_confidence=prediction.decision_confidence,
        estimated_cost_usd=estimated_cost(prediction),
    )
    return data


def markdown_report(summary: dict[str, dict[str, Any]]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    def number(value: float | None, suffix: str = "") -> str:
        return "n/a" if value is None else f"{value:.2f}{suffix}"

    lines = [
        "# Jev evaluation result",
        "",
        "| Provider | Route | Urgent | Refund | Exact | Coverage | Auto accuracy | Brier U/R | Repeat agreement | Mean/P95 latency | Failures | Est. cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, values in summary.items():
        cost = values["estimated_total_cost_usd"]
        lines.append(
            f"| {provider} | {pct(values['route_accuracy'])} | {pct(values['urgent_accuracy'])} | "
            f"{pct(values['refund_accuracy'])} | {pct(values['exact_match_accuracy'])} | "
            f"{pct(values['automation_coverage'])} | {pct(values['automated_exact_accuracy'])} | "
            f"{number(values['urgent_brier_score'])}/{number(values['refund_brier_score'])} | "
            f"{pct(values['repeat_agreement'])} | "
            f"{number(values['mean_latency_ms'])}/{number(values['p95_latency_ms'])} ms | "
            f"{values['parse_or_call_failures']} | {'n/a' if cost is None else f'${cost:.6f}'} |"
        )
    lines.extend(
        [
            "",
            "Coverage is the share of all cases whose minimum decision confidence met the configured threshold.",
            "Automated accuracy is exact-match accuracy only within that auto-action subset.",
            "Brier scores measure probability calibration for urgent/refund; lower is better and 0 is perfect.",
            "Repeat agreement is reported only when the benchmark uses two or more repeats.",
        ]
    )
    return "\n".join(lines) + "\n"
