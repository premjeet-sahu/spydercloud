from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
# Codex UI polish: larger result boxes and explicit metric units.
_CODEX_UI_POLISH = """
<style>
textarea {
    min-height: 220px !important;
}
pre, code {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
}
div[data-testid="stJson"] pre,
div[data-testid="stCodeBlock"] pre {
    max-height: 720px !important;
}
div[data-testid="stMetricValue"] {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    font-size: 1.15rem !important;
}
div[data-testid="stDataFrame"] {
    width: 100% !important;
}
</style>
"""

_ORIGINAL_SET_PAGE_CONFIG = st.set_page_config
def _set_page_config_with_codex_ui(*args, **kwargs):
    result = _ORIGINAL_SET_PAGE_CONFIG(*args, **kwargs)
    st.markdown(_CODEX_UI_POLISH, unsafe_allow_html=True)
    return result
st.set_page_config = _set_page_config_with_codex_ui

_ORIGINAL_METRIC = st.metric
def _metric_with_units(label, value, *args, **kwargs):
    unit_map = {
        "latency": "ms",
        "processing": "ms",
        "wall": "ms",
        "duration": "ms",
        "input token": "tokens",
        "output token": "tokens",
        "total token": "tokens",
        "cost": "USD",
    }
    label_text = str(label)
    lower = label_text.lower()
    if "(" not in label_text:
        for key, unit in unit_map.items():
            if key in lower:
                label_text = f"{label_text} ({unit})"
                break
    return _ORIGINAL_METRIC(label_text, value, *args, **kwargs)
st.metric = _metric_with_units


from jev_poc.evaluation import estimated_cost, load_cases
from jev_poc.providers import QUESTION_TEXT, ROUTE_CRITERIA, Prediction, get_provider, load_dotenv
from jev_poc.runtime import ProviderError, credential_hint, sanitize, system_diagnostics


ROOT = Path(__file__).resolve().parent
load_dotenv(str(ROOT / ".env"))
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("jev_poc.app")

st.set_page_config(
    page_title="Jev vs Claude",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: 0;}
      [data-testid="stMetric"] {border: 1px solid #d8dee8; padding: 0.75rem; border-radius: 6px;}
      [data-testid="stMetricLabel"] {font-size: 0.8rem;}
      .result-ok {color: #137333; font-weight: 600;}
      .result-bad {color: #b3261e; font-weight: 600;}
      .muted {color: #5f6368; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def sample_cases() -> list[dict[str, Any]]:
    return load_cases(ROOT / "data" / "tickets.jsonl")


def initialize_state(samples: list[dict[str, Any]]) -> None:
    sample = samples[0]
    defaults = {
        "ticket_text": sample["text"],
        "expected_route": sample["route"],
        "expected_urgent": sample["urgent"],
        "expected_refund": sample["refund_requested"],
        "loaded_sample_id": sample["id"],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_selected_sample(sample: dict[str, Any]) -> None:
    st.session_state.ticket_text = sample["text"]
    st.session_state.expected_route = sample["route"]
    st.session_state.expected_urgent = sample["urgent"]
    st.session_state.expected_refund = sample["refund_requested"]
    st.session_state.loaded_sample_id = sample["id"]
    st.session_state.pop("comparison", None)


def invoke(provider_name: str, text: str, run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        logger.info("provider_call_started run_id=%s provider=%s", run_id, provider_name)
        provider = get_provider(provider_name)
        prediction = provider.predict(text)
        total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "provider_call_completed run_id=%s provider=%s model=%s total_ms=%.2f",
            run_id,
            provider_name,
            prediction.model,
            total_ms,
        )
        return {
            "prediction": prediction,
            "operation_id": f"{run_id}:{provider_name}",
            "started_at_utc": started_at,
            "total_processing_ms": total_ms,
            "error": None,
        }
    except ProviderError as exc:
        logger.warning(
            "provider_call_failed run_id=%s provider=%s code=%s retryable=%s",
            run_id,
            provider_name,
            exc.code,
            exc.retryable,
        )
        return {
            "prediction": None,
            "operation_id": f"{run_id}:{provider_name}",
            "started_at_utc": started_at,
            "total_processing_ms": (time.perf_counter() - started) * 1000,
            "error": exc.to_dict(),
        }
    except Exception as exc:
        logger.exception("provider_call_unexpected_error run_id=%s provider=%s", run_id, provider_name)
        return {
            "prediction": None,
            "operation_id": f"{run_id}:{provider_name}",
            "started_at_utc": started_at,
            "total_processing_ms": (time.perf_counter() - started) * 1000,
            "error": {
                "provider": provider_name,
                "code": "unexpected_error",
                "message": "An unexpected internal error occurred. Use the operation ID when investigating logs.",
                "retryable": False,
                "details": {"exception_type": type(exc).__name__},
            },
        }


def clean_json(value: Any) -> Any:
    return sanitize(json.loads(json.dumps(value, default=str)))


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def displayed_binary_probability(yes_probability: float, result: bool) -> float:
    """Return probability for the displayed Yes/No result, not always raw P(Yes)."""
    return yes_probability if result else 1 - yes_probability


def binary_probability_rationale(yes_probability: float, result: bool, provider_rationale: str | None) -> str:
    shown_probability = displayed_binary_probability(yes_probability, result)
    raw_note = f"Raw yes probability: {yes_probability * 100:.0f}%; displayed {yes_no(result)} probability: {shown_probability * 100:.0f}%."
    if provider_rationale:
        return f"{raw_note} {provider_rationale}"
    return raw_note


def render_result(
    title: str,
    result: dict[str, Any],
    expected: dict[str, Any] | None,
) -> None:
    st.subheader(title)
    if result["error"]:
        error = result["error"]
        st.error(error["message"])
        with st.expander("Debug", expanded=True):
            st.json(
                {
                    "status": "error",
                    "operation_id": result["operation_id"],
                    "started_at_utc": result["started_at_utc"],
                    "error": error,
                    "total_processing_ms": round(result["total_processing_ms"], 2),
                }
            )
        return

    prediction: Prediction = result["prediction"]

    def pct(value: float) -> str:
        return f"{value * 100:.0f}%"

    urgent_confidence = abs(2 * prediction.urgent_probability - 1)
    refund_confidence = abs(2 * prediction.refund_probability - 1)
    urgent_display_probability = displayed_binary_probability(prediction.urgent_probability, prediction.urgent)
    refund_display_probability = displayed_binary_probability(prediction.refund_probability, prediction.refund_requested)
    field_rows = [
        {
            "Field": "Route",
            "Result": prediction.route.title(),
            "Probability": "-",
            "Confidence": pct(prediction.route_confidence),
            "Rationale": prediction.route_rationale or "-",
        },
        {
            "Field": "Urgency",
            "Result": yes_no(prediction.urgent),
            "Probability": pct(urgent_display_probability),
            "Confidence": pct(urgent_confidence),
            "Rationale": binary_probability_rationale(
                prediction.urgent_probability,
                prediction.urgent,
                prediction.urgent_rationale,
            ),
        },
        {
            "Field": "Refund requested",
            "Result": yes_no(prediction.refund_requested),
            "Probability": pct(refund_display_probability),
            "Confidence": pct(refund_confidence),
            "Rationale": binary_probability_rationale(
                prediction.refund_probability,
                prediction.refund_requested,
                prediction.refund_rationale,
            ),
        },
    ]
    st.dataframe(field_rows, hide_index=True, width="stretch")

    if expected:
        checks = {
            "route": prediction.route == expected["route"],
            "urgent": prediction.urgent == expected["urgent"],
            "refund": prediction.refund_requested == expected["refund_requested"],
        }
        correct = sum(checks.values())
        css_class = "result-ok" if correct == 3 else "result-bad"
        details = ", ".join(f"{name}: {'correct' if passed else 'wrong'}" for name, passed in checks.items())
        st.markdown(
            f'<span class="{css_class}">{correct}/3 correct</span>'
            f'<span class="muted"> - {details}</span>',
            unsafe_allow_html=True,
        )

    cost = estimated_cost(prediction)
    total_tokens = (
        prediction.input_tokens + (prediction.output_tokens or 0)
        if prediction.input_tokens is not None
        else None
    )
    local_overhead = max(0.0, result["total_processing_ms"] - prediction.latency_ms)
    debug = {
        "status": "ok",
        "operation_id": result["operation_id"],
        "started_at_utc": result["started_at_utc"],
        "provider": prediction.provider,
        "model": prediction.model,
        "tokens": {
            "input": prediction.input_tokens,
            "output": prediction.output_tokens,
            "total": total_tokens,
        },
        "estimated_cost_usd": cost,
        "timing_ms": {
            "api_round_trip": round(prediction.latency_ms, 2),
            "local_overhead": round(local_overhead, 2),
            "total_processing": round(result["total_processing_ms"], 2),
        },
        "classification": {
            "route": prediction.route,
            "route_confidence": prediction.route_confidence,
            "urgent_raw_yes_probability": prediction.urgent_probability,
            "urgent_displayed_result_probability": urgent_display_probability,
            "urgent_confidence": urgent_confidence,
            "refund_raw_yes_probability": prediction.refund_probability,
            "refund_displayed_result_probability": refund_display_probability,
            "refund_confidence": refund_confidence,
            "weakest_field_confidence": prediction.decision_confidence,
            "rationales": {
                "route": prediction.route_rationale,
                "urgent": prediction.urgent_rationale,
                "refund_requested": prediction.refund_rationale,
            },
        },
        "provider_diagnostics": prediction.diagnostics or {},
    }

    with st.expander("Debug"):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Input tokens", prediction.input_tokens or "n/a")
        metric_cols[1].metric("Output tokens", prediction.output_tokens or "n/a")
        metric_cols[2].metric("API latency", f"{prediction.latency_ms:.1f} ms")
        metric_cols[3].metric("Total time", f"{result['total_processing_ms']:.1f} ms")
        st.code("Cost unavailable" if cost is None else f"Estimated cost: ${cost:.8f}", language="text")
        debug_tab, raw_tab = st.tabs(["Normalized", "Raw provider response"])
        with debug_tab:
            st.json(debug)
        with raw_tab:
            st.json(clean_json(prediction.raw or {}))


samples = sample_cases()
initialize_state(samples)

with st.sidebar:
    st.header("Configuration")
    jev_model = st.text_input("Jev model", os.environ.get("JEV_MODEL", "jev-latest"))
    bedrock_model = st.text_input(
        "Claude Bedrock model / profile",
        os.environ.get(
            "BEDROCK_MODEL_ID",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
    )
    aws_region = st.text_input("AWS Region", os.environ.get("AWS_REGION", "us-east-1"))
    aws_profile = st.text_input("AWS profile (optional)", os.environ.get("AWS_PROFILE", ""))

    st.divider()
    st.subheader("Cost inputs")
    jev_input_price = st.number_input(
        "Jev input $ / 1M tokens",
        min_value=0.0,
        value=float(os.environ.get("JEV_INPUT_USD_PER_MILLION", "0.042")),
        format="%.6f",
    )
    bedrock_input_price = st.number_input(
        "Claude input $ / 1M tokens",
        min_value=0.0,
        value=float(os.environ.get("BEDROCK_INPUT_USD_PER_MILLION", "0")),
        format="%.6f",
    )
    bedrock_output_price = st.number_input(
        "Claude output $ / 1M tokens",
        min_value=0.0,
        value=float(os.environ.get("BEDROCK_OUTPUT_USD_PER_MILLION", "0")),
        format="%.6f",
    )
    st.divider()
    if os.environ.get("TYPESAFE_API_KEY") not in (None, "", "replace-me"):
        st.success("Jev API key found")
    else:
        st.warning("Set TYPESAFE_API_KEY in .env")
    aws_credential_hint = credential_hint()
    if aws_credential_hint["source"] == "incomplete_environment_access_keys":
        st.error("AWS access-key configuration is incomplete")
    elif aws_credential_hint["source"] == "environment_access_keys":
        st.success(f"AWS access keys found ({aws_credential_hint['access_key']})")
    elif aws_credential_hint["source"] == "aws_profile":
        st.info(f"AWS profile configured: {aws_credential_hint['profile']}")
    else:
        st.info("AWS will use the Boto3 default credential chain")
    with st.expander("System diagnostics"):
        diagnostics = system_diagnostics(ROOT)
        diagnostics["aws"]["region"] = aws_region
        diagnostics["aws"]["model"] = bedrock_model
        st.json(diagnostics)

st.title("Jev vs Claude Decision Comparator")
st.caption("One context, one classification contract, two model paths.")

sample_labels = {
    f"{sample['id']} - {sample.get('notes', sample['route'])}": sample for sample in samples
}
sample_col, load_col = st.columns([5, 1])
with sample_col:
    selected_label = st.selectbox("Sample ticket", list(sample_labels))
with load_col:
    st.write("")
    st.write("")
    if st.button("Load sample", width="stretch"):
        load_selected_sample(sample_labels[selected_label])
        st.rerun()

ticket_text = st.text_area("Shared context", key="ticket_text", height=140)

with st.expander("Shared classification questions and reference answer", expanded=True):
    question_rows = [
        {"Field": "route", "Question": QUESTION_TEXT["route"], "Answer shape": ", ".join(ROUTE_CRITERIA)},
        {"Field": "urgent", "Question": QUESTION_TEXT["urgent"], "Answer shape": "Yes probability (0-1)"},
        {
            "Field": "refund_requested",
            "Question": QUESTION_TEXT["refund_requested"],
            "Answer shape": "Yes probability (0-1)",
        },
    ]
    st.dataframe(question_rows, hide_index=True, width="stretch")
    use_reference = st.checkbox("Score against reference answer", value=True)
    answer_cols = st.columns(3)
    with answer_cols[0]:
        st.selectbox("Expected route", list(ROUTE_CRITERIA), key="expected_route")
    with answer_cols[1]:
        st.checkbox("Expected urgent", key="expected_urgent")
    with answer_cols[2]:
        st.checkbox("Expected refund request", key="expected_refund")

run_clicked = st.button("Run comparison", type="primary", width="stretch")

if run_clicked:
    if not ticket_text.strip():
        st.error("Enter a context before running the comparison.")
    else:
        os.environ["JEV_MODEL"] = jev_model.strip()
        os.environ["BEDROCK_MODEL_ID"] = bedrock_model.strip()
        os.environ["AWS_REGION"] = aws_region.strip()
        os.environ["JEV_INPUT_USD_PER_MILLION"] = str(jev_input_price)
        os.environ["BEDROCK_INPUT_USD_PER_MILLION"] = str(bedrock_input_price)
        os.environ["BEDROCK_OUTPUT_USD_PER_MILLION"] = str(bedrock_output_price)
        if aws_profile.strip():
            os.environ["AWS_PROFILE"] = aws_profile.strip()
        else:
            os.environ.pop("AWS_PROFILE", None)

        with st.spinner("Running Jev and Claude concurrently..."):
            run_id = uuid.uuid4().hex[:12]
            with ThreadPoolExecutor(max_workers=2) as executor:
                jev_future = executor.submit(invoke, "jev", ticket_text, run_id)
                claude_future = executor.submit(invoke, "bedrock", ticket_text, run_id)
                st.session_state.comparison = {
                    "run_id": run_id,
                    "context": ticket_text,
                    "jev": jev_future.result(),
                    "claude": claude_future.result(),
                }

comparison = st.session_state.get("comparison")
if comparison:
    if comparison["context"] != ticket_text:
        st.warning("The context changed after the last run. Run the comparison again to refresh results.")
    expected = None
    if use_reference:
        expected = {
            "route": st.session_state.expected_route,
            "urgent": st.session_state.expected_urgent,
            "refund_requested": st.session_state.expected_refund,
        }
    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        render_result("Jev", comparison["jev"], expected)
    with right:
        render_result("Claude on Amazon Bedrock", comparison["claude"], expected)
else:
    st.info("Load a sample or enter a ticket, then run the comparison.")

