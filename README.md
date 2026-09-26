# Jev evaluation POC

This is a local, measurable comparison of Jev, an Amazon Bedrock model, and a small deterministic rules baseline. It answers a practical question:

> For bounded decisions inside an application, does Jev provide enough improvement in speed, cost, calibrated confidence, and operational simplicity to justify another model service?

The POC uses support-ticket triage because it has clear labels and exercises Jev's three core ideas: one shared state, several atomic questions, and typed answers. It does **not** ask either model to write a response.

## Run the interactive UI

Install [uv](https://docs.astral.sh/uv/) once, then create `.env` from the supplied example:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `TYPESAFE_API_KEY`. Configure AWS credentials through your normal profile, SSO, or environment-variable process. Then launch the application with one command:

```powershell
python main.py
```

`main.py` uses uv to install and synchronize dependencies when necessary, starts Streamlit, and opens `http://localhost:8501`. The page provides a shared context and reference answer, Jev on the left, Claude through Amazon Bedrock on the right, and a separate debug panel under each result.

## What you will learn

After running the benchmark, you will have evidence for:

- Classification quality: route, urgency, refund intent, and all-three exact match.
- Latency: mean and p95 response time measured from your laptop.
- Structured-output reliability: call and parsing failures.
- Confidence usefulness: accuracy and coverage above an automation threshold.
- Probability calibration: Brier scores for urgency and refund intent.
- Repeatability: agreement across repeated calls on identical input.
- Estimated request cost from observed token usage.

The included 32-ticket dataset is a harness and smoke benchmark. It is too small for a purchase decision. After the harness works, replace it with 100-300 anonymized tickets labeled by people who understand your process.

## Architecture

```text
Labeled JSONL tickets
        |
        v
Local Python benchmark
   |         |          |
   v         v          v
 Rules      Jev       Bedrock Converse
baseline  System One   + forced tool schema
   |         |          |
   +---------+----------+
             |
             v
  normalized predictions
             |
             v
 accuracy, Brier score, latency,
 cost, failures, coverage, stability
             |
             v
 results/latest/report.md + results.json
```

The exact business action remains code. A production policy might say: auto-route only when confidence is at least 0.80; otherwise send the ticket to a human queue. This is the right boundary for both Jev and Bedrock: model judgment supplies signals, while code owns policy.

## Prerequisites

- Python 3.10 or newer.
- A TypeSafe AI account and API key from the [TypeSafe console](https://console.typesafe.ai/).
- AWS credentials with `bedrock:InvokeModel` permission.
- Claude Haiku 4.5 available through Amazon Bedrock in your selected Region. The default US inference profile is `us.anthropic.claude-haiku-4-5-20251001-v1:0`.
- Optional but useful: AWS CLI v2 for verifying the active identity.

Jev's current quickstart documents `POST https://api.typesafe.ai/v1/systemone`, the `Choice`, `Noul`, and `Score` primitives, and the Python SDK. This POC calls the HTTP API directly so the raw contract is visible. See the [Jev quickstart](https://docs.typesafe.ai/introduction/quickstart), [primitives](https://docs.typesafe.ai/primitives), and [current model table](https://docs.typesafe.ai/models).

The Bedrock adapter uses the Converse API and forces a named tool call with a JSON schema. This gives Claude a fair structured-output path instead of relying on fragile JSON-in-prose prompting. See [AWS Converse](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api-ex-python.html) and the [Claude Haiku 4.5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html).

## 1. Set up the project

From PowerShell in this directory:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

Expected result: four tests pass. These tests require no network access or credentials.

## 2. Prove the harness offline

```powershell
jev-poc smoke --provider rules
jev-poc evaluate --providers rules --dataset data/tickets.jsonl --output results/rules
```

Open `results/rules/report.md`. The rules baseline should be fast but imperfect. It is intentionally simple: the goal is to prove that an AI service beats a cheap, understandable alternative.

## 3. Configure and smoke-test Jev

Create the local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `TYPESAFE_API_KEY`. Keep `JEV_MODEL=jev-1.13.0` pinned during the evaluation. An alias such as `jev-latest` can move to a newer model and make before/after results incomparable.

Run one call:

```powershell
jev-poc smoke --provider jev --text "Production checkout is down and customers cannot pay. Fix this immediately."
```

Check that the result contains:

- `route: technical`
- `urgent: true`
- `refund_requested: false`
- the resolved model version, token usage, latency, and confidence values

If it fails with `401`, verify the API key. If it fails with `429`, retry later or reduce call frequency. Do not put the key into the dataset, command line, or source control.

## 4. Configure and smoke-test Bedrock

Authenticate using whichever mechanism your company supports: AWS SSO, a named profile, or temporary environment credentials. For SSO, the usual flow is:

```powershell
aws configure sso
aws sso login --profile YOUR_PROFILE
$env:AWS_PROFILE = "YOUR_PROFILE"
aws sts get-caller-identity
```

Edit `.env` and set:

```text
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Make sure the selected model is available to the account in that Region. The calling identity needs `bedrock:InvokeModel`; scope the IAM resource to the selected model or inference profile under your organization's policy.

Run one call:

```powershell
jev-poc smoke --provider bedrock --text "Production checkout is down and customers cannot pay. Fix this immediately."
```

If you receive `AccessDeniedException`, check the active identity, Region, model availability, and IAM permission. If your organization uses an inference profile, put its ID or ARN in `BEDROCK_MODEL_ID`.

## 5. Run the first comparison

Start with one pass to catch configuration problems and limit unnecessary calls:

```powershell
jev-poc evaluate `
  --providers rules,jev,bedrock `
  --dataset data/tickets.jsonl `
  --threshold 0.80 `
  --repeats 1 `
  --output results/first-run
```

Then run three repeats to measure consistency:

```powershell
jev-poc evaluate `
  --providers rules,jev,bedrock `
  --dataset data/tickets.jsonl `
  --threshold 0.80 `
  --repeats 3 `
  --output results/repeated
```

Each remote provider receives 96 calls in the repeated run: 32 cases times 3. Jev evaluates all three questions inside each call. The benchmark saves:

- `report.md`: the comparison table.
- `results.json`: every normalized prediction, error, token count, confidence, and expected label.

Raw provider responses are intentionally not saved because they can be large and may contain metadata you do not want in an evaluation artifact.

## 6. Add cost figures

Jev's model page currently lists Jev 1.13 at `$0.042` per million input tokens with output tokens free; the code applies that value to observed usage. Verify the current [TypeSafe model pricing](https://docs.typesafe.ai/models) before presenting results.

Bedrock pricing depends on model and Region. Look up the exact prices for your selection and add them to `.env`:

```text
BEDROCK_INPUT_USD_PER_MILLION=YOUR_INPUT_PRICE
BEDROCK_OUTPUT_USD_PER_MILLION=YOUR_OUTPUT_PRICE
```

Re-run the benchmark. If both values remain zero, Bedrock cost is reported as `n/a` rather than pretending it is free.

## 7. Read the results correctly

Use these measures in this order:

1. **Failures**: any schema, parsing, authentication, throttling, or service failure. Read its exact error in `results.json`.
2. **Exact match**: all three decisions must be correct for a case. This approximates whether the full downstream action would be safe.
3. **Automated accuracy and coverage**: accuracy among cases at or above the confidence threshold, and the fraction of work that threshold covers.
4. **Brier score**: probability error for binary decisions; lower is better, zero is perfect. Accuracy can be high while confidence is poorly calibrated.
5. **P95 latency**: the slow-end user experience. Compare from the same laptop and network, in alternating runs, at similar times.
6. **Repeat agreement**: whether the final route/urgent/refund decisions remain identical over three runs.
7. **Cost**: use the measured token totals and current contracted prices, not marketing examples.

A reasonable initial gate for a larger pilot is:

- Zero parsing/schema failures in the controlled run.
- At least 95% exact accuracy within the auto-action subset.
- At least 50% automation coverage at that quality level.
- Urgency and refund Brier scores below 0.15.
- Repeat agreement above 98%.
- A material latency or cost improvement over the Bedrock model, if those are the claimed differentiators.

Do not treat these numbers as universal. If a wrong refund action has financial impact, use a stricter per-question threshold and require deterministic validation before execution.

## 8. Replace the toy data with your data

Create a new JSONL file using the same fields:

```json
{"id":"CASE-001","text":"An anonymized ticket","route":"billing","urgent":false,"refund_requested":true}
```

For a credible result:

1. Sample 100-300 recent tickets across common, rare, and adversarial cases.
2. Remove names, email addresses, account numbers, secrets, and regulated data unless both vendors are approved for that data class.
3. Write label definitions before labeling.
4. Have two people label independently without seeing model answers.
5. Resolve disagreements through an adjudicator; keep the resolved label as ground truth.
6. Keep 20-30% as a final test set. Tune questions and thresholds only on the calibration set.
7. Freeze the prompts, model IDs, dataset hash, Region, date, and threshold before the final run.

Useful edge cases include negation ("not urgent"), historical mentions ("the refund arrived"), competing intents, vague wording, sarcasm, quoted email threads, very long tickets, and deliberate prompt-injection text inside a ticket.

## 9. Run threshold sensitivity

Run the same predictions at thresholds such as 0.60, 0.70, 0.80, 0.90, and 0.95. The current CLI calls providers each time, so for a small run use separate commands. For a larger evaluation, compute thresholds from one saved `results.json` to avoid paying for repeated inference.

The useful decision chart is **coverage versus automated accuracy**. A service is valuable when it can confidently automate a meaningful share of cases while routing uncertain cases to people. A single overall accuracy number hides this.

## 10. What this POC can and cannot prove

It can test Jev's claims for one bounded decision workflow:

- Typed outputs and probability-bearing judgments.
- Batched independent questions over one state.
- Latency, token cost, consistency, and confidence-gated automation.

It cannot show that Jev replaces a general LLM. Jev does not generate customer replies, perform deep multi-step reasoning, calculate exact values, or execute business actions. If the workflow needs a response, a sensible production composition is:

```text
ticket -> Jev decision signals -> code policy -> Bedrock drafts response -> human/API action
```

The key architectural question is therefore not "Jev or Bedrock?" It is "which judgments deserve a specialized decision model, and which steps still require deterministic code or a generative model?"
