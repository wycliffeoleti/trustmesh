# Email classifier evaluation — local walkthrough script

**Status.** This is a written script for a local, unpublished demo. No Loom or
other recording has been made from it, no blog post exists, and nobody has been
contacted about it. Wicky may use it later to record a demo entirely at his own
discretion; recording it is not part of this task.

**Purpose.** Show, with exact commands, how TrustMesh's offline email-classifier
evaluator (see the ["Email classifier evaluation (offline, staged draft)"
section](../README.md#email-classifier-evaluation-offline-staged-draft) of the
README) detects a prompt/config regression before it ships — without a paid
model call, a network request, or a message to Slack or any other external
system. Every command below was run against this repository and its output is
reproduced verbatim.

## Prerequisites

Run once, from the repository root: `uv sync --group dev`.

## Step 1 — run the evaluator against the checked-in baseline

```bash
uv run python evals/run_email_classifier_eval.py
```

This loads the checked-in `prompts/email_classifier/v1.yaml` prompt config and
the 16-case `evals/email_classifier_staged/v1.json` staged dataset, runs the
offline evaluation engine, and compares the result to the checked-in baseline
run (`evals/baselines/email_classifier_staged_v1.json`). Observed output:

```json
{
  "report_id": "v1__1",
  "dataset_status": "staged_draft",
  "headline_status": "pass",
  "previous_pass_rate": 0.625,
  "current_pass_rate": 0.625,
  "overall_delta_percentage_points": 0.0,
  "regressed_case_count": 0,
  "improved_case_count": 0
}
```

Exit code is `0`. The current pass rate (0.625, i.e. 10/16) is unchanged from
the baseline — the checked-in prompt config was not degraded, and this staged
dataset is a small, AI-authored draft, not a golden or human-reviewed set (see
the README's provenance warning), so the pass rate itself is not a quality
claim.

## Step 2 — inspect the JSON and HTML report

The same command also writes `report.json`, `report.html`, and `alert.json` to
`eval-artifacts/email_classifier/` (this directory is git-ignored and never
committed). Open `report.json` in an editor or:

```bash
cat eval-artifacts/email_classifier/report.json
```

and open `eval-artifacts/email_classifier/report.html` in a local browser
(`file://` URL — no server, no network). Both show run metadata (prompt
version, dataset schema version, timestamps), the baseline scorecard, per-case
regressions (empty at this point), and the same thresholds used in step 4.

## Step 3 — deliberately degrade a *temporary copy* of the prompt config

Never edit the checked-in `prompts/email_classifier/v1.yaml` for this demo.
Copy it out of the repository first:

```bash
DEMO_DIR=$(mktemp -d)
cp prompts/email_classifier/v1.yaml "$DEMO_DIR/v1-demo.yaml"
```

Then remove one fallback keyword — `refund` — from the `billing` category in
the copy only:

```bash
sed -i 's/billing: \["invoice", "charge", "charged", "payment", "refund", "billing", "subscription"\]/billing: ["invoice", "charge", "charged", "payment", "billing", "subscription"]/' \
  "$DEMO_DIR/v1-demo.yaml"
```

`git status` should show no repository changes at this point, since the edit
only touched the temporary copy in `$DEMO_DIR`.

## Step 4 — rerun the evaluator against the temporary copy

```bash
uv run python evals/run_email_classifier_eval.py \
  --prompt-path "$DEMO_DIR/v1-demo.yaml" \
  --output-dir "$DEMO_DIR/report-out"
```

The dataset and baseline are unchanged; only the prompt config path changed.
Observed output:

```json
{
  "report_id": "v1__1",
  "dataset_status": "staged_draft",
  "headline_status": "warn",
  "previous_pass_rate": 0.625,
  "current_pass_rate": 0.5625,
  "overall_delta_percentage_points": -6.25,
  "regressed_case_count": 1,
  "improved_case_count": 0
}
```

`report.json` under `$DEMO_DIR/report-out/` names the exact regressed case:

```json
{
  "case_id": "billing-002",
  "expected_category": "billing",
  "previous_category": "billing",
  "current_category": "general",
  "current_summary": "Not sure if this is a refund thing or my plan renewing wrong, but something about what I paid this month looks off."
}
```

Removing one keyword flipped case `billing-002` from a correct `billing` match
to the `general` default, moving the overall pass rate down by 6.25
percentage points — enough to cross the 3-point warning threshold but not the
8-point critical one, so `headline_status` is `warn` and the process still
exits `0` (see `trustmesh/email_classifier_eval_cli.py`, which exits nonzero
only on `critical`). A larger or differently placed keyword change could push
this past the critical threshold instead; that is a further exercise, not
required for this walkthrough.

## Step 5 — the alert payload is in-memory and offline, not delivered anywhere

The JSON printed to stdout in steps 1 and 4, and the `alert.json` file written
alongside the report, are produced by
`trustmesh.email_classifier_eval_alert.build_email_classifier_eval_alert_payload`.
That module's own docstring states it directly: *"there is no webhook client,
no environment variable read, no network request, and no Slack (or any other
transport) dependency here."* This is enforced, not just documented — see
`tests/test_email_classifier_eval_reporting_offline_guardrails.py`, which
parses the module's AST and fails if it imports `requests`, `httpx`,
`socket`, `smtplib`, `urllib`, or `slack_sdk`, or reads `os.environ`/`os.getenv`.
Nothing produced by this evaluator is sent to Slack, any other chat tool, or
any external service; the payload only exists on local disk and in the
terminal that ran the command.

## Step 6 — clean up

```bash
rm -rf "$DEMO_DIR"
rm -rf eval-artifacts
```

`eval-artifacts/` is already git-ignored, so it was never at risk of being
committed, but removing it locally keeps the working tree clean for the next
run. `git status --short` should be empty of anything produced by this
walkthrough.

## Evidence / claim boundary

- Everything above ran locally, offline, deterministically, and without a
  model/provider call. Nothing in this walkthrough was executed in CI,
  recorded, published, or sent to any third party or external system.
- The classifier under test is a deterministic test double (exact-example and
  keyword-fallback matching), not a real LLM call — see the module docstring
  in `trustmesh/email_classifier.py`. This walkthrough demonstrates the
  regression-detection *mechanism*, not real LLM output-quality evaluation.
- The dataset is a 16-case, AI-authored **staged draft** (`evals/email_classifier_staged/v1.json`),
  not human-reviewed, not golden, and not at BASWE's suggested 50–100-case
  scale. Pass-rate numbers describe this small draft set only.
- The alert payload is in-memory/offline only, as shown in step 5. No Slack
  webhook, email, or other notification integration exists in this
  repository.
- CI (`.github/workflows/email-classifier-eval.yml`) runs this same evaluator
  on pushes/PRs that touch the relevant paths and uploads the report as a
  build artifact; it fails the job only on a `critical` result. It does not
  post PR comments, and no branch-protection/merge-blocking configuration is
  claimed or verified by this repository's files.
- No Loom, video, blog post, or public claim has been produced from this
  script. If Wicky records a demo from it later, that recording and any
  publication decision is his own, separate from this documentation.
