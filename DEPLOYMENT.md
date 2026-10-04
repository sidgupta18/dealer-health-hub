# DealerHub release preparation

Design is frozen after the final semantic-status polish. No deployment has been performed.

## Exact GitHub upload contents

`DEPLOYMENT_FILES.txt` is the exact, reviewed upload allowlist. Preserve its relative paths; do not upload the entire working directory. It includes:

- `app.py`, `requirements.txt`, `.gitignore`, `README.md`, `METHODOLOGY.md`, this document and the manifest.
- `.streamlit/config.toml` and `.streamlit/secrets.toml.example` only.
- All `modules/*.py`, `config/*.json` and `assets/ui.css`.
- `data/sample_dealers.csv`, `data/sample_dealers_v2.csv`, `data/sample_field_assessments.json` (fictional synthetic data).
- `artifacts/results_v2.json`.
- The JSON and joblib pair `artifacts/overall_deterioration/5c7f68a32fb070dab1091d66`.
- The JSON and joblib pair `artifacts/inventory_warning/ce10d5a104565009ef2bce18`.
- The JSON and joblib pair `artifacts/971d3c8fabc68d7d8fb987ea` for the existing optional forecasting appendix.

Do not upload `.streamlit/secrets.toml`, any `.env`, `.venv`, SQLite databases, `data/action_backups/`, private backups, credentials or stale model artifacts. Tests and training scripts may be kept in a development repository but are not required for this hosted runtime. Do not omit or rename the matching model files: fingerprint-based loading deliberately rejects stale artifacts.

Use the contents of `deployment_upload/` as the new repository root. There is no `.git` repository in this workspace yet. Ignore rules and this allowlist are prepared, but tracked-file/history verification must happen when the GitHub repository is created. Ignoring a file does not remove a secret already committed; never drag the whole folder into GitHub.

## Community Cloud configuration

Choose repository root entry point **`app.py`** and **Python 3.12** in Advanced settings. `requirements.txt` pins the validated runtime, including the scikit-learn version used by saved artifacts. A fresh Linux dependency installation has not been verified locally.

Use Streamlit's private Secrets editor, not a committed TOML file. Setting names:

- `AI_LIVE_ENABLED` (disabled by default; enable only after live review and chat pass validation)
- `AI_API_KEY`
- `AI_BASE_URL`
- `AI_MODEL_ID`
- `AI_FREE_PLAN_CONFIRMED` (confirm actual provider entitlement before enabling live requests)
- `DEMO_RESET_ENABLED` (**false** on hosted/shared instances; omitting it also defaults to disabled)
- `AI_REQUEST_TOKEN_BUDGET` (optional; only lower the provider/model request budget)

The example contains a placeholder API key and a disabled reset setting. Local secrets are never part of the upload. Provider endpoint and model identifiers are configuration, not credentials.

Official instructions: [deploy and select Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [private cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Startup and persistence

Startup uses project-relative paths, deterministic synthetic inputs and saved overall-health artifacts. It does not train models. Optional inventory/research sections load matching artifacts on demand; explicit training controls remain existing features. Torch is optional research tooling and is not required for core startup or the saved selected tabular prediction. GRU-only experimental inference may require the optional research environment.

Approved actions and status history use `data/actions.sqlite3`, created on the hosted filesystem. **Hosted records are temporary and may reset after restart or redeployment.** Community Cloud is not a durable action-record backend. Do not treat this demo as the system of record. Existing human approval and readiness checks remain active. Reset controls must remain disabled on the hosted demo.

## Release checks and remaining limits

The colour layer does not change thresholds, sorting, urgency, forecasts, prerequisites or approval. Current Critical dealers stay urgent even when no deterioration is flagged. Forecasts remain experimental and disclose evaluation limitations.

Saved overall, inventory and optional tabular artifacts load with the bundled synthetic data. Entry point compiles. Functional tests exercise navigation, selection, approval, status updates and isolated empty/populated tracker states. Browser smoke checks use demo briefs and never save real actions.

Live hosted AI review and follow-up chat are not confirmed end to end: prior provider responses failed grounding/validation checks. Treat live AI as an outstanding release verification item; deterministic demo remains clearly labelled. No live API request is required by this polish pass.

Final verification: 64 automated tests passed, including semantic contrast (at least 4.5:1 for every badge text/background pair), HTML escaping, unchanged table values and urgent Critical/no-flag priority. Chrome rendered all five pages at normal zoom at 1440×900 and 1024×768 without page-level horizontal overflow. Inspected Portfolio, Diagnosis, AI Review and the empty tracker screenshots; badges and signed changes remained readable. Populated tracker approval/status changes were verified with isolated test databases, not real saved records. Linux hosting and live-AI success remain unverified.

The deployment package defaults to deterministic **AI Demo** and disables live chat gracefully, even when credentials are configured, until `AI_LIVE_ENABLED` is explicitly enabled after successful verification. No API key is needed to use the demo, forecasts or approval workflow.


## Clean upload package verification

`deployment_upload/` contains exactly 45 allowlisted files (about 1.73 MiB), preserving relative paths. Its credential scan checked known local credential bytes and common key/private-key patterns without printing values. Real secrets, environment files, databases, action/recording history, backups and caches are absent. All three provider configuration strings in the example are placeholders; live AI and hosted reset default to disabled.

The isolated package passed smoke checks with no secrets and no network calls: startup, matching dependency versions, all three saved model loaders, selected-dealer/month navigation, deterministic AI Demo generation, disabled chat including stale pending questions, fresh SQLite schema initialization, explicit approval and real status update, and Data & Methodology. Predictive training entry points were blocked during the smoke. Disposable package test records were removed afterward. The full functional suite passed 64 tests, and the targeted AI suite passed 14 checks after the disabled-request guards were added.

Linux/Python 3.12 wheel verification remains incomplete: the first PyPI attempt timed out; a retry stalled during a wheel download and was stopped. No dependency-version conflict was reported before that point. Local installed versions match the pinned requirements, but a fresh Linux installation and actual Community Cloud startup are still release-verification items.
