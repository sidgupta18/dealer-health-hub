# Dealer Health & Action Hub
Python + Streamlit + SQLite decision dashboard. Current holistic health, scoped operational readiness and feasible interventions stay separate. The final manager workflow requires an overall sustained-deterioration forecast. Its event definition is missing, so the three-month outlook is explicitly unavailable. Inventory and payment forecasts remain preserved technical experiments. No Node, React, Docker or source compilation.

## Launch
From `/Users/siddhant/dealer-health-hub`:

```sh
.venv/bin/python -m streamlit run app.py
```

Core startup does **not load predictive artifacts, train models or require PyTorch**. Portfolio and Diagnosis contain no future payment probabilities. Operational AI evidence uses current health and scoped readiness; superseded inventory/payment forecasts are excluded. Index comparisons and sensitivity remain available. Preserved experimental results are loaded only after selecting **Data & Methodology → Exploratory forecasting → Load experimental forecasting results**; the payment target requires business validation.

## Exploratory forecasting — secondary technical appendix

```sh
.venv/bin/python scripts/train_models.py
.venv/bin/python scripts/train_models.py --data path/to/dealers.csv
```

Alternatively use **Data & Methodology → Exploratory forecasting → Load experimental forecasting results → Training controls & evaluation protocol → Train / reuse matching predictive artifact**. Artifacts are keyed by data, score/configuration and implementation fingerprints. Existing matches are reused. `--without-gru` permits training with no deep-learning package. `--force` explicitly retrains a matching artifact and reruns the held-out experiment; do not use it to tune against test results. If adding GRU to a previously cached non-GRU experiment, explicitly retrain with `--force`.

Restart the app after editing the central scoring/readiness policy files. Core dependencies are in `requirements.txt` (scikit-learn is needed for health-index PCA). Experimental training dependencies are in `requirements-experiments.txt`; optional GRU dependencies are in `requirements-gru.txt`. Neither payment predictive engine nor deep-learning package is imported by the core workflow. The verified prebuilt install command is:

```sh
.venv/bin/python -m pip install --only-binary=:all: -r requirements-gru.txt
```

PyTorch 2.14.1's `cp312-macosx_14_0_arm64` wheel ran successfully in this project's Python 3.12 Apple Silicon environment. Only the project `.venv` was updated. See the [official binary installation guidance](https://pytorch.org/get-started/locally/). No developer tools or base Conda changes. Missing torch does not prevent scoring, reviews or actions; the core never loads a predictive artifact; explicitly opted-in logistic experiments can run without torch.

## Analytical approach
- Business pillars remain Commercial 25%, Financial 30%, Inventory 25%, Customer/service 20%.
- Compare equal pillars and experimental **PCA-derived variance weights** using a frozen first-12-month historical reference. Show indicator/implied pillar weights, score/rank/class disagreements, and correlation flags. Do not auto-promote PCA.
- Reproducible sensitivity: 100 relative pillar-weight perturbations ±20%, and 100 independent anchor perturbations ±5% of original spans. Report raw ranges, raw/final stability and rank movement; these are not confidence intervals.
- Persistence, logistic, forest and a 24-unit GRU share six consecutive calendar months of complete inputs. No dealer IDs, future predictors, missing-month bridging or cross-dealer sequences.
- Three expanding validation folds; chronological inner GRU early stopping; train-only scalers. Select by pooled validation average precision. Select each alert threshold on validation F1. Hold out latest four eligible months for all models on identical sequences.
- Current v2 test: May–August 2026, 238 sequences / 12 events. Logistic test AP 0.427, ROC-AUC 0.856, precision 0.600, recall 0.500, P@10/month 0.225, Brier 0.0366. GRU test AP is 0.439 but it loses validation AP (0.564 vs logistic 0.660), so logistic remains selected. No calibrated-probability or production-validity claim.

Detailed formulas, measurement definition, prediction/label dates, splits and all actual model results are in [METHODOLOGY.md](METHODOLOGY.md). Machine-readable results are in `artifacts/results_v2.json`; model objects remain local and ignored by Git. Earlier fingerprinted artifacts are inactive historical runs; the explicitly opted-in appendix loads only the exact current match.

## Dataset, CSV and saved actions
Default synthetic demo is separately versioned `data/sample_dealers_v2.csv`, with persistence, seasonality, decaying random shocks and noisy delay outcomes. The original `data/sample_dealers.csv` and v1 generator remain preserved. No generator change was made to make the GRU win.

Upload contract uses the sample's columns. Dates normalize to month starts; duplicate keys, invalid numeric types/ranges, fractional unit counts and inconsistent dealer metadata are rejected. Missing scoring values yield Unassessed; incomplete six-month prediction windows are excluded. Invalid uploads block analysis rather than substituting sample data. Uploads stay in session.

Actions persist in `data/actions.sqlite3`. Manager approval is required for saving; owner, deadline, priority, status history and evidence snapshots are retained. Analytical exploration and training never rewrite original action evidence. Existing actions keep their original dataset/configuration/model context.

## Hosted Llama API
Configure `AI_API_KEY`, `AI_BASE_URL` (provider's versioned OpenAI-compatible root, excluding `/chat/completions`) and `AI_MODEL_ID` through environment variables or `.streamlit/secrets.toml`. `.env.example` contains placeholders; .env is not loaded automatically. Secrets are ignored. No provider/model ID is assumed. HTTP uses standard-library urllib. Without credentials, the labelled **AI Demo — no AI model connected** supports evidence and editable proposals; hosted follow-up requires credentials. Live provider verification remains pending configuration.

## Verification and demo

```sh
.venv/bin/python -m unittest discover -s tests -v
```

29 automated tests pass. Checks cover scoring/overrides, missing inputs, PCA reference-only fitting/formula/constants, sensitivity reproducibility/direction, correct dealer/calendar sequences and labels, temporal boundaries, selection invariance to changed future test inputs, optional GRU reproducibility, immutable SQLite evidence, API failures, and navigation/explicit approval. Chrome inspections use 1440×900 and 1280×800, including populated and isolated empty states. The Apple-inspired desktop layout uses a system font stack, neutral cards, native callback navigation and a small stylesheet scoped to stable elements and explicit container keys. Checks include overflow, keyboard focus, dealer/month preservation, explicit approval and real status updates in an isolated database.

## Operational readiness and demo sequence
Field evidence uses the append-only `field_assessments` SQLite table in `data/actions.sqlite3`. Assessments capture exact model/campaign/service scope, one of five checks, Ready / Blocked / Not assessed, a concrete observation, evidence reference, assessor, assessment date and review/expiry date. Revision explicitly supersedes an existing version; old observations are retained. Independent contradictory current assessments require verification. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

Rules are in `config/readiness.json`. Evidence expires at its review date or the 30-day maximum age, whichever is earlier. Blank scope never verifies a different scope. Demand-growth and stock interventions require a named model/campaign/service scope even when all unscoped checks are Ready. A historical reporting view excludes evidence assessed **or recorded** after month end; later records remain visible with a clear label. The optional current-evidence control explicitly labels later evidence. Approval rechecks current evidence inside the same SQLite write transaction, and saves both the original review bundle and approval-time readiness.

Lead growth / model campaigns require a usable scoped demo vehicle and trained sales staff; service growth requires scoped service capacity; stock changes require model/variant demand fit; site-dependent actions require access/facility evidence. Blocked prerequisites prevent approval. Unknown/stale/conflicting prerequisites require a separate verification action. Separate corrective and verification actions remain approvable. Edited action types and common demand-growth instructions are checked in deterministic code; managers must classify the intervention and site dependence correctly. Free-form intent recognition is conservative, not a general natural-language proof system.

Synthetic field scenarios live separately in `data/sample_field_assessments.json`; they are never written over user-entered assessments and never claim actual site visits. They apply only to the sample KPI dataset, not imported CSVs. Fixed September 2026 scenarios intentionally expire over time:

1. Portfolio → Aarav Motors / D001 → Atlas SUV: strong observed sales, weak financial indicators, verified demand fit but unavailable demo vehicle. Generate the deterministic review; approve **Arrange a usable demo vehicle**, not a lead-growth intervention. No causal conversion claim is made.
2. D002 / Service workshop: fictional service opportunity, blocked equipment/technician/essential-parts capacity. Recommend capacity verification/remediation before service demand growth.
3. D003 / Atlas SUV: all five checks verified for this scope. D004 / Atlas SUV: stale July evidence. D005 or another scope: missing evidence.
4. Record a field assessment; use **Assessment version** to create an explicit revision. For September reporting, a new October record is later evidence; opt into current field evidence to inspect it without retroactively changing September conclusions.
5. Edit, assign and explicitly approve a feasible action; update its real tracker status. Existing saved approvals retain their original snapshots, including historical prediction fields.
6. Explore Index Methods, sensitivity and the opt-in forecasting appendix. PCA remains exploratory; forecast target meaning requires business validation.

Financial definitions: the supplied data dictionary does not establish whose receivables are overdue, the denominator, which payment obligations are delayed, or the exact margin accounting basis. These remain observed illustrative KPIs in the unchanged Financial pillar; confirm dealer-customer versus dealer-OEM/lender counterparties and accounting definitions before operational interpretation. Do not infer cash-flow distress or insolvency.


Calendar cutoffs use the configurable business timezone in `config/readiness.json` (Asia/Kolkata for this workspace). Entry timestamps stay UTC for audit and are converted to that business calendar before historical applicability checks. Assessment dates, current-readiness dates and freshness use the same calendar. This excludes newly entered backdated evidence even across UTC/local midnight.

## Inventory early warning — historical extension, now outside the core workflow

Train explicitly: `.venv/bin/python scripts/train_inventory_warning.py` (optional `--data path/to/dealers.csv`). The matching artifact is reused; `--force` retrains. UI control: Data & Methodology → Inventory warning. Launch: `.venv/bin/python -m streamlit run app.py`. Restart after code/config changes. No generator, health weights, payment experiments or saved evidence were changed.

The provisional demo event is aged-stock share strictly above 30% in any of the next three calendar months, requiring all three outcomes. Only currently below-threshold dealer-months enter the forecast population. Current breaches are observed review needs, separate from future flags. Six consecutive months of observed inventory, sales achievement/growth, margin and lead conversion provide predictors; no IDs, local-market assumptions or field evidence enter the model. Lead conversion is an observed sales-funnel indicator, not a causal explanation.

Two usable expanding validation folds cover August–November 2025 (203 samples, 27 events, 13.3% prevalence); an earlier fold lacked support. Each training label is fully known before its fold freeze. Final training has 457 samples/69 events through November 2025, outcomes known by February 28, 2026. Untouched test predictions cover March–June 2026, with 206 samples/37 events (18.0% prevalence), outcomes through September. Minimum support is 60 samples and two training months; first fold has only 104 samples/16 events, a material limitation.

Select pooled-validation average precision (AP); challengers need +0.02 absolute AP over the trend baseline. Each threshold maximizes validation F2, weighting recall four times precision; ties choose the higher threshold. Test results never select models or thresholds.

| Model | Validation AP | Test AP | Test ROC-AUC | Precision | Recall | P@10 flagged/month | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trend baseline | .449 | .412 | .837 | .458 | .730 | .450 | .1182 | .25 |
| Regularized logistic | .847 | .817 | .943 | .660 | .838 | .725 | .0644 | .25 |
| Regularized forest (selected) | .866 | .858 | .960 | .857 | .649 | .856 | .0615 | .50 |

P@10 averages precision among at most 10 **eligible flagged** dealers each month; months without flags are omitted and disclosed (zero in this run). Reliability uses five held-out score bins and sample counts. Scores are not established as calibrated probabilities. Synthetic generator patterns, overlapping horizons and repeated dealers reduce independence; this tests future periods for existing dealers, not new dealers or production data. The trend baseline maps extrapolated distance to a heuristic score using training residual scale.

Artifacts in `artifacts/inventory_warning/` include data, configuration and implementation fingerprints. A dataset-wide provenance hash is not a predictor. Missing/mismatched artifacts show unavailable; missing calendar history shows insufficient history; historical months before the final training freeze are outside the forecast window. Ordinary navigation never trains. New approved action snapshots contain model version, data hash, prediction date and horizon; earlier records remain immutable.

Demo: open Portfolio → inspect the separate inventory review reason → Review dealer → compare current health/readiness with observed inventory and the Early Warning panel → Prepare review → generate a deterministic review → verify evidence and prerequisites before manager approval. No saved actions are fabricated. Inventory flags support investigation, not automatic promotions, stock transfers or purchasing restrictions. Live AI credentials are optional; arbitrary prose is not semantically proven by numeric/evidence-ID validation.

Verification: 35 automated tests pass, including exact calendar labels, strict threshold boundaries, missing horizons, train-only preprocessing, purged freezes, held-out outcome perturbation, fingerprint invalidation and AI/approval snapshots. Chrome inspected all five pages at 1440×900 and 1280×800, including inventory evaluation and generated demo review; no page-level horizontal overflow. Wide analytical tables scroll internally. Original action evidence was compared byte-for-byte and preserved, along with subsequent manager-approved records. Live AI is unverified without credentials. Current unchanged synthetic data yields five September inventory flags, all already At Risk or Critical; there is no Healthy/Watchlist flagged demonstration in this fitted run. Do not imply otherwise or change the generator to create one.

## Hosted dealer review and manager chat

The AI Review page now has explicit hosted generation and a separate **AI Demo** control. Demo content is labelled **AI Demo—no live AI response**. Live errors never silently produce a demo. Reviews and chat share one bounded selected-dealer evidence bundle: observed KPIs/changes, rule scores/overrides, gaps, scoped field evidence and the available inventory forecast. The saved overall-health deterioration model provides a separately labelled experimental three-month outlook. Superseded payment forecasts are excluded. Review/chat cannot approve or save actions. The manager supplies the owner/deadline, reviews prerequisites and explicitly approves; readiness is rechecked inside the saving transaction.

Local setup:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. In that **local file**, replace `YOUR_API_KEY` with your Groq key. Keep `AI_BASE_URL = "https://api.groq.com/openai/v1"` and configure `AI_MODEL_ID` explicitly; the example uses `openai/gpt-oss-20b`.
3. Verify your own account and that exact model have free-plan entitlement. Only then set `AI_FREE_PLAN_CONFIRMED = true`. Llama requires confirmation for that particular model. Model-list access alone is insufficient. The app never purchases credits, changes billing, upgrades plans or substitutes models/providers.
4. Restart: `.venv/bin/python -m streamlit run app.py`.
5. Open AI Review and click Generate dealer review. After generation, submit a starter question or chat message. Calls occur only on those explicit actions.

Equivalent environment variables are supported, including `AI_FREE_PLAN_CONFIRMED=true`. Environment variables take precedence; `.env` is **not** loaded automatically. Keys stay server-side, never appear in UI/logs or evidence. The real secrets file is ignored; distribute only the placeholder example, exclude `.env` and real secrets from any source ZIP.

For Streamlit Community Cloud, place the same TOML values in the app's **Settings → Secrets**, not in the repository. This documents setup only; no deployment was performed.

Groq's [official free-plan limits](https://console.groq.com/docs/rate-limits) list GPT-OSS 20B/120B, but limits apply to the organization and can change. Confirm entitlement in your account. The [OpenAI-compatible endpoint](https://console.groq.com/docs/openai) is used with model/messages/max_completion_tokens, without assuming temperature, tools or strict-schema support. Responses must report the exact configured model. API failures distinguish missing configuration, entitlement, authentication, unavailable model/parameters, rate limit and timeout. Requests time out after 25 seconds, have no automatic network retries and permit at most one JSON/schema repair. Retry-after is respected via a session cooldown without sleeping; normal requests have a five-second cooldown. Output is capped at 1,800 tokens and 16 KB of structured content. Conversation is limited to six exchanges, context sends the latest six messages, and questions to 600 characters.

Validated reviews are cached within the user session by dealer/month/evidence/provider/model. Ordinary reruns preserve chat; changing dealer/month/data/evidence/model resets it. Failed requests retain the previous review and pending question, clearly labelled as previously generated. Quoted numeric claims and units, references, categories and deterministic prerequisites are checked. This is **not proof that every narrative statement is factually correct**. Imported notes/messages remain untrusted evidence. Chat is read-only and cannot browse or access another dealer.

Explicit live verification (connectivity, blocked demo-vehicle review and follow-up; no database writes):

```sh
.venv/bin/python scripts/verify_hosted_ai.py
```

Live verification is currently **pending**: no server-side API credentials or account entitlement are configured. Mock tests are not live verification. Once configured, this script uses only the selected provider/model and prints verification metadata, never credentials. No actual provider/model request has been made; the example configuration is Groq / `openai/gpt-oss-20b`.

## Final manager-first UX pass

Main Portfolio queue: Dealer, Current status, Main concern, Three-month outlook and Review priority. Filters govern the computed headline and featured dealer. Show queue details exposes scores, changes, region and exact action scope; sorting and every dealer remain available. Diagnosis shows actual status overrides beside the score, previous available month with date, pillar weakness text, next review action and action-specific readiness. Model/formula/history details use progressive disclosure. AI controls say Generate/Regenerate review, provide a disabled live control with setup path when unavailable, and show one mode badge. Chat sits beside the findings. Tracker cards show compact action/dealer/owner/deadline/priority/status summaries, with description/evidence/status changes expanded. Caption contrast is increased without changing the neutral/blue design.

### Forecast audit and outstanding blocker

No overall sustained-deterioration module, target configuration or corresponding artifact was found. Existing inventory artifacts predict aged-stock >30% within three months; old payment artifacts predict payment delay. Neither was relabelled or retrained. Both remain in Methodology, outside operational AI evidence and manager priority. The core **Three-month outlook says Forecast unavailable**. To finish the agreed overall forecast, the required business definition is missing: score-drop threshold, reference score and how many consecutive future months constitute sustained deterioration. This has been requested from the user. No labels, training results, model scores or “no alert” claims are fabricated. This is a blocker to completing that forecast, not a cosmetic issue.

### Duplicate approvals and recording reset

New submissions use review-generation/proposal identity plus the approved edits/deadline, with a unique submission record inside the same SQLite write transaction. Repeated identical submissions return the same saved action. A separate proposal, new generated review or meaningful approved edit can create a separate action. Existing historical repetitions are preserved; the audit found identical action/owner/deadline records #3 and #4 and did not delete either.

Reset demo actions lives in Data & Methodology → Demo controls. It is disabled by default. For a **private local recording workspace only**, set `DEMO_RESET_ENABLED = true` in server-side Streamlit secrets, or `DEMO_RESET_ENABLED=true` in the server environment, then restart. Never enable it on a public/shared deployment; this local app has no authenticated administrative role. Merely visiting the app cannot enable it.

The reset previews the current count, requires a count-specific confirmation, checks the count again under a write lock, creates/verifies a recoverable SQLite backup under `data/action_backups/` (permissions 0600), and transactionally removes all tracked actions, their associated action-only history and submission keys. Review/chat/action draft state in the current session is cleared. Dealer data, field assessments, scoring rules, artifacts, credentials and configuration remain intact. Backups contain only action tables/evidence, never credential/configuration files; they are ignored by Git and must be excluded from source ZIPs. No actual reset was performed during implementation.

Backups contain tables actions, history and approval_submissions and can be inspected with SQLite. For recovery, stop the app, preserve the current database, and restore only these three tables from the backup in a single transaction after checking for ID conflicts. Do not replace the entire application database, which would discard field assessments.

Launch: `.venv/bin/python -m streamlit run app.py`. Live AI remains unverified without credentials and confirmed entitlement. No deployment performed.

Final verification: 49 functional tests passed, with targeted hosted/chat tests rerun after final copy/layout changes. Chrome inspected all five pages at 1440×900 and 1280×800 at normal zoom; queue, override explanation, native navigation/selection, compact populated tracker and final nearby chat controls were reviewed for clipping, contrast and page overflow. The intentional empty tracker was visually verified at both sizes in a disposable preview database. The reset flow and post-reset approval were verified through AppTest against isolated databases. No real actions were reset or inserted, and original actions/history/field assessments were compared unchanged. The temporary preview was stopped; the real app remains available on localhost:8501. Live AI and a populated live conversation remain unverified without credentials. Overall-deterioration forecasting remains blocked on the missing event definition requested from the user.


## Overall health outlook — implemented 4 October 2026

This supersedes earlier statements that the overall event was undefined/unavailable. Portfolio, Diagnosis and AI/chat context now use **persistent health-score deterioration over the next three months**: raw fixed business-weighted score at least 10 points below the current month in at least two of the next three consecutive calendar months (not necessarily consecutive qualifying months). All three outcomes are required for labels; latest unlabelled months remain inference eligible. This is an index-decline prototype, not business failure or a causal diagnosis. Configuration: `config/overall_deterioration.json`.

Six consecutive months with valid fixed health scores supply all four pillar KPIs, raw score, pillar levels, one/three-month changes and six-month slopes. Past missing scoring inputs/gaps make history insufficient; no score or negative target is fabricated. Logistic regression uses training-only median imputation and scaling, C=0.2. Forest: 180 trees, depth 4, minimum leaf 12. Recent-three-month health trend extrapolates monthly scores, using training residual scale for an uncalibrated distance score. Training prevalence is constant per fitted fold. No IDs, action outcomes, readiness assessments or PCA weights enter predictors/labels. The scripted D001 Aarav, D002 Neel, D003 Surya and D004 Kiran cases are identified by exact ID/name and excluded from training and reported validation/test; inference remains available for them.

Shared temporal validation windows: Aug–Sep 2025 and Oct–Nov 2025. Each training label is fully observable before its validation freeze. All pooled validation outcomes precede the 1 March 2026 final-test freeze. Final training: Mar–Nov 2025, 504 samples / 27 events; latest outcome 28 February 2026. Test: Mar–Jun 2026, 224 samples / 10 events (4.46%). Validation: 224 samples / 14 events (6.25%). An earlier candidate fold lacks training support and is disclosed as skipped. All candidates share evaluation rows. Repeated dealers and overlapping horizons reduce independence; this tests later months of the existing synthetic network, not unseen dealers.

Predeclared selection: validation AP; ML must exceed the better baseline by 0.02 and have at least five validation events. Alert threshold maximises validation F2 (recall weighted four times precision), with higher-threshold ties. Test outcomes never select or retune models/thresholds. Selected **regularised logistic regression**, frozen threshold **0.65**.

| Candidate | Validation AP | Test AP | Test ROC-AUC | Test Brier | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| Recent health trend | 0.039 | 0.029 | 0.203 | 0.1259 | 0.045 | 1.000 |
| Training prevalence | 0.068 | 0.045 | 0.500 | 0.0427 | 0.045 | 1.000 |
| Regularised logistic regression | 0.378 | 0.217 | 0.848 | 0.0429 | 0.000 | 0.000 |
| Constrained random forest | 0.196 | 0.191 | 0.776 | 0.0400 | 0.125 | 0.400 |

**Material limitation:** selected-model test confusion matrix is TN=213, FP=1, FN=10, TP=0. The frozen threshold catches none of ten test events despite improved ranking AP. Validation precision/recall are 0.667/0.571, Brier 0.0672. This is an implemented experimental outlook, not a demonstrated reliable alert system; synthetic sample sizes and calibration do not establish production predictive usefulness. No threshold was adjusted against test results. Methodology displays this warning, candidate metrics, sample counts and confusion matrices.

Saved artifact includes all fitted pipelines/baselines, selected model, threshold, schema, target/config, scoring hash, training cutoff, fold audit and reports. Fingerprint covers canonical input data, scoring/config and implementation bytes (including reused inventory split/report helpers); it does not use dealer identity as a predictor. Changed data/scoring/config/code produce a stale-artifact message. Normal app startup only loads a matching artifact and predicts; it never trains. Missing history, stale/missing artifacts, historical pre-freeze months and model errors have specific reasons. Scores appear as raw decimals under Details, never percentage chances. Critical/urgent status and readiness/approval gates remain unchanged. Inventory results remain a separate secondary Methodology tab.

From the repository root:

```sh
.venv/bin/python scripts/train_overall_deterioration.py
# For a validated import instead of the unchanged synthetic demo:
.venv/bin/python scripts/train_overall_deterioration.py --csv /path/to/dealers.csv
.venv/bin/python -m unittest discover -s tests
.venv/bin/streamlit run app.py --server.port 8501
```

Full audit/report: `artifacts/overall_deterioration/5c7f68a32fb070dab1091d66.json`. Deploy the matching joblib artifact together with the exact data/config/source and install `requirements.txt`; no extra deep-learning dependency is needed. Imports need explicit offline training; short/single-class data may lack sufficient purged validation support, reported with exact eligible/positive counts. No deployment was performed. Real-world policy acceptance, sufficient representative event data and prospective threshold/calibration validation remain production blockers; hosted AI account access is independent of the local forecast.

Verification for this implementation: 56 functional checks passed, including exact target boundaries, calendar gaps, future-feature invariance, purged splits, unchanged selection after test-outcome perturbation, training-only imputation/scaling, latest-month inference, Critical priority, AI target/version, and isolated approval/persistence. Chrome rendered all five pages and Health outlook methodology at 1440×900 and 1280×800 with no page-level horizontal overflow; diagnosis forecast and approval controls were inspected. Browser checks generated deterministic demo briefs without live AI calls or saving real records. Live provider output for the new forecast was not verified, and no production deployment occurred.


## Prepared cloud demo

Upload the contents of `deployment_upload/` as the repository root; use `app.py` and Python 3.12. Consult `DEPLOYMENT.md` for the exact file list and private Streamlit settings. The app works without an API key: **AI Demo** is an explicitly labelled deterministic evidence-based briefing, and live chat is disabled gracefully. Live AI is disabled by default pending successful verification; configuring credentials and free-plan entitlement alone does not enable it. `AI_LIVE_ENABLED` is an optional explicit server-side opt-in after verification. Keep hosted demo reset disabled. Action records on the hosted filesystem are temporary and may reset after restart/redeployment.
