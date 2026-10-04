### Revised decision architecture — current policy
Current health is the unchanged business-weighted index. Review priority uses current final severity, critical overrides, missing inputs and observed deterioration only. Readiness is separate evidence coverage and intervention-specific blockers, never an additional score. Payment prediction is **superseded in the core workflow**: no probabilities, risk columns or forecasts enter operational priorities, reviews or new approval bundles. Model artifacts, old snapshots and measured results below are preserved exclusively as exploratory forecasting. The payment target requires business validation.

### Financial data dictionary limitation
`operating_margin_pct` is a supplied operating-margin percentage; its accounting basis is not defined. `overdue_receivables_pct` is a supplied overdue percentage, but the source does not establish whose receivables or its denominator. `avg_payment_delay_days` is a supplied observed average delay, but payer, payee and obligation type are unspecified. It is unsafe to invent dealer-to-OEM, customer-to-dealer or lender definitions. Verify counterparties, denominator, invoice/due-date rules and reporting policy. Financial scores and critical policy overrides are illustrative until this is established. No insolvency or cash-flow distress inference is warranted.

### Versioned field evidence and intervention feasibility
`config/readiness.json` defines five checks, 30-day maximum age and prerequisite mappings. SQLite assessments are append-only versions keyed by dealer, exact scope and check. Status is Ready, Blocked or Not assessed. Concrete observation, evidence reference, assessor and assessment date are required; review/expiry date is optional. An explicit revision supersedes its referenced version. Multiple independent active versions with contradictory statuses require verification; newest alone does not override a conflict.

Evidence is applicable only to its exact model/campaign/service scope. Empty scope is unscoped and cannot certify a named model. As-of reporting excludes observations assessed **or entered** after the reporting cutoff; later evidence is visible and labelled, never retroactively applied. A separately labelled current-evidence mode supports present-day decisions against historical monthly health. Expired, more-than-30-day-old, missing and conflicting evidence is Not assessed, never Ready. Ready coverage means five fresh scoped checks, not operational causality or a universal action gate.

Increasing leads or a model campaign requires demo vehicle + trained staff; service growth requires service capacity; stock changes require demand fit; site-dependent actions require facility/access. Known blockers deny approval. Unknown/stale evidence requires a separately approved verification action. Corrective actions resolving a blocker do not require that blocker to be already resolved. Combined prose that also directs demand growth remains subject to the relevant prerequisites. Rules check declared type, explicit site dependence and common edited-text intent patterns. Correct classification remains a manager responsibility; arbitrary prose cannot be fully understood by regex.

At approval, application code acquires a SQLite write transaction and rereads the current scoped field evidence. Concurrent assessment revisions cannot slip between the check and saved action. Original monthly bundle, validated review and approved proposal are saved with an additional approval-time readiness/prerequisite snapshot. No historical evidence JSON is migrated or rewritten. Imported datasets never inherit synthetic field scenarios. Demo scenarios are separately labelled fictional observations, not site visits.

### Operational health policy
The business index remains the operational default: Commercial 25%, Financial 30%, Inventory 25%, Customer/service 20%. KPI weights within each pillar are equal. Rules are centrally configured in `config/scoring.json`; experimental settings are in `config/analysis.json`.

Sales attainment = sales units / target units ×100. Each direction-aligned health score is `clip(100 × (value − bad)/(good − bad), 0, 100)`. Reversed good/bad anchors encode lower-is-better metrics. These are illustrative policy anchors, not validated OEM standards.

Healthy ≥80; Watchlist ≥65; At Risk ≥45; Critical <45. Overdue receivables ≥30% or average delay ≥40 days forces at least Critical; stock older than 90 days ≥30% forces at least At Risk. Worst severity wins. Any required input missing yields Unassessed across all methods, including zero-weight PCA indicators. Overrides never improve a classification; raw scores/classes remain visible separately from final statuses.

Review priority: missing data or Critical P1; At Risk or index decline ≥8 points P2; Watchlist P3; otherwise P4. Trajectory compares the previous available dealer record, which may not be consecutive. Separate growth hypothesis = 0.5 × clipped sales-target shortfall + 0.5 × satisfaction. It is not a financial forecast.

### Weighting alternatives and frozen PCA reference
Equal pillars assign 25% to each pillar, with equal KPI weights inside each pillar. The experimental alternative is **PCA-derived variance weights**, not proven business importance.

For the v2 demo, use only October 2024–September 2025 as the historical reference (first 12 calendar observations; 720 complete rows). Drop incomplete reference rows; do not impute missing scoring inputs. Fit StandardScaler and full-SVD PCA only on the direction-aligned reference health scores. Constant reference indicators are excluded from PCA and assigned zero weight; they remain required scoring inputs. Fewer than 30 complete rows or fewer than two varying indicators disables PCA with a reason. A later month beyond the reference period is required.

Retain the smallest number K of components with cumulative explained variance ≥80%. For indicator j, `importance_j = Σ(k=1..K) explained_variance_ratio_k × component_coefficient_kj²`; normalize importance across indicators to sum to one. The alternative index is the weighted sum of the original 0–100 direction-aligned scores. Standardized values are used to derive weights, not directly as the health index. Indicator contributions group into the existing pillars. The fitted scaler, PCA and derived weights are frozen for later months. Comparisons inside the reference are explicitly retrospective.

Actual v2 fit retains 2 components explaining 80.14% variance. Implied pillar weights: Commercial 18.40%, Financial 28.92%, Inventory 20.62%, Customer/service 32.07%. There are no constant reference indicators in this sample. Correlation diagnostics use the same complete reference scores; |Pearson r|≥0.85 flags potential redundancy. Three pairs are flagged. No indicators are automatically deleted. PCA weights depend on synthetic correlations and can emphasize redundant latent signals.

Scores, within-month ranks, raw classes, final statuses and disagreements against business weighting appear in Data & Methodology. September 2026 has 59 assessed and one Unassessed dealer: equal pillars have no class/status disagreements; PCA has two raw-class and two final-status disagreements. Explained variance is not a reason to replace the operational index. Exploration does not change saved actions or their evidence snapshots.

### Reproducible sensitivity, not confidence intervals
Seed 1729; 100 scenarios per family, with overrides fixed. Weight family: independently perturb each original pillar weight by a uniform relative amount in [−20%,+20%], then renormalize. Final normalized changes may exceed the pre-normalization relative bounds. Anchor family: independently move each good and bad anchor by up to ±5% of its original absolute span, with business weights unchanged. This leaves at least 90% of the original span and preserves direction.

Raw score envelopes include the original policy. Raw-class and override-constrained final-status stability are measured separately as the fraction of the 100 perturbed scenarios matching the baseline. Maximum absolute rank movement uses within-month average ranks and excludes Unassessed scores. Missing rows have no raw range/rank; their final status remains Unassessed.

For September 2026, weight perturbations have a mean raw range width of 2.71 points, maximum 8.55; mean raw-class stability 98.34%, mean final-status stability 98.37%, maximum rank movement 4 places. Anchor perturbations have mean width 3.94, maximum 4.43; mean raw-class stability 96.95%, mean final-status stability 97.00%, maximum rank movement 3 places. These are finite sensitivity scenarios, not confidence intervals or production uncertainty estimates.

### Dataset version and target measurement
The original `generate()` and `data/sample_dealers.csv` (v1) remain preserved. The default demo is separately versioned `generate_v2()` / `data/sample_dealers_v2.csv`: seed 42, 60 fictional Indian-market dealers, October 2024–September 2026. Four named cases demonstrate cash stress despite strong sales, weak sales with strong service/finances, improvement, and missing inputs.

V1 already had autoregressive payment delay and noise; it lacked explicit seasonality and shocks. V2 adds an annual sinusoidal market phase, dealer-level two-sided random shocks with decay, and independent stochastic innovations to delay, sales and inventory aggregates. Payment delay evolves from its previous value and a financial-pressure proxy, with seasonal, shock and noise terms. No health classification is used to generate outcomes, and the generator was not adjusted to favour a model.

Compatibility target: **next calendar month's `avg_payment_delay_days` strictly exceeds 20 days**. The input is interpreted as the monthly arithmetic mean of `max(actual receipt date − contractual due date, 0)` over payments settled during that reporting month; on-time payments contribute zero. Outstanding receivables are represented separately. This is an assumed input measurement contract: the synthetic generator simulates the aggregate, not invoice records. A real source must align to this definition.

The `month` key is the reporting period start. Prediction is made at that reporting month's end after all that month's metrics are available. The next-month label becomes available at the next month's end, assuming zero reporting latency. Unknown final-month outcomes, gaps and missing next-month targets are excluded from labelled evaluation.

### Common six-month information window
Each model gets reporting months M−5 through M, exactly six consecutive calendar months from one dealer, ending at the prediction month. Require all 11 input KPIs in all six months to be finite. No gap bridging, cross-dealer sequences or imputation. Dealer ID is metadata only and is excluded from predictors. Tabular models receive every lag (66 numeric features); the GRU receives the equivalent 6×11 tensor. Persistence uses the current breach status, a subset of the same permitted information.

Models: persistence baseline (next breach equals current breach; outputs 0/1), logistic regression, shallow random forest, and optional CPU PyTorch GRU with one recurrent layer of 24 units and one binary output. Prebuilt PyTorch 2.14.1 was installed into the project virtual environment for Python 3.12 / macOS Apple Silicon; no base Conda changes, source builds or developer tools. Seeds are fixed, CPU threads =1, deterministic algorithms enabled. GRU budget ≤60 epochs, batch 64, Adam learning rate 0.003, patience 8, clipped gradients.

Training runs only through `scripts/train_models.py` or the explicit Methodology control. Ordinary Streamlit reruns load local artifacts keyed by validated data, scoring config, experimental config and implementation fingerprints. No torch import is needed for startup or inference by the selected logistic model. Missing deep-learning packages disable the challenger transparently. A selected GRU without its backend suppresses predictions but leaves scoring/reviews/actions usable. Local artifact files are produced by this app; uploaded pickle/model files are never accepted.

### Expanding temporal validation and final freeze
Split by unique eligible prediction months across dealers. Labels in a training window must be available strictly before its validation reporting month starts: this conservatively freezes before the month and prevents outcome-boundary leakage. Test-selection labels must likewise be available before test start. Preprocessing fits independently in each window.

Actual v2 folds:

| Fold | Training prediction months | Training samples / events | Validation months | Validation samples / events |
|---|---|---:|---|---:|
| 1 | Mar–Aug 2025 | 360 / 17 | Oct–Nov 2025 | 120 / 24 |
| 2 | Mar–Oct 2025 | 480 / 43 | Dec 2025–Jan 2026 | 120 / 15 |
| 3 | Mar–Dec 2025 | 600 / 62 | Feb–Mar 2026 | 120 / 8 |

GRU early stopping uses a separate chronological subset within each outer training window: final two training prediction months are the inner stopping period; inner training labels must be observable before that period begins. The selected epoch budget then refits on the full outer training set before predicting outer validation. Thus outer validation labels do not drive GRU early stopping. Best inner epochs are [59, 4, 23]. The final GRU refit uses their median, 23, with no test early stopping. Scalers use only the corresponding training data.

Model selection uses highest **pooled out-of-fold validation average precision** across 360 sequences / 47 events (13.06%). Each model's alert threshold independently maximizes pooled-validation F1 over the predeclared 0.10–0.90 grid, step 0.05; ties choose the higher threshold. Degenerate training folds are skipped with reasons; AP and ROC-AUC are undefined on single-class evaluation sets. Insufficient eligible history disables the experiment.

After selection, refit each challenger on all pre-test labels observable before 1 May 2026: March 2025–March 2026 prediction months, 780 sequences / 77 events (9.87%). Latest test prediction months May–August 2026 are untouched until final evaluation. All models share 238 eligible test sequences / 12 events (5.04%). April prediction labels are not available at the freeze and cannot enter selection/refit. September is eligible for live inference where its six-month inputs exist, but its unknown October outcome is not evaluated. Former architecture (superseded): Portfolio predictions before May 2026 were suppressed. There are no operational payment forecasts; those payment temporal inference limits apply only to the explicitly opted-in technical appendix. The authorized inventory extension below has its own target and freezes.

### Actual predictive comparison

| Model | Validation AP | Test AP | Test ROC-AUC | Test precision | Test recall | Test P@10/month | Test Brier | Alert threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.424 | 0.256 | 0.662 | 0.667 | 0.333 | 0.125 | 0.0420 | 0.90 |
| Logistic regression | **0.660** | 0.427 | **0.856** | 0.600 | 0.500 | **0.225** | 0.0366 | 0.35 |
| Random forest | 0.560 | 0.396 | 0.781 | 0.412 | 0.583 | 0.175 | 0.0405 | 0.20 |
| GRU | 0.564 | 0.439 | 0.840 | 0.538 | 0.583 | 0.200 | 0.0346 | 0.20 |

**Logistic regression remains selected**, with threshold 0.35. The GRU has slightly higher AP and lower Brier on the small test set, but lower validation AP; the selection is not changed using test results. The comparison is not a significance claim. The test experiment is run once per fingerprinted artifact; explicit forced retraining reruns it and must not be used to tune against test outcomes.

Precision@10/month averages monthly precision among the ten highest-scoring eligible dealers, using all if fewer than ten and breaking score ties by dealer ID. Test reliability plots use five equal-width score bins, omit empty bins and show counts. No probability calibration claim: binary persistence scores and uncalibrated model probabilities are shown honestly. Global standardized lag coefficients / forest impurity importance are distinguished from current observations. No genuine dealer-specific attribution or GRU feature explanation is implemented.

This evaluates future months for existing dealers, not new-dealer generalization. Only 12 test events and substantial validation/test base-rate change make the estimates fragile. PCA and predictive results depend on synthetic assumptions and do not establish production validity.

### Reviews, persisted evidence and limitations
Hosted OpenAI-compatible integration, reference validation, follow-up questions and clearly labelled deterministic demo remain supported. Current operational evidence includes the health method/config fingerprint, monthly KPI changes and scoped field evidence IDs; it excludes predictive artifact fingerprints, probabilities and thresholds. Previous architecture snapshots included those predictive fields; this architecture is superseded, but saved historical JSON remains untouched. Existing SQLite actions and evidence JSON are never rewritten by index exploration, model training or dataset version changes. Only explicit approved additions/status updates modify tracking records.

No automated outreach, email or credit decisions. This local prototype has no multi-user authentication. Real use requires representative data, reviewed metric definitions and risk policies, monitoring, access controls, retention and provider review. Uploaded CSV stays in session; approved actions persist. The hosted provider receives selected evidence only when a hosted review is requested.

The native light dashboard remains concise; analytical comparisons, sensitivity, folds and reliability live in Methodology. Browser checks cover normal laptop layouts. GRU execution is genuinely implemented; live hosted API verification remains dependent on configured credentials.

Calendar cutoffs use the configurable business timezone in `config/readiness.json` (Asia/Kolkata for this workspace). Entry timestamps stay UTC for audit and are converted to that business calendar before historical applicability checks. Assessment dates, current-readiness dates and freshness use the same calendar. This excludes newly entered backdated evidence even across UTC/local midnight.

## Inventory early warning — authorized scope extension

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

## Hosted review and contextual manager chat

Implemented server-side OpenAI-compatible transport, configurable provider/model and explicit free-entitlement gate; bounded shared evidence catalog; structured findings/actions/chat; metric/unit/reference/category validation; deterministic readiness prerequisites and transaction-time approval recheck; explicit manager owner entry; immutable generation provenance in approval snapshots; explicit demo mode; session cache, bounded chat, context resets, cooldown and nonblocking Retry-After; 25-second timeout and at most one validation repair. No tools, browsing, vector store, new analytics or deployment.

Verification: 43 automated tests pass, including mocked hosted generation/chat UI, ordinary reruns, dealer/context reset, separate sessions, malformed JSON, unknown IDs, mismatched metrics/units, unknown categories, blocked campaign prerequisites, owner entry, authentication/network/rate-limit paths and original persistence checks. Chrome inspected all five pages at 1440×900 and 1280×800; the explicit demo review and inline chat setup state have readable, unclipped starter controls and no page-level horizontal overflow. Hosted/populated chat was checked with mocked AppTest responses, not an actual provider conversation. All six pre-existing action evidence snapshots are byte-for-byte unchanged; tests/browser checks created no real saved actions.

Live status: Not configured. No actual provider/model request has been made. Suggested explicit configuration is Groq / openai/gpt-oss-20b, listed in official free-plan documentation; account entitlement must still be confirmed. Missing credentials prevent live connectivity/review/chat and the live blocked-demo-vehicle scenario. scripts/verify_hosted_ai.py completes those checks once configured and never writes records. Detailed local and Community Cloud secret instructions are in README. No deployment performed. Sustained deterioration is explicitly unavailable; existing inventory forecast and superseded payment appendix remain correctly distinguished and unchanged.

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
