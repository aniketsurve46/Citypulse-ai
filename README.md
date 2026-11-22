# CityPulse AI — Chicago City Operations Center

An AI-powered city operations platform built on **live Chicago 311 data**. It detects
emerging infrastructure problems, predicts which service requests are likely to breach a
7-day service window, and produces a transparent, blended priority queue that a city
operations manager can act on — approve, reject or escalate — with a human in charge of
every decision.

This is a decision-support system, not a dashboard: a trained model scores every open
request, an ops agent recommends and (on human approval) executes actions, citizens can
file and track complaints inside the system, and a grounded knowledge assistant answers
questions from the project's own data — never from invented numbers.

## The problem

Chicago receives thousands of 311 complaints a day — potholes, streetlight outages,
garbage, water leaks, tree emergencies. Crews are limited, so the real operational
question is not "what was reported?" but **"what should we fix first, and why?"**
Worked purely first-come-first-served, the queue spends its first capacity on quick,
low-impact jobs while the requests that will take longest — and matter most — wait.

## What CityPulse does

1. **Ingests live data** from the Chicago Data Portal (Socrata dataset `v6vf-nfxy`,
   unauthenticated API, cursor-paginated and cached locally).
2. **Predicts delay risk** — an XGBoost model estimates the probability that a request
   will take more than 7 days to close, using only features knowable at creation time.
3. **Decides what to fix first with a blended priority** (below) — FIFO is one factor,
   not the whole decision.
4. **Detects duplicates** — repeat reports of the same type at the same address within
   14 days are clustered, so one crew visit can close what looks like five jobs.
5. **Detects emerging hotspots** — community-area × type volume in the last 7 days vs
   the prior 28-day daily average (z-score / ratio).
6. **Forecasts volume** — next 7 days of citywide request volume from a lag/rolling
   feature model, benchmarked against a naive baseline.
7. **Ops Agent** — runs a monitoring cycle (fetch live → score → detect → recommend),
   and on manager approval executes work orders and citizen status updates, logging
   every step.
8. **Citizen reports** — residents can file a complaint inside CityPulse, get a receipt
   with a tracking ID, a priority score and a queue position, and track its status as
   managers act on it. **Honesty note: the Socrata API is read-only, so citizen reports
   live only in CityPulse's local system — they are NOT submitted to the City of
   Chicago's official 311.**
9. **Knowledge Assistant (RAG)** — retrieval over the Socrata data dictionary, per-type
   and per-area knowledge cards computed from the real data, hotspot snapshots and this
   README. With no LLM API key it answers extractively from retrieved chunks and cites
   sources; it never invents a number.

## The blended priority decision

Priority is a transparent weighted blend of five factors (weights for the default
**Balanced** mode; the app also offers FIFO-first and Urgency-first modes that
re-weight the same components live):

| Factor | Balanced weight | What it means |
|---|---|---|
| Delay risk | 0.30 | Model probability of taking > 7 days to close |
| Severity | 0.25 | Hand-set tier per request type (safety / public impact) |
| Waiting time (FIFO) | 0.20 | Days already open, capped at 30 — old requests steadily rise, nothing starves |
| Hotspot | 0.15 | Area surge for this type vs its normal level |
| People affected | 0.10 | Repeat reports at the same address (duplicate group size) |

**Severity tiers are policy, not data** — they are hand-set in one config dict in
`priority.py` (`SEVERITY_RULES`), shown in the app, and editable:

| Tier | Score | Types (reasoning) |
|---|---|---|
| Critical | 1.00 | Traffic signal out, tree emergency, check for leak, no water, sewer/pavement cave-in, open fire hydrant, water on street — immediate safety hazard or essential service out |
| High | 0.75 | Pothole in street / alley pothole, street & alley light out, rodent baiting, sanitation code, dead animal, fly dumping, water in basement, building & plumbing violations, stop-sign repair — road/health hazards and outages |
| Medium | 0.50 | Garbage/recycling carts, missed pickup, yard waste, tree debris/trim, weeds, abandoned vehicle, street cleaning, animal services — service disruption |
| Low | 0.25 | Graffiti, e-scooter / Divvy parking, parking-code and consumer/administrative complaints — cosmetic or administrative |

Every request shows its breakdown and a plain-English reason naming its top
contributing factor, e.g. *"Top factor: Delay risk · Severity: High (Alley Pothole
Complaint) · Waiting 53d · Delay risk 0.90 · 2 reports at this address · Hotspot area
(3.0x normal)"*.

## Measured results (real data, 2026-11-22)

Data: **220,000 completed requests** (created 2026-08-03 → 2026-11-22, after excluding
`311 INFORMATION ONLY CALL` and `Aircraft Noise Complaint`, which close instantly and
are not field work) and **40,000 open requests** (created 2026-09-25 → 2026-11-22).
Temporal split at 2026-10-21: 176,000 train / 44,000 test.

**Delay-risk model (target: closed in more than 7 days — a working SLA proxy; Chicago
publishes no single SLA for all 311 types):**

| Model | AUROC | AUPRC | Brier |
|---|---|---|---|
| XGBoost (creation-time features only) | **0.9219** | **0.6431** | **0.0843** |
| Baseline — per-type historical breach rate | 0.9012 | 0.5608 | 0.1051 |
| Baseline — always predict overall breach rate | 0.5000 | 0.1416 | 0.1874 |

**Which order should a crew work? Three strategies, top-K = 1,000 from the test backlog:**

| Strategy | Avg days-to-close of top-K | Share of top-K that breached > 7d | Breaches captured | Recall of all test breaches |
|---|---|---|---|---|
| FIFO | 4.93 | 25.9% | 259 | 4.2% |
| Model risk only | 10.67 | 76.1% | 761 | 12.2% |
| **CityPulse blended** | **11.60** | **86.2%** | **862** | **13.8%** |

Reading this honestly: the strategies that prioritise risk surface the jobs that took
longest — that is the point. Under FIFO only a quarter of the first 1,000 jobs a crew
would start were eventual breaches; under the blended order, 86% were. (Backtest note:
hotspot and duplicate signals are live point-in-time only, so the blended backtest
renormalises the Balanced weights over delay risk + severity + waiting time.)

**Other measured results**

- **Duplicates:** 4,624 open requests sit in 1,933 duplicate clusters (same type +
  address, ≤ 14 days apart). The city's own duplicate flag is set on only 36.7% of the
  requests inside those clusters (9.3% of all open requests).
- **Hotspots:** 37 community-area × type combinations currently flagged as emerging
  hotspots (≥ 20 requests in 7 days, ≥ 1.5× normal, z ≥ 2).
- **Forecast:** 14-day holdout (2026-11-09 → 2026-11-22) MAE **231.9 requests/day** for
  the lag-feature model vs **293.6** for the naive same-day-last-week baseline.
- **Potholes, highlighted:** "Pothole in Street Complaint" — 8,215 completed in the
  data, **median 22.0 days to close, 76.4% took more than 7 days**, handled by CDOT;
  2,540 pothole requests are open in the current queue. It is a first-class category in
  training, filters, map, per-type stats and the citizen form (listed first).
- **Fairness:** across 78 community areas with ≥ 30 open requests, mean priority ranges
  0.436 → 0.538 (spread 0.101); historical median days-to-close by area ranges
  0.08 → 8.53 days. Areas with slower historical service score higher priority — the
  intended direction — but the spread is reported so no area is silently left at the
  back of the queue. Community areas only; no demographic data is used or inferred.
- **Top of today's queue (real examples):** SR26-02000442, Alley Pothole Complaint,
  10146 S WINSTON AVE, score 0.834 — delay risk 0.90, waiting 53 days, 2 reports at the
  address, hotspot area at 3.0× normal.

## Architecture

```
Chicago 311 Socrata API (live, read-only)
        |
   ingest.py  (cursor pagination, local parquet cache)
        |
   train_model.py -> XGBoost delay-risk model (model.pkl)
                  -> duplicates, hotspots, forecast, fairness (outputs/*.csv)
                  -> prioritized_open.csv (blended priority, priority.py)
        |
   app.py (Streamlit ops center)  ---- citizen.py (report / receipt / tracking)
        |                            |- agent.py   (Ops Agent: tools, recommendations,
        |                            |              work orders, activity log)
        |                            |- rag.py     (TF-IDF RAG over grounded sources)
        +-- Live mode polls the API and scores new requests on arrival
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # or use Anaconda
pip install -r requirements.txt
python ingest.py        # downloads + caches real Chicago 311 data (~4 min)
python train_model.py   # trains model, builds all outputs + metrics.json (~2 min)
streamlit run app.py    # the operations center
python agent.py         # optional: run one Ops Agent monitoring cycle + brief
python rag.py           # optional: test the Knowledge Assistant on sample questions
jupyter notebook CityPulse.ipynb   # optional: step-by-step walkthrough
```

No API key is required for anything, including the agent and the knowledge assistant.
If `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set, the Knowledge Assistant uses it to
compose grounded, source-citing answers; otherwise it answers extractively.

## App tour

- **Operations Center** — KPI cards, dark pydeck map (risk / hotspot / duplicate /
  citizen layers with tooltips), the blended priority queue with working filters
  (department, type, area, ward, risk band, date, source, search, LIVE-only), a
  priority-mode selector, request drill-down with score gauge, 5-factor breakdown,
  duplicate group members, area trend and similar past requests with real outcomes,
  Approve / Reject / Escalate with a persistent, exportable decision log.
- **Hotspots & Forecast** — emerging hotspot chart + table, volume history with the
  7-day forecast, interval and naive comparison, per-type median close times.
- **Model & Evaluation** — model vs baselines, calibration curve, feature importance,
  risk distribution, the three-strategy comparison, the severity policy table, and the
  fairness-by-area analysis.
- **Live Feed** — poll the API now (or auto-refresh) and watch brand-new requests get
  scored and mapped, marked LIVE with their real creation timestamps.
- **Report a Problem** — citizen complaint form (pothole first), instant scoring,
  receipt with complaint ID + queue position (downloadable), and complaint tracking
  wired to the same decision store managers use.
- **Ops Agent** — run monitoring cycles, approve/reject agent recommendations, see
  executed work orders and the full agent activity log, draft the daily ops brief.
- **Knowledge Assistant** — ask about request types, departments, neighbourhoods,
  fields, hotspots or a specific request ID; answers cite their grounded sources.

## Limitations (stated, not hidden)

- **SLA is a proxy.** "Breach" = took more than 7 days. Chicago does not publish one
  universal SLA across all 311 types; per-type medians are reported so the proxy can be
  judged type by type.
- **Right-censoring in recent data.** The completed-requests dataset only contains
  requests that have already closed, so the most recent weeks under-represent slow
  requests (they are still open). This is visible in the split: breach prevalence is
  39.8% in train vs 14.2% in the newest test window. The model is therefore best read
  as a ranking tool, and its probabilities are optimistic for very recent cohorts.
- **Prioritisation, not dispatch.** No crew locations, skills, shifts or travel times
  are modelled — a human manager approves every action.
- **Severity tiers are hand-set policy**, documented above and in `priority.py`; they
  are not learned from data and reasonable people could set them differently.
- **Source freshness.** The app can poll frequently, but the Socrata source does not
  guarantee instant updates; timestamps shown are the city's own `created_date`.
- **Citizen reports are local to CityPulse** (read-only API) and locations for typed
  addresses are ZIP-level approximations (median of real requests with that ZIP),
  labelled approximate — never fake precise pins.
- **Fairness check is geographic only** (community areas). No demographic data is used
  or inferred anywhere in this project.

## Resume bullets

- Built CityPulse AI, a live city-operations platform on the real Chicago 311 Socrata
  API (220K completed + 40K open requests), with an XGBoost delay-risk model
  (AUROC 0.92, AUPRC 0.64 vs 0.14 for a naive baseline) using creation-time features
  only and a temporal train/test split.
- Designed a transparent blended prioritisation (model risk + hand-set severity policy
  + FIFO waiting time + hotspot surge + duplicate reports); in a top-1,000 backtest it
  concentrated 86% eventual SLA breaches vs 26% under first-come-first-served.
- Engineered duplicate clustering (1,933 clusters covering 4,624 open requests),
  emerging-hotspot detection (37 flagged area/type surges) and a volume forecast that
  beat a naive baseline by 21% MAE on a 14-day holdout.
- Built an ops agent that monitors live requests, recommends escalations/merges and
  executes manager-approved work orders with a full activity log; a citizen
  report-and-track flow scored by the same model; and a TF-IDF RAG assistant grounded
  in the dataset dictionary and computed per-type/per-area knowledge cards.
- Shipped a dark ops-center Streamlit app with pydeck mapping, live API scoring,
  per-request explainability, fairness-by-area reporting and an exportable human
  decision log.
