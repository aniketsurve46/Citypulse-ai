"""
CityPulse AI - refresh the open queue from the live Chicago 311 API.

What this does (no retraining - the saved model in outputs/model.pkl is reused):
  1. Downloads ALL currently-open 311 requests from the Socrata API
     (same exclusions as ingest.py) -> data/open_requests.parquet
  2. Rescores every open request with the saved model, recomputes duplicate
     clusters, hotspots and the blended priority queue
     -> outputs/prioritized_open.csv (+ hotspots.csv, area_hotspots.csv,
        fairness.csv)
  3. Updates the open-queue fields in outputs/metrics.json (data_as_of,
     open_rows, open_date_min/max, high-risk count, duplicates, hotspots)
     so the app header and KPIs match the refreshed queue.

Run directly (`python refresh.py`) or from the app's sidebar button.
"""
import json
import os
import pickle

import pandas as pd
import requests

import ingest
import priority as prio_mod
import train_model as tm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
EXCL = " AND ".join([f"sr_type != '{t}'" for t in ingest.EXCLUDED_TYPES])


def refresh(log=print):
    # ---------------------------------------------------------- 1. download
    # Pull the FULL current open queue: fetch_cursor walks newest-first and
    # stops on its own when a page returns no new sr_numbers, so a very high
    # target simply means "everything that is open right now".
    log("Downloading the full current open queue from Chicago 311...")
    open_raw = ingest.fetch_cursor(f"status='Open' AND {EXCL}", 1_000_000, "open")
    ingest.save(open_raw, "open_requests.parquet")

    # ------------------------------------------------------------- 2. rescore
    closed = tm.prep(tm.load("closed_requests"))
    closed["days_to_close"] = ((closed["closed_date"] - closed["created_date"])
                                .dt.total_seconds() / 86400.0)
    closed = closed[(closed["days_to_close"] >= 0)
                    & (closed["days_to_close"] <= 365)].copy()
    closed["breach"] = (closed["days_to_close"] > tm.SLA_DAYS).astype(int)

    open_df = tm.prep(tm.load("open_requests"))
    open_df, dup_stats = tm.add_duplicate_clusters(open_df)
    log(f"duplicates: {dup_stats}")

    all_df = pd.concat([
        closed[["created_date", "community_area", "sr_type", "latitude", "longitude"]],
        open_df[["created_date", "community_area", "sr_type", "latitude", "longitude"]],
    ], ignore_index=True)
    hotspots, area_hot, asof = tm.compute_hotspots(all_df)
    hotspots.to_csv(os.path.join(OUT, "hotspots.csv"), index=False)
    area_hot.to_csv(os.path.join(OUT, "area_hotspots.csv"), index=False)
    n_hot = int(hotspots["is_hotspot"].sum())
    log(f"hotspots flagged: {n_hot} (as of {asof})")

    with open(os.path.join(OUT, "model.pkl"), "rb") as f:
        pipe = pickle.load(f)
    hs_lookup = {(r.community_area, r.sr_type): (r.ratio, r.z_score)
                 for r in hotspots.itertuples() if pd.notna(r.ratio)}
    open_df["breach_prob"] = pipe.predict_proba(open_df[tm.FEATURES])[:, 1]
    ratios = open_df.apply(
        lambda r: hs_lookup.get((r["community_area"], r["sr_type"]), (1.0, 0.0)), axis=1)
    open_df["hotspot_ratio"] = [x[0] for x in ratios]
    open_df["source"] = "Official 311"
    open_df = prio_mod.compute_priority(open_df, mode="Balanced",
                                        now=open_df["created_date"].max())
    cols = ["sr_number", "sr_type", "sr_short_code", "owner_department", "status",
            "origin", "created_date", "street_address", "zip_code", "community_area",
            "ward", "duplicate_flag", "latitude", "longitude", "dup_cluster_id",
            "dup_group_size", "breach_prob", "hotspot_ratio", "hotspot_norm",
            "dup_norm", "severity_tier", "severity_score", "wait_days", "wait_norm",
            "priority_score", "risk_band", "reason", "source"]
    prio_out = open_df.sort_values("priority_score", ascending=False)[cols]
    prio_out.to_csv(os.path.join(OUT, "prioritized_open.csv"), index=False)

    # ------------------------------------------------------------- 3. fairness
    fair_test = (closed.groupby("community_area")
                 .agg(n=("sr_number", "count"), median_days=("days_to_close", "median"),
                      breach_rate=("breach", "mean")).reset_index())
    fair_open = (prio_out.groupby("community_area")
                 .agg(open_n=("sr_number", "count"),
                      mean_priority=("priority_score", "mean"),
                      mean_breach_prob=("breach_prob", "mean")).reset_index())
    fairness = fair_open.merge(fair_test, on="community_area", how="left")
    fairness = fairness[fairness["open_n"] >= 30].sort_values(
        "mean_priority", ascending=False)
    fairness.to_csv(os.path.join(OUT, "fairness.csv"), index=False)

    # -------------------------------------------------------------- 4. metrics
    mpath = os.path.join(OUT, "metrics.json")
    with open(mpath) as f:
        m = json.load(f)
    m["data_as_of"] = asof
    m["open_rows"] = int(len(prio_out))
    m["open_date_min"] = str(prio_out["created_date"].min())
    m["open_date_max"] = str(prio_out["created_date"].max())
    m["high_risk_open"] = int((prio_out["risk_band"] == "High").sum())
    m["mean_open_breach_prob"] = round(float(prio_out["breach_prob"].mean()), 4)
    m["hotspots_flagged"] = n_hot
    m["duplicates"] = dup_stats
    if len(fairness):
        m["fairness"] = {
            "areas_compared": int(len(fairness)),
            "mean_priority_max": round(float(fairness["mean_priority"].max()), 4),
            "mean_priority_min": round(float(fairness["mean_priority"].min()), 4),
            "mean_priority_spread": round(float(fairness["mean_priority"].max()
                                                  - fairness["mean_priority"].min()), 4),
            "median_days_max": round(float(fairness["median_days"].max()), 2),
            "median_days_min": round(float(fairness["median_days"].min()), 2),
            "highest_priority_area": str(fairness.iloc[0]["community_area"]),
            "lowest_priority_area": str(fairness.iloc[-1]["community_area"]),
            "note": "Community areas only; no demographic data is used or inferred.",
        }
    with open(mpath, "w") as f:
        json.dump(m, f, indent=1)

    summary = {"open_rows": int(len(prio_out)),
               "date_min": str(prio_out["created_date"].min().date()),
               "date_max": str(prio_out["created_date"].max().date()),
               "data_as_of": asof, "hotspots": n_hot}
    log(f"Refresh complete: {summary}")
    return summary


if __name__ == "__main__":
    refresh()
