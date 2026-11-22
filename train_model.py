"""
CityPulse AI - model training + all offline analysis.

Produces (in outputs/):
  model.pkl               trained pipeline (preprocessing + classifier)
  metrics.json            every measured number used by the app and README
  prioritized_open.csv    open queue scored + ranked, with reasons
  hotspots.csv            community_area x sr_type emerging hotspots
  area_hotspots.csv       community-area level hotspot aggregates (for the map)
  forecast.csv            next-7-days volume forecast (+ naive comparison)
  calibration.csv         predicted vs actual breach rate by decile
  feature_importance.csv  top features from the model
  fairness.csv            per-community-area priority / wait-time spread
  type_stats.csv          per-sr_type median days-to-close + breach rate
  daily_volume.csv        daily total request volume (all cached data)

Target definition (working SLA proxy):
  breach = a completed request took MORE THAN 7 DAYS from creation to close.
  Chicago does not publish one universal SLA for all 311 types, so 7 days is a
  transparent proxy; per-type medians are reported in type_stats.csv.

Features are knowable AT CREATION TIME ONLY (no leakage):
  sr_type, owner_department, origin, community_area, ward, zip_code,
  created_hour, created_day_of_week, created_month, duplicate flag.
Temporal split: oldest 80% train, newest 20% test, by created_date.
"""
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

CAT = ["sr_type", "owner_department", "origin", "community_area", "ward", "zip_code"]
NUM = ["created_hour", "created_day_of_week", "created_month", "duplicate_flag"]
FEATURES = CAT + NUM
SLA_DAYS = 7
FIFO_K = 1000  # plausible daily crew capacity used in the FIFO simulation


def load(name):
    p = os.path.join(DATA, name + ".parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.read_csv(os.path.join(DATA, name + ".csv"), low_memory=False)


def prep(df):
    df = df.copy()
    for c in ["created_date", "closed_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["created_hour", "created_day_of_week", "created_month"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["duplicate_flag"] = (
        df["duplicate"].astype(str).str.lower().isin(["true", "1"]).astype(int)
        if "duplicate" in df.columns else 0
    )
    for c in CAT:
        df[c] = df[c].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    return df


# ---------------------------------------------------------------- duplicates
def add_duplicate_clusters(open_df):
    """Group open requests by sr_type + normalized street address, splitting a
    group when reports are more than 14 days apart. Returns df with
    dup_cluster_id and dup_group_size."""
    df = open_df.sort_values("created_date").copy()
    df["addr_norm"] = (
        df["street_address"].astype(str).str.upper().str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df["dup_cluster_id"] = -1
    df["dup_group_size"] = 1
    cid = 0
    for (sr_type, addr), idx in df.groupby(["sr_type", "addr_norm"]).groups.items():
        if addr in ("NAN", "", "NONE"):
            continue
        sub = df.loc[idx].sort_values("created_date")
        gaps = sub["created_date"].diff().dt.days.fillna(0)
        cluster_no = (gaps > 14).cumsum()
        for _, sub_idx in sub.groupby(cluster_no).groups.items():
            if len(sub_idx) > 1:
                df.loc[sub_idx, "dup_cluster_id"] = cid
                df.loc[sub_idx, "dup_group_size"] = len(sub_idx)
                cid += 1
    n_grouped = int((df["dup_group_size"] > 1).sum())
    grouped = df[df["dup_group_size"] > 1]
    agree = float(grouped["duplicate_flag"].mean()) if len(grouped) else 0.0
    stats = {
        "open_in_duplicate_groups": n_grouped,
        "duplicate_clusters": int(df.loc[df["dup_cluster_id"] >= 0, "dup_cluster_id"].nunique()),
        "city_flag_rate_within_our_groups": round(agree, 4),
        "city_flag_rate_overall_open": round(float(df["duplicate_flag"].mean()), 4),
    }
    return df, stats


# ------------------------------------------------------------------ hotspots
def compute_hotspots(all_df):
    max_date = all_df["created_date"].max().normalize()
    last7_start = max_date - pd.Timedelta(days=6)
    prior_start = max_date - pd.Timedelta(days=34)
    recent = all_df[all_df["created_date"] >= last7_start]
    prior = all_df[(all_df["created_date"] >= prior_start) &
                   (all_df["created_date"] < last7_start)]
    r = recent.groupby(["community_area", "sr_type"]).size().rename("count_7d")
    p = prior.groupby(["community_area", "sr_type"]).size().rename("count_prior_28d")
    h = pd.concat([r, p], axis=1).fillna(0).reset_index()
    h["expected_7d"] = h["count_prior_28d"] / 28.0 * 7.0
    h["ratio"] = np.where(h["expected_7d"] > 0, h["count_7d"] / h["expected_7d"], np.nan)
    h["z_score"] = np.where(h["expected_7d"] > 0,
                            (h["count_7d"] - h["expected_7d"]) / np.sqrt(h["expected_7d"]), 0)
    h["is_hotspot"] = (h["count_7d"] >= 20) & (h["ratio"] >= 1.5) & (h["z_score"] >= 2)
    h = h.sort_values("z_score", ascending=False)
    # area-level aggregate for the map
    cent = (all_df.dropna(subset=["latitude", "longitude"])
            .groupby("community_area")[["latitude", "longitude"]].mean().reset_index())
    area = h.groupby("community_area").agg(
        count_7d=("count_7d", "sum"), expected_7d=("expected_7d", "sum"),
        hot_spot_types=("is_hotspot", "sum")).reset_index()
    area["ratio"] = area["count_7d"] / area["expected_7d"].replace(0, np.nan)
    area = area.merge(cent, on="community_area", how="left")
    return h, area, str(max_date.date())


# ------------------------------------------------------------------ forecast
def run_forecast(all_df):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    daily = (all_df.set_index("created_date").resample("D").size()
             .rename("volume").reset_index())
    daily = daily[daily["volume"] > 0]
    d = daily.copy()
    for lag in (1, 7, 14):
        d[f"lag{lag}"] = d["volume"].shift(lag)
    d["roll7"] = d["volume"].shift(1).rolling(7).mean()
    d["dow"] = d["created_date"].dt.dayofweek
    d["month"] = d["created_date"].dt.month
    d = d.dropna().reset_index(drop=True)
    feats = ["lag1", "lag7", "lag14", "roll7", "dow", "month"]
    train, test = d.iloc[:-14], d.iloc[-14:]
    model = HistGradientBoostingRegressor(max_iter=300, random_state=42)
    model.fit(train[feats], train["volume"])
    pred = model.predict(test[feats])
    mae_model = float(mean_absolute_error(test["volume"], pred))
    mae_naive = float(mean_absolute_error(test["volume"], test["lag7"]))
    resid_std = float(np.std(test["volume"] - pred))

    # recursive next-7-days forecast
    hist = list(daily["volume"].values)
    dates = pd.date_range(daily["created_date"].max() + pd.Timedelta(days=1), periods=7)
    rows = []
    work = list(hist)
    for dt in dates:
        x = pd.DataFrame([{
            "lag1": work[-1], "lag7": work[-7], "lag14": work[-14],
            "roll7": float(np.mean(work[-7:])),
            "dow": dt.dayofweek, "month": dt.month}])[feats]
        f = float(model.predict(x)[0])
        rows.append({"date": dt.date().isoformat(), "forecast": round(f, 1),
                     "naive_lag7": round(float(work[-7]), 1),
                     "lower": round(max(f - 1.28 * resid_std, 0), 1),
                     "upper": round(f + 1.28 * resid_std, 1)})
        work.append(f)
    fc = pd.DataFrame(rows)
    stats = {"holdout_days": 14, "mae_model": round(mae_model, 2),
             "mae_naive_lag7": round(mae_naive, 2),
             "holdout_start": str(test["created_date"].min().date()),
             "holdout_end": str(test["created_date"].max().date())}
    return daily, fc, stats


# ---------------------------------------------------------------------- main
def main():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    closed = prep(load("closed_requests"))
    open_df = prep(load("open_requests"))
    print(f"closed raw {len(closed):,} | open raw {len(open_df):,}")

    closed["days_to_close"] = (closed["closed_date"] - closed["created_date"]).dt.total_seconds() / 86400.0
    closed = closed[(closed["days_to_close"] >= 0) & (closed["days_to_close"] <= 365)].copy()
    closed["breach"] = (closed["days_to_close"] > SLA_DAYS).astype(int)
    print(f"closed usable {len(closed):,} | breach rate {closed['breach'].mean():.3f}")

    type_stats = (closed.groupby("sr_type")
                  .agg(n=("sr_number", "count"),
                       median_days=("days_to_close", "median"),
                       breach_rate=("breach", "mean"))
                  .reset_index().sort_values("n", ascending=False))
    type_stats.to_csv(os.path.join(OUT, "type_stats.csv"), index=False)

    # temporal split
    closed = closed.sort_values("created_date")
    cut = closed["created_date"].quantile(0.8)
    train = closed[closed["created_date"] <= cut]
    test = closed[closed["created_date"] > cut].copy()
    print(f"train {len(train):,} (<= {cut.date()}) | test {len(test):,}")

    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUM),
    ])
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.06,
                            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                            n_jobs=4, random_state=42)
        model_name = "XGBoost"
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print(f"XGBoost unavailable ({e}); using HistGradientBoosting")
        clf = HistGradientBoostingClassifier(max_iter=350, random_state=42)
        model_name = "HistGradientBoosting"
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train[FEATURES], train["breach"])
    prob = pipe.predict_proba(test[FEATURES])[:, 1]
    test["pred_prob"] = prob

    def mets(y, p):
        return {"auroc": round(float(roc_auc_score(y, p)), 4),
                "auprc": round(float(average_precision_score(y, p)), 4),
                "brier": round(float(brier_score_loss(y, p)), 4)}

    model_m = mets(test["breach"], prob)
    base_rate = float(train["breach"].mean())
    dumb_m = mets(test["breach"], np.full(len(test), base_rate))
    type_rate = train.groupby("sr_type")["breach"].mean()
    type_pred = test["sr_type"].map(type_rate).fillna(base_rate).values
    type_m = mets(test["breach"], type_pred)
    print("model", model_m, "| dumb", dumb_m, "| type-only", type_m)

    # calibration by decile
    test["decile"] = pd.qcut(test["pred_prob"], 10, duplicates="drop")
    cal = (test.groupby("decile", observed=True)
           .agg(mean_pred=("pred_prob", "mean"), actual_rate=("breach", "mean"),
                n=("breach", "size")).reset_index(drop=True))
    cal.to_csv(os.path.join(OUT, "calibration.csv"), index=False)

    # feature importance (aggregate one-hot back to source feature)
    try:
        oh = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        names = list(oh.get_feature_names_out(CAT)) + NUM
        imp = pipe.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"feature": names, "importance": imp})
        fi["source"] = fi["feature"].apply(
            lambda f: next((c for c in CAT if f.startswith(c + "_")), f))
        fi_agg = (fi.groupby("source")["importance"].sum()
                  .sort_values(ascending=False).reset_index()
                  .rename(columns={"source": "feature"}))
        top = fi.sort_values("importance", ascending=False).head(25)
        pd.concat([fi_agg.assign(level="source_feature"),
                   top.assign(level="encoded_feature")]).to_csv(
            os.path.join(OUT, "feature_importance.csv"), index=False)
    except Exception as e:
        print(f"feature importance skipped: {e}")

    # ---- Three-strategy comparison on the test set -------------------------
    # Scenario: on the last day of the test window a manager can start K jobs
    # from the backlog. FIFO = earliest created first. Model-only = highest
    # breach probability. Blended = CityPulse priority (delay risk + severity
    # + waiting time; hotspot/duplicate components are point-in-time live
    # signals, so in this backtest the remaining weights are renormalised over
    # the three available components - stated honestly, not hidden).
    import priority as prio_mod
    review_day = test["created_date"].max()
    test["wait_days_bt"] = (review_day - test["created_date"]).dt.total_seconds() / 86400.0
    test["wait_norm_bt"] = (test["wait_days_bt"] / prio_mod.WAIT_CAP_DAYS).clip(0, 1)
    sev_bt = test["sr_type"].apply(prio_mod.severity_for)
    test["severity_score_bt"] = [s[1] for s in sev_bt]
    wb = prio_mod.PRIORITY_MODES["Balanced"]
    avail = wb["breach"] + wb["severity"] + wb["wait"]
    test["blended_bt"] = (wb["breach"] * test["pred_prob"]
                          + wb["severity"] * test["severity_score_bt"]
                          + wb["wait"] * test["wait_norm_bt"]) / avail
    total_breaches = int(test["breach"].sum())

    def strat(order_col, ascending=False):
        topk = test.sort_values(order_col, ascending=ascending).head(FIFO_K)
        return {
            "mean_days_to_close_topk": round(float(topk["days_to_close"].mean()), 2),
            "breach_share_in_topk": round(float(topk["breach"].mean()), 4),
            "breaches_captured_topk": int(topk["breach"].sum()),
            "breach_recall_topk": round(float(topk["breach"].sum() / max(total_breaches, 1)), 4),
        }

    strategies = {
        "k": FIFO_K,
        "total_breaches_in_test": total_breaches,
        "fifo": strat("created_date", ascending=True),
        "model_only": strat("pred_prob"),
        "blended": strat("blended_bt"),
        "note": ("All three strategies surface requests that took a long time - "
                 "that is the point: the jobs most likely to breach are exactly the "
                 "ones that wait longest under FIFO. Read breach_share / recall in "
                 "the top-K as 'how well does this order concentrate the eventual "
                 "breaches into the first K jobs a crew would start'. Blended "
                 "backtest renormalises Balanced weights over breach+severity+wait "
                 "because hotspot and duplicate signals are live point-in-time only."),
    }
    fifo_sim = {  # backwards-compatible summary used in the app header text
        "k": FIFO_K,
        "fifo_mean_days_to_close": strategies["fifo"]["mean_days_to_close_topk"],
        "priority_mean_days_to_close": strategies["blended"]["mean_days_to_close_topk"],
        "fifo_breach_rate_in_topk": strategies["fifo"]["breach_share_in_topk"],
        "priority_breach_rate_in_topk": strategies["blended"]["breach_share_in_topk"],
        "note": strategies["note"],
    }
    print("Strategies:", json.dumps(strategies, indent=1))

    with open(os.path.join(OUT, "model.pkl"), "wb") as f:
        pickle.dump(pipe, f)

    # ------------------------------------------------------------ open queue
    open_df, dup_stats = add_duplicate_clusters(open_df)
    print("duplicates:", dup_stats)

    all_df = pd.concat([
        closed[["created_date", "community_area", "sr_type", "latitude", "longitude"]],
        open_df[["created_date", "community_area", "sr_type", "latitude", "longitude"]],
    ], ignore_index=True)
    hotspots, area_hot, asof = compute_hotspots(all_df)
    hotspots.to_csv(os.path.join(OUT, "hotspots.csv"), index=False)
    area_hot.to_csv(os.path.join(OUT, "area_hotspots.csv"), index=False)
    n_hot = int(hotspots["is_hotspot"].sum())
    print(f"hotspots flagged: {n_hot} (as of {asof})")

    hs_lookup = {(r.community_area, r.sr_type): (r.ratio, r.z_score)
                 for r in hotspots.itertuples() if pd.notna(r.ratio)}
    open_df["breach_prob"] = pipe.predict_proba(open_df[FEATURES])[:, 1]
    ratios = open_df.apply(lambda r: hs_lookup.get((r["community_area"], r["sr_type"]), (1.0, 0.0)), axis=1)
    open_df["hotspot_ratio"] = [x[0] for x in ratios]
    open_df["source"] = "Official 311"
    # Blended priority: delay risk + severity + waiting time (FIFO) +
    # hotspot + repeat reports. See priority.py for weights and tiers.
    open_df = prio_mod.compute_priority(open_df, mode="Balanced",
                                        now=open_df["created_date"].max())
    sev_table = prio_mod.severity_table(
        list(closed["sr_type"].unique()) + list(open_df["sr_type"].unique()))
    sev_table.to_csv(os.path.join(OUT, "severity_table.csv"), index=False)
    cols = ["sr_number", "sr_type", "sr_short_code", "owner_department", "status",
            "origin", "created_date", "street_address", "zip_code", "community_area",
            "ward", "duplicate_flag", "latitude", "longitude", "dup_cluster_id",
            "dup_group_size", "breach_prob", "hotspot_ratio", "hotspot_norm",
            "dup_norm", "severity_tier", "severity_score", "wait_days", "wait_norm",
            "priority_score", "risk_band", "reason", "source"]
    prio_out = open_df.sort_values("priority_score", ascending=False)[cols]
    prio_out.to_csv(os.path.join(OUT, "prioritized_open.csv"), index=False)

    # --------------------------------------------------------------- fairness
    fair_test = (closed.groupby("community_area")
                 .agg(n=("sr_number", "count"), median_days=("days_to_close", "median"),
                      breach_rate=("breach", "mean")).reset_index())
    fair_open = (prio_out.groupby("community_area")
                 .agg(open_n=("sr_number", "count"),
                      mean_priority=("priority_score", "mean"),
                      mean_breach_prob=("breach_prob", "mean")).reset_index())
    fairness = fair_open.merge(fair_test, on="community_area", how="left")
    fairness = fairness[fairness["open_n"] >= 30].sort_values("mean_priority", ascending=False)
    fairness.to_csv(os.path.join(OUT, "fairness.csv"), index=False)
    fairness_stats = {
        "areas_compared": int(len(fairness)),
        "mean_priority_max": round(float(fairness["mean_priority"].max()), 4),
        "mean_priority_min": round(float(fairness["mean_priority"].min()), 4),
        "mean_priority_spread": round(float(fairness["mean_priority"].max() - fairness["mean_priority"].min()), 4),
        "median_days_max": round(float(fairness["median_days"].max()), 2),
        "median_days_min": round(float(fairness["median_days"].min()), 2),
        "highest_priority_area": str(fairness.iloc[0]["community_area"]),
        "lowest_priority_area": str(fairness.iloc[-1]["community_area"]),
        "note": "Community areas only; no demographic data is used or inferred.",
    }
    print("fairness:", fairness_stats)

    # --------------------------------------------------------------- forecast
    daily, fc, fc_stats = run_forecast(all_df)
    daily.to_csv(os.path.join(OUT, "daily_volume.csv"), index=False)
    fc.to_csv(os.path.join(OUT, "forecast.csv"), index=False)
    print("forecast:", fc_stats)

    metrics = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_as_of": asof,
        "model_name": model_name,
        "sla_definition": f"breach = closed in more than {SLA_DAYS} days (working proxy)",
        "closed_rows_raw": int(len(load('closed_requests'))),
        "closed_rows_used": int(len(closed)),
        "closed_date_min": str(closed["created_date"].min().date()),
        "closed_date_max": str(closed["created_date"].max().date()),
        "open_rows": int(len(open_df)),
        "open_date_min": str(open_df["created_date"].min().date()),
        "open_date_max": str(open_df["created_date"].max().date()),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "split_cutoff": str(cut.date()),
        "overall_breach_rate": round(base_rate, 4),
        "model": model_m, "baseline_global_rate": dumb_m, "baseline_type_only": type_m,
        "fifo_simulation": fifo_sim,
        "strategy_comparison": strategies,
        "priority_modes": prio_mod.PRIORITY_MODES,
        "pothole": {
            "sr_type": "Pothole in Street Complaint",
            "closed_n": int((closed["sr_type"] == "Pothole in Street Complaint").sum()),
            "open_n": int((open_df["sr_type"] == "Pothole in Street Complaint").sum()),
            "median_days": round(float(closed.loc[closed["sr_type"] == "Pothole in Street Complaint", "days_to_close"].median()), 2),
            "breach_rate": round(float(closed.loc[closed["sr_type"] == "Pothole in Street Complaint", "breach"].mean()), 4),
        },
        "duplicates": dup_stats,
        "hotspots_flagged": n_hot,
        "forecast": fc_stats,
        "fairness": fairness_stats,
        "high_risk_open": int((prio_out["risk_band"] == "High").sum()),
        "mean_open_breach_prob": round(float(open_df["breach_prob"].mean()), 4),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
