"""
CityPulse AI - Ops Agent.

A plain-Python tool-calling agent (no LLM key needed, no framework required).
Its decisions come from the trained model + the transparent priority rules,
which makes every step explainable. It runs a monitoring cycle:

  observe   -> fetch_live_requests (Socrata) + current queue state
  analyse   -> score_requests (trained model), detect_duplicates,
               detect_hotspots
  recommend -> a list of RECOMMENDED ACTIONS (escalate / merge duplicates /
               approve top-K / update citizen status) with reasons
  human     -> a manager approves or rejects each recommendation in the app
  execute   -> on approval the agent executes in the stores:
               create_work_order (work_orders table), update_citizen_status
               (decisions table - the same store citizen tracking reads),
               and writes every step to the Agent Activity Log (agent_log).

Stores: outputs/decisions.db (decisions, work_orders, agent_log,
recommendations tables). Core pipeline/app do not depend on this module.
"""
import json
import os
import pickle
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

import priority as prio_mod

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DB_PATH = os.path.join(OUT, "decisions.db")
API = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
EXCLUDED = ("311 INFORMATION ONLY CALL", "Aircraft Noise Complaint")
CAT = ["sr_type", "owner_department", "origin", "community_area", "ward", "zip_code"]


def _con():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
      CREATE TABLE IF NOT EXISTS decisions (sr_number TEXT, decision TEXT, ts TEXT);
      CREATE TABLE IF NOT EXISTS work_orders (id INTEGER PRIMARY KEY AUTOINCREMENT,
        sr_number TEXT, action TEXT, reason TEXT, created_by TEXT, ts TEXT);
      CREATE TABLE IF NOT EXISTS agent_log (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, stage TEXT, detail TEXT);
      CREATE TABLE IF NOT EXISTS recommendations (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, kind TEXT, target TEXT, reason TEXT, status TEXT,
        decided_ts TEXT, executed TEXT);
    """)
    return con


def log(stage, detail):
    con = _con()
    con.execute("INSERT INTO agent_log (ts, stage, detail) VALUES (?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), stage, str(detail)))
    con.commit(); con.close()


def get_log(limit=200):
    con = _con()
    df = pd.read_sql(f"SELECT * FROM agent_log ORDER BY id DESC LIMIT {limit}", con)
    con.close()
    return df


# --------------------------------------------------------------------- tools
def fetch_live_requests(hours=6, limit=500):
    # Same timezone-proof approach as app.fetch_live: window relative to
    # the data's own newest created_date, never the wall clock.
    excl = " AND ".join([f"sr_type != '{t}'" for t in EXCLUDED])
    params = {"$limit": 1500, "$order": "created_date DESC",
              "$where": excl,
              "$select": ("sr_number,sr_type,sr_short_code,owner_department,status,origin,"
                          "created_date,street_address,zip_code,community_area,ward,"
                          "duplicate,latitude,longitude,created_hour,created_day_of_week,created_month")}
    # Socrata intermittently returns 503 — retry up to 3 times before failing.
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise last_err
    df = pd.DataFrame(r.json())
    if not df.empty:
        cd = pd.to_datetime(df["created_date"], errors="coerce")
        df = df[cd >= cd.max() - pd.Timedelta(hours=hours)].head(limit).reset_index(drop=True)
    return df


def score_requests(df):
    if df.empty:
        return df
    with open(os.path.join(OUT, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    d = df.copy()
    d["created_date"] = pd.to_datetime(d["created_date"], errors="coerce")
    for c in ["created_hour", "created_day_of_week", "created_month"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["duplicate_flag"] = d["duplicate"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    for c in CAT:
        d[c] = d[c].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    d["breach_prob"] = model.predict_proba(
        d[CAT + ["created_hour", "created_day_of_week", "created_month", "duplicate_flag"]])[:, 1]
    hotspots = pd.read_csv(os.path.join(OUT, "hotspots.csv"))
    hs = {(str(r.community_area), r.sr_type): r.ratio for r in hotspots.itertuples()
          if pd.notna(r.ratio)}
    d["hotspot_ratio"] = d.apply(
        lambda r: hs.get((str(r["community_area"]), r["sr_type"]), 1.0), axis=1)
    d["dup_group_size"] = 1
    return prio_mod.compute_priority(d, mode="Balanced", now=pd.Timestamp.now())


def detect_duplicates(open_df=None):
    if open_df is None:
        open_df = pd.read_csv(os.path.join(OUT, "prioritized_open.csv"))
    clusters = (open_df[open_df["dup_group_size"] >= 3]
                .groupby("dup_cluster_id")
                .agg(n=("sr_number", "count"), sr_type=("sr_type", "first"),
                     address=("street_address", "first"),
                     top_sr=("sr_number", "first"),
                     max_priority=("priority_score", "max"))
                .reset_index().sort_values("n", ascending=False))
    return clusters


def detect_hotspots():
    h = pd.read_csv(os.path.join(OUT, "hotspots.csv"))
    return h[h["is_hotspot"] == True].sort_values("z_score", ascending=False)


def create_work_order(sr_number, action, reason, created_by="Ops Agent"):
    con = _con()
    con.execute("INSERT INTO work_orders (sr_number, action, reason, created_by, ts) "
                "VALUES (?,?,?,?,?)",
                (sr_number, action, reason, created_by,
                 datetime.now().isoformat(timespec="seconds")))
    con.commit(); con.close()
    log("execute", f"Work order created: {action} for {sr_number} — {reason}")


def update_citizen_status(cp_id, decision):
    """Writes to the SAME decisions store the citizen tracking page reads."""
    con = _con()
    con.execute("INSERT INTO decisions VALUES (?,?,?)",
                (cp_id, decision, datetime.now().isoformat(timespec="seconds")))
    con.commit(); con.close()
    log("execute", f"Citizen status updated: {cp_id} -> {decision}")


def draft_brief():
    m = json.load(open(os.path.join(OUT, "metrics.json")))
    q = pd.read_csv(os.path.join(OUT, "prioritized_open.csv"))
    hot = detect_hotspots()
    top = q.head(3)
    lines = [
        f"CITYPULSE DAILY OPS BRIEF — {datetime.now():%Y-%m-%d %H:%M}",
        f"Open requests in queue: {len(q):,} · High priority: "
        f"{int((q['risk_band'] == 'High').sum()):,} · Mean predicted breach "
        f"probability: {m['mean_open_breach_prob']}",
        f"Emerging hotspots: {len(hot)} area/type combinations "
        f"(top: area {hot.iloc[0]['community_area']} {hot.iloc[0]['sr_type']}, "
        f"{hot.iloc[0]['ratio']:.1f}x normal)" if len(hot) else "Emerging hotspots: none flagged",
        f"Forecast: model MAE {m['forecast']['mae_model']} vs naive "
        f"{m['forecast']['mae_naive_lag7']} requests/day (14-day holdout).",
        "Top of queue right now:",
    ]
    for _, r in top.iterrows():
        lines.append(f"  - {r['sr_number']} {r['sr_type']} @ {r['street_address']} "
                     f"(score {r['priority_score']:.2f}) — {r['reason']}")
    return "\n".join(lines)


# ------------------------------------------------------------- the cycle
def _recommend(kind, target, reason):
    con = _con()
    cur = con.execute("INSERT INTO recommendations (ts, kind, target, reason, status) "
                      "VALUES (?,?,?,?, 'Pending')",
                      (datetime.now().isoformat(timespec="seconds"), kind, target, reason))
    con.commit(); rid = cur.lastrowid; con.close()
    log("recommend", f"[{kind}] {target} — {reason}")
    return rid


def run_monitoring_cycle(live_hours=6):
    log("observe", f"Monitoring cycle started (live window {live_hours}h)")
    recs = []
    try:
        live = fetch_live_requests(hours=live_hours)
        log("observe", f"Fetched {len(live)} live requests from Socrata")
        scored = score_requests(live)
        if not scored.empty:
            log("analyse", f"Scored live requests; mean priority "
                           f"{scored['priority_score'].mean():.2f}, "
                           f"{int((scored['risk_band'] == 'High').sum())} high")
            for _, r in scored[scored["priority_score"] >= 0.6].head(5).iterrows():
                recs.append(_recommend(
                    "Escalate", r["sr_number"],
                    f"LIVE request scored {r['priority_score']:.2f} — {r['reason']}"))
    except Exception as e:
        log("observe", f"Live fetch failed ({e}); continuing with cached queue")

    clusters = detect_duplicates()
    log("analyse", f"Found {len(clusters)} duplicate clusters with 3+ reports")
    for _, c in clusters.head(3).iterrows():
        recs.append(_recommend(
            "Merge duplicates", c["top_sr"],
            f"{int(c['n'])} reports of {c['sr_type']} at {c['address']} — "
            f"merge into one work order to avoid duplicate crew visits"))

    hot = detect_hotspots()
    if len(hot):
        h0 = hot.iloc[0]
        log("analyse", f"{len(hot)} hotspots flagged; strongest: area "
                       f"{h0['community_area']} {h0['sr_type']} at {h0['ratio']:.1f}x normal")

    q = pd.read_csv(os.path.join(OUT, "prioritized_open.csv"))
    decided = set()
    if os.path.exists(DB_PATH):
        con = _con()
        decided = {r[0] for r in con.execute("SELECT sr_number FROM decisions").fetchall()}
        con.close()
    fresh = q[~q["sr_number"].isin(decided)].head(3)
    for _, r in fresh.iterrows():
        kind = "Approve citizen report" if str(r["sr_number"]).startswith("CP-") else "Approve"
        recs.append(_recommend(
            kind, r["sr_number"],
            f"Top of queue (score {r['priority_score']:.2f}) — {r['reason']}"))
    log("recommend", f"Cycle complete: {len(recs)} recommendations awaiting a manager")
    return recs


def list_recommendations(status=None):
    con = _con()
    q = "SELECT * FROM recommendations"
    if status:
        q += f" WHERE status='{status}'"
    df = pd.read_sql(q + " ORDER BY id DESC", con)
    con.close()
    return df


def decide_recommendation(rec_id, approve=True):
    con = _con()
    row = con.execute("SELECT kind, target, reason FROM recommendations WHERE id=?",
                      (rec_id,)).fetchone()
    if not row:
        con.close(); return False
    kind, target, reason = row
    status = "Approved" if approve else "Rejected"
    con.execute("UPDATE recommendations SET status=?, decided_ts=? WHERE id=?",
                (status, datetime.now().isoformat(timespec="seconds"), rec_id))
    con.commit(); con.close()
    log("human decision", f"Recommendation #{rec_id} ({kind} {target}) -> {status}")
    if approve:
        if kind == "Escalate":
            create_work_order(target, "Escalated work order", reason, "Ops Agent (approved)")
            if target.startswith("CP-"):
                update_citizen_status(target, "Escalated")
        elif kind == "Merge duplicates":
            create_work_order(target, "Merged duplicate cluster", reason, "Ops Agent (approved)")
        elif kind in ("Approve", "Approve citizen report"):
            create_work_order(target, "Approved work order", reason, "Ops Agent (approved)")
            if target.startswith("CP-"):
                update_citizen_status(target, "Approved")
    return True


def list_work_orders(limit=100):
    con = _con()
    df = pd.read_sql(f"SELECT * FROM work_orders ORDER BY id DESC LIMIT {limit}", con)
    con.close()
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "brief":
        print(draft_brief())
    else:
        ids = run_monitoring_cycle()
        print(f"Recommendations created: {ids}")
        print(draft_brief())
