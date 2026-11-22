"""
CityPulse AI - Chicago City Operations Center
Run:  streamlit run app.py
Reads cached outputs from train_model.py (no retraining at startup).
Live mode polls the Socrata API and scores brand-new requests with model.pkl.
"""
import json
import os
import pickle
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

import citizen
import priority as prio_mod

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")
API = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
DB_PATH = os.path.join(OUT, "decisions.db")
CAT = ["sr_type", "owner_department", "origin", "community_area", "ward", "zip_code"]
EXCLUDED = ("311 INFORMATION ONLY CALL", "Aircraft Noise Complaint")

st.set_page_config(page_title="CityPulse AI — Operations Center", layout="wide",
                   page_icon="🏙️", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background:#0b1220; color:#e6edf7; }
  section[data-testid="stSidebar"] { background:#0e1628; }
  .kpi { background:linear-gradient(135deg,#131f38,#0f1930); border:1px solid #22345c;
         border-radius:14px; padding:16px 18px; min-height:118px; }
  .kpi .label { color:#8fa8cc; font-size:12px; letter-spacing:.12em; text-transform:uppercase; }
  .kpi .value { color:#ffffff; font-size:32px; font-weight:700; line-height:1.15; }
  .kpi .delta { color:#7ee2a8; font-size:13px; }
  .kpi .delta.neg { color:#ff9c9c; }
  .hdr { font-size:30px; font-weight:800; color:#fff; margin-bottom:0; }
  .sub { color:#8fa8cc; margin-top:2px; }
  .badge-live { background:#b91c1c; color:#fff; padding:2px 8px; border-radius:8px;
                font-size:11px; font-weight:700; letter-spacing:.08em; }
  .badge-risk { background:#7f1d1d; color:#fecaca; padding:2px 8px; border-radius:8px; font-size:11px; }
  div[data-testid="stMetric"] { background:#131f38; border:1px solid #22345c; border-radius:12px; padding:10px; }
  .stDataFrame { border:1px solid #22345c; border-radius:10px; }
  h2, h3 { color:#dbe7fb !important; }
</style>
""", unsafe_allow_html=True)

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d8f2"))


# ------------------------------------------------------------------ decisions
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS decisions
                   (sr_number TEXT, decision TEXT, ts TEXT)""")
    return con


def add_decision(sr, decision):
    con = db()
    con.execute("INSERT INTO decisions VALUES (?,?,?)",
                (sr, decision, datetime.now().isoformat(timespec="seconds")))
    con.commit(); con.close()


def load_decisions():
    con = db()
    df = pd.read_sql("SELECT * FROM decisions ORDER BY ts DESC", con)
    con.close()
    return df


# ---------------------------------------------------------------------- loads
@st.cache_data(show_spinner=False)
def load_csv(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(os.path.join(OUT, "metrics.json")) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model():
    with open(os.path.join(OUT, "model.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_raw(name):
    p = os.path.join(DATA, name + ".parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.read_csv(os.path.join(DATA, name + ".csv"), low_memory=False)


metrics = load_metrics()
prio = load_csv("prioritized_open.csv")
hotspots = load_csv("hotspots.csv")
area_hot = load_csv("area_hotspots.csv")
forecast = load_csv("forecast.csv")
calibration = load_csv("calibration.csv")
featimp = load_csv("feature_importance.csv")
fairness = load_csv("fairness.csv")
type_stats = load_csv("type_stats.csv")
daily_vol = load_csv("daily_volume.csv")
severity_table = load_csv("severity_table.csv")

if prio.empty:
    st.error("No outputs found. Run `python ingest.py` then `python train_model.py` first.")
    st.stop()

prio["created_date"] = pd.to_datetime(prio["created_date"])
prio["is_live"] = False
decisions = load_decisions()
decided_srs = set(decisions["sr_number"]) if not decisions.empty else set()

# ------------------------------------------------------------------ live mode
def fetch_live(hours=6, limit=800):
    # NOTE: Socrata created_date values are Chicago local time and the feed
    # lags real time, so filtering by wall clock returns 0 rows on a Chicago
    # laptop. Fetch the newest records and window them relative to the
    # data's own newest timestamp instead — timezone-proof.
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


def score_live(df):
    if df.empty:
        return df
    model = load_model()
    d = df.copy()
    d["created_date"] = pd.to_datetime(d["created_date"], errors="coerce")
    for c in ["created_hour", "created_day_of_week", "created_month"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["duplicate_flag"] = d["duplicate"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    for c in CAT:
        d[c] = d[c].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
    d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
    d["breach_prob"] = model.predict_proba(d[CAT + ["created_hour", "created_day_of_week",
                                                   "created_month", "duplicate_flag"]])[:, 1]
    hs = {(str(r.community_area), r.sr_type): r.ratio for r in hotspots.itertuples()
          if pd.notna(r.ratio)} if not hotspots.empty else {}
    d["hotspot_ratio"] = d.apply(lambda r: hs.get((str(r["community_area"]), r["sr_type"]), 1.0), axis=1)
    d["dup_group_size"] = 1
    d = prio_mod.compute_priority(d, mode="Balanced", now=pd.Timestamp.now())
    d["reason"] = "LIVE · " + d["reason"]
    d["is_live"] = True
    d["source"] = "LIVE (311 API)"
    keep = ["sr_number", "sr_type", "sr_short_code", "owner_department", "status", "origin",
            "created_date", "street_address", "zip_code", "community_area", "ward",
            "duplicate_flag", "latitude", "longitude", "breach_prob", "hotspot_ratio",
            "hotspot_norm", "dup_norm", "dup_group_size", "severity_tier", "severity_score",
            "wait_days", "wait_norm", "priority_score", "risk_band",
            "reason", "is_live", "source"]
    for c in keep:
        if c not in d.columns:
            d[c] = np.nan
    return d[keep].sort_values("priority_score", ascending=False)


if "live_df" not in st.session_state:
    st.session_state.live_df = pd.DataFrame()
    st.session_state.live_at = None

# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    dept = st.multiselect("Department", sorted(prio["owner_department"].dropna().unique()))
    types = st.multiselect("Request type", sorted(prio["sr_type"].dropna().unique()))
    areas = st.multiselect("Community area", sorted(prio["community_area"].dropna().astype(str).unique()))
    wards = st.multiselect("Ward", sorted(prio["ward"].dropna().astype(str).unique()))
    bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    dmin, dmax = prio["created_date"].min().date(), prio["created_date"].max().date()
    drange = st.date_input("Created date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    live_only = st.toggle("🔴 LIVE only", value=False)
    search = st.text_input("Search SR # / street", "")
    st.markdown("---")
    st.markdown("### ⚖️ Priority mode")
    mode = st.radio("How should the queue be ordered?",
                    ["Balanced", "FIFO-first", "Urgency-first"],
                    help="Re-weights the blend live: delay risk, severity, waiting time "
                         "(FIFO), hotspot and repeat reports. Balanced is the default "
                         "CityPulse recommendation.")
    w = prio_mod.PRIORITY_MODES[mode]
    st.caption(" · ".join([f"{k} {v:.0%}" for k, v in
                            [("Risk", w["breach"]), ("Severity", w["severity"]),
                             ("Wait/FIFO", w["wait"]), ("Hotspot", w["hotspot"]),
                             ("Reports", w["dup"])]]))
    sources_sel = st.multiselect("Source", ["Official 311", "Citizen Report (CityPulse)", "LIVE (311 API)"],
                                 default=["Official 311", "Citizen Report (CityPulse)", "LIVE (311 API)"])
    st.markdown("---")
    st.markdown("### 🗺️ Map layers")
    lyr_risk = st.checkbox("High-priority requests", value=True)
    lyr_hot = st.checkbox("Hotspot areas", value=True)
    lyr_dup = st.checkbox("Duplicate clusters", value=False)
    lyr_cit = st.checkbox("Citizen reports", value=True)
    st.markdown("---")
    if st.button("🔴 Fetch live requests now", use_container_width=True):
        try:
            with st.spinner("Polling Chicago 311 API…"):
                st.session_state.live_df = score_live(fetch_live())
                st.session_state.live_at = datetime.now()
            st.success(f"Scored {len(st.session_state.live_df)} live requests.")
        except Exception as e:
            st.warning(f"Live fetch failed ({e}). Showing cached data.")
    if st.session_state.live_at:
        st.caption(f"Last live fetch: {st.session_state.live_at:%Y-%m-%d %H:%M:%S} · "
                   f"{len(st.session_state.live_df)} requests")
    st.markdown("---")
    with st.expander("ℹ️ How priority works"):
        st.markdown(
            "**Priority is a blend, not just the model score:**\n\n"
            "- **Delay risk (30%)**: model probability this request takes > 7 days to close, "
            "from features known at creation time only.\n"
            "- **Severity (25%)**: hand-set tier per request type (safety / public impact) — "
            "policy, not learned from data. Full table under Model & Evaluation.\n"
            "- **Waiting time / FIFO (20%)**: days already open, capped at 30, so old "
            "requests steadily rise and nothing starves forever.\n"
            "- **Hotspot (15%)**: how far above normal this type is running in its "
            "community area (last 7 days vs prior 28-day average).\n"
            "- **People affected (10%)**: repeat reports at the same address within 14 days.\n\n"
            "Use **Priority mode** above to re-weight the blend (FIFO-first / Urgency-first). "
            "A human approves, rejects or escalates every recommendation. "
            "This is prioritisation, not dispatch — real crew constraints are not modelled.")
    with st.expander("🚦 Severity tiers (hand-set policy)"):
        st.markdown("Critical = traffic signal out, tree emergency, water leak / no water, "
                    "sewer or pavement cave-in, open hydrant. High = potholes, streetlight "
                    "outages, rodent / sanitation hazards, building violations. Medium = "
                    "carts, missed pickups, tree and street services. Low = graffiti and "
                    "administrative / cosmetic issues. Editable in `priority.py`.")

base_queue = prio.copy()
try:
    reps = citizen.load_reports()
    if not reps.empty:
        for c in base_queue.columns:
            if c not in reps.columns:
                reps[c] = np.nan
        base_queue = pd.concat([base_queue, reps[base_queue.columns]], ignore_index=True)
except Exception:
    pass

if live_only:
    queue = st.session_state.live_df.copy() if not st.session_state.live_df.empty else base_queue.iloc[0:0]
elif not st.session_state.live_df.empty:
    queue = pd.concat([st.session_state.live_df, base_queue], ignore_index=True)
else:
    queue = base_queue

# Re-weight the blend live according to the selected priority mode (cached
# per mode + data shape so widget interactions stay responsive).
_cache_key = (mode, len(queue), int(queue["priority_score"].sum() * 1000) if len(queue) else 0)
if st.session_state.get("blend_key") != _cache_key:
    queue = prio_mod.compute_priority(queue, mode=mode, now=pd.Timestamp.now())
    st.session_state.blend_key = _cache_key
    st.session_state.blend_queue = queue
else:
    queue = st.session_state.blend_queue
if sources_sel:
    queue = queue[queue["source"].isin(sources_sel)]
if dept:
    queue = queue[queue["owner_department"].isin(dept)]
if types:
    queue = queue[queue["sr_type"].isin(types)]
if areas:
    queue = queue[queue["community_area"].astype(str).isin(areas)]
if wards:
    queue = queue[queue["ward"].astype(str).isin(wards)]
if bands:
    queue = queue[queue["risk_band"].astype(str).isin(bands)]
if len(drange) == 2:
    queue = queue[(queue["created_date"].dt.date >= drange[0]) &
                  (queue["created_date"].dt.date <= drange[1])]
if search:
    s = search.lower()
    queue = queue[queue["sr_number"].str.lower().str.contains(s) |
                  queue["street_address"].astype(str).str.lower().str.contains(s)]
queue = queue.sort_values("priority_score", ascending=False)

# --------------------------------------------------------------------- header
newest = prio["created_date"].max()
st.markdown('<div class="hdr">🏙️ CityPulse AI — Chicago Operations Center</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub">Live 311 prioritisation · Data as of {metrics["data_as_of"]} · '
            f'Newest request in data: {newest:%Y-%m-%d %H:%M} · Model: {metrics["model_name"]} · '
            f'SLA proxy: breach = &gt; {7} days to close</div>', unsafe_allow_html=True)

today_dec = 0
if not decisions.empty:
    today_dec = int((pd.to_datetime(decisions["ts"], format="mixed", errors="coerce").dt.date == datetime.now().date()).sum())
last24 = int((prio["created_date"] >= newest - pd.Timedelta(days=1)).sum())
kpis = [
    ("Open requests (cached)", f"{len(prio):,}", f"+{last24:,} created in last 24h of data"),
    ("High-risk queue", f"{metrics['high_risk_open']:,}",
     f"{metrics['high_risk_open']/max(len(prio),1)*100:.1f}% of open requests"),
    ("Emerging hotspots", f"{metrics['hotspots_flagged']}", "area × type, ≥1.5× normal, z ≥ 2"),
    ("Avg predicted breach prob", f"{metrics['mean_open_breach_prob']:.2f}",
     f"historical breach rate {metrics['overall_breach_rate']:.2f}"),
    ("Decisions made today", f"{today_dec}", f"{len(decisions)} total in log"),
]
cols = st.columns(len(kpis))
for c, (label, value, delta) in zip(cols, kpis):
    c.markdown(f'<div class="kpi"><div class="label">{label}</div>'
               f'<div class="value">{value}</div><div class="delta">{delta}</div></div>',
               unsafe_allow_html=True)
st.markdown("")

tab_ops, tab_hot, tab_model, tab_live, tab_report, tab_agent, tab_kb = st.tabs(
    ["🗂️ Operations Center", "🔥 Hotspots & Forecast", "🤖 Model & Evaluation", "🔴 Live Feed",
     "📝 Report a Problem", "🤖 Ops Agent", "💬 Knowledge Assistant"])

# ============================================================ OPERATIONS TAB
with tab_ops:
    map_df = queue.dropna(subset=["latitude", "longitude"]).head(3000)
    layers = []
    if lyr_risk and not map_df.empty:
        md = map_df.copy()
        md["color"] = md["risk_band"].astype(str).map(
            {"High": [255, 70, 70, 190], "Medium": [255, 180, 60, 160],
             "Low": [90, 160, 255, 140]}).tolist()
        layers.append(pdk.Layer("ScatterplotLayer", md, get_position="[longitude, latitude]",
                                get_fill_color="color", get_radius=90, pickable=True,
                                auto_highlight=True))
    if lyr_dup:
        dup_df = queue[(queue["dup_group_size"] > 1)].dropna(subset=["latitude", "longitude"])
        if not dup_df.empty:
            layers.append(pdk.Layer("ScatterplotLayer", dup_df,
                                    get_position="[longitude, latitude]",
                                    get_fill_color=[190, 90, 255, 200], get_radius=160,
                                    pickable=True))
    if lyr_cit:
        cit_df = queue[queue["source"] == "Citizen Report (CityPulse)"].dropna(subset=["latitude", "longitude"])
        if not cit_df.empty:
            layers.append(pdk.Layer("ScatterplotLayer", cit_df,
                                    get_position="[longitude, latitude]",
                                    get_fill_color=[60, 220, 130, 220], get_radius=170,
                                    pickable=True))
    if lyr_hot and not area_hot.empty:
        ah = area_hot.dropna(subset=["latitude", "longitude"]).copy()
        ah = ah[ah["ratio"] >= 1.3]
        if not ah.empty:
            layers.append(pdk.Layer("ScatterplotLayer", ah,
                                    get_position="[longitude, latitude]",
                                    get_fill_color=[255, 120, 40, 150],
                                    get_radius="count_7d * 6", pickable=True))
    if layers:
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(
            latitude=41.84, longitude=-87.68, zoom=10, pitch=35),
            map_style="dark",
            tooltip={"text": "{sr_number}\n{sr_type}\n{street_address}\n"
                             "Priority {priority_score} · {reason}"}), use_container_width=True)
    else:
        st.info("No map layers selected or no geocoded requests match the filters.")

    st.subheader(f"Priority queue — {len(queue):,} requests after filters")
    show_cols = ["sr_number", "sr_type", "owner_department", "street_address",
                 "community_area", "ward", "created_date", "priority_score",
                 "breach_prob", "dup_group_size", "risk_band", "source", "is_live", "reason"]
    st.dataframe(queue[show_cols].head(200), use_container_width=True, height=380,
                 column_config={
                     "priority_score": st.column_config.ProgressColumn("Priority", format="%.2f", min_value=0, max_value=1),
                     "breach_prob": st.column_config.NumberColumn("Delay risk", format="%.2f"),
                     "created_date": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD HH:mm"),
                     "is_live": st.column_config.CheckboxColumn("LIVE")})

    # ------------------------------------------------------------- drill-down
    st.subheader("🔍 Request drill-down")
    options = queue["sr_number"].head(500).tolist()
    if not options:
        st.info("No requests match the current filters — widen them to use the drill-down.")
    else:
        sel = st.selectbox("Select a request (queue order)", options)
        row = queue[queue["sr_number"] == sel]
        if row.empty:
            row = base_queue[base_queue["sr_number"] == sel]
        if row.empty and not st.session_state.live_df.empty:
            row = st.session_state.live_df[st.session_state.live_df["sr_number"] == sel]
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3 = st.columns([1.1, 1.4, 1.5])
            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=float(r["priority_score"]) * 100,
                    title={"text": "Priority score"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "#ff4646" if r["priority_score"] > 0.6 else "#ffb43c"},
                           "bgcolor": "#131f38"}))
                fig.update_layout(**PLOT, height=240, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
                src_badge = ("🟢 **Citizen Report (CityPulse)** — filed in this app, not official 311"
                             if str(r.get("source", "")) == "Citizen Report (CityPulse)"
                             else f"Source: {r.get('source', 'Official 311')}")
                st.markdown(f"**{r['sr_type']}**  \n{r['street_address']} · Area {r['community_area']} · Ward {r['ward']}  \n"
                            f"Dept: {r['owner_department']} · Origin: {r['origin']}  \n"
                            f"Created: {pd.to_datetime(r['created_date']):%Y-%m-%d %H:%M} · Status: {r['status']}  \n{src_badge}")
                if sel in decided_srs:
                    last = decisions[decisions["sr_number"] == sel].iloc[0]
                    st.success(f"Decision on file: **{last['decision']}** at {last['ts']}")
                b1, b2, b3 = st.columns(3)
                if b1.button("✅ Approve", key=f"ap_{sel}", use_container_width=True):
                    add_decision(sel, "Approved"); st.rerun()
                if b2.button("🚫 Reject", key=f"rj_{sel}", use_container_width=True):
                    add_decision(sel, "Rejected"); st.rerun()
                if b3.button("⬆️ Escalate", key=f"es_{sel}", use_container_width=True):
                    add_decision(sel, "Escalated"); st.rerun()
            with c2:
                st.markdown(f"**Why this score?** (mode: {mode})")
                contrib = pd.DataFrame({
                    "Component": [f"Delay risk (×{w['breach']:.2f})", f"Severity {r.get('severity_tier','')} (×{w['severity']:.2f})",
                                  f"Waiting {float(r.get('wait_days',0)):.0f}d / FIFO (×{w['wait']:.2f})",
                                  f"Hotspot (×{w['hotspot']:.2f})", f"Repeat reports (×{w['dup']:.2f})"],
                    "Contribution": [w["breach"] * float(r["breach_prob"]),
                                     w["severity"] * float(r.get("severity_score", 0)),
                                     w["wait"] * float(r.get("wait_norm", 0)),
                                     w["hotspot"] * float(r.get("hotspot_norm", 0)),
                                     w["dup"] * float(r.get("dup_norm", 0))]})
                fig2 = px.bar(contrib, x="Contribution", y="Component", orientation="h",
                              range_x=[0, 0.45], color="Component", text="Contribution")
                fig2.update_traces(texttemplate="%{x:.2f}")
                fig2.update_layout(**PLOT, height=220, showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)
                st.info(r["reason"])
                if int(r.get("dup_group_size", 1)) > 1 and "dup_cluster_id" in prio.columns:
                    members = prio[prio["dup_cluster_id"] == r["dup_cluster_id"]]
                    st.markdown(f"**Duplicate group — {len(members)} reports at this address**")
                    st.dataframe(members[["sr_number", "sr_type", "created_date", "origin", "duplicate_flag"]],
                                 use_container_width=True, height=150)
            with c3:
                st.markdown("**Area trend — daily requests (all cached data)**")
                try:
                    closed_raw = load_raw("closed_requests")
                    open_raw = load_raw("open_requests")
                    all_raw = pd.concat([
                        closed_raw[["created_date", "community_area"]],
                        open_raw[["created_date", "community_area"]]], ignore_index=True)
                    all_raw["created_date"] = pd.to_datetime(all_raw["created_date"])
                    trend = (all_raw[all_raw["community_area"].astype(str) == str(r["community_area"])]
                             .set_index("created_date").resample("D").size().rename("requests")
                             .reset_index().tail(45))
                    fig3 = px.area(trend, x="created_date", y="requests")
                    fig3.update_layout(**PLOT, height=200, margin=dict(t=10, b=10),
                                       xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.caption(f"Area trend unavailable ({e})")
                st.markdown("**Similar past requests — same type + area, actual outcomes**")
                try:
                    closed_raw = load_raw("closed_requests")
                    cr = closed_raw.copy()
                    cr["created_date"] = pd.to_datetime(cr["created_date"])
                    cr["closed_date"] = pd.to_datetime(cr["closed_date"])
                    cr["days"] = (cr["closed_date"] - cr["created_date"]).dt.total_seconds() / 86400
                    sim = cr[(cr["sr_type"] == r["sr_type"]) &
                             (cr["community_area"].astype(str) == str(r["community_area"]))]
                    if len(sim):
                        st.markdown(f"n = {len(sim):,} · median **{sim['days'].median():.1f} days** · "
                                    f"{(sim['days'] > 7).mean()*100:.0f}% took > 7 days")
                        st.dataframe(sim.sort_values("created_date", ascending=False)
                                     [["sr_number", "created_date", "closed_date", "days"]]
                                     .head(6), use_container_width=True, height=170)
                    else:
                        st.caption("No similar completed requests in cache.")
                except Exception as e:
                    st.caption(f"Similar-request lookup unavailable ({e})")

    # ------------------------------------------------------- quick actions row
    with st.expander("⚡ Quick actions — top 10 in queue"):
        for _, r in queue.head(10).iterrows():
            a, b, c, d, e = st.columns([3.2, 1.6, 1, 1, 1])
            a.markdown(f"**{r['sr_number']}** · {r['sr_type']}  \n"
                       f"<span style='color:#8fa8cc;font-size:12px'>{r['street_address']} · score {r['priority_score']:.2f}</span>",
                       unsafe_allow_html=True)
            b.markdown(f"<span style='color:#8fa8cc;font-size:12px'>{r['reason']}</span>", unsafe_allow_html=True)
            if c.button("✅", key=f"qa_ap_{r['sr_number']}"):
                add_decision(r["sr_number"], "Approved"); st.rerun()
            if d.button("🚫", key=f"qa_rj_{r['sr_number']}"):
                add_decision(r["sr_number"], "Rejected"); st.rerun()
            if e.button("⬆️", key=f"qa_es_{r['sr_number']}"):
                add_decision(r["sr_number"], "Escalated"); st.rerun()

    # ------------------------------------------------------------- decision log
    st.subheader("📋 Decision log")
    decisions = load_decisions()
    if decisions.empty:
        st.info("No decisions yet. Approve, reject or escalate a request above and it will be logged here.")
    else:
        counts = decisions["decision"].value_counts()
        st.markdown(" · ".join([f"**{k}: {v}**" for k, v in counts.items()]))
        st.dataframe(decisions.head(100), use_container_width=True, height=220)
        st.download_button("⬇️ Export decision log (CSV)", decisions.to_csv(index=False),
                           file_name="citypulse_decisions.csv", mime="text/csv")

# ============================================================ HOTSPOTS TAB
with tab_hot:
    st.subheader("Emerging hotspots — last 7 days vs prior 28-day average")
    if not hotspots.empty:
        flagged = hotspots[hotspots["is_hotspot"] == True] if "is_hotspot" in hotspots else hotspots.head(0)
        if len(flagged):
            top = flagged.head(15)
            fig = px.bar(top, x="z_score", y=top["community_area"].astype(str) + " · " + top["sr_type"],
                         orientation="h", color="ratio", title="Top emerging hotspots (z-score)")
            fig.update_layout(**PLOT, height=460, yaxis=dict(autorange="reversed"),
                              yaxis_title="", xaxis_title="z-score vs expected volume")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(flagged[["community_area", "sr_type", "count_7d", "expected_7d",
                                  "ratio", "z_score"]].head(50),
                         use_container_width=True, height=320)
        else:
            st.info("No area × type combination currently meets the hotspot rule "
                    "(≥20 requests in 7 days, ≥1.5× normal, z ≥ 2).")
    st.subheader("Request volume — history and next 7 days")
    if not daily_vol.empty and not forecast.empty:
        dv = daily_vol.copy()
        dv["created_date"] = pd.to_datetime(dv["created_date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dv["created_date"].tail(90), y=dv["volume"].tail(90),
                                 mode="lines", name="Actual daily volume"))
        fc = forecast.copy()
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc["date"]), y=fc["forecast"],
                                 mode="lines+markers", name="Forecast", line=dict(color="#ffb43c")))
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc["date"]), y=fc["upper"], mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc["date"]), y=fc["lower"], mode="lines",
                                 line=dict(width=0), fill="tonexty", name="~80% interval"))
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc["date"]), y=fc["naive_lag7"],
                                 mode="lines", name="Naive (same day last week)",
                                 line=dict(dash="dot", color="#8fa8cc")))
        fig.update_layout(**PLOT, height=380)
        st.plotly_chart(fig, use_container_width=True)
        fm = metrics["forecast"]
        st.markdown(f"Holdout ({fm['holdout_start']} → {fm['holdout_end']}, {fm['holdout_days']} days): "
                    f"**model MAE {fm['mae_model']}** vs naive lag-7 MAE **{fm['mae_naive_lag7']}** requests/day.")
        st.dataframe(forecast, use_container_width=True)
    if not type_stats.empty:
        st.subheader("Median days-to-close by request type (historical, top 20 by volume)")
        ts = type_stats.head(20).sort_values("median_days")
        fig = px.bar(ts, x="median_days", y="sr_type", orientation="h", color="breach_rate",
                     color_continuous_scale="Reds")
        fig.update_layout(**PLOT, height=480, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================== MODEL TAB
with tab_model:
    st.subheader(f"Delay-risk model — {metrics['model_name']}")
    st.markdown(f"Target: **{metrics['sla_definition']}**. Features are knowable at creation time only "
                f"(type, department, origin, area, ward, zip, hour, weekday, month, duplicate flag). "
                f"Temporal split at {metrics['split_cutoff']}: {metrics['train_rows']:,} train / "
                f"{metrics['test_rows']:,} test requests.")
    comp = pd.DataFrame([
        {"Model": metrics["model_name"], **metrics["model"]},
        {"Model": "Baseline — always predict overall breach rate", **metrics["baseline_global_rate"]},
        {"Model": "Baseline — per-type historical breach rate", **metrics["baseline_type_only"]},
    ])
    st.dataframe(comp, use_container_width=True)
    st.caption("AUROC / AUPRC higher is better · Brier score lower is better.")
    c1, c2 = st.columns(2)
    with c1:
        if not calibration.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=calibration["mean_pred"], y=calibration["actual_rate"],
                                     mode="lines+markers", name="Model"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                                     line=dict(dash="dot", color="#8fa8cc")))
            fig.update_layout(**PLOT, height=340, title="Calibration (predicted vs actual, by decile)",
                              xaxis_title="Mean predicted probability", yaxis_title="Actual breach rate")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if not featimp.empty:
            src = featimp[featimp["level"] == "source_feature"].sort_values("importance")
            if len(src):
                fig = px.bar(src, x="importance", y="feature", orientation="h",
                             title="Feature importance (grouped)")
                fig.update_layout(**PLOT, height=340, yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
    st.subheader("Risk distribution — open queue")
    fig = px.histogram(prio, x="breach_prob", nbins=40, color="risk_band",
                       color_discrete_map={"High": "#ff4646", "Medium": "#ffb43c", "Low": "#5aa0ff"})
    fig.update_layout(**PLOT, height=300, xaxis_title="Predicted breach probability", yaxis_title="Requests")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which order should a crew work? — three strategies compared")
    sc = metrics["strategy_comparison"]
    strat_df = pd.DataFrame([
        {"Strategy": "FIFO (first-come, first-served)", **sc["fifo"]},
        {"Strategy": "Model risk only", **sc["model_only"]},
        {"Strategy": "CityPulse blended (recommended)", **sc["blended"]},
    ]).rename(columns={"mean_days_to_close_topk": "Avg days-to-close of top-K",
                       "breach_share_in_topk": "Share of top-K that breached >7d",
                       "breaches_captured_topk": "Breaches captured in top-K",
                       "breach_recall_topk": "Recall of all test breaches"})
    st.markdown(f"Scenario: on the last day of the test window, a manager can start "
                f"**K = {sc['k']:,}** jobs from the backlog ({sc['total_breaches_in_test']:,} of the "
                f"test requests eventually breached the 7-day proxy). Which order puts the "
                f"eventual breaches first?")
    st.dataframe(strat_df, use_container_width=True)
    fig = go.Figure(data=[
        go.Bar(name="FIFO", x=["Share of top-K that breached"], y=[sc["fifo"]["breach_share_in_topk"]]),
        go.Bar(name="Model only", x=["Share of top-K that breached"], y=[sc["model_only"]["breach_share_in_topk"]]),
        go.Bar(name="Blended (recommended)", x=["Share of top-K that breached"], y=[sc["blended"]["breach_share_in_topk"]])])
    fig.update_layout(**PLOT, height=280, barmode="group", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(sc["note"])

    st.subheader("Severity tiers — hand-set policy (not learned from data)")
    st.markdown("Severity is a policy judgement about safety and public impact, set by hand in "
                "`priority.py` (`SEVERITY_RULES`) and editable there. Critical 1.00 · High 0.75 · "
                "Medium 0.50 · Low 0.25. Pothole in Street is **High** — and the data backs the "
                f"attention: potholes took a median of **{metrics['pothole']['median_days']} days** to "
                f"close and **{metrics['pothole']['breach_rate']*100:.0f}%** breached the 7-day proxy "
                f"({metrics['pothole']['closed_n']:,} completed in the data).")
    if not severity_table.empty:
        st.dataframe(severity_table.sort_values(["severity_score", "sr_type"],
                                                ascending=[False, True]),
                     use_container_width=True, height=300)

    st.subheader("Fairness check — community areas")
    fsx = metrics["fairness"]
    st.markdown(f"Across {fsx['areas_compared']} community areas with ≥30 open requests, mean priority "
                f"ranges from **{fsx['mean_priority_min']}** (area {fsx['lowest_priority_area']}) to "
                f"**{fsx['mean_priority_max']}** (area {fsx['highest_priority_area']}) — a spread of "
                f"**{fsx['mean_priority_spread']}**. Historical median days-to-close by area ranges "
                f"{fsx['median_days_min']}–{fsx['median_days_max']} days. "
                f"{fsx['note']} Areas with genuinely slower historical service score higher priority, "
                f"which is the intended direction, but the spread should be monitored so no area is "
                f"systematically left at the back of the queue.")
    if not fairness.empty:
        f2 = fairness.sort_values("mean_priority")
        fig = px.bar(f2, x="mean_priority", y=f2["community_area"].astype(str), orientation="h",
                     color="median_days", title="Mean priority score by community area")
        fig.update_layout(**PLOT, height=520, yaxis_title="Community area")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fairness, use_container_width=True, height=300)
    st.markdown("---")
    dm = metrics["duplicates"]
    st.markdown(f"**Duplicates:** {dm['open_in_duplicate_groups']:,} open requests sit in "
                f"{dm['duplicate_clusters']:,} duplicate clusters (same type + address, ≤14 days apart). "
                f"The city's own duplicate flag is set on {dm['city_flag_rate_within_our_groups']*100:.0f}% "
                f"of the requests inside our clusters (and {dm['city_flag_rate_overall_open']*100:.1f}% of all open requests).")

# ================================================================ LIVE TAB
with tab_live:
    st.subheader("🔴 Live feed — newest Chicago 311 requests, scored on arrival")
    st.markdown("Polls the Socrata API for requests created in the last 6 hours and scores them "
                "immediately with the trained model. The source does not guarantee instant updates, "
                "so timestamps shown are the city's own `created_date` values.")
    auto = st.toggle("Auto-refresh every 2 minutes", value=False)

    @st.fragment(run_every=120 if auto else None)
    def live_panel():
        if st.button("🔴 Fetch live requests now", key="live_btn"):
            try:
                with st.spinner("Polling Chicago 311 API…"):
                    st.session_state.live_df = score_live(fetch_live())
                    st.session_state.live_at = datetime.now()
            except Exception as e:
                st.warning(f"Live fetch failed ({e}). Showing last successful fetch, if any.")
        lv = st.session_state.live_df
        if st.session_state.live_at:
            st.caption(f"Last fetched {st.session_state.live_at:%Y-%m-%d %H:%M:%S} · {len(lv)} requests scored")
        if lv is None or lv.empty:
            st.info("No live requests fetched yet in this session — hit the fetch button.")
            return
        newest_live = pd.to_datetime(lv["created_date"]).max()
        st.markdown(f"Newest live request: **{newest_live:%Y-%m-%d %H:%M}** · "
                    f"High-risk among live: **{int((lv['risk_band'].astype(str) == 'High').sum())}** · "
                    f"Mean delay risk: **{lv['breach_prob'].mean():.2f}**")
        st.dataframe(lv[["sr_number", "sr_type", "street_address", "community_area",
                         "created_date", "priority_score", "breach_prob", "risk_band", "reason"]],
                     use_container_width=True, height=420,
                     column_config={"priority_score": st.column_config.ProgressColumn(
                         "Priority", format="%.2f", min_value=0, max_value=1)})
        lmap = lv.dropna(subset=["latitude", "longitude"])
        if not lmap.empty:
            st.pydeck_chart(pdk.Deck(
                layers=[pdk.Layer("ScatterplotLayer", lmap.head(1000),
                                  get_position="[longitude, latitude]",
                                  get_fill_color=[255, 60, 60, 190], get_radius=110, pickable=True)],
                initial_view_state=pdk.ViewState(latitude=41.84, longitude=-87.68, zoom=10),
                map_style="dark",
                tooltip={"text": "{sr_number}\n{sr_type}\n{street_address}\nPriority {priority_score}"}),
                use_container_width=True)

    live_panel()

# ============================================================ CITIZEN TAB
with tab_report:
    st.subheader("📝 Report a Problem")
    st.warning("This files your complaint inside **CityPulse's local system for demo purposes**. "
               "The Chicago 311 Socrata API is read-only, so this does **NOT** submit anything to "
               "the City of Chicago's official 311. To file an official complaint, use the City's "
               "311 service.")
    type_options = ["Pothole in Street Complaint"] + \
        [t for t in type_stats.sort_values("n", ascending=False)["sr_type"].tolist()
         if t != "Pothole in Street Complaint"] if not type_stats.empty else ["Pothole in Street Complaint"]
    with st.form("citizen_form"):
        f1, f2 = st.columns(2)
        c_type = f1.selectbox("Problem type", type_options,
                              help="Pothole is listed first — it is one of Chicago's slowest "
                                   "request types (see Model & Evaluation).")
        c_addr = f1.text_input("Street address *", placeholder="e.g. 2517 W FULLERTON AVE")
        c_zip = f1.text_input("ZIP code *", placeholder="e.g. 60647")
        c_area = f2.text_input("Community area (optional, number 1-77)", "")
        c_ward = f2.text_input("Ward (optional)", "")
        c_urg = f2.selectbox("How urgent does it feel to you? (optional)",
                             ["", "Low", "Medium", "High", "Safety hazard right now"])
        c_desc = st.text_area("Description (optional)", placeholder="What is the problem, exactly where, since when?")
        c_name = st.text_input("Your name (optional)", "")
        c_contact = st.text_input("Email or phone (optional)", "")
        submitted = st.form_submit_button("📨 Submit complaint to CityPulse")
    if submitted:
        try:
            rep = citizen.submit_report(
                c_type, c_addr, c_zip,
                community_area=c_area or None, ward=c_ward or None,
                description=c_desc, urgency=c_urg,
                reporter_name=c_name, reporter_contact=c_contact)
            st.session_state.last_receipt = rep
            st.success(f"✅ Complaint received — your Complaint ID is **{rep['sr_number']}**. "
                       "Save it to track your complaint below.")
        except Exception as e:
            st.error(f"Could not file the complaint: {e}")
    rep = st.session_state.get("last_receipt")
    if rep:
        st.markdown("### 🧾 Your receipt")
        rc1, rc2 = st.columns([1.3, 1])
        with rc1:
            st.markdown(f"""
| | |
|---|---|
| **Complaint ID** | {rep['sr_number']} |
| **Filed** | {rep['created_date']} |
| **Problem** | {rep['sr_type']} |
| **Address** | {rep['street_address']}, Chicago, IL {rep['zip_code']} |
| **Department** | {rep['owner_department']} |
| **Priority score** | {rep['priority_score']} ({rep['risk_band']}) |
| **Queue position** | #{rep['queue_position']} of {rep['queue_total']} open requests |
| **Status** | {rep['status_label']} |
""")
            st.info(f"Why this priority: {rep['reason']}")
            if rep.get("location_approx"):
                st.caption("Map location is approximate: the median location of real requests "
                           "with the same ZIP code — not a precise pin for your address.")
        with rc2:
            st.download_button("⬇️ Download receipt (.txt)", citizen.receipt_text(rep),
                               file_name=f"{rep['sr_number']}_receipt.txt", mime="text/plain")
            st.markdown("Your complaint now appears in the **Operations Center** queue and map, "
                        "badged **Citizen Report (CityPulse)**. The operations team reviews it "
                        "alongside official 311 requests.")
    st.markdown("---")
    st.subheader("🔎 Track your complaint")
    track_id = st.text_input("Enter your Complaint ID", placeholder="CP-2026-0001", key="track_id")
    if st.button("Track", key="track_btn"):
        info = citizen.tracking(track_id) if track_id.strip() else None
        if info is None:
            st.error("No complaint found with that ID in CityPulse's local system.")
        else:
            st.markdown(f"**{info['sr_number']}** — {info['sr_type']} at {info['street_address']}")
            st.markdown(f"Status: **{info['status_label']}** · Priority score: "
                        f"**{info['priority_score']}** ({info['risk_band']}) · "
                        f"Last updated: {info['last_updated']}")
            st.caption(f"Why this priority: {info['reason']}")

# ============================================================== AGENT TAB
with tab_agent:
    st.subheader("🤖 CityPulse Ops Agent")
    st.markdown("The agent runs a monitoring cycle with real tools — fetch live requests → score "
                "them with the model → detect duplicates and hotspots → **recommend actions** → a "
                "manager approves or rejects → the agent **executes** approved actions in the "
                "stores (work orders, citizen status updates) and logs every step. No LLM key is "
                "needed: its decisions come from the trained model and the transparent rules.")
    try:
        import agent as ops_agent
        a1, a2 = st.columns(2)
        if a1.button("▶️ Run monitoring cycle", use_container_width=True):
            with st.spinner("Agent is observing, scoring and recommending…"):
                ops_agent.run_monitoring_cycle()
            st.rerun()
        if a2.button("📰 Draft daily ops brief", use_container_width=True):
            st.session_state.brief = ops_agent.draft_brief()
        if st.session_state.get("brief"):
            st.code(st.session_state.brief, language=None)
        pend = ops_agent.list_recommendations("Pending")
        st.markdown(f"### Pending recommendations ({len(pend)})")
        if pend.empty:
            st.info("Nothing pending. Run a monitoring cycle to generate recommendations.")
        for _, r in pend.iterrows():
            x, y, z = st.columns([4, 1, 1])
            x.markdown(f"**#{r['id']} · {r['kind']}** → `{r['target']}`  \n"
                       f"<span style='color:#8fa8cc;font-size:12px'>{r['reason']} · {r['ts']}</span>",
                       unsafe_allow_html=True)
            if y.button("✅ Approve & execute", key=f"ag_ap_{r['id']}", use_container_width=True):
                ops_agent.decide_recommendation(int(r["id"]), True); st.rerun()
            if z.button("🚫 Reject", key=f"ag_rj_{r['id']}", use_container_width=True):
                ops_agent.decide_recommendation(int(r["id"]), False); st.rerun()
        wo = ops_agent.list_work_orders()
        st.markdown(f"### Work orders created ({len(wo)})")
        if not wo.empty:
            st.dataframe(wo, use_container_width=True, height=200)
        log_df = ops_agent.get_log()
        st.markdown("### 🪵 Agent activity log")
        if not log_df.empty:
            st.dataframe(log_df[["ts", "stage", "detail"]], use_container_width=True, height=300)
        else:
            st.info("No agent activity yet this install — run a monitoring cycle.")
    except Exception as e:
        st.warning(f"Ops Agent unavailable ({e}). The rest of CityPulse is unaffected.")

# ================================================================ KB TAB
with tab_kb:
    st.subheader("💬 Knowledge Assistant")
    st.markdown("Answers come **only** from grounded CityPulse sources: the Socrata dataset "
                "dictionary, per-type and per-area cards computed from the real data, the hotspot "
                "snapshot, and the project documentation. Every answer lists its sources and says "
                "whether it was composed by an LLM (only if an API key is configured) or "
                "extractively from the retrieved material.")
    examples = ["How long does a pothole complaint usually take to close?",
                "Which department handles street light outages?",
                "What is happening in Logan Square this week?",
                "How does the priority score work?"]
    ex_cols = st.columns(len(examples))
    for c, ex in zip(ex_cols, examples):
        if c.button(ex, use_container_width=True):
            st.session_state.kb_input = ex
            st.session_state.kb_run = True
    q = st.text_input("Ask a question", key="kb_input",
                      placeholder="Try a request type, a neighbourhood, a field name, or an SR number…")
    ask_clicked = st.button("Ask", key="kb_ask")
    run_flag = st.session_state.pop("kb_run", False)
    if (ask_clicked or run_flag) and q.strip():
        try:
            import rag
            with st.spinner("Retrieving from the knowledge base…"):
                ans, srcs, mode_used = rag.assistant().answer(q)
            st.markdown(f"_Answered in mode: **{mode_used}**_")
            st.markdown(ans)
            with st.expander("📚 Sources used"):
                for s in srcs:
                    st.markdown(f"- {s}")
        except Exception as e:
            st.warning(f"Knowledge Assistant unavailable ({e}). The rest of CityPulse is unaffected.")

st.markdown("---")
st.caption("CityPulse AI · Data: Chicago Data Portal, 311 Service Requests (v6vf-nfxy), unauthenticated Socrata API · "
           "SLA is a 7-day working proxy, not an official city SLA · Prioritisation only — no crew dispatch or routing · "
           "Citizen reports filed in this app live only in CityPulse's local system — the Socrata API is read-only and "
           "nothing is submitted to the City of Chicago's official 311 · "
           "All figures on this page are measured from the cached real data shown in Model & Evaluation.")
