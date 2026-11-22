"""
CityPulse AI - citizen complaints ("Report a Problem").

HONESTY RULE: the Chicago Socrata API is READ-ONLY. A complaint filed here is
stored in CityPulse's LOCAL system only (data/citizen_reports.csv). It is NOT
submitted to the City of Chicago's official 311. The UI and the receipt say
this explicitly.

Flow: form -> validate -> local CP-2026-XXXX id -> scored immediately with the
trained model + duplicate check + hotspot context (blended priority, same as
official requests) -> appears in the Operations queue/map badged
"Citizen Report (CityPulse)" -> manager decisions in the shared decisions
store change the status the citizen sees in "Track your complaint".
"""
import os
import pickle
import sqlite3
from datetime import datetime

import pandas as pd

import priority as prio_mod

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "data", "citizen_reports.csv")
OUT = os.path.join(HERE, "outputs")
DB_PATH = os.path.join(OUT, "decisions.db")
SOURCE_LABEL = "Citizen Report (CityPulse)"
STATUS_MAP = {"Approved": "Approved for action", "Rejected": "Rejected by manager",
              "Escalated": "Escalated"}
DEFAULT_STATUS = "Waiting for review by operations team"


def _load(name):
    p = os.path.join(HERE, "data", name + ".parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.read_csv(
        os.path.join(HERE, "data", name + ".csv"), low_memory=False)


def load_reports():
    if os.path.exists(STORE):
        df = pd.read_csv(STORE)
        # Parse element-wise: this pandas build's format="mixed" path still
        # infers one strict format and crashes on the other date style
        # present in this file (app-written ISO vs seed space-separated).
        df["created_date"] = df["created_date"].map(lambda v: pd.to_datetime(v, errors="coerce"))
        return df
    return pd.DataFrame()


def _next_id(existing):
    return f"CP-2026-{len(existing) + 1:04d}"


def _zip_context(zip_code):
    """Median lat/lon + modal community_area/ward of REAL requests with this
    ZIP from the cached data. Returns (lat, lon, area, ward) or Nones.
    The location is an approximation from the ZIP, never a fake exact pin."""
    try:
        frames = []
        for name in ("open_requests", "closed_requests"):
            d = _load(name)[["zip_code", "latitude", "longitude", "community_area", "ward"]]
            frames.append(d)
        allr = pd.concat(frames, ignore_index=True)
        allr["zip_code"] = allr["zip_code"].astype(str).str[:5]
        allr["latitude"] = pd.to_numeric(allr["latitude"], errors="coerce")
        allr["longitude"] = pd.to_numeric(allr["longitude"], errors="coerce")
        m = allr[allr["zip_code"] == str(zip_code)[:5]].dropna(subset=["latitude", "longitude"])
        if len(m) >= 5:
            return (float(m["latitude"].median()), float(m["longitude"].median()),
                    str(m["community_area"].mode().iloc[0]), str(m["ward"].mode().iloc[0]))
    except Exception:
        pass
    return None, None, None, None


def submit_report(sr_type, street_address, zip_code, community_area=None, ward=None,
                  description="", urgency="", reporter_name="", reporter_contact=""):
    """Validate, score and persist one citizen report. Returns the report dict
    including queue position and receipt fields."""
    if not sr_type or not str(street_address).strip() or not str(zip_code).strip():
        raise ValueError("Problem type, street address and ZIP code are required.")
    zip_code = str(zip_code).strip()[:5]
    now = pd.Timestamp.now()
    existing = load_reports()

    closed = _load("closed_requests")
    dept_mode = closed[closed["sr_type"] == sr_type]["owner_department"].mode()
    department = dept_mode.iloc[0] if len(dept_mode) else "311 City Services"

    lat, lon, zip_area, zip_ward = _zip_context(zip_code)
    area = str(community_area) if community_area else (zip_area or "Unknown")
    ward_v = str(ward) if ward else (zip_ward or "Unknown")

    with open(os.path.join(OUT, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    row = pd.DataFrame([{
        "sr_type": sr_type, "owner_department": department, "origin": "CityPulse Web",
        "community_area": area, "ward": ward_v, "zip_code": zip_code,
        "created_hour": now.hour, "created_day_of_week": now.dayofweek + 1,
        "created_month": now.month, "duplicate_flag": 0}])
    for c in ["sr_type", "owner_department", "origin", "community_area", "ward", "zip_code"]:
        row[c] = row[c].astype(str)
    breach = float(model.predict_proba(row[["sr_type", "owner_department", "origin",
                                            "community_area", "ward", "zip_code",
                                            "created_hour", "created_day_of_week",
                                            "created_month", "duplicate_flag"]])[:, 1][0])

    queue = pd.read_csv(os.path.join(OUT, "prioritized_open.csv"))
    addr_norm = str(street_address).upper().strip()
    same = queue[(queue["sr_type"] == sr_type) &
                 (queue["street_address"].astype(str).str.upper().str.strip() == addr_norm)]
    dup_group_size = int(len(same)) + 1

    hotspots = pd.read_csv(os.path.join(OUT, "hotspots.csv"))
    hs = hotspots[(hotspots["community_area"].astype(str) == area) &
                  (hotspots["sr_type"] == sr_type)]
    hotspot_ratio = float(hs["ratio"].iloc[0]) if len(hs) and pd.notna(hs["ratio"].iloc[0]) else 1.0

    scored = prio_mod.compute_priority(pd.DataFrame([{
        "sr_type": sr_type, "created_date": now, "breach_prob": breach,
        "hotspot_ratio": hotspot_ratio, "dup_group_size": dup_group_size}]),
        mode="Balanced", now=now).iloc[0]

    report_id = _next_id(existing)
    report = {
        "sr_number": report_id, "sr_type": sr_type, "sr_short_code": "",
        "owner_department": department, "status": "Open", "origin": "CityPulse Web",
        "created_date": now.isoformat(timespec="seconds"),
        "street_address": str(street_address).strip(), "zip_code": zip_code,
        "community_area": area, "ward": ward_v, "duplicate_flag": 0,
        "latitude": lat, "longitude": lon, "location_approx": bool(lat is not None),
        "dup_cluster_id": -1, "dup_group_size": dup_group_size,
        "breach_prob": round(breach, 4), "hotspot_ratio": hotspot_ratio,
        "hotspot_norm": float(scored["hotspot_norm"]), "dup_norm": float(scored["dup_norm"]),
        "severity_tier": scored["severity_tier"], "severity_score": float(scored["severity_score"]),
        "wait_days": 0.0, "wait_norm": 0.0,
        "priority_score": round(float(scored["priority_score"]), 4),
        "risk_band": str(scored["risk_band"]), "reason": scored["reason"],
        "source": SOURCE_LABEL, "description": description, "urgency": urgency,
        "reporter_name": reporter_name, "reporter_contact": reporter_contact,
        "is_live": False,
    }
    all_reports = pd.concat([existing, pd.DataFrame([report])], ignore_index=True)
    all_reports.to_csv(STORE, index=False)

    position = int((queue["priority_score"] > report["priority_score"]).sum()) + 1
    report["queue_position"] = position
    report["queue_total"] = int(len(queue)) + 1
    report["status_label"] = DEFAULT_STATUS
    return report


def receipt_text(report):
    return (
        "CityPulse AI - Complaint Receipt\n"
        "================================\n"
        f"Complaint ID   : {report['sr_number']}\n"
        f"Filed          : {report['created_date']}\n"
        f"Problem type   : {report['sr_type']}\n"
        f"Address        : {report['street_address']}, Chicago, IL {report['zip_code']}\n"
        f"Department     : {report['owner_department']}\n"
        f"Priority score : {report['priority_score']} ({report['risk_band']} priority)\n"
        f"Why            : {report['reason']}\n"
        f"Queue position : #{report['queue_position']} of {report['queue_total']} open requests\n"
        f"Status         : {DEFAULT_STATUS}\n\n"
        "Save your Complaint ID to track your complaint in CityPulse.\n"
        "IMPORTANT: This complaint was filed in CityPulse's local system for demo\n"
        "purposes only. The Chicago 311 Socrata API is read-only, so it was NOT\n"
        "submitted to the City of Chicago's official 311 system.\n"
    )


def tracking(cp_id):
    """Current status of a citizen report, read from the SAME decisions store
    the operations manager uses. Returns dict or None."""
    reports = load_reports()
    m = reports[reports["sr_number"] == str(cp_id).strip().upper()]
    if m.empty:
        return None
    r = m.iloc[-1].to_dict()
    status, updated = DEFAULT_STATUS, r["created_date"]
    if os.path.exists(DB_PATH):
        con = sqlite3.connect(DB_PATH)
        try:
            rows = con.execute(
                "SELECT decision, ts FROM decisions WHERE sr_number=? ORDER BY ts DESC",
                (r["sr_number"],)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        con.close()
        if rows:
            status = STATUS_MAP.get(rows[0][0], rows[0][0])
            updated = rows[0][1]
    r["status_label"] = status
    r["last_updated"] = str(updated)
    return r
