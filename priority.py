"""
CityPulse AI - blended priority decision.

The priority score is a TRANSPARENT BLEND of five factors. It is not just the
model score, and FIFO alone is not the decision either:

  1. Delay risk   - trained model's probability the request breaches the
                    7-day working SLA proxy.
  2. Severity     - hand-set tier per request type, based on safety and public
                    impact (see SEVERITY_RULES below). These tiers are a policy
                    choice made by the product, NOT learned from the data, and
                    they are listed in the app and README. Edit them here in
                    this one config to change policy.
  3. Waiting time - days the request has already been open (FIFO factor),
                    normalised and capped at 30 days so old requests steadily
                    rise and nothing starves at the bottom forever.
  4. People affected - duplicate group size: repeat reports of the same type
                    at the same address within 14 days.
  5. Hotspot      - emerging surge of this type in this community area
                    (last 7 days vs prior 28-day daily average).

Priority modes (manager-selectable in the app) re-weight the same components:
  Balanced (default), FIFO-first, Urgency-first.
"""
import pandas as pd

# --------------------------------------------------------------- severity ---
# (keywords matched against the sr_type, first match wins, case-insensitive)
# Tier scores: Critical 1.00, High 0.75, Medium 0.50, Low 0.25, Default 0.40
SEVERITY_RULES = [
    # Critical - immediate safety hazard or essential service out
    ("Critical", 1.00, ["traffic signal out", "tree emergency", "check for leak",
                        "no water complaint", "sewer cave-in", "open fire hydrant",
                        "water on street", "pavement cave-in"]),
    # High - safety / health hazard, road hazard, or outage
    ("High", 0.75, ["pothole", "street light out", "alley light out",
                    "viaduct light out", "street light pole damage",
                    "rodent", "sanitation code", "dead animal", "fly dumping",
                    "water in basement", "low water pressure", "vicious animal",
                    "building violation", "plumbing violation",
                    "vacant/abandoned building", "sign repair request - stop sign",
                    "sign repair request - do not enter"]),
    # Medium - service disruption or neighbourhood quality issue
    ("Medium", 0.50, ["garbage cart", "missed garbage", "blue recycling cart",
                      "recycling pick up", "yard waste", "tree debris",
                      "tree trim", "weed removal", "abandoned vehicle",
                      "street cleaning", "sewer cleaning", "alley sewer",
                      "injured animal", "stray animal", "nuisance animal",
                      "clean vacant lot", "inspect public way", "sign repair",
                      "snow", "ice and snow"]),
    # Low - cosmetic / informational / administrative
    ("Low", 0.25, ["graffiti", "e-scooter", "divvy", "parking code",
                   "vehicle sticker", "cab feedback", "consumer", "business complaint",
                   "restaurant complaint", "bicycle request", "tree planting",
                   "lead test", "information", "feedback"]),
]
DEFAULT_SEVERITY = ("Medium", 0.40)


def severity_for(sr_type):
    """Return (tier_name, tier_score) for a request type."""
    t = str(sr_type).lower()
    for name, score, keywords in SEVERITY_RULES:
        if any(k in t for k in keywords):
            return name, score
    return DEFAULT_SEVERITY


def severity_table(sr_types):
    rows = [{"sr_type": s, "severity_tier": severity_for(s)[0],
             "severity_score": severity_for(s)[1]} for s in sorted(set(sr_types))]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- weights --
PRIORITY_MODES = {
    "Balanced":     {"breach": 0.30, "severity": 0.25, "wait": 0.20, "hotspot": 0.15, "dup": 0.10},
    "FIFO-first":   {"breach": 0.15, "severity": 0.15, "wait": 0.50, "hotspot": 0.10, "dup": 0.10},
    "Urgency-first": {"breach": 0.35, "severity": 0.40, "wait": 0.05, "hotspot": 0.10, "dup": 0.10},
}
WAIT_CAP_DAYS = 30


def compute_priority(df, mode="Balanced", now=None):
    """df must carry: sr_type, created_date, breach_prob, hotspot_ratio (or
    hotspot_norm), dup_group_size. Adds component + score + reason columns."""
    w = PRIORITY_MODES[mode]
    d = df.copy()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    # Element-wise parse — version-proof for mixed date styles (see citizen.py).
    d["created_date"] = d["created_date"].map(lambda v: pd.to_datetime(v, errors="coerce"))
    sev = d["sr_type"].apply(severity_for)
    d["severity_tier"] = [s[0] for s in sev]
    d["severity_score"] = [s[1] for s in sev]
    d["wait_days"] = ((now - d["created_date"]).dt.total_seconds() / 86400).clip(lower=0).fillna(0)
    d["wait_norm"] = (d["wait_days"] / WAIT_CAP_DAYS).clip(0, 1)
    if "hotspot_norm" not in d.columns:
        d["hotspot_norm"] = ((d.get("hotspot_ratio", 1.0) - 1.0) / 2.0).clip(0, 1)
    if "dup_norm" not in d.columns:
        d["dup_norm"] = ((d.get("dup_group_size", 1) - 1) / 4.0).clip(0, 1)
    d["priority_score"] = (w["breach"] * d["breach_prob"] + w["severity"] * d["severity_score"]
                           + w["wait"] * d["wait_norm"] + w["hotspot"] * d["hotspot_norm"]
                           + w["dup"] * d["dup_norm"])
    d["risk_band"] = pd.cut(d["priority_score"], [-0.01, 0.35, 0.55, 1.01],
                            labels=["Low", "Medium", "High"])

    def reason(r):
        factors = {
            "Delay risk": w["breach"] * r.breach_prob,
            "Severity": w["severity"] * r.severity_score,
            "Waiting time": w["wait"] * r.wait_norm,
            "Hotspot": w["hotspot"] * r.hotspot_norm,
            "Repeat reports": w["dup"] * r.dup_norm,
        }
        top = max(factors, key=factors.get)
        parts = [f"Top factor: {top}",
                 f"Severity: {r.severity_tier} ({r.sr_type})",
                 f"Waiting {r.wait_days:.0f}d",
                 f"Delay risk {r.breach_prob:.2f}"]
        if r.dup_group_size and r.dup_group_size > 1:
            parts.append(f"{int(r.dup_group_size)} reports at this address")
        hr = r.get("hotspot_ratio", 1.0)
        if pd.notna(hr) and hr >= 1.5:
            parts.append(f"Hotspot area ({hr:.1f}x normal)")
        return " · ".join(parts)

    d["reason"] = d.apply(reason, axis=1)
    return d
