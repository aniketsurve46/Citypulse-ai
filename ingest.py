"""
CityPulse AI - data ingestion from the Chicago Data Portal (Socrata).

Dataset: 311 Service Requests (v6vf-nfxy)
Endpoint: https://data.cityofchicago.org/resource/v6vf-nfxy.json

Downloads (unauthenticated, cached locally):
  (a) recent COMPLETED requests  -> data/closed_requests.parquet  (training data)
  (b) current OPEN requests      -> data/open_requests.parquet

Pagination: cursor-based on created_date DESC (deep $offset pagination is
extremely slow on this dataset). Each page is de-duplicated by sr_number.

Exclusions (documented):
  - sr_type '311 INFORMATION ONLY CALL' : informational phone calls, not field work,
    close instantly and would swamp the training set (~651k in the last 12 months).
  - sr_type 'Aircraft Noise Complaint'  : logged and closed the same second,
    not a field-work request (~455k in the last 12 months).
Both distort the days-to-close target, so they are excluded from training AND
from the open queue (the open queue is about field-work prioritisation).

Status values in this dataset: 'Completed' is the closed status (there is
essentially one legacy 'Closed' row), 'Open' is open, 'Canceled' is excluded.
"""
import os
import time
import requests
import pandas as pd

BASE = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIELDS = [
    "sr_number", "sr_type", "sr_short_code", "owner_department", "status",
    "origin", "created_date", "closed_date", "last_modified_date",
    "street_address", "zip_code", "community_area", "ward", "duplicate",
    "latitude", "longitude", "created_hour", "created_day_of_week",
    "created_month",
]
EXCLUDED_TYPES = ("311 INFORMATION ONLY CALL", "Aircraft Noise Complaint")
PAGE = 50000
SINCE = "2025-11-22T00:00:00"          # last ~12 months
CLOSED_TARGET = 220000
OPEN_TARGET = 1000000  # effectively "all open": fetch_cursor stops on its own
                       # when a page returns no new sr_numbers


def get(params):
    for attempt in range(4):
        try:
            r = requests.get(BASE, params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry {attempt+1} ({e})", flush=True)
            time.sleep(3 * (attempt + 1))


def fetch_cursor(base_where, target, label):
    """Newest-first cursor pagination on created_date."""
    seen, rows, cursor = set(), [], None
    while len(rows) < target:
        where = base_where
        if cursor:
            where += f" AND created_date <= '{cursor}'"
        batch = get({
            "$select": ",".join(FIELDS), "$where": where,
            "$order": "created_date DESC", "$limit": min(PAGE, target - len(rows) + 500),
        })
        if not batch:
            break
        new = [b for b in batch if b.get("sr_number") not in seen]
        for b in new:
            seen.add(b.get("sr_number"))
        rows.extend(new)
        cursor = batch[-1].get("created_date")
        print(f"  {label}: {len(rows):,} rows (cursor {cursor})", flush=True)
        if len(new) == 0:
            break
        time.sleep(0.3)
    return pd.DataFrame(rows[:target])


def save(df, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        print(f"  parquet failed ({e}); saving CSV instead")
        path = path.replace(".parquet", ".csv")
        df.to_csv(path, index=False)
    print(f"  saved {len(df):,} rows -> {path}", flush=True)
    return path


def main():
    excl = " AND ".join([f"sr_type != '{t}'" for t in EXCLUDED_TYPES])
    print("Downloading COMPLETED requests (training set)...", flush=True)
    closed = fetch_cursor(
        f"status='Completed' AND closed_date IS NOT NULL "
        f"AND created_date >= '{SINCE}' AND {excl}",
        CLOSED_TARGET, "closed")
    save(closed, "closed_requests.parquet")

    print("Downloading OPEN requests (live queue)...", flush=True)
    open_df = fetch_cursor(f"status='Open' AND {excl}", OPEN_TARGET, "open")
    save(open_df, "open_requests.parquet")
    print("Ingest complete.", flush=True)


if __name__ == "__main__":
    main()
