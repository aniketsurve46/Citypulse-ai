"""End-to-end test of the citizen complaint loop:
submit a POTHOLE complaint -> verify it is scored + stored + receipt computed ->
record a manager decision in the shared decisions store -> verify tracking
reflects the new status. Run: python test_citizen_flow.py
The test report stays in the store as the worked sample (labelled in its
description) so the loop is visible in the app."""
import sqlite3

import citizen

rep = citizen.submit_report(
    sr_type="Pothole in Street Complaint",
    street_address="2517 W FULLERTON AVE",
    zip_code="60647",
    description="Verification sample: large pothole in the street near the curb, "
                "filed by the CityPulse end-to-end test.",
    urgency="High",
    reporter_name="CityPulse Test",
)
print("SUBMITTED:", rep["sr_number"], "| score", rep["priority_score"],
      "| band", rep["risk_band"], "| queue", f"#{rep['queue_position']} of {rep['queue_total']}")
print("REASON:", rep["reason"])
print("LOCATION approx:", rep["location_approx"], rep["latitude"], rep["longitude"],
      "| area", rep["community_area"], "| ward", rep["ward"], "| dept", rep["owner_department"])
assert rep["sr_number"].startswith("CP-2026-")
assert rep["priority_score"] > 0

store = citizen.load_reports()
assert rep["sr_number"] in set(store["sr_number"]), "report missing from store"
print("STORE OK:", len(store), "citizen report(s) persisted in data/citizen_reports.csv")

receipt = citizen.receipt_text(rep)
assert rep["sr_number"] in receipt and "NOT" in receipt and "official 311" in receipt
print("RECEIPT OK (includes the not-official-311 honesty line)")

before = citizen.tracking(rep["sr_number"])
print("TRACKING BEFORE DECISION:", before["status_label"])
assert before["status_label"] == citizen.DEFAULT_STATUS

# simulate the operations manager approving it in the Operations Center
con = sqlite3.connect(citizen.DB_PATH)
con.execute("CREATE TABLE IF NOT EXISTS decisions (sr_number TEXT, decision TEXT, ts TEXT)")
con.execute("INSERT INTO decisions VALUES (?, 'Approved', datetime('now'))", (rep["sr_number"],))
con.commit(); con.close()

after = citizen.tracking(rep["sr_number"])
print("TRACKING AFTER MANAGER APPROVAL:", after["status_label"], "| updated", after["last_updated"])
assert after["status_label"] == "Approved for action"
print("CITIZEN LOOP TEST PASSED: submit -> receipt -> manager decision -> tracking")
