"""
Seed the ResQLink database with demo resources and sample incidents
so the portal looks populated when you first open it.
Run once: python seed_demo_data.py
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import random

DB_PATH = Path(__file__).resolve().parent.parent / "resqlink_incidents.db"


def connect():
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    return db


def seed():
    db = connect()

    # ── Resources ────────────────────────────────────────────────
    resources = [
        ("ambulance", "Ambulance Unit 1",   "CMCH, Egmore, Chennai",          "2 paramedics"),
        ("ambulance", "Ambulance Unit 2",   "Vadapalani, Chennai",             "1 stretcher"),
        ("ambulance", "Ambulance Unit 3",   "Anna Nagar West, Chennai",        "ICU equipped"),
        ("hospital",  "Apollo Hospital",    "Greams Road, Chennai",            "ICU: 4 beds"),
        ("hospital",  "MIOT International","Manapakkam, Chennai",              "Trauma: 6 beds"),
        ("police",    "Traffic Police OP1", "Koyambedu Junction",              "2 officers"),
        ("police",    "Traffic Police OP2", "Anna Salai, near Central Station","4 officers"),
        ("fire",      "Fire Station 3",     "Ashok Nagar, Chennai",            "1 tender"),
        ("fire",      "Fire Station 7",     "Perungudi, OMR, Chennai",         "Heavy rescue"),
    ]

    existing = db.execute("SELECT COUNT(*) FROM response_resources").fetchone()[0]
    if existing == 0:
        for agency_type, name, location, cap in resources:
            db.execute(
                "INSERT INTO response_resources(agency_type,resource_name,location_text,capacity_note) VALUES(?,?,?,?)",
                (agency_type, name, location, cap)
            )
        print(f"Seeded {len(resources)} resources.")
    else:
        print(f"Resources already seeded ({existing} records). Skipping.")

    # ── Sample incidents ──────────────────────────────────────────
    sample_incidents = [
        (26, 260, "Car",        42.3, 135.2, 0.82, "High",     "Verified",       "Near Anna Flyover, NH-45", 1, 1, 0),
        (44, 320, "Motorcycle", 67.1, 92.5,  1.10, "Critical", "Under Review",   "Koyambedu Junction",       1, 1, 1),
        (12, 105, "Truck",      15.4, 210.0, 0.31, "Medium",   "Closed",         "OMR, near Tidel Park",     0, 1, 0),
        (88, 490, "Car",        55.9, 88.3,  0.74, "High",     "New",            "ECR, Kovalam",             0, 0, 0),
        (7,  34,  "Auto",       28.2, 45.6,  0.22, "Low",      "New",            "",                         0, 0, 0),
        (63, 412, "Bus",        9.1,  310.5, 0.44, "Medium",   "Under Review",   "Poonamallee High Road",    0, 1, 0),
    ]

    existing_inc = db.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    if existing_inc == 0:
        base = datetime.now()
        for i, (vid, frame, vtype, mov, angle, decel, sev, status, loc, amb, pol, fire) in enumerate(sample_incidents):
            ts = (base - timedelta(hours=i*3 + random.randint(0, 2))).isoformat(timespec="seconds")
            key = f"demo:{i}:{frame}:{vid}"
            notes = "Candidate from AI pipeline. Human review pending." if status == "New" else ""
            db.execute("""
                INSERT OR IGNORE INTO incidents
                (candidate_key, vehicle_id, frame, vehicle_type,
                 movement_per_sec, direction_change_angle, deceleration,
                 status, severity, notes, location_text,
                 ambulance_assigned, police_assigned, fire_assigned,
                 created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (key, vid, frame, vtype, mov, angle, decel,
                  status, sev, notes, loc, amb, pol, fire, ts, ts))
        print(f"Seeded {len(sample_incidents)} sample incidents.")
    else:
        print(f"Incidents already exist ({existing_inc} records). Skipping.")

    # ── Citizen reports ───────────────────────────────────────────
    existing_rpt = db.execute("SELECT COUNT(*) FROM citizen_reports").fetchone()[0]
    if existing_rpt == 0:
        reports = [
            ("admin", "Two cars collided at the traffic signal. One person injured.",
             "2026-10-11 08:22:00", "Anna Flyover, NH-45, Chennai", 13.0827, 80.2707, "9876543210", "Submitted"),
            ("admin", "Motorcycle skidded on wet road. Rider needs help.",
             "2026-10-10 18:45:00", "ECR near Kovalam beach junction", 12.8406, 80.2534, None, "Submitted"),
        ]
        for reporter, desc, inc_time, loc, lat, lon, contact, status in reports:
            db.execute("""
                INSERT INTO citizen_reports
                (reporter_username, description, incident_time, location_text,
                 latitude, longitude, contact, status)
                VALUES (?,?,?,?,?,?,?,?)
            """, (reporter, desc, inc_time, loc, lat, lon, contact, status))
        print(f"Seeded {len(reports)} citizen reports.")
    else:
        print(f"Reports already exist ({existing_rpt} records). Skipping.")

    db.commit()
    db.close()
    print("\nDemo data seeded successfully.")
    print("Visit http://127.0.0.1:5000 — login: admin / Admin@ResQLink1")


if __name__ == "__main__":
    seed()
