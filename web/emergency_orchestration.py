"""Prototype emergency orchestration for CCTV-originated detections.

The module is deliberately self-contained: it never assumes a vendor SMS
gateway, an unauthorised phone location feed, or a live hospital system.  It
creates role-scoped *in-app* notifications and exposes a driver location page
that can use a phone's browser location only after that driver has logged in
and granted permission.
"""

from __future__ import annotations

import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

from flask import abort, flash, jsonify, redirect, render_template_string, request, url_for


CONTROL_ROLES = {"admin", "operator"}
AMBULANCE_ROLES = {"ambulance", "amb_crew", "ambulance_crew", "driver", "responder"}
DEPARTMENT_ROLES = {"police", "traffic", "fire"}
SIMULATION_SCENARIOS = {
    "standard": [
        ("Ambulance acknowledges assignment (simulated).", 0.0, "ambulance"),
        ("Ambulance en route — simulated GPS update.", 0.2, "incident"),
        ("En-route progress — simulated GPS update.", 0.55, "incident"),
        ("Ambulance arrived at scene (simulated).", 1.0, "incident"),
        ("Hospital readiness confirmed (simulated).", 1.0, "incident"),
        ("Patient onboard — simulated handover status.", 1.0, "incident"),
        ("Arrived at hospital — simulated handover complete.", 1.0, "hospital"),
    ],
    "driver_no_ack": [
        ("Dispatch alert queued in the portal (simulation only).", 0.0, "ambulance"),
        ("Driver acknowledgement timeout — no real call or radio contact was made.", 0.0, "ambulance"),
        ("Manual follow-up required: use the agency's approved phone/radio procedure.", 0.0, "ambulance"),
    ],
    "gps_lost": [
        ("Ambulance acknowledges assignment (simulated).", 0.0, "ambulance"),
        ("Ambulance en route — simulated GPS update.", 0.2, "incident"),
        ("GPS signal lost — last simulated position is now stale.", 0.35, "incident"),
        ("Operator follow-up required; no real GPS or driver contact was attempted.", 0.35, "incident"),
    ],
    "hospital_unavailable": [
        ("Ambulance acknowledges assignment (simulated).", 0.0, "ambulance"),
        ("Ambulance en route — simulated GPS update.", 0.2, "incident"),
        ("Hospital reports no capacity (simulated); no real hospital was contacted.", 0.55, "incident"),
        ("Operator diversion required: verify alternate hospital capacity manually.", 0.55, "incident"),
    ],
}


@contextmanager
def _connect(db_path):
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        yield db
    except BaseException:
        db.rollback()
        raise
    else:
        db.commit()
    finally:
        db.close()


def _distance_km(lat_a, lon_a, lat_b, lon_b):
    if None in (lat_a, lon_a, lat_b, lon_b):
        return 9999.0
    radius = 6371.0
    d_lat = math.radians(lat_b - lat_a)
    d_lon = math.radians(lon_b - lon_a)
    angle = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat_a)) * math.cos(math.radians(lat_b)) * math.sin(d_lon / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(angle), math.sqrt(1 - angle))


def ensure_response_schema(db):
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS response_ambulances (
            ambulance_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            driver_phone TEXT NOT NULL,
            driver_username TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'Available',
            last_location_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS response_hospitals (
            hospital_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            contact_username TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            emergency_beds INTEGER NOT NULL,
            emergency_department TEXT NOT NULL,
            preparedness_status TEXT NOT NULL DEFAULT 'Ready',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS response_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_number TEXT NOT NULL UNIQUE,
            source_upload_id INTEGER,
            location TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL DEFAULT 'High',
            status TEXT NOT NULL,
            analysis_state TEXT NOT NULL,
            ambulance_id TEXT,
            hospital_id TEXT,
            duplicate_of_case_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS response_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            target_role TEXT NOT NULL,
            target_username TEXT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'New',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES response_cases(id)
        );
        CREATE TABLE IF NOT EXISTS response_location_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            ambulance_id TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            accuracy_meters REAL,
            source TEXT NOT NULL DEFAULT 'manual',
            note TEXT,
            captured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES response_cases(id)
        );
        CREATE TABLE IF NOT EXISTS response_simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL UNIQUE,
            case_id INTEGER NOT NULL,
            scenario TEXT NOT NULL,
            step INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'Running',
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(case_id) REFERENCES response_cases(id)
        );
        CREATE TABLE IF NOT EXISTS response_simulation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL,
            step INTEGER NOT NULL,
            message TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            captured_at TEXT NOT NULL,
            UNIQUE(run_key, step),
            FOREIGN KEY(run_key) REFERENCES response_simulation_runs(run_key)
        );
        """
    )
    ambulance_columns = {row[1] for row in db.execute("PRAGMA table_info(response_ambulances)")}
    if "driver_username" not in ambulance_columns:
        db.execute("ALTER TABLE response_ambulances ADD COLUMN driver_username TEXT")
    hospital_columns = {row[1] for row in db.execute("PRAGMA table_info(response_hospitals)")}
    if "contact_username" not in hospital_columns:
        db.execute("ALTER TABLE response_hospitals ADD COLUMN contact_username TEXT")
    notification_columns = {row[1] for row in db.execute("PRAGMA table_info(response_notifications)")}
    if "target_username" not in notification_columns:
        db.execute("ALTER TABLE response_notifications ADD COLUMN target_username TEXT")
    location_columns = {row[1] for row in db.execute("PRAGMA table_info(response_location_updates)")}
    if "accuracy_meters" not in location_columns:
        db.execute("ALTER TABLE response_location_updates ADD COLUMN accuracy_meters REAL")
    if "source" not in location_columns:
        db.execute("ALTER TABLE response_location_updates ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'")
    demo_mode = (
        os.environ.get("RESQLINK_DEMO_MODE") == "1"
        and os.environ.get("RESQLINK_ENV", "development").lower() != "production"
    )
    if demo_mode:
        db.executemany(
            """
            INSERT OR IGNORE INTO response_ambulances
            (ambulance_id, label, driver_phone, driver_username, latitude, longitude, status)
            VALUES (?, ?, ?, ?, ?, ?, 'Available')
            """,
            [
                ("AMB-DEMO-01", "Ambulance 01", "+91 90000 00001", "amb_crew1", 13.0785, 80.2640),
                ("AMB-DEMO-02", "Ambulance 02", "+91 90000 00002", None, 13.0624, 80.2492),
            ],
        )
        db.execute(
            """UPDATE response_ambulances SET driver_username='amb_crew1'
               WHERE ambulance_id='AMB-DEMO-01'
                 AND EXISTS (SELECT 1 FROM users WHERE username='amb_crew1' AND role='responder')"""
        )
        db.executemany(
            """
            INSERT OR IGNORE INTO response_hospitals
            (hospital_id, name, contact_username, latitude, longitude, emergency_beds, emergency_department)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("HOS-DEMO-01", "ResQLink City Emergency Hospital", "hospital1", 13.0871, 80.2785, 8, "24×7 Emergency & Trauma"),
                ("HOS-DEMO-02", "ResQLink Central Medical Centre", None, 13.0705, 80.2551, 5, "Emergency Department"),
            ],
        )
        db.execute(
            """UPDATE response_hospitals SET contact_username='hospital1'
               WHERE hospital_id='HOS-DEMO-01'
                 AND EXISTS (SELECT 1 FROM users WHERE username='hospital1' AND role='hospital')"""
        )
    db.commit()


def _notify(db, case_id, role, title, body, *, target_username=None):
    db.execute(
        """INSERT INTO response_notifications
           (case_id, target_role, target_username, title, body) VALUES (?, ?, ?, ?, ?)""",
        (case_id, role, target_username, title, body),
    )


def _nearest_available_ambulance(db, latitude, longitude):
    ambulances = db.execute(
        "SELECT * FROM response_ambulances WHERE status = 'Available' AND driver_username IS NOT NULL"
    ).fetchall()
    return min(ambulances, key=lambda item: _distance_km(latitude, longitude, item["latitude"], item["longitude"]), default=None)


def _nearest_ready_hospital(db, latitude, longitude):
    hospitals = db.execute(
        "SELECT * FROM response_hospitals WHERE emergency_beds > 0 AND contact_username IS NOT NULL"
    ).fetchall()
    return min(hospitals, key=lambda item: _distance_km(latitude, longitude, item["latitude"], item["longitude"]), default=None)


def _simulation_coordinates(db, case, ambulance_id, fraction, destination):
    ambulance = db.execute(
        "SELECT latitude,longitude FROM response_ambulances WHERE ambulance_id=?",
        (ambulance_id,),
    ).fetchone()
    if not ambulance:
        return None, None
    start = (ambulance["latitude"], ambulance["longitude"])
    if destination == "hospital":
        target = db.execute(
            "SELECT latitude,longitude FROM response_hospitals WHERE hospital_id=?",
            (case["hospital_id"],),
        ).fetchone()
    elif destination == "incident":
        target = case
    else:
        return start
    if not target or target["latitude"] is None or target["longitude"] is None:
        return None, None
    progress = max(0.0, min(1.0, fraction))
    return (
        start[0] + (target["latitude"] - start[0]) * progress,
        start[1] + (target["longitude"] - start[1]) * progress,
    )


def _simulation_payload(db, run):
    events = db.execute(
        """SELECT step,message,latitude,longitude,captured_at
           FROM response_simulation_events WHERE run_key=?
           ORDER BY step""",
        (run["run_key"],),
    ).fetchall()
    steps = SIMULATION_SCENARIOS[run["scenario"]]
    last_event = events[-1] if events else None
    return {
        "run_key": run["run_key"],
        "scenario": run["scenario"],
        "state": run["state"],
        "step": run["step"],
        "total_steps": len(steps),
        "progress_percent": round(max(0, run["step"] + 1) / len(steps) * 100),
        "latitude": last_event["latitude"] if last_event else None,
        "longitude": last_event["longitude"] if last_event else None,
        "events": [dict(event) for event in events],
    }


def process_cctv_detection(db_path, upload_id, camera_id, location, latitude, longitude, confidence):
    """Create or merge an unverified CCTV signal for control-room review."""
    with _connect(db_path) as db:
        ensure_response_schema(db)
        upload_columns = {row[1] for row in db.execute("PRAGMA table_info(cctv_video_uploads)")}
        if "response_case_id" not in upload_columns:
            db.execute("ALTER TABLE cctv_video_uploads ADD COLUMN response_case_id INTEGER")
        if confidence < 0.70:
            db.execute(
                "UPDATE cctv_video_uploads SET analysis_status = 'Low-confidence detection — operator review required' WHERE id = ?",
                (upload_id,),
            )
            db.commit()
            return None

        candidates = db.execute(
            """
            SELECT * FROM response_cases
            WHERE created_at >= datetime('now', '-30 minutes')
              AND status NOT IN ('Resolved', 'Closed')
            ORDER BY id DESC
            """
        ).fetchall()
        match = next(
            (
                item for item in candidates
                if _distance_km(latitude, longitude, item["latitude"], item["longitude"]) <= 0.8
            ),
            None,
        )
        if match:
            db.execute(
                "UPDATE cctv_video_uploads SET analysis_status = ?, linked_incident_id = ?, response_case_id = ? WHERE id = ?",
                (f"Duplicate CCTV evidence linked to {match['case_number']}", match["id"], match["id"], upload_id),
            )
            _notify(
                db, match["id"], "operator", "Duplicate CCTV evidence linked",
                f"Upload #{upload_id} from {camera_id} was merged into {match['case_number']}; operator review is available.",
            )
            db.commit()
            merged_case = dict(match)
            merged_case["is_duplicate"] = True
            return merged_case

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cursor = db.execute(
            """
            INSERT INTO response_cases
            (case_number, source_upload_id, location, latitude, longitude, confidence,
             status, analysis_state)
            VALUES (?, ?, ?, ?, ?, ?, 'Awaiting operator verification',
                    'Prototype CCTV signal — operator verification required')
            """,
            (
                f"CCTV-{timestamp}", upload_id, location or f"Camera {camera_id}", latitude, longitude,
                confidence,
            ),
        )
        case_id = cursor.lastrowid
        db.execute(
            "UPDATE cctv_video_uploads SET analysis_status = ?, linked_incident_id = ?, response_case_id = ? WHERE id = ?",
            (f"Awaiting operator verification: CCTV-{timestamp}", case_id, case_id, upload_id),
        )
        control_body = (
            f"{camera_id} reported a {confidence:.0%} collision signal at "
            f"{location or 'the camera location'}. Verify before dispatch."
        )
        _notify(db, case_id, "admin", "CCTV signal requires verification", control_body)
        _notify(db, case_id, "operator", "CCTV signal requires verification", control_body)
        db.commit()
        return dict(db.execute("SELECT * FROM response_cases WHERE id = ?", (case_id,)).fetchone())


def dispatch_verified_case(db_path, case_id, actor):
    """Dispatch a case only after an authorised control-room verification."""
    with _connect(db_path) as db:
        ensure_response_schema(db)
        db.execute("BEGIN IMMEDIATE")
        case = db.execute("SELECT * FROM response_cases WHERE id = ?", (case_id,)).fetchone()
        if not case:
            return False, "Case not found."
        if case["status"] != "Awaiting operator verification":
            return False, "Case is not awaiting operator verification."

        ambulance = _nearest_available_ambulance(db, case["latitude"], case["longitude"])
        hospital = _nearest_ready_hospital(db, case["latitude"], case["longitude"])
        if not ambulance or not hospital:
            status = "Operator verified — manual dispatch required"
            db.execute(
                "UPDATE response_cases SET status=?,analysis_state=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (status, "Verified by control room; no available ambulance or hospital bed.", case_id),
            )
            _notify(
                db, case_id, "admin", "Manual dispatch required",
                "The case was verified, but no available ambulance or emergency bed was found.",
            )
            _notify(
                db, case_id, "operator", "Manual dispatch required",
                "The case was verified, but no available ambulance or emergency bed was found.",
            )
            db.commit()
            return False, status

        db.execute(
            "UPDATE response_ambulances SET status='Dispatched' WHERE ambulance_id=?",
            (ambulance["ambulance_id"],),
        )
        db.execute(
            """UPDATE response_hospitals
               SET emergency_beds=emergency_beds-1,
                   preparedness_status='Patient preparation in progress',
                   updated_at=CURRENT_TIMESTAMP
               WHERE hospital_id=? AND emergency_beds > 0""",
            (hospital["hospital_id"],),
        )
        if db.execute("SELECT changes()").fetchone()[0] != 1:
            raise RuntimeError("Selected hospital capacity changed before dispatch.")

        status = "Ambulance assigned — prototype in-app alerts queued (not external dispatch)"
        db.execute(
            """UPDATE response_cases SET ambulance_id=?,hospital_id=?,status=?,
               analysis_state='Operator verified; dispatch initiated.',
               updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (ambulance["ambulance_id"], hospital["hospital_id"], status, case_id),
        )
        db.execute(
            """INSERT INTO response_location_updates
               (case_id,ambulance_id,latitude,longitude,source,note)
               VALUES (?,?,?,?,'registry','Registered ambulance location; not live driver GPS')""",
            (case_id, ambulance["ambulance_id"], ambulance["latitude"], ambulance["longitude"]),
        )
        _notify(
            db, case_id, "operator", "Operator-confirmed assignment recorded",
            f"{actor} verified the CCTV signal. {ambulance['label']} and {hospital['name']} were assigned in this prototype. No external service was contacted.",
        )
        _notify(
            db, case_id, "hospital", "Incoming emergency patient (portal notification)",
            f"Prototype portal notification only; no hospital system was contacted. Prepare {hospital['emergency_department']} at {hospital['name']}; one demo emergency bed is reserved.",
            target_username=hospital["contact_username"],
        )
        if ambulance["driver_username"]:
            _notify(
                db, case_id, "responder", "Ambulance assignment (portal notification)",
                f"{ambulance['label']} is assigned. This is an in-app prototype notification, not a phone call, SMS, or radio dispatch. Share GPS only after signing in and granting consent.",
                target_username=ambulance["driver_username"],
            )
        _notify(
            db, case_id, "police", "Road incident notification",
            f"Prototype portal notification only; no police service was contacted. Verified emergency at {case['location']}.",
        )
        _notify(
            db, case_id, "traffic", "Traffic control notification",
            f"Prototype portal notification only; no traffic service was contacted. Coordinate traffic control for {case['location']}.",
        )
        _notify(
            db, case_id, "fire", "Fire and rescue notification",
            f"Prototype portal notification only; no fire/rescue service was contacted. Stand by for required rescue support at {case['location']}.",
        )
        db.commit()
        return True, status


BASE_STYLE = """
<style>
.response-wrap{max-width:1200px;margin:0 auto}.response-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}.response-card{padding:1.15rem;border:1px solid rgba(128,150,180,.25);border-radius:12px;background:rgba(35,42,55,.55)}.response-card h3{margin-top:0}.response-actions{display:flex;flex-wrap:wrap;gap:.6rem;margin-top:1rem}.status-live{color:#39d98a;font-weight:700}.muted{color:#aeb7c4}.case-table{width:100%;border-collapse:collapse}.case-table th,.case-table td{padding:.85rem;text-align:left;border-bottom:1px solid rgba(128,150,180,.2)}@media(max-width:800px){.response-grid{grid-template-columns:1fr}}
</style>
"""


RESPONSE_CENTER_PAGE = """
{% extends 'base.html' %}{% block title %}Live Response Centre | ResQLink{% endblock %}{% block content %}
""" + BASE_STYLE + """
<div class="response-wrap"><div class="page-header"><div><h1>Live Response Centre</h1><p>CCTV signals wait for control-room verification before dispatch or responder notification.</p></div><a class="btn btn-primary" href="{{ url_for('cctv_demo_upload') }}">Upload CCTV demo</a></div>
<div class="alert alert-info"><strong>Prototype notice:</strong> video analysis, driver GPS, bed availability, and notifications are simulated inside this portal. Operator confirmation is required before response teams are alerted.</div>
<table class="case-table"><thead><tr><th>Case</th><th>Location / confidence</th><th>Ambulance</th><th>Hospital preparation</th><th>Live status</th><th></th></tr></thead><tbody>{% for case in cases %}<tr><td><strong>{{ case.case_number }}</strong><br><small class="muted">{{ case.created_at }}</small></td><td>{{ case.location }}<br><small>{{ '%.0f'|format(case.confidence*100) }}% CCTV signal</small></td><td>{{ case.ambulance_id or 'Awaiting assignment' }}</td><td>{{ case.hospital_name or 'Awaiting hospital' }}{% if case.emergency_beds is not none %}<br><small>{{ case.emergency_beds }} emergency beds currently available</small>{% endif %}</td><td class="status-live">{{ case.status }}</td><td><a class="btn btn-sm btn-outline" href="{{ url_for('response_case', case_id=case.id) }}">Open</a></td></tr>{% else %}<tr><td colspan="6" class="muted">No CCTV signals await review. Upload authorised footage to create an operator-review case.</td></tr>{% endfor %}</tbody></table></div>{% endblock %}
"""


ALERTS_PAGE = """
{% extends 'base.html' %}{% block title %}My Emergency Alerts | ResQLink{% endblock %}{% block content %}
""" + BASE_STYLE + """
<div class="response-wrap"><div class="page-header"><div><h1>My Emergency Alerts</h1><p>Role-scoped alerts for {{ user.role|replace('_', ' ')|title }}.</p></div></div><div class="response-grid">{% for alert in alerts %}<article class="response-card"><h3>{{ alert.title }}</h3><p>{{ alert.body }}</p><small class="muted">{{ alert.created_at }} · {{ alert.state }}</small><div class="response-actions"><a class="btn btn-sm btn-primary" href="{{ url_for('response_case', case_id=alert.case_id) }}">Open live case</a></div></article>{% else %}<p class="muted">There are no alerts for your role.</p>{% endfor %}</div></div>{% endblock %}
"""


CASE_PAGE = """
{% extends 'base.html' %}{% block title %}{{ case.case_number }} | ResQLink{% endblock %}{% block content %}
""" + BASE_STYLE + """
<div class="response-wrap">
  <div class="alert alert-warning"><strong>Prototype only:</strong> this response centre does not connect to emergency service radio, SMS, phone dispatch, hospital systems, or external responders. “Alerts queued” means in-app database notifications. CCTV confidence is not an accident probability.</div>
  <div class="page-header"><div><h1>{{ case.case_number }}</h1><p>{{ case.location }} · {{ '%.0f'|format(case.confidence * 100) }}% prototype CCTV signal</p></div><a class="btn btn-outline" href="{{ url_for('response_alerts') }}">My alerts</a></div>
  {% if can_verify %}<section class="response-card"><h3>Operator verification required</h3><p>Review the source video and evidence. This candidate signal is not a confirmed collision; no responder or hospital has been contacted.</p><form method="post" action="{{ url_for('response_verify', case_id=case.id) }}"><input type="hidden" name="_csrf_token" value="{{ csrf_token() }}"><button class="btn btn-primary" type="submit">Verify evidence &amp; record prototype assignment</button></form></section>{% endif %}
  <div class="response-grid">
    <section class="response-card"><h3>Case and notification status</h3><p class="status-live">{{ case.status }}</p><p>{{ case.analysis_state }}</p><p class="muted">This status records the application's workflow only. Confirm real-world dispatch directly with the responsible agency.</p>{% if can_update_status %}<form method="post" action="{{ url_for('response_status', case_id=case.id) }}"><input type="hidden" name="_csrf_token" value="{{ csrf_token() }}"><label>Status update</label><select name="status">{% for status in allowed_statuses %}<option>{{ status }}</option>{% endfor %}</select><button class="btn btn-primary" type="submit">Update case status</button></form>{% else %}<p class="muted">Your linked role can view this case but cannot change control-room status.</p>{% endif %}</section>
    <section class="response-card"><h3>Ambulance &amp; driver GPS</h3><p><strong>{{ case.ambulance_id or 'Not assigned' }}</strong></p>{% if ambulance %}<p>Driver contact number: {{ ambulance.driver_phone }} <small class="muted">(contact only; not a tracking method)</small></p><p>Linked driver account: {{ ambulance.driver_username or 'No verified driver account linked' }}</p>{% endif %}<p class="muted">Location is obtained from the assigned driver's signed-in account and phone/browser GPS permission—not from the phone number or IP address.</p>{% if can_update_gps and case.ambulance_id and user.role in ('responder','ambulance','amb_crew','ambulance_crew','driver') %}<p id="device-gps-state" aria-live="polite">Device GPS sharing is stopped.</p><button class="btn btn-primary" type="button" id="toggle-device-gps">Start consented phone GPS</button><p class="muted">Starts only after you press the button and grant permission. Requires HTTPS outside localhost. Stop sharing or close this page to end it.</p>{% elif can_update_gps and case.ambulance_id %}<form method="post" action="{{ url_for('response_gps', case_id=case.id) }}"><input type="hidden" name="_csrf_token" value="{{ csrf_token() }}"><label>Manual control-room position (not driver GPS)</label><div class="response-grid"><input name="latitude" type="number" step="any" placeholder="Latitude" required><input name="longitude" type="number" step="any" placeholder="Longitude" required></div><input name="note" maxlength="140" placeholder="Explain the source of this manually entered position"><button class="btn btn-outline" type="submit">Record manual position</button></form>{% endif %}<div id="actual-gps-summary"><strong>Latest recorded position:</strong> {% if locations %}{{ locations[0].latitude }}, {{ locations[0].longitude }} · {{ locations[0].captured_at }}<br><small>{{ locations[0].source|replace('_',' ')|title }}{% if locations[0].accuracy_meters is not none %} · reported accuracy {{ '%.0f'|format(locations[0].accuracy_meters) }} m{% endif %} · {{ locations[0].note }}</small>{% else %}No position received.{% endif %}</div></section>
    <section class="response-card"><h3>Hospital readiness</h3>{% if hospital %}<p><strong>{{ hospital.name }}</strong></p><p>{{ hospital.emergency_department }}</p><p class="status-live">{{ hospital.preparedness_status }}</p><p>{{ hospital.emergency_beds }} demo emergency beds after reservation.</p><p class="muted">This is local prototype data; the hospital has not been contacted through an external system.</p>{% else %}<p>Awaiting hospital assignment.</p>{% endif %}</section>
  </div>
  {% if can_simulate %}
  <section class="response-card" style="margin-top:1rem;border:2px solid #e9a23b">
    <h3>Run response simulation (not a real dispatch)</h3>
    <p>Demonstrates how a response could progress and how common failures should be handled. It does not call, message, notify, or track a real driver/hospital; it does not change actual case status, resource availability, or actual GPS history.</p>
    {% if simulation and simulation.state == 'Running' %}<p class="status-live">Simulation running · {{ simulation.scenario|replace('_',' ')|title }} · step {{ simulation.step + 1 }} of {{ simulation.total_steps }}</p>{% elif simulation %}<p>Last simulation: {{ simulation.scenario|replace('_',' ')|title }} · {{ simulation.state }}</p>{% endif %}
    {% if not simulation or simulation.state != 'Running' %}<form method="post" action="{{ url_for('response_simulation_start', case_id=case.id) }}"><input type="hidden" name="_csrf_token" value="{{ csrf_token() }}"><label for="simulation-scenario">Scenario</label><select id="simulation-scenario" name="scenario"><option value="standard">Standard response and hospital handover</option><option value="driver_no_ack">Driver does not acknowledge</option><option value="gps_lost">GPS signal lost while en route</option><option value="hospital_unavailable">Hospital has no capacity; manual diversion needed</option></select><button class="btn btn-warning" type="submit">Start simulation</button></form>{% endif %}
    <div id="simulation-display" data-run-key="{{ simulation.run_key if simulation else '' }}" data-state="{{ simulation.state if simulation else '' }}" data-advance-url="{{ simulation_advance_url or '' }}" style="display:{% if simulation %}block{% else %}none{% endif %}">
      <p id="simulation-state" aria-live="polite">{% if simulation %}SIMULATION ONLY — {{ simulation.state }}{% endif %}</p><div style="height:10px;background:#263347;border-radius:8px"><div id="simulation-progress" style="height:10px;width:{{ simulation.progress_percent if simulation else 0 }}%;background:#f0a43c;border-radius:8px"></div></div>
      <p>Simulated ambulance position: <span id="simulation-position">{% if simulation and simulation.latitude is not none %}{{ '%.5f'|format(simulation.latitude) }}, {{ '%.5f'|format(simulation.longitude) }}{% else %}No simulated GPS fix yet{% endif %}</span></p>
      <div id="simulation-events">{% if simulation %}{% for event in simulation.events %}<p><strong>{{ event.captured_at }}</strong> — {{ event.message }}{% if event.latitude is not none %}<br><small>SIMULATED coordinates: {{ event.latitude }}, {{ event.longitude }}</small>{% endif %}</p>{% endfor %}{% endif %}</div>
    </div>
  </section>
  {% endif %}
  <section class="response-card" style="margin-top:1rem"><h3>Actual location history (not simulation)</h3>{% for point in locations %}<p><strong>{{ point.captured_at }}</strong> — {{ point.latitude }}, {{ point.longitude }} · {{ point.source|replace('_',' ')|title }}{% if point.accuracy_meters is not none %} · accuracy ±{{ '%.0f'|format(point.accuracy_meters) }} m{% endif %}<br><small>{{ point.note }}</small></p>{% else %}<p class="muted">No actual device GPS received yet. The registry/depot coordinate recorded when dispatch was assigned is not a live location.</p>{% endfor %}</section>
</div>
{% endblock %}{% block extra_js %}
<script>
(function(){
  const token=document.querySelector('meta[name="csrf-token"]').content;
  const gpsButton=document.getElementById('toggle-device-gps'),gpsState=document.getElementById('device-gps-state');
  let watchId=null,lastSent=0;
  async function sendGps(position){
    const now=Date.now();if(now-lastSent<10000)return;lastSent=now;
    const c=position.coords;
    try{
      const response=await fetch('{{ response_gps_url }}',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify({latitude:c.latitude,longitude:c.longitude,accuracy_meters:c.accuracy,note:'Consented driver device GPS'})});
      const result=await response.json();if(!response.ok)throw Error(result.error||'GPS update was rejected');
      gpsState.textContent='Device GPS sent to ResQLink at '+new Date().toLocaleTimeString()+' (reported accuracy ±'+Math.round(c.accuracy)+' m).';
    }catch(error){gpsState.textContent='GPS update failed: '+error.message;}
  }
  if(gpsButton)gpsButton.addEventListener('click',()=>{
    if(watchId!==null){navigator.geolocation.clearWatch(watchId);watchId=null;gpsButton.textContent='Start consented phone GPS';gpsState.textContent='Device GPS sharing stopped.';return;}
    if(!navigator.geolocation){gpsState.textContent='This browser does not support device GPS.';return;}
    watchId=navigator.geolocation.watchPosition(sendGps,error=>{gpsState.textContent='GPS unavailable: '+error.message;},{enableHighAccuracy:true,maximumAge:5000,timeout:20000});
    gpsButton.textContent='Stop phone GPS';gpsState.textContent='Waiting for phone permission and GPS fix…';
  });
  window.addEventListener('pagehide',()=>{if(watchId!==null)navigator.geolocation.clearWatch(watchId);});
  const simulation=document.getElementById('simulation-display');
  if(simulation&&simulation.dataset.runKey){
    const events=document.getElementById('simulation-events'),status=document.getElementById('simulation-state'),progress=document.getElementById('simulation-progress'),position=document.getElementById('simulation-position');
    async function advance(){
      try{
        const response=await fetch(simulation.dataset.advanceUrl,{method:'POST',headers:{'X-CSRF-Token':token},credentials:'same-origin'});
        const data=await response.json();if(!response.ok)throw Error(data.error||'Simulation step failed');
        status.textContent='SIMULATION ONLY — '+data.state+' · '+data.scenario.replaceAll('_',' ');
        progress.style.width=data.progress_percent+'%';
        position.textContent=data.latitude==null?'No simulated GPS fix':Number(data.latitude).toFixed(5)+', '+Number(data.longitude).toFixed(5)+' (simulated)';
        events.replaceChildren();
        data.events.forEach(event=>{
          const line=document.createElement('p'),stamp=document.createElement('strong');stamp.textContent=event.captured_at+' — ';
          line.append(stamp,document.createTextNode(event.message));
          if(event.latitude!=null){const coords=document.createElement('small');coords.textContent='SIMULATED coordinates: '+event.latitude+', '+event.longitude;line.append(document.createElement('br'),coords);}
          events.append(line);
        });
        if(data.state==='Completed')clearInterval(timer);
      }catch(error){status.textContent='Simulation paused — '+error.message;}
    }
    const timer=setInterval(advance,4000);advance();
  }
})();
</script>{% endblock %}
"""


def register_emergency_orchestration(app, current_user, db_path):
    def _user():
        return current_user()

    def _allowed_case(user, case_id):
        if not user:
            return False
        if user["role"] in CONTROL_ROLES:
            return True
        with _connect(db_path) as db:
            ensure_response_schema(db)
            if user["role"] in AMBULANCE_ROLES:
                return bool(db.execute(
                    """SELECT 1 FROM response_cases c
                       JOIN response_ambulances a ON a.ambulance_id=c.ambulance_id
                       WHERE c.id=? AND a.driver_username=?""",
                    (case_id, user["username"]),
                ).fetchone())
            if user["role"] == "hospital":
                return bool(db.execute(
                    """SELECT 1 FROM response_cases c
                       JOIN response_hospitals h ON h.hospital_id=c.hospital_id
                       WHERE c.id=? AND h.contact_username=?""",
                    (case_id, user["username"]),
                ).fetchone())
            return bool(db.execute(
                """SELECT 1 FROM response_notifications
                   WHERE case_id=? AND target_role=?
                     AND (target_username IS NULL OR target_username=?)""",
                (case_id, user["role"], user["username"]),
            ).fetchone())

    @app.route("/response-centre")
    def response_centre():
        user = _user()
        if not user:
            return redirect(url_for("login", next=request.path))
        if user["role"] not in CONTROL_ROLES:
            abort(403)
        with _connect(db_path) as db:
            ensure_response_schema(db)
            cases = db.execute(
                """SELECT c.*, h.name AS hospital_name, h.emergency_beds
                   FROM response_cases c LEFT JOIN response_hospitals h ON h.hospital_id = c.hospital_id
                   ORDER BY c.id DESC LIMIT 30"""
            ).fetchall()
        return render_template_string(RESPONSE_CENTER_PAGE, cases=cases)

    @app.route("/response-alerts")
    def response_alerts():
        user = _user()
        if not user:
            return redirect(url_for("login", next=request.path))
        with _connect(db_path) as db:
            ensure_response_schema(db)
            alerts = db.execute(
                """SELECT * FROM response_notifications
                   WHERE target_role=? AND (target_username IS NULL OR target_username=?)
                   ORDER BY id DESC LIMIT 40""",
                (user["role"], user["username"]),
            ).fetchall()
            db.execute(
                """UPDATE response_notifications SET state='Seen'
                   WHERE target_role=? AND (target_username IS NULL OR target_username=?)
                     AND state='New'""",
                (user["role"], user["username"]),
            )
            db.commit()
        return render_template_string(ALERTS_PAGE, user=user, alerts=alerts)

    @app.route("/response-case/<int:case_id>")
    def response_case(case_id):
        user = _user()
        if not user:
            return redirect(url_for("login", next=request.path))
        if not _allowed_case(user, case_id):
            abort(403)
        with _connect(db_path) as db:
            ensure_response_schema(db)
            case = db.execute("SELECT * FROM response_cases WHERE id = ?", (case_id,)).fetchone()
            if not case:
                abort(404)
            ambulance = db.execute("SELECT * FROM response_ambulances WHERE ambulance_id = ?", (case["ambulance_id"],)).fetchone() if case["ambulance_id"] else None
            hospital = db.execute("SELECT * FROM response_hospitals WHERE hospital_id = ?", (case["hospital_id"],)).fetchone() if case["hospital_id"] else None
            locations = db.execute("SELECT * FROM response_location_updates WHERE case_id = ? ORDER BY id DESC LIMIT 15", (case_id,)).fetchall()
            simulation_run = db.execute(
                "SELECT * FROM response_simulation_runs WHERE case_id=? ORDER BY id DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            simulation = _simulation_payload(db, simulation_run) if simulation_run else None
        role = user["role"]
        if role in CONTROL_ROLES:
            allowed_statuses = (
                ["False alarm"] if not case["ambulance_id"] and case["status"] != "False alarm"
                else (["Under control", "Resolved"] if case["ambulance_id"] else [])
            )
        elif role in AMBULANCE_ROLES:
            allowed_statuses = ["Ambulance en route", "At scene", "Patient onboard", "Arrived at hospital"]
        elif role == "hospital":
            allowed_statuses = ["Hospital ready", "Patient received"]
        else:
            allowed_statuses = [case["status"]]
        return render_template_string(
            CASE_PAGE, user=user, case=case, ambulance=ambulance, hospital=hospital,
            locations=locations, allowed_statuses=allowed_statuses,
            can_update_status=role in CONTROL_ROLES | AMBULANCE_ROLES | {"hospital"},
            can_update_gps=(
                role in CONTROL_ROLES
                or (
                    role in AMBULANCE_ROLES
                    and ambulance
                    and ambulance["driver_username"] == user["username"]
                )
            ),
            can_verify=role in CONTROL_ROLES and case["status"] == "Awaiting operator verification",
            can_simulate=role in CONTROL_ROLES and bool(case["ambulance_id"] and case["hospital_id"]),
            simulation=simulation,
            simulation_advance_url=(
                url_for("response_simulation_advance", run_key=simulation["run_key"])
                if simulation else ""
            ),
            response_gps_url=url_for("response_gps", case_id=case_id),
        )

    @app.route("/response-case/<int:case_id>/verify-dispatch", methods=["POST"])
    def response_verify(case_id):
        user = _user()
        if not user or user["role"] not in CONTROL_ROLES:
            abort(403)
        dispatched, message = dispatch_verified_case(db_path, case_id, user["username"])
        if message == "Case not found.":
            abort(404)
        flash(
            "Operator verification recorded and dispatch started."
            if dispatched else message,
            "success" if dispatched else "error",
        )
        return redirect(url_for("response_case", case_id=case_id))

    @app.post("/response-case/<int:case_id>/simulation/start")
    def response_simulation_start(case_id):
        user = _user()
        if not user or user["role"] not in CONTROL_ROLES:
            abort(403)
        scenario = (request.form.get("scenario") or "standard").strip()
        if scenario not in SIMULATION_SCENARIOS:
            abort(400)
        run_key = uuid.uuid4().hex
        timestamp = datetime.now().isoformat(timespec="seconds")
        with _connect(db_path) as db:
            ensure_response_schema(db)
            case = db.execute(
                "SELECT * FROM response_cases WHERE id=?", (case_id,)
            ).fetchone()
            if not case:
                abort(404)
            if not case["ambulance_id"] or not case["hospital_id"]:
                flash("Verify and assign a response before running the dispatch simulation.", "warning")
                return redirect(url_for("response_case", case_id=case_id))
            active_run = db.execute(
                """SELECT id FROM response_simulation_runs
                   WHERE case_id=? AND state='Running'""",
                (case_id,),
            ).fetchone()
            if active_run:
                flash("A simulation is already running for this case.", "warning")
                return redirect(url_for("response_case", case_id=case_id))
            db.execute(
                """INSERT INTO response_simulation_runs
                   (run_key,case_id,scenario,step,state,started_at,updated_at)
                   VALUES(?,?,?,-1,'Running',?,?)""",
                (run_key, case_id, scenario, timestamp, timestamp),
            )
            db.execute(
                """INSERT INTO response_simulation_events
                   (run_key,step,message,captured_at)
                   VALUES(?, -1, ?, ?)""",
                (
                    run_key,
                    "SIMULATION STARTED — no real emergency service, driver, hospital, or department was contacted; actual case/GPS data is unchanged.",
                    timestamp,
                ),
            )
            db.commit()
        flash("Response simulation started. All following events and positions are simulated only.", "success")
        return redirect(url_for("response_case", case_id=case_id))

    @app.post("/response-simulation/<run_key>/advance")
    def response_simulation_advance(run_key):
        user = _user()
        if not user or user["role"] not in CONTROL_ROLES:
            abort(403)
        with _connect(db_path) as db:
            ensure_response_schema(db)
            run = db.execute(
                "SELECT * FROM response_simulation_runs WHERE run_key=?",
                (run_key,),
            ).fetchone()
            if not run:
                abort(404)
            case = db.execute(
                "SELECT * FROM response_cases WHERE id=?", (run["case_id"],)
            ).fetchone()
            if not case:
                abort(404)
            steps = SIMULATION_SCENARIOS[run["scenario"]]
            next_step = run["step"] + 1
            if run["state"] == "Running" and next_step < len(steps):
                message, fraction, destination = steps[next_step]
                latitude, longitude = _simulation_coordinates(
                    db, case, case["ambulance_id"], fraction, destination
                )
                state = "Completed" if next_step == len(steps) - 1 else "Running"
                timestamp = datetime.now().isoformat(timespec="seconds")
                db.execute(
                    """INSERT INTO response_simulation_events
                       (run_key,step,message,latitude,longitude,captured_at)
                       VALUES(?,?,?,?,?,?)""",
                    (run_key, next_step, message, latitude, longitude, timestamp),
                )
                db.execute(
                    """UPDATE response_simulation_runs
                       SET step=?,state=?,updated_at=? WHERE run_key=?""",
                    (next_step, state, timestamp, run_key),
                )
                run = db.execute(
                    "SELECT * FROM response_simulation_runs WHERE run_key=?",
                    (run_key,),
                ).fetchone()
            payload = _simulation_payload(db, run)
            db.commit()
        return jsonify(payload)

    @app.route("/response-case/<int:case_id>/status", methods=["POST"])
    def response_status(case_id):
        user = _user()
        if not user or not _allowed_case(user, case_id):
            abort(403)
        status = (request.form.get("status") or "").strip()[:100]
        with _connect(db_path) as db:
            ensure_response_schema(db)
            case = db.execute("SELECT status,ambulance_id FROM response_cases WHERE id = ?", (case_id,)).fetchone()
            if not case:
                abort(404)
            if user["role"] in CONTROL_ROLES:
                if case["ambulance_id"]:
                    permitted = {"Under control", "Resolved"}
                elif case["status"] != "False alarm":
                    permitted = {"False alarm"}
                else:
                    permitted = set()
            elif user["role"] in AMBULANCE_ROLES:
                permitted = {"Ambulance en route", "At scene", "Patient onboard", "Arrived at hospital"}
            elif user["role"] == "hospital":
                permitted = {"Hospital ready", "Patient received"}
            else:
                permitted = set()
            if status not in permitted:
                abort(403)
            if status == "False alarm":
                db.execute(
                    """UPDATE response_cases SET status=?,analysis_state=?,
                       updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                    (status, "Operator reviewed and marked this signal as a false alarm.", case_id),
                )
            else:
                db.execute("UPDATE response_cases SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, case_id))
            _notify(db, case_id, "operator", "Case status updated", f"{user['role'].title()} updated the live case to: {status}.")
            db.commit()
        flash("Live incident status updated.", "success")
        return redirect(url_for("response_case", case_id=case_id))

    @app.route("/response-case/<int:case_id>/gps", methods=["POST"])
    def response_gps(case_id):
        user = _user()
        if not user or user["role"] not in CONTROL_ROLES | AMBULANCE_ROLES or not _allowed_case(user, case_id):
            abort(403)
        payload = request.get_json(silent=True) if request.is_json else request.form
        payload = payload or {}
        try:
            latitude, longitude = float(payload["latitude"]), float(payload["longitude"])
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError
            raw_accuracy = payload.get("accuracy_meters")
            accuracy = float(raw_accuracy) if raw_accuracy not in (None, "") else None
            if accuracy is not None and not 0 <= accuracy <= 100000:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            if request.is_json:
                return {"error": "Enter valid GPS coordinates and accuracy."}, 400
            flash("Enter valid GPS coordinates.", "error")
            return redirect(url_for("response_case", case_id=case_id))
        with _connect(db_path) as db:
            ensure_response_schema(db)
            case = db.execute("SELECT ambulance_id FROM response_cases WHERE id = ?", (case_id,)).fetchone()
            if not case or not case["ambulance_id"]:
                abort(404)
            ambulance = db.execute(
                "SELECT driver_username FROM response_ambulances WHERE ambulance_id=?",
                (case["ambulance_id"],),
            ).fetchone()
            if user["role"] not in CONTROL_ROLES and (
                not ambulance or ambulance["driver_username"] != user["username"]
            ):
                abort(403)
            note = (payload.get("note") or "Live location update")
            if not isinstance(note, str):
                if request.is_json:
                    return {"error": "Location note must be text."}, 400
                flash("Enter a valid location note.", "error")
                return redirect(url_for("response_case", case_id=case_id))
            source = "device_gps" if request.is_json else "manual"
            db.execute("UPDATE response_ambulances SET latitude = ?, longitude = ?, last_location_at = CURRENT_TIMESTAMP WHERE ambulance_id = ?", (latitude, longitude, case["ambulance_id"]))
            db.execute(
                """INSERT INTO response_location_updates
                   (case_id,ambulance_id,latitude,longitude,accuracy_meters,source,note)
                   VALUES(?,?,?,?,?,?,?)""",
                (case_id, case["ambulance_id"], latitude, longitude, accuracy, source, note[:140]),
            )
            _notify(db, case_id, "operator", "Ambulance GPS update", f"{'Driver device' if source == 'device_gps' else 'Control-room manual entry'} location for {case['ambulance_id']}: {latitude:.5f}, {longitude:.5f}.")
            db.commit()
        if request.is_json:
            return {"ok": True, "captured_at": datetime.now().isoformat(timespec="seconds"), "accuracy_meters": accuracy}
        flash("Driver GPS update shared with the response team.", "success")
        return redirect(url_for("response_case", case_id=case_id))
