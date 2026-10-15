"""
ResQLink AI — Complete Web Portal
Smart Road Accident Detection & Emergency Response Coordination System
"""

import sqlite3, secrets, os, json, math, hashlib, threading, time, hmac, logging, tempfile
import click
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from urllib.parse import urljoin, urlparse

import bcrypt
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_from_directory, abort)
from flask_socketio import SocketIO, join_room
from werkzeug.utils import secure_filename

DEPLOYMENT_ENV = os.environ.get("RESQLINK_ENV", "development").lower()
configured_secret = os.environ.get("RESQLINK_SECRET_KEY")
configured_db_path = os.environ.get("RESQLINK_DATABASE_PATH")
if DEPLOYMENT_ENV == "production" and (not configured_secret or len(configured_secret) < 32):
    raise RuntimeError("Production requires RESQLINK_SECRET_KEY with at least 32 characters.")
if DEPLOYMENT_ENV == "production" and not configured_db_path:
    raise RuntimeError("Production requires an explicit RESQLINK_DATABASE_PATH.")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DB_PATH   = Path(configured_db_path).expanduser().resolve() if configured_db_path else BASE_DIR / "resqlink_incidents.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = configured_secret or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    CCTV_DEMO_CAMERAS_ENABLED=(
        DEPLOYMENT_ENV != "production"
        and os.environ.get("RESQLINK_DEMO_MODE") == "1"
    ),
    CCTV_INLINE_WORKER_ENABLED=DEPLOYMENT_ENV != "production",
    SESSION_COOKIE_SECURE=os.environ.get(
        "RESQLINK_COOKIE_SECURE", "1" if DEPLOYMENT_ENV == "production" else "0"
    ) == "1",
)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "mov", "avi"}

# Same-origin browser clients are the only supported deployment model.  Allowing
# every origin here would let a foreign site attach to the live notification feed.
socketio = SocketIO(app, cors_allowed_origins=None, async_mode="threading")

# ── Constants ─────────────────────────────────────────────────────────────────
ROLES = {
    "admin":     "Administrator",
    "operator":  "Emergency Operator",
    "responder": "Emergency Responder",
    "hospital":  "Hospital Staff",
    "police":    "Police / Traffic",
    "fire":      "Fire & Rescue",
    "traffic":   "Traffic Control",
    "public":    "Public User",
    "viewer":    "Read-Only Viewer",
}

PERMISSIONS = {
    "admin":    {"manage_users","view_incidents","review_incidents","manage_resources",
                 "view_audit_log","view_reports","manage_agencies","view_analytics",
                 "manage_registries","view_blockchain"},
    "operator": {"view_incidents","review_incidents","manage_resources",
                 "view_reports","manage_agencies","view_analytics","view_blockchain"},
    "responder":{"view_assigned_incidents","update_response_status"},
    "hospital": {"view_hospital_requests","update_hospital_status","submit_report"},
    "police":   {"view_assigned_incidents","update_response_status"},
    "fire":     {"view_assigned_incidents","update_response_status"},
    "traffic":  {"view_assigned_incidents","update_response_status"},
    "public":   {"submit_report","view_own_reports"},
    "viewer":   {"view_incidents"},
}

LIFECYCLE_STATES = [
    "REPORTED","POSSIBLE","UNDER_VERIFICATION","VERIFIED","DISPATCHED",
    "ACCEPTED","EN_ROUTE","ON_SCENE","PATIENT_ASSESSED","TRANSPORTING",
    "HOSPITAL_SELECTED","HOSPITAL_NOTIFIED","ARRIVED","HANDOVER_COMPLETE",
    "CLOSED","FALSE_ALARM"
]
LIFECYCLE_NEXT = {
    "REPORTED":"POSSIBLE","POSSIBLE":"UNDER_VERIFICATION",
    "UNDER_VERIFICATION":"VERIFIED","VERIFIED":"DISPATCHED",
    "DISPATCHED":"ACCEPTED","ACCEPTED":"EN_ROUTE","EN_ROUTE":"ON_SCENE",
    "ON_SCENE":"PATIENT_ASSESSED","PATIENT_ASSESSED":"TRANSPORTING",
    "TRANSPORTING":"HOSPITAL_SELECTED","HOSPITAL_SELECTED":"HOSPITAL_NOTIFIED",
    "HOSPITAL_NOTIFIED":"ARRIVED","ARRIVED":"HANDOVER_COMPLETE",
    "HANDOVER_COMPLETE":"CLOSED",
}
LIFECYCLE_COLORS = {
    "REPORTED":"secondary","POSSIBLE":"secondary","UNDER_VERIFICATION":"primary",
    "VERIFIED":"info","DISPATCHED":"warning","ACCEPTED":"warning","EN_ROUTE":"warning",
    "ON_SCENE":"danger","PATIENT_ASSESSED":"danger","TRANSPORTING":"warning",
    "HOSPITAL_SELECTED":"info","HOSPITAL_NOTIFIED":"info","ARRIVED":"success",
    "HANDOVER_COMPLETE":"success","CLOSED":"secondary","FALSE_ALARM":"dark",
}

STATUSES   = ["New","Under Review","Verified","Closed"]
SEVERITIES = ["Low","Medium","High","Critical"]
SEVERITY_COLORS = {"Low":"success","Medium":"warning","High":"danger","Critical":"dark"}
STATUS_COLORS   = {"New":"secondary","Under Review":"primary","Verified":"success","Closed":"light"}
RESOURCE_STATUSES = {"Available", "Busy", "Offline"}
AMBULANCE_STATUSES = {"AVAILABLE", "DISPATCHED", "EN_ROUTE", "ON_SCENE", "TRANSPORTING", "AT_HOSPITAL", "OUT_OF_SERVICE"}
CAMERA_STATUSES = {"Online", "Offline", "Maintenance", "Unknown"}
HOSPITAL_STATUSES = {"Open", "Divert", "Closed"}
_blockchain_lock = threading.Lock()
_LOGGER = logging.getLogger(__name__)

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db():
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db

def query(sql, params=(), one=False):
    db = get_db()
    try:
        cur = db.execute(sql, params)
        return cur.fetchone() if one else cur.fetchall()
    finally:
        db.close()

def mutate(sql, params=()):
    db = get_db()
    try:
        db.execute(sql, params)
        db.commit()
    finally:
        db.close()

def _sio_emit(event, data, room=None):
    try: socketio.emit(event, data, to=room)
    except Exception: pass

# ── Blockchain helper ─────────────────────────────────────────────────────────
def _blockchain_record(event_type, event_data_str):
    # Keep height allocation and insertion atomic across requests/background jobs.
    with _blockchain_lock:
        db = get_db()
        try:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT sha256_hash, block_height FROM blockchain_log ORDER BY block_height DESC LIMIT 1").fetchone()
            prev_hash = last["sha256_hash"] if last else "0"*64
            height    = (last["block_height"] + 1) if last else 1
            ts        = datetime.now().isoformat(timespec="seconds")
            raw       = event_type + event_data_str + prev_hash + ts
            h         = hashlib.sha256(raw.encode()).hexdigest()
            db.execute("INSERT INTO blockchain_log(event_type,event_data,sha256_hash,prev_hash,block_height,created_at) VALUES(?,?,?,?,?,?)",
                       (event_type, event_data_str, h, prev_hash, height, ts))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

# ── Notification helper ───────────────────────────────────────────────────────
def create_notification(recipient_role, recipient_username, title, body, notif_type, incident_id=None):
    db = get_db()
    db.execute("INSERT INTO notifications(recipient_role,recipient_username,title,body,notif_type,incident_id) VALUES(?,?,?,?,?,?)",
               (recipient_role, recipient_username, title, body, notif_type, incident_id))
    db.commit(); db.close()
    _sio_emit("new_notification", {"title": title, "type": notif_type}, room=f"user:{recipient_username}")

def _emit_incident_event(event, data, incident_id):
    """Send live event metadata only to command staff and assigned units."""
    _sio_emit(event, data, room="role:admin")
    _sio_emit(event, data, room="role:operator")
    assignees = query("SELECT u.username FROM incident_assignments ia JOIN users u ON u.id=ia.assignee_user_id WHERE ia.incident_id=?", (incident_id,))
    for assignee in assignees:
        _sio_emit(event, data, room=f"user:{assignee['username']}")

def notify_role(role, title, body, notif_type, incident_id=None):
    users = query("SELECT username FROM users WHERE role=? AND active=1", (role,))
    for u in users:
        create_notification(role, u["username"], title, body, notif_type, incident_id)

def notify_hospital_resource(resource_id, title, body, notif_type, incident_id=None):
    """Notify only staff linked to the receiving hospital resource."""
    users = query("""SELECT u.username FROM users u
        JOIN hospital_registry hr ON hr.linked_user_id=u.id
        JOIN response_resources rr ON lower(rr.resource_name)=lower(hr.name)
        WHERE rr.id=? AND u.active=1""", (resource_id,))
    for user in users:
        create_notification("hospital", user["username"], title, body, notif_type, incident_id)

# ── Audit ─────────────────────────────────────────────────────────────────────
def audit(username, action, details=""):
    mutate("INSERT INTO audit_log(username,action,details) VALUES(?,?,?)", (username, action, details))

# ── Initialize DB ─────────────────────────────────────────────────────────────
def initialize_database():
    db = get_db()

    # ── Core tables ──────────────────────────────────────────────
    db.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash BLOB NOT NULL,
        role TEXT NOT NULL DEFAULT 'public',
        agency_name TEXT NOT NULL DEFAULT '',
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_key TEXT NOT NULL UNIQUE,
        vehicle_id INTEGER, frame INTEGER, vehicle_type TEXT,
        movement_per_sec REAL, direction_change_angle REAL, deceleration REAL,
        status TEXT NOT NULL DEFAULT 'New',
        severity TEXT NOT NULL DEFAULT 'Medium',
        notes TEXT NOT NULL DEFAULT '',
        ambulance_assigned INTEGER NOT NULL DEFAULT 0,
        police_assigned INTEGER NOT NULL DEFAULT 0,
        fire_assigned INTEGER NOT NULL DEFAULT 0,
        location_text TEXT NOT NULL DEFAULT '',
        lifecycle_status TEXT NOT NULL DEFAULT 'REPORTED',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS citizen_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reporter_username TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'Citizen Web',
        description TEXT NOT NULL,
        incident_time TEXT NOT NULL,
        location_text TEXT NOT NULL,
        latitude REAL, longitude REAL, contact TEXT,
        evidence_filename TEXT,
        status TEXT NOT NULL DEFAULT 'Submitted',
        linked_incident_id INTEGER,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS response_resources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agency_type TEXT NOT NULL,
        resource_name TEXT NOT NULL,
        location_text TEXT NOT NULL DEFAULT '',
        latitude REAL, longitude REAL,
        status TEXT NOT NULL DEFAULT 'Available',
        capacity_note TEXT NOT NULL DEFAULT '',
        capability TEXT NOT NULL DEFAULT 'Basic',
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, action TEXT NOT NULL, details TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS incident_locations (
        incident_id INTEGER PRIMARY KEY,
        latitude REAL NOT NULL, longitude REAL NOT NULL,
        address TEXT NOT NULL DEFAULT ''
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS ambulance_gps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resource_id INTEGER NOT NULL, incident_id INTEGER,
        latitude REAL NOT NULL, longitude REAL NOT NULL,
        accuracy_meters REAL,
        speed_kmh REAL NOT NULL DEFAULT 0,
        heading REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'EN_ROUTE',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    gps_columns = {row["name"] for row in db.execute("PRAGMA table_info(ambulance_gps)")}
    if "accuracy_meters" not in gps_columns:
        db.execute("ALTER TABLE ambulance_gps ADD COLUMN accuracy_meters REAL")
    db.execute("""CREATE TABLE IF NOT EXISTS incident_assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER NOT NULL,
        assignment_role TEXT NOT NULL,
        assignee_user_id INTEGER NOT NULL,
        resource_label TEXT NOT NULL DEFAULT '',
        assignment_status TEXT NOT NULL DEFAULT 'DISPATCHED',
        assigned_by INTEGER,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(incident_id, assignment_role, assignee_user_id)
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS camera_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id INTEGER NOT NULL,
        external_event_id TEXT NOT NULL,
        incident_id INTEGER,
        observed_at TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        confidence REAL NOT NULL,
        vehicle_type TEXT NOT NULL DEFAULT 'Unknown',
        payload TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(camera_id, external_event_id)
    )""")

    # ── New feature tables ───────────────────────────────────────
    db.execute("""CREATE TABLE IF NOT EXISTS lifecycle_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER NOT NULL,
        from_status TEXT NOT NULL DEFAULT '',
        to_status TEXT NOT NULL,
        username TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER NOT NULL,
        source TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        location_text TEXT NOT NULL DEFAULT '',
        confidence REAL NOT NULL DEFAULT 0.5,
        raw_data TEXT NOT NULL DEFAULT '',
        evidence_score REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient_role TEXT NOT NULL,
        recipient_username TEXT NOT NULL,
        title TEXT NOT NULL,
        body TEXT NOT NULL DEFAULT '',
        notif_type TEXT NOT NULL DEFAULT 'INFO',
        incident_id INTEGER,
        read INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS replan_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER NOT NULL,
        trigger_reason TEXT NOT NULL,
        old_plan TEXT NOT NULL DEFAULT '',
        new_plan TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'PENDING',
        operator_username TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        resolved_at TEXT
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL UNIQUE,
        location_text TEXT NOT NULL DEFAULT '',
        latitude REAL, longitude REAL,
        stream_url TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'Online',
        last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS blockchain_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_data TEXT NOT NULL DEFAULT '',
        sha256_hash TEXT NOT NULL,
        prev_hash TEXT NOT NULL,
        block_height INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS hospital_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER NOT NULL,
        hospital_resource_id INTEGER,
        severity TEXT NOT NULL DEFAULT 'Medium',
        eta_minutes INTEGER NOT NULL DEFAULT 15,
        patient_count INTEGER NOT NULL DEFAULT 1,
        alert_status TEXT NOT NULL DEFAULT 'SENT',
        acknowledged_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── Registries ───────────────────────────────────────────────
    db.execute("""CREATE TABLE IF NOT EXISTS ambulance_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_id TEXT NOT NULL UNIQUE,
        registration_number TEXT NOT NULL DEFAULT '',
        organization TEXT NOT NULL DEFAULT '',
        contact_number TEXT NOT NULL DEFAULT '',
        crew_count INTEGER NOT NULL DEFAULT 2,
        capability TEXT NOT NULL DEFAULT 'Basic',
        base_latitude REAL, base_longitude REAL,
        base_location TEXT NOT NULL DEFAULT '',
        operational_status TEXT NOT NULL DEFAULT 'AVAILABLE',
        current_incident_id INTEGER,
        driver_name TEXT NOT NULL DEFAULT '',
        paramedic_name TEXT NOT NULL DEFAULT '',
        linked_user_id INTEGER,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS hospital_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hospital_code TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        address TEXT NOT NULL DEFAULT '',
        latitude REAL, longitude REAL,
        contact_number TEXT NOT NULL DEFAULT '',
        emergency_contact TEXT NOT NULL DEFAULT '',
        total_beds INTEGER NOT NULL DEFAULT 0,
        available_beds INTEGER NOT NULL DEFAULT 0,
        icu_beds INTEGER NOT NULL DEFAULT 0,
        icu_available INTEGER NOT NULL DEFAULT 0,
        trauma_center INTEGER NOT NULL DEFAULT 0,
        capability_level TEXT NOT NULL DEFAULT 'General',
        operating_status TEXT NOT NULL DEFAULT 'Open',
        linked_user_id INTEGER,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS department_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dept_code TEXT NOT NULL UNIQUE,
        dept_type TEXT NOT NULL,
        name TEXT NOT NULL,
        jurisdiction TEXT NOT NULL DEFAULT '',
        address TEXT NOT NULL DEFAULT '',
        latitude REAL, longitude REAL,
        contact_number TEXT NOT NULL DEFAULT '',
        unit_count INTEGER NOT NULL DEFAULT 0,
        available_units INTEGER NOT NULL DEFAULT 0,
        operational_status TEXT NOT NULL DEFAULT 'Active',
        linked_user_id INTEGER,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── Migrations on existing columns ───────────────────────────
    inc_cols = {r[1] for r in db.execute("PRAGMA table_info(incidents)").fetchall()}
    for col, defn in [("fire_assigned","INTEGER NOT NULL DEFAULT 0"),
                      ("location_text","TEXT NOT NULL DEFAULT ''"),
                      ("lifecycle_status","TEXT NOT NULL DEFAULT 'REPORTED'")]:
        if col not in inc_cols:
            db.execute(f"ALTER TABLE incidents ADD COLUMN {col} {defn}")
    res_cols = {r[1] for r in db.execute("PRAGMA table_info(response_resources)").fetchall()}
    if "capability" not in res_cols:
        db.execute("ALTER TABLE response_resources ADD COLUMN capability TEXT NOT NULL DEFAULT 'Basic'")
    amb_cols = {r[1] for r in db.execute("PRAGMA table_info(ambulance_registry)").fetchall()}
    if "gps_resource_id" not in amb_cols:
        db.execute("ALTER TABLE ambulance_registry ADD COLUMN gps_resource_id INTEGER")

    demo_mode = DEPLOYMENT_ENV != "production" and os.environ.get("RESQLINK_DEMO_MODE") == "1"

    # ── Seed default demo accounts ────────────────────────────────
    for uname, pwd, role, agency in ([
        ("admin",     "Admin@ResQLink1",     "admin",     "ResQLink HQ"),
        ("operator1", "Operator@ResQLink1",  "operator",  "ResQLink OPS"),
        ("amb_crew1", "AmbCrew@ResQLink1",   "responder", "CMCH Ambulance"),
        ("hospital1", "Hospital@ResQLink1",  "hospital",  "Apollo Hospital"),
        ("police1",   "Police@ResQLink1",    "police",    "Chennai Police"),
        ("fire1",     "Fire@ResQLink1",       "fire",      "Chennai Fire"),
        ("traffic1",  "Traffic@ResQLink1",   "traffic",   "Chennai Traffic"),
        ("public1",   "Public@ResQLink1",    "public",    ""),
    ] if demo_mode else []):
        existing = db.execute("SELECT id FROM users WHERE username=?", (uname,)).fetchone()
        if not existing:
            h = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt())
            db.execute("INSERT INTO users(username,password_hash,role,agency_name) VALUES(?,?,?,?)",
                       (uname, h, role, agency))


    # ── Seed dispatch resources for a clean prototype database ─────
    resource_seed = [
        ("ambulance", "Ambulance Unit 1", "Egmore, Chennai", "Available", "ALS"),
        ("ambulance", "Ambulance Unit 2", "Vadapalani, Chennai", "Available", "Basic"),
        ("ambulance", "Ambulance Unit 3", "Anna Nagar, Chennai", "Available", "ICU"),
        ("hospital", "Apollo Hospital", "Greams Road, Chennai", "Available", "Trauma"),
        ("hospital", "MIOT International", "Manapakkam, Chennai", "Available", "ICU"),
        ("police", "Traffic Police OP1", "Koyambedu, Chennai", "Available", "Basic"),
        ("police", "Traffic Police OP2", "Anna Salai, Chennai", "Available", "Basic"),
        ("fire", "Fire Station 3", "Ashok Nagar, Chennai", "Available", "Trauma"),
        ("fire", "Fire Station 7", "Perungudi, Chennai", "Available", "Trauma"),
    ] if demo_mode else []
    for agency_type, name, location, status, capability in resource_seed:
        if not db.execute("SELECT id FROM response_resources WHERE agency_type=? AND resource_name=?", (agency_type, name)).fetchone():
            db.execute("""INSERT INTO response_resources
                (agency_type,resource_name,location_text,status,capability) VALUES(?,?,?,?,?)""",
                (agency_type, name, location, status, capability))

    # ── Seed resource GPS ─────────────────────────────────────────
    for name, lat, lng in ([
        ("Ambulance Unit 1",13.0814,80.2735),("Ambulance Unit 2",13.0525,80.2191),
        ("Ambulance Unit 3",13.0870,80.2102),("Apollo Hospital",13.0562,80.2477),
        ("MIOT International",13.0124,80.1802),("Traffic Police OP1",13.0694,80.1993),
        ("Traffic Police OP2",13.0563,80.2503),("Fire Station 3",13.0324,80.2072),
        ("Fire Station 7",12.9678,80.2416),
    ] if demo_mode else []):
        db.execute("UPDATE response_resources SET latitude=?,longitude=? WHERE resource_name=? AND latitude IS NULL",(lat,lng,name))

    # Older demo seeds could leave locations for incident IDs that do not
    # exist. Such orphan rows can collide with IDs of newly-created incidents.
    db.execute("""DELETE FROM incident_locations
        WHERE incident_id NOT IN (SELECT id FROM incidents)""")

    # ── Seed cameras ──────────────────────────────────────────────
    cameras_seed = [
        ("CAM-001","Anna Flyover, NH-45",13.0606,80.2605,"rtsp://cam.resqlink.local/cam001","Online"),
        ("CAM-002","Koyambedu Junction",13.0694,80.1993,"rtsp://cam.resqlink.local/cam002","Online"),
        ("CAM-003","OMR Toll Plaza",12.9678,80.2416,"rtsp://cam.resqlink.local/cam003","Online"),
        ("CAM-004","Poonamallee High Road",13.0756,80.2125,"rtsp://cam.resqlink.local/cam004","Offline"),
        ("CAM-005","ECR Kovalam",12.8406,80.2534,"rtsp://cam.resqlink.local/cam005","Online"),
    ] if demo_mode else []
    for cid,loc,lat,lng,url,st in cameras_seed:
        db.execute("INSERT OR IGNORE INTO cameras(camera_id,location_text,latitude,longitude,stream_url,status) VALUES(?,?,?,?,?,?)",(cid,loc,lat,lng,url,st))

    # ── Seed ambulance registry ───────────────────────────────────
    amb_seed = [
        ("AMB-001","TN09AB1234","CMCH","044-28152000",2,"ALS",13.0814,80.2735,"Egmore, Chennai","AVAILABLE","Rajan Kumar","Dr. Priya"),
        ("AMB-002","TN09CD5678","Vadapalani GH","044-23651555",2,"Basic",13.0525,80.2191,"Vadapalani, Chennai","AVAILABLE","Suresh P","Nurse Lakshmi"),
        ("AMB-003","TN09EF9012","Anna Nagar GH","044-26151234",2,"ICU",13.0870,80.2102,"Anna Nagar, Chennai","AVAILABLE","Venkat R","Dr. Anand"),
        ("AMB-004","TN09GH3456","Apollo Ambulance","044-28293333",3,"Trauma",13.0562,80.2477,"Greams Road, Chennai","OUT_OF_SERVICE","Kumar S","Dr. Meena"),
        ("AMB-005","TN09IJ7890","MIOT Ambulance","044-22492288",2,"ALS",13.0124,80.1802,"Manapakkam, Chennai","AVAILABLE","Babu T","Nurse Saranya"),
    ] if demo_mode else []
    for uid,reg,org,cont,crew,cap,lat,lng,bloc,ops,drv,para in amb_seed:
        db.execute("""INSERT OR IGNORE INTO ambulance_registry
            (unit_id,registration_number,organization,contact_number,crew_count,capability,
             base_latitude,base_longitude,base_location,operational_status,driver_name,paramedic_name)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(uid,reg,org,cont,crew,cap,lat,lng,bloc,ops,drv,para))
    # The prototype's seeded registry IDs align with their response resources;
    # store the link explicitly for live-driver GPS updates.
    if demo_mode:
        db.execute("""UPDATE ambulance_registry SET gps_resource_id=id
            WHERE gps_resource_id IS NULL AND id IN
            (SELECT id FROM response_resources WHERE agency_type='ambulance')""")

    # ── Seed hospital registry ────────────────────────────────────
    hosp_seed = [
        ("HOSP-001","Apollo Hospital","Greams Road, Chennai",13.0562,80.2477,"044-28293333","044-28296000",500,45,40,8,1,"Trauma","Open"),
        ("HOSP-002","MIOT International","Manapakkam, Chennai",13.0124,80.1802,"044-22492288","044-22492289",350,30,25,5,1,"ICU","Open"),
        ("HOSP-003","CMCH","Egmore, Chennai",13.0814,80.2735,"044-28152000","044-28152001",1200,120,80,15,1,"Trauma","Open"),
        ("HOSP-004","Kilpauk Medical College","Kilpauk, Chennai",13.0819,80.2337,"044-26441111","044-26441112",900,85,60,10,0,"General","Open"),
        ("HOSP-005","Stanley Medical College","Park Town, Chennai",13.0878,80.2785,"044-25281111","044-25281112",1000,95,70,12,1,"ICU","Open"),
    ] if demo_mode else []
    for code,name,addr,lat,lng,cont,econt,tb,ab,ib,ia,trauma,cap,ops in hosp_seed:
        db.execute("""INSERT OR IGNORE INTO hospital_registry
            (hospital_code,name,address,latitude,longitude,contact_number,emergency_contact,
             total_beds,available_beds,icu_beds,icu_available,trauma_center,capability_level,operating_status)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(code,name,addr,lat,lng,cont,econt,tb,ab,ib,ia,trauma,cap,ops))

    # ── Seed department registry ──────────────────────────────────
    dept_seed = [
        ("POLICE-001","police","Chennai Central Police","Egmore Zone","Egmore, Chennai",13.0814,80.2735,"100",25,18,"Active"),
        ("POLICE-002","police","Chennai Traffic Police HQ","City Zone","Anna Salai, Chennai",13.0563,80.2503,"103",40,30,"Active"),
        ("POLICE-003","police","Koyambedu Police Station","West Zone","Koyambedu, Chennai",13.0694,80.1993,"044-24710100",12,8,"Active"),
        ("FIRE-001","fire","Chennai Fire Station 3","Ashok Nagar","Ashok Nagar, Chennai",13.0324,80.2072,"101",8,6,"Active"),
        ("FIRE-002","fire","Chennai Fire Station 7","OMR Zone","Perungudi, Chennai",12.9678,80.2416,"044-24912345",6,4,"Active"),
        ("TRAFFIC-001","traffic","Anna Salai Traffic Post","Anna Salai","Anna Salai, Chennai",13.0563,80.2503,"103",5,5,"Active"),
        ("TRAFFIC-002","traffic","OMR Traffic Control","OMR Corridor","OMR, Chennai",12.9678,80.2416,"103",4,3,"Active"),
        ("RESCUE-001","rescue","Chennai Rescue Team","City-wide","Park Town, Chennai",13.0878,80.2785,"044-25281234",10,7,"Active"),
    ] if demo_mode else []
    for code,dtype,name,juri,addr,lat,lng,cont,uc,au,ops in dept_seed:
        db.execute("""INSERT OR IGNORE INTO department_registry
            (dept_code,dept_type,name,jurisdiction,address,latitude,longitude,
             contact_number,unit_count,available_units,operational_status)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",(code,dtype,name,juri,addr,lat,lng,cont,uc,au,ops))

    # Link the supplied demonstration accounts to their matching operational
    # records.  Only fill empty links so real assignments are never replaced.
    for username, table, key_column, key_value in ([
        ("amb_crew1", "ambulance_registry", "unit_id", "AMB-001"),
        ("hospital1", "hospital_registry", "hospital_code", "HOSP-001"),
        ("police1", "department_registry", "dept_code", "POLICE-001"),
        ("fire1", "department_registry", "dept_code", "FIRE-001"),
        ("traffic1", "department_registry", "dept_code", "TRAFFIC-001"),
    ] if demo_mode else []):
        user = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if user:
            db.execute(
                f"UPDATE {table} SET linked_user_id=? "
                f"WHERE {key_column}=? AND (linked_user_id IS NULL OR linked_user_id=?)",
                (user["id"], key_value, user["id"]),
            )

    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_blockchain_log_height ON blockchain_log(block_height)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_assignment_user ON incident_assignments(assignee_user_id, incident_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ambulance_gps_latest ON ambulance_gps(resource_id, updated_at DESC)")

    db.commit(); db.close()


@app.cli.command("create-admin")
def create_admin_command():
    """Create an administrator account without a built-in production password."""
    initialize_database()
    username = click.prompt("Administrator username").strip()
    if not 3 <= len(username) <= 64:
        raise click.ClickException("Username must be between 3 and 64 characters.")
    if query("SELECT 1 FROM users WHERE username=?", (username,), one=True):
        raise click.ClickException("That username already exists.")
    password = click.prompt(
        "Administrator password", hide_input=True, confirmation_prompt=True
    )
    if len(password) < 12:
        raise click.ClickException("Password must be at least 12 characters.")
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    mutate(
        "INSERT INTO users(username,password_hash,role,agency_name) VALUES(?,?,?,?)",
        (username, password_hash, "admin", "System Administration"),
    )
    click.echo(f"Administrator account {username!r} created.")


@app.cli.command("backup-database")
@click.argument("destination", type=click.Path(path_type=Path, dir_okay=False))
def backup_database_command(destination):
    """Create a consistent, integrity-checked SQLite backup without overwriting."""
    source_path = Path(DB_PATH).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise click.ClickException(f"Portal database does not exist: {source_path}")
    if destination == source_path:
        raise click.ClickException("Backup destination must differ from the portal database.")
    if destination.exists():
        raise click.ClickException(f"Backup destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise click.ClickException("Backup destination directory must already exist.")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    source = None
    target = None
    try:
        source = sqlite3.connect(str(source_path), timeout=15)
        target = sqlite3.connect(str(temporary_path), timeout=15)
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise sqlite3.DatabaseError("SQLite backup integrity check failed.")
        target.close()
        target = None
        source.close()
        source = None
        os.replace(temporary_path, destination)
    except (OSError, sqlite3.Error) as error:
        raise click.ClickException(f"Database backup failed: {error}") from error
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()
        temporary_path.unlink(missing_ok=True)

    click.echo(f"Database backup created and verified: {destination}")


# ── Auth helpers ──────────────────────────────────────────────────────────────
def current_user(): return session.get("user")

def csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token

app.jinja_env.globals["csrf_token"] = csrf_token

@app.before_request
def enforce_csrf():
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return
    if request.endpoint == "ingest_camera_detection_service":
        expected_key = os.environ.get("RESQLINK_CAMERA_INGEST_KEY", "")
        supplied_key = request.headers.get("X-ResQLink-Camera-Key", "")
        if expected_key and hmac.compare_digest(expected_key, supplied_key):
            return
    expected = session.get("_csrf_token", "")
    supplied = request.form.get("_csrf_token") or request.headers.get("X-CSRF-Token", "")

    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
         app.logger.warning("CSRF validation failed: endpoint=%s method=%s "
        "session_token_present=%s submitted_token_present=%s",
         request.endpoint,
         request.method,
         bool(expected),
         bool(supplied),
         )
         abort(400)


def _safe_redirect_target(target):
    if not target:
        return None
    base = urlparse(request.host_url)
    candidate = urlparse(urljoin(request.host_url, target))
    return target if candidate.scheme in ("http", "https") and candidate.netloc == base.netloc else None

def _active_session_user():
    user = current_user()
    if not user:
        return None
    row = query("SELECT id,username,role,agency_name,active FROM users WHERE id=?", (user.get("id"),), one=True)
    if not row or not row["active"]:
        session.clear()
        return None
    # Authorisation values always come from the database, not a stale session.
    updated = {"id": row["id"], "username": row["username"], "role": row["role"],
               "agency_name": row["agency_name"], "role_name": ROLES.get(row["role"], row["role"])}
    session["user"] = updated
    return updated

def has_permission(role, perm):
    return perm in PERMISSIONS.get(role, set())

def login_required(f):
    @wraps(f)
    def d(*a, **kw):
        if not _active_session_user():
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*a, **kw)
    return d

def permission_required(perm):
    def decorator(f):
        @wraps(f)
        def d(*a, **kw):
            u = _active_session_user()
            if not u: return redirect(url_for("login"))
            if not has_permission(u["role"], perm):
                flash("Access denied.", "danger")
                return redirect(url_for("dashboard"))
            return f(*a, **kw)
        return d
    return decorator

def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def _number(value, field, minimum=None, maximum=None, integer=False):
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a valid number.")
    if not math.isfinite(result) or (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        bounds = f" between {minimum} and {maximum}" if minimum is not None and maximum is not None else " within the permitted range"
        raise ValueError(f"{field} must be{bounds}.")
    return result

def _coordinates(latitude, longitude):
    if latitude in (None, "") and longitude in (None, ""):
        return None, None
    if latitude in (None, "") or longitude in (None, ""):
        raise ValueError("Enter both latitude and longitude, or leave both blank.")
    return (_number(latitude, "Latitude", -90, 90),
            _number(longitude, "Longitude", -180, 180))

def _incident_or_404(iid):
    incident = query("SELECT * FROM incidents WHERE id=?", (iid,), one=True)
    if not incident:
        abort(404)
    return incident

def _hospital_receives_incident(user_id, iid):
    return bool(query("""SELECT 1 FROM hospital_alerts ha
        JOIN response_resources rr ON rr.id=ha.hospital_resource_id
        JOIN hospital_registry hr ON lower(hr.name)=lower(rr.resource_name)
        WHERE ha.incident_id=? AND hr.linked_user_id=? LIMIT 1""", (iid, user_id), one=True))

def _can_view_incident(user, incident):
    role = user["role"]
    if has_permission(role, "view_incidents") or has_permission(role, "review_incidents"):
        return True
    if role == "responder":
        return _is_user_assigned_to_incident(user["id"], incident["id"], role)
    if role in ("police", "traffic"):
        return _is_user_assigned_to_incident(user["id"], incident["id"], role)
    if role == "fire":
        return _is_user_assigned_to_incident(user["id"], incident["id"], role)
    if role == "hospital":
        return _hospital_receives_incident(user["id"], incident["id"])
    return False

def _can_update_incident_lifecycle(user, incident):
    if has_permission(user["role"], "review_incidents"):
        return True
    if not _can_view_incident(user, incident):
        return False
    return user["role"] in {"responder", "police", "fire", "traffic", "hospital"}

def _find_matching_incident(latitude, longitude, observed_at=None):
    """Find an open incident within 500 m and 30 minutes of an observation."""
    if latitude is None or longitude is None:
        return None
    reference = observed_at or datetime.now()
    rows = query("""SELECT i.*, il.latitude, il.longitude
        FROM incidents i JOIN incident_locations il ON il.incident_id=i.id
        WHERE i.status != 'Closed'""")
    for incident in rows:
        try:
            created = datetime.fromisoformat(incident["created_at"])
        except (TypeError, ValueError):
            continue
        if abs((reference - created).total_seconds()) > 1800:
            continue
        if _haversine_km(latitude, longitude, incident["latitude"], incident["longitude"]) <= 0.5:
            return incident
    return None

def _create_or_merge_incident(source, confidence, location_text, latitude, longitude,
                              vehicle_type="Unknown", observed_at=None, raw_data=""):
    observed_at = observed_at or datetime.now()
    match = _find_matching_incident(latitude, longitude, observed_at)
    if match:
        add_evidence(match["id"], source, confidence, location_text, raw_data)
        return match["id"], True
    now = observed_at.isoformat(timespec="seconds")
    key = f"{source.lower()}:{secrets.token_hex(12)}"
    db = get_db()
    try:
        cur = db.execute("""INSERT INTO incidents
            (candidate_key,vehicle_type,status,severity,location_text,lifecycle_status,created_at,updated_at)
            VALUES(?,?, 'New','Medium',?,'REPORTED',?,?)""",
            (key, vehicle_type[:80], location_text[:500], now, now))
        incident_id = cur.lastrowid
        db.execute("INSERT INTO incident_locations(incident_id,latitude,longitude,address) VALUES(?,?,?,?)",
                   (incident_id, latitude, longitude, location_text[:500]))
        db.commit()
    finally:
        db.close()
    add_evidence(incident_id, source, confidence, location_text, raw_data)
    notify_role("operator", f"New {source.title()} Incident #{incident_id}", location_text[:200], "NEW_INCIDENT", incident_id)
    notify_role("admin", f"New {source.title()} Incident #{incident_id}", location_text[:200], "NEW_INCIDENT", incident_id)
    _blockchain_record("INCIDENT_CREATED", f"id={incident_id} source={source}")
    return incident_id, False


# The application can be started either with ``python flask_app.py`` or with
# ``flask run``.  The latter does not execute the ``__main__`` block, so make
# schema setup lazy as well.  This keeps an older or freshly-created database
# from causing a 500 on the first dashboard request.
_database_ready = False
_database_ready_lock = threading.Lock()


@app.before_request
def ensure_database_ready():
    global _database_ready
    if _database_ready:
        return None
    with _database_ready_lock:
        if not _database_ready:
            initialize_database()
            _database_ready = True
    return None

def _is_user_assigned_to_incident(user_id, incident_id, role):
    return bool(query("""SELECT 1 FROM incident_assignments
        WHERE incident_id=? AND assignee_user_id=? AND assignment_role=?
        AND assignment_status NOT IN ('CANCELLED','COMPLETED') LIMIT 1""", (incident_id, user_id, role), one=True))

# ── Stats ─────────────────────────────────────────────────────────────────────
def get_stats():
    def c(sql, p=()): return query(sql, p, one=True)["c"]
    return {
        "total":       c("SELECT COUNT(*) as c FROM incidents"),
        "new":         c("SELECT COUNT(*) as c FROM incidents WHERE status='New'"),
        "under_review":c("SELECT COUNT(*) as c FROM incidents WHERE status='Under Review'"),
        "verified":    c("SELECT COUNT(*) as c FROM incidents WHERE status='Verified'"),
        "closed":      c("SELECT COUNT(*) as c FROM incidents WHERE status='Closed'"),
        "critical":    c("SELECT COUNT(*) as c FROM incidents WHERE severity='Critical'"),
        "high":        c("SELECT COUNT(*) as c FROM incidents WHERE severity='High'"),
        "reports":     c("SELECT COUNT(*) as c FROM citizen_reports"),
        "resources":   c("SELECT COUNT(*) as c FROM response_resources WHERE active=1"),
        "ambulances":  c("SELECT COUNT(*) as c FROM response_resources WHERE agency_type='ambulance' AND status='Available' AND active=1"),
        "cameras_online": c("SELECT COUNT(*) as c FROM cameras WHERE status='Online'"),
        "unread_notifs":  0,
    }

def get_role_stats(user):
    """Return only totals a role is entitled to see."""
    role = user["role"]
    if role in {"admin", "operator", "viewer"}:
        return get_stats()
    if role in {"responder", "police", "fire", "traffic"}:
        assignments = query("""SELECT COUNT(*) AS c FROM incident_assignments ia JOIN incidents i ON i.id=ia.incident_id
            WHERE ia.assignee_user_id=? AND ia.assignment_role=? AND ia.assignment_status NOT IN ('CANCELLED','COMPLETED')""",
            (user["id"], role), one=True)["c"]
        return {"total": assignments, "new": 0, "under_review": 0, "verified": 0, "closed": 0,
                "critical": 0, "high": 0, "reports": 0, "resources": 0, "ambulances": 0,
                "cameras_online": 0, "unread_notifs": 0}
    if role == "hospital":
        alerts = query("""SELECT COUNT(*) AS c FROM hospital_alerts ha JOIN response_resources rr ON rr.id=ha.hospital_resource_id
            JOIN hospital_registry hr ON lower(hr.name)=lower(rr.resource_name)
            WHERE hr.linked_user_id=? AND ha.alert_status='SENT'""", (user["id"],), one=True)["c"]
        return {"total": alerts, "new": 0, "under_review": 0, "verified": 0, "closed": 0,
                "critical": 0, "high": 0, "reports": 0, "resources": 0, "ambulances": 0,
                "cameras_online": 0, "unread_notifs": 0}
    reports = query("SELECT COUNT(*) AS c FROM citizen_reports WHERE reporter_username=?", (user["username"],), one=True)["c"]
    return {"total": reports, "new": 0, "under_review": 0, "verified": 0, "closed": 0,
            "critical": 0, "high": 0, "reports": reports, "resources": 0, "ambulances": 0,
            "cameras_online": 0, "unread_notifs": 0}

# ── Map / GPS helpers ─────────────────────────────────────────────────────────
def _haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0; dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def _bearing(lat1,lon1,lat2,lon2):
    dlon=math.radians(lon2-lon1); x=math.sin(dlon)*math.cos(math.radians(lat2))
    y=(math.cos(math.radians(lat1))*math.sin(math.radians(lat2))-math.sin(math.radians(lat1))*math.cos(math.radians(lat2))*math.cos(dlon))
    return (math.degrees(math.atan2(x,y))+360)%360

def _simulate_ambulance_pos(alat,alng,ilat,ilng,t_sec):
    td=_haversine_km(alat,alng,ilat,ilng); speed=60.0
    travel_s=(td/speed)*3600 if td>0 else 1
    frac=min(t_sec/travel_s,1.0)
    lat=alat+frac*(ilat-alat)
    lng=alng+frac*(ilng-alng)
    hdg=_bearing(alat,alng,ilat,ilng)
    if frac>=1.0: return ilat,ilng,0,hdg,"ON_SCENE"
    elif frac>0.85: return lat,lng,25.0,hdg,"ON_SCENE"
    else: return lat,lng,speed,hdg,"EN_ROUTE"

def _utc_datetime(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

# ── Allocation scoring ────────────────────────────────────────────────────────
def _allocation_score(severity, capability, availability, dist_km, conflict=False):
    SEV_MAP  = {"Critical":1.0,"High":0.75,"Medium":0.5,"Low":0.25}
    CAP_MAP  = {"Trauma":1.0,"ICU":0.85,"ALS":0.7,"Basic":0.5,"General":0.4}
    AVAIL_MAP= {"Available":1.0,"Busy":0.3,"Offline":0.0,"Open":1.0,"Divert":0.2}
    s  = SEV_MAP.get(severity,0.5)
    cp = CAP_MAP.get(capability,0.5)
    av = AVAIL_MAP.get(availability,0.5)
    prox = max(0, 1 - dist_km/50)
    tt   = max(0, 1 - dist_km/100)
    cf   = 1 if conflict else 0
    return round(0.25*s + 0.20*cp + 0.25*av + 0.15*prox - 0.10*tt - 0.05*cf, 4)

# ── Evidence fusion ───────────────────────────────────────────────────────────
def _evidence_score(confidence, created_at_str):
    try:
        dt = (datetime.now() - datetime.fromisoformat(created_at_str)).total_seconds()
    except Exception:
        dt = 0
    freshness = math.exp(-0.001 * max(dt, 0))
    return round(confidence * freshness, 4)

def add_evidence(incident_id, source, confidence, location_text="", raw_data="", db=None):
    now = _utc_datetime().isoformat(timespec="seconds")
    score = _evidence_score(confidence, now)
    sql = "INSERT INTO evidence(incident_id,source,timestamp,location_text,confidence,raw_data,evidence_score,created_at) VALUES(?,?,?,?,?,?,?,?)"
    values = (incident_id, source, now, location_text, confidence, raw_data, score, now)
    if db is None:
        mutate(sql, values)
    else:
        db.execute(sql, values)

# ── Adaptive re-planning background thread ────────────────────────────────────
def _replan_checker():
    while True:
        time.sleep(30)
        try:
            active = query("SELECT * FROM incidents WHERE lifecycle_status NOT IN ('CLOSED','FALSE_ALARM')")
            for inc in active:
                reasons = []
                # Check if any assigned ambulance went offline
                if inc["ambulance_assigned"]:
                    offline = query("SELECT id FROM response_resources WHERE agency_type='ambulance' AND status='Offline' AND active=1 LIMIT 1")
                    if offline:
                        reasons.append("Assigned ambulance unit went Offline")
                # Check EN_ROUTE > 30 min without ON_SCENE
                if inc["lifecycle_status"] == "EN_ROUTE":
                    last_ev = query("SELECT created_at FROM lifecycle_events WHERE incident_id=? AND to_status='EN_ROUTE' ORDER BY id DESC LIMIT 1",(inc["id"],),one=True)
                    if last_ev:
                        try:
                            dt = (datetime.now()-datetime.fromisoformat(last_ev["created_at"])).total_seconds()
                            if dt > 1800:
                                reasons.append(f"EN_ROUTE for {int(dt/60)} minutes without reaching ON_SCENE")
                        except Exception:
                            pass
                for reason in reasons:
                    existing = query("SELECT id FROM replan_events WHERE incident_id=? AND trigger_reason=? AND status='PENDING'",(inc["id"],reason),one=True)
                    if not existing:
                        mutate("INSERT INTO replan_events(incident_id,trigger_reason,old_plan,new_plan,status) VALUES(?,?,?,?,'PENDING')",
                               (inc["id"], reason, f"status={inc['lifecycle_status']}", "Requires operator review"))
                        notify_role("operator", f"Re-plan Required — Incident #{inc['id']}", reason, "RE_PLAN", inc["id"])
                        notify_role("admin", f"Re-plan Required — Incident #{inc['id']}", reason, "RE_PLAN", inc["id"])
        except Exception:
            pass

def start_replan_checker():
    """Start one checker only when the portal process is actually serving."""
    thread = threading.Thread(target=_replan_checker, daemon=True, name="resqlink-replan-checker")
    thread.start()
    return thread

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health/live")
def health_live():
    """Process liveness probe; does not disclose configuration or data."""
    return jsonify(status="ok")


@app.get("/health/ready")
def health_ready():
    """Readiness probe for required database schema and SQLite availability."""
    db = get_db()
    try:
        integrity = db.execute("PRAGMA quick_check(1)").fetchone()
        if not integrity or integrity[0] != "ok":
            raise sqlite3.DatabaseError("SQLite integrity check failed.")
        required_tables = {"users", "incidents", "audit_log"}
        existing_tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('users','incidents','audit_log')"
            )
        }
        if required_tables - existing_tables:
            _LOGGER.error("Readiness check failed: required database schema is missing.")
            return jsonify(status="not_ready"), 503
        return jsonify(status="ready")
    except sqlite3.Error:
        _LOGGER.exception("Readiness check could not validate the portal database.")
        return jsonify(status="not_ready"), 503
    finally:
        db.close()


@app.route("/")
def index():
    return redirect(url_for("dashboard")) if current_user() else render_template("landing.html")

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET","POST"])
def login():
    if current_user(): return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        row = query("SELECT id,username,password_hash,role,agency_name FROM users WHERE username=? AND active=1",(username,),one=True)
        ok = False
        if row:
            stored = row["password_hash"]
            if isinstance(stored,str): stored=stored.encode()
            try: ok = bcrypt.checkpw(password.encode(), stored)
            except Exception: ok=False
        if ok:
            session.clear()
            session["user"] = {"id":row["id"],"username":row["username"],"role":row["role"],
                               "agency_name":row["agency_name"],"role_name":ROLES.get(row["role"],row["role"])}
            csrf_token()
            audit(username,"LOGIN")
            return redirect(_safe_redirect_target(request.args.get("next")) or url_for("dashboard"))
        error = "Invalid username or password."
        audit(username,"LOGIN_FAILED")
    return render_template("login.html", error=error)

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    u = current_user()
    if u: audit(u["username"],"LOGOUT")
    session.clear()
    flash("Logged out.","info")
    return redirect(url_for("index"))

@app.route("/register", methods=["GET","POST"])
def register():
    if current_user(): return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        confirm  = request.form.get("confirm_password","")
        # Public registration must not create emergency-service accounts.  Those
        # identities are provisioned by an administrator after verification.
        role     = "public"
        agency   = request.form.get("agency_name","").strip()

        if len(username)<3: error="Username must be ≥3 characters."
        elif len(password)<12: error="Password must be ≥12 characters."
        elif password!=confirm: error="Passwords do not match."
        elif query("SELECT id FROM users WHERE username=?",(username,),one=True): error="Username already taken."
        else:
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            db = get_db()
            db.execute("INSERT INTO users(username,password_hash,role,agency_name) VALUES(?,?,?,?)",(username,hashed,role,agency))
            db.commit(); db.close()
            audit(username,"REGISTER",f"role={role}")
            flash("Citizen account created! Please log in.","success")
            return redirect(url_for("login"))
    return render_template("register.html", roles=ROLES, error=error)

# ── Dashboard (role-routed) ───────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    stats = get_role_stats(user)
    # unread notifications
    unread = query("SELECT COUNT(*) as c FROM notifications WHERE recipient_username=? AND read=0",(user["username"],),one=True)["c"]
    stats["unread_notifs"] = unread

    role = user["role"]

    if role in ("admin",):
        recent = query("SELECT * FROM incidents ORDER BY id DESC LIMIT 10")
        recent_reports = query("SELECT * FROM citizen_reports ORDER BY id DESC LIMIT 5")
        sev_data  = query("SELECT severity, COUNT(*) as cnt FROM incidents GROUP BY severity")
        st_data   = query("SELECT status, COUNT(*) as cnt FROM incidents GROUP BY status")
        trend_data= query("SELECT DATE(created_at) as day, COUNT(*) as cnt FROM incidents WHERE created_at>=DATE('now','-7 days') GROUP BY day ORDER BY day")
        return render_template("dashboard_admin.html", user=user, stats=stats,
            recent_incidents=[dict(r) for r in recent],
            recent_reports=[dict(r) for r in recent_reports],
            severity_colors=SEVERITY_COLORS, status_colors=STATUS_COLORS,
            sev_labels=json.dumps([r["severity"] for r in sev_data]),
            sev_counts=json.dumps([r["cnt"] for r in sev_data]),
            st_labels=json.dumps([r["status"] for r in st_data]),
            st_counts=json.dumps([r["cnt"] for r in st_data]),
            trend_labels=json.dumps([r["day"] for r in trend_data]),
            trend_counts=json.dumps([r["cnt"] for r in trend_data]))

    elif role == "operator":
        pending = query("SELECT * FROM incidents WHERE status IN ('New','Under Review') ORDER BY id DESC LIMIT 20")
        replans = query("SELECT r.*,i.severity,i.location_text FROM replan_events r JOIN incidents i ON i.id=r.incident_id WHERE r.status='PENDING' ORDER BY r.id DESC LIMIT 10")
        resources = query("SELECT * FROM response_resources WHERE active=1 ORDER BY agency_type")
        return render_template("dashboard_operator.html", user=user, stats=stats,
            pending=[dict(r) for r in pending],
            replans=[dict(r) for r in replans],
            resources=[dict(r) for r in resources],
            severity_colors=SEVERITY_COLORS, status_colors=STATUS_COLORS)

    elif role == "hospital":
        my_hosp = query("SELECT * FROM hospital_registry WHERE linked_user_id=? LIMIT 1",(user["id"],),one=True)
        alerts = []
        if my_hosp:
            alerts = query("""SELECT ha.*,i.severity,i.location_text,i.vehicle_type
                FROM hospital_alerts ha
                JOIN incidents i ON i.id=ha.incident_id
                JOIN response_resources rr ON rr.id=ha.hospital_resource_id
                WHERE ha.alert_status='SENT' AND lower(rr.resource_name)=lower(?)
                ORDER BY ha.id DESC LIMIT 20""", (my_hosp["name"],))
        return render_template("dashboard_hospital.html", user=user, stats=stats,
            alerts=[dict(r) for r in alerts],
            my_hospital=dict(my_hosp) if my_hosp else None)

    elif role in ("responder","police","fire","traffic"):
        my_amb = query("SELECT * FROM ambulance_registry WHERE linked_user_id=? LIMIT 1",(user["id"],),one=True)
        my_dept= query("SELECT * FROM department_registry WHERE linked_user_id=? LIMIT 1",(user["id"],),one=True)
        assigned = query("""SELECT i.* FROM incidents i JOIN incident_assignments ia ON ia.incident_id=i.id
            WHERE ia.assignee_user_id=? AND ia.assignment_role=?
              AND ia.assignment_status NOT IN ('CANCELLED','COMPLETED') AND i.status != 'Closed'
            ORDER BY i.updated_at DESC LIMIT 10""", (user["id"], role))
        return render_template("dashboard_responder.html", user=user, stats=stats,
            my_ambulance=dict(my_amb) if my_amb else None,
            my_dept=dict(my_dept) if my_dept else None,
            assigned=[dict(r) for r in assigned],
            severity_colors=SEVERITY_COLORS, status_colors=STATUS_COLORS,
            lifecycle_colors=LIFECYCLE_COLORS)

    elif role == "viewer":
        recent = query("SELECT * FROM incidents ORDER BY id DESC LIMIT 20")
        return render_template("dashboard_viewer.html", user=user, stats=stats,
            recent_incidents=[dict(r) for r in recent],
            severity_colors=SEVERITY_COLORS, status_colors=STATUS_COLORS)

    # Citizen dashboard contains only the person's own reports and linked cases.
    if role == "public":
        my_reports = query("SELECT * FROM citizen_reports WHERE reporter_username=? ORDER BY id DESC LIMIT 20", (user["username"],))
        return render_template("dashboard_public.html", user=user, stats=stats,
                               reports=[dict(r) for r in my_reports])

    # Safe fallback for unknown roles.
    return render_template("dashboard_viewer.html", user=user, stats=stats,
        recent_incidents=[], severity_colors=SEVERITY_COLORS, status_colors=STATUS_COLORS)

# ── Incidents ─────────────────────────────────────────────────────────────────
@app.route("/incidents")
@login_required
def incidents():
    user = current_user()
    if not (has_permission(user["role"], "view_incidents") or has_permission(user["role"], "view_assigned_incidents")):
        abort(403)
    sf = request.args.get("status",""); svf = request.args.get("severity",""); q = request.args.get("q","").strip()
    sql="SELECT * FROM incidents WHERE 1=1"; params=[]
    if user["role"] in {"responder", "police", "traffic", "fire"}:
        sql += " AND EXISTS (SELECT 1 FROM incident_assignments ia WHERE ia.incident_id=incidents.id AND ia.assignee_user_id=? AND ia.assignment_role=? AND ia.assignment_status NOT IN ('CANCELLED','COMPLETED'))"
        params.extend([user["id"], user["role"]])
    if sf: sql+=" AND status=?"; params.append(sf)
    if svf: sql+=" AND severity=?"; params.append(svf)
    if q:
        sql+=" AND (vehicle_type LIKE ? OR notes LIKE ? OR CAST(vehicle_id AS TEXT) LIKE ?)"
        params+=[f"%{q}%",f"%{q}%",f"%{q}%"]
    sql+=" ORDER BY id DESC LIMIT 200"
    rows=query(sql,params)
    return render_template("incidents.html", user=user, incidents=[dict(r) for r in rows],
        statuses=STATUSES, severities=SEVERITIES, severity_colors=SEVERITY_COLORS,
        status_colors=STATUS_COLORS, status_filter=sf, severity_filter=svf, search=q,
        has_permission=has_permission, lifecycle_colors=LIFECYCLE_COLORS)

@app.route("/incidents/<int:iid>", methods=["GET","POST"])
@login_required
def incident_detail(iid):
    user = current_user()
    row = _incident_or_404(iid)
    if not _can_view_incident(user, row): abort(403)
    incident = dict(row)
    if request.method == "POST":
        if not has_permission(user["role"],"review_incidents"):
            flash("Access denied.","danger"); return redirect(url_for("incident_detail",iid=iid))
        action = request.form.get("action"); now = datetime.now().isoformat(timespec="seconds")
        if action == "update":
            status   = request.form.get("status", incident["status"])
            severity = request.form.get("severity", incident["severity"])
            notes    = request.form.get("notes", incident["notes"])
            location = request.form.get("location_text", incident.get("location_text",""))
            if status not in STATUSES or severity not in SEVERITIES:
                flash("Invalid incident status or severity.","danger")
                return redirect(url_for("incident_detail",iid=iid))
            old_sev  = incident["severity"]
            mutate("UPDATE incidents SET status=?,severity=?,notes=?,location_text=?,updated_at=? WHERE id=?",(status,severity,notes,location,now,iid))
            audit(user["username"],"UPDATE_INCIDENT",f"id={iid} status={status}")
            if severity != old_sev and SEVERITIES.index(severity) > SEVERITIES.index(old_sev):
                notify_role("operator",f"Severity Escalated — Incident #{iid}",f"Changed from {old_sev} to {severity}","SEVERITY_ESCALATION",iid)
                notify_role("admin",f"Severity Escalated — Incident #{iid}",f"Changed from {old_sev} to {severity}","SEVERITY_ESCALATION",iid)
                _blockchain_record("INCIDENT_SEVERITY_ESCALATED",f"id={iid} from={old_sev} to={severity}")
            if status == "Verified":
                _blockchain_record("INCIDENT_VERIFIED",f"id={iid} severity={severity}")
            flash("Incident updated.","success")
        elif action == "assign":
            flash(
                "Select a specific linked unit in the Dispatch Console; broad role-level assignment is disabled.",
                "danger",
            )
        return redirect(url_for("incident_detail",iid=iid))

    related   = query("SELECT * FROM citizen_reports WHERE linked_incident_id=? ORDER BY id DESC",(iid,))
    ev_list   = query("SELECT * FROM evidence WHERE incident_id=? ORDER BY id DESC",(iid,))
    lc_events = query("SELECT * FROM lifecycle_events WHERE incident_id=? ORDER BY id ASC",(iid,))
    replans   = query("SELECT * FROM replan_events WHERE incident_id=? ORDER BY id DESC",(iid,))
    alerts    = query("SELECT * FROM hospital_alerts WHERE incident_id=? ORDER BY id DESC",(iid,))
    loc       = query("SELECT * FROM incident_locations WHERE incident_id=?", (iid,), one=True)
    # Update evidence scores
    for e in ev_list:
        new_score = _evidence_score(e["confidence"], e["created_at"])
        if abs(new_score - e["evidence_score"]) > 0.01:
            mutate("UPDATE evidence SET evidence_score=? WHERE id=?",(new_score,e["id"]))
    ev_list = query("SELECT * FROM evidence WHERE incident_id=? ORDER BY id DESC",(iid,))
    hospitals = query("SELECT * FROM response_resources WHERE agency_type='hospital' AND active=1")
    operational_users = []
    if has_permission(user["role"], "manage_resources"):
        operational_users = query("""SELECT username,role,agency_name FROM users
            WHERE active=1 AND role IN ('responder','hospital','police','fire','traffic')
            ORDER BY role,username""")
    assignments = query("""SELECT ia.*,u.username,u.agency_name FROM incident_assignments ia
        JOIN users u ON u.id=ia.assignee_user_id WHERE ia.incident_id=? ORDER BY ia.id DESC""", (iid,))
    return render_template("incident_detail.html", user=user, incident=incident,
        related=[dict(r) for r in related],
        evidence=[dict(r) for r in ev_list],
        lifecycle_events=[dict(r) for r in lc_events],
        replans=[dict(r) for r in replans],
        alerts=[dict(r) for r in alerts],
        incident_loc=dict(loc) if loc else None,
        hospitals=[dict(r) for r in hospitals],
        operational_users=[dict(r) for r in operational_users],
        assignments=[dict(r) for r in assignments],
        statuses=STATUSES, severities=SEVERITIES,
        lifecycle_states=LIFECYCLE_STATES, lifecycle_colors=LIFECYCLE_COLORS,
        next_lifecycle_state=LIFECYCLE_NEXT.get(incident["lifecycle_status"]),
        can_advance_lifecycle=_can_update_incident_lifecycle(user, incident),
        severity_colors=SEVERITY_COLORS, status_colors=STATUS_COLORS,
        has_permission=has_permission)

@app.route("/incidents/new", methods=["GET","POST"])
@permission_required("review_incidents")
def new_incident():
    user = current_user()
    if request.method == "POST":
        now=datetime.now().isoformat(timespec="seconds")
        vid=request.form.get("vehicle_id",""); frame=request.form.get("frame","")
        key=f"manual:{now}:{vid}:{frame}"
        vtype=request.form.get("vehicle_type","Unknown")
        sev=request.form.get("severity","Medium"); notes=request.form.get("notes","")
        loc=request.form.get("location_text","")
        if sev not in SEVERITIES:
            flash("Invalid incident severity.","danger")
            return redirect(url_for("new_incident"))
        try:
            vid = _number(vid, "Vehicle ID", 0, integer=True) if vid else None
            frame = _number(frame, "Frame", 0, integer=True) if frame else None
            movement = _number(request.form.get("movement_per_sec", "0"), "Movement", 0)
            direction = _number(request.form.get("direction_change_angle", "0"), "Direction change", 0, 360)
            deceleration = _number(request.form.get("deceleration", "0"), "Deceleration", 0)
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("new_incident"))
        mutate("""INSERT OR IGNORE INTO incidents
            (candidate_key,vehicle_id,frame,vehicle_type,movement_per_sec,direction_change_angle,deceleration,
             status,severity,notes,location_text,ambulance_assigned,police_assigned,fire_assigned,
             lifecycle_status,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,'New',?,?,?,0,0,0,'REPORTED',?,?)""",
            (key,vid,frame,vtype,movement,direction,deceleration,
             sev,notes,loc,now,now))
        r=query("SELECT id FROM incidents WHERE candidate_key=?",(key,),one=True)
        iid=r["id"]
        add_evidence(iid,"OPERATOR",0.9,loc,f"Manually created by {user['username']}")
        notify_role("operator",f"New Incident #{iid}",f"Severity: {sev}, Location: {loc}","NEW_INCIDENT",iid)
        notify_role("admin",f"New Incident #{iid}",f"Severity: {sev}, Location: {loc}","NEW_INCIDENT",iid)
        audit(user["username"],"CREATE_INCIDENT",f"id={iid}")
        flash("Incident created.","success")
        return redirect(url_for("incident_detail",iid=iid))
    return render_template("new_incident.html", user=user, severities=SEVERITIES)

# ── Incident lifecycle ────────────────────────────────────────────────────────
@app.route("/incidents/<int:iid>/lifecycle", methods=["POST"])
@login_required
def advance_lifecycle(iid):
    user = current_user()
    inc = _incident_or_404(iid)
    if not _can_update_incident_lifecycle(user, inc):
        return jsonify(error="Access denied"), 403
    payload = request.get_json(silent=True) or {} if request.is_json else request.form
    new_state = payload.get("to_state", "")
    notes = str(payload.get("notes", ""))[:2000]
    old_state = inc["lifecycle_status"]
    allowed_state = LIFECYCLE_NEXT.get(old_state)
    can_false_alarm = has_permission(user["role"], "review_incidents") and old_state not in {"CLOSED", "FALSE_ALARM"}
    if new_state != allowed_state and not (can_false_alarm and new_state == "FALSE_ALARM"):
        flash("Invalid lifecycle state.","danger")
        if request.is_json:
            return jsonify(error="Lifecycle transitions must follow the defined sequence."), 400
        return redirect(url_for("incident_detail",iid=iid))
    status_for_state = {
        "UNDER_VERIFICATION": "Under Review",
        "VERIFIED": "Verified",
        "CLOSED": "Closed",
        "FALSE_ALARM": "Closed",
    }.get(new_state, inc["status"])
    mutate("UPDATE incidents SET lifecycle_status=?,status=?,updated_at=? WHERE id=?",
           (new_state, status_for_state, datetime.now().isoformat(timespec="seconds"), iid))
    mutate("INSERT INTO lifecycle_events(incident_id,from_status,to_status,username,notes) VALUES(?,?,?,?,?)",(iid,old_state,new_state,user["username"],notes))
    audit(user["username"],"LIFECYCLE_ADVANCE",f"id={iid} {old_state}→{new_state}")
    if new_state in ("DISPATCHED","CLOSED","HOSPITAL_NOTIFIED"):
        _blockchain_record(f"INCIDENT_{new_state}",f"id={iid} by={user['username']}")
    _emit_incident_event("incident_update", {"id":iid,"lifecycle_status":new_state}, iid)
    if request.is_json:
        return jsonify(ok=True, new_state=new_state)
    flash(f"Lifecycle advanced to {new_state}.","success")
    return redirect(url_for("incident_detail",iid=iid))

# ── AI Recommendation / Allocation Scoring ────────────────────────────────────
@app.route("/incidents/<int:iid>/recommend")
@login_required
def recommend_resources(iid):
    inc = _incident_or_404(iid)
    if not _can_view_incident(current_user(), inc):
        return jsonify(error="Access denied"), 403
    loc = query("SELECT latitude,longitude FROM incident_locations WHERE incident_id=?",(iid,),one=True)
    ilat = loc["latitude"]  if loc else 13.0827
    ilng = loc["longitude"] if loc else 80.2707

    ambs = query("SELECT * FROM response_resources WHERE agency_type='ambulance' AND active=1")
    amb_scored = []
    for a in ambs:
        if not a["latitude"]: continue
        d = _haversine_km(a["latitude"],a["longitude"],ilat,ilng)
        sc = _allocation_score(inc["severity"], a["capability"] or "Basic", a["status"], d)
        amb_scored.append({"id":a["id"],"name":a["resource_name"],"status":a["status"],
                           "capability":a["capability"] or "Basic","distance_km":round(d,2),
                           "score":sc,"location_text":a["location_text"]})
    amb_scored.sort(key=lambda x:-x["score"])

    hosps = query("SELECT * FROM response_resources WHERE agency_type='hospital' AND active=1")
    hosp_scored = []
    for h in hosps:
        if not h["latitude"]: continue
        d = _haversine_km(h["latitude"],h["longitude"],ilat,ilng)
        sc = _allocation_score(inc["severity"], h["capability"] or "General", h["status"], d)
        hosp_scored.append({"id":h["id"],"name":h["resource_name"],"status":h["status"],
                            "capability":h["capability"] or "General","distance_km":round(d,2),
                            "score":sc,"location_text":h["location_text"]})
    hosp_scored.sort(key=lambda x:-x["score"])
    return jsonify(ambulances=amb_scored[:3], hospitals=hosp_scored[:2], incident_id=iid)

@app.route("/incidents/<int:iid>/accept_recommendation", methods=["POST"])
@permission_required("manage_resources")
def accept_recommendation(iid):
    _incident_or_404(iid)
    flash(
        "Resource recommendations are advisory. Verify the incident, then select a specific linked unit in the Dispatch Console.",
        "warning",
    )
    return redirect(url_for("incident_detail",iid=iid))

# ── Evidence ──────────────────────────────────────────────────────────────────
@app.route("/incidents/<int:iid>/evidence")
@login_required
def incident_evidence(iid):
    incident = _incident_or_404(iid)
    if not _can_view_incident(current_user(), incident):
        return jsonify(error="Access denied"), 403
    rows = query("SELECT * FROM evidence WHERE incident_id=? ORDER BY id DESC",(iid,))
    ev = []
    for r in rows:
        d = dict(r); d["evidence_score"] = _evidence_score(d["confidence"],d["created_at"]); ev.append(d)
    return jsonify(evidence=ev)

@app.route("/incidents/<int:iid>/evidence/add", methods=["POST"])
@login_required
def add_evidence_route(iid):
    user = current_user()
    incident = _incident_or_404(iid)
    if not _can_update_incident_lifecycle(user, incident):
        return jsonify(error="Access denied"), 403
    data = request.get_json(silent=True) or {}
    source = str(data.get("source", "OPERATOR")).strip().upper()
    loc = str(data.get("location_text", "")).strip()[:500]
    raw = str(data.get("raw_data", ""))[:10000]
    if source not in {"AI", "CAMERA", "CITIZEN", "OPERATOR", "RESPONDER", "HOSPITAL", "POLICE", "FIRE"}:
        return jsonify(error="Invalid evidence source."), 400
    try:
        conf = _number(data.get("confidence", 0.7), "Confidence", 0, 1)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    add_evidence(iid, source, conf, loc, raw)
    audit(user["username"],"ADD_EVIDENCE",f"id={iid} src={source}")
    return jsonify(ok=True)

# ── Re-planning ───────────────────────────────────────────────────────────────
@app.route("/incidents/<int:iid>/replan")
@login_required
def view_replan(iid):
    user = current_user()
    if not has_permission(user["role"],"review_incidents"):
        return jsonify(error="Access denied"), 403
    _incident_or_404(iid)
    rows = query("SELECT * FROM replan_events WHERE incident_id=? ORDER BY id DESC",(iid,))
    return jsonify(replans=[dict(r) for r in rows])

@app.route("/incidents/<int:iid>/replan/trigger", methods=["POST"])
@permission_required("review_incidents")
def trigger_replan(iid):
    user = current_user()
    _incident_or_404(iid)
    reason = request.form.get("reason","Manual trigger by operator").strip()[:500]
    if not reason:
        flash("A re-plan reason is required.", "danger")
        return redirect(url_for("incident_detail", iid=iid))
    mutate("INSERT INTO replan_events(incident_id,trigger_reason,old_plan,new_plan,status) VALUES(?,?,?,?,'PENDING')",
           (iid,reason,"Operator-triggered","Awaiting allocation scoring"))
    notify_role("operator",f"Re-plan Triggered — Incident #{iid}",reason,"RE_PLAN",iid)
    audit(user["username"],"TRIGGER_REPLAN",f"id={iid}")
    flash("Re-plan triggered.","success")
    return redirect(url_for("incident_detail",iid=iid))

@app.route("/incidents/<int:iid>/replan/<int:rid>/approve", methods=["POST"])
@permission_required("review_incidents")
def approve_replan(iid, rid):
    user = current_user()
    _incident_or_404(iid)
    if not query("SELECT id FROM replan_events WHERE id=? AND incident_id=? AND status='PENDING'", (rid, iid), one=True):
        abort(404)
    now = datetime.now().isoformat(timespec="seconds")
    mutate("UPDATE replan_events SET status='APPROVED',operator_username=?,resolved_at=? WHERE id=?",(user["username"],now,rid))
    _blockchain_record("REPLAN_APPROVED",f"incident_id={iid} replan_id={rid}")
    audit(user["username"],"APPROVE_REPLAN",f"incident={iid} replan={rid}")
    flash("Re-plan approved.","success")
    return redirect(url_for("incident_detail",iid=iid))

@app.route("/incidents/<int:iid>/replan/<int:rid>/reject", methods=["POST"])
@permission_required("review_incidents")
def reject_replan(iid, rid):
    user = current_user()
    _incident_or_404(iid)
    if not query("SELECT id FROM replan_events WHERE id=? AND incident_id=? AND status='PENDING'", (rid, iid), one=True):
        abort(404)
    now = datetime.now().isoformat(timespec="seconds")
    mutate("UPDATE replan_events SET status='REJECTED',operator_username=?,resolved_at=? WHERE id=?",(user["username"],now,rid))
    audit(user["username"],"REJECT_REPLAN",f"incident={iid} replan={rid}")
    flash("Re-plan rejected.","success")
    return redirect(url_for("incident_detail",iid=iid))

# ── Hospital pre-alerts ───────────────────────────────────────────────────────
@app.route("/incidents/<int:iid>/alert_hospital", methods=["POST"])
@permission_required("manage_resources")
def alert_hospital(iid):
    user = current_user()
    inc = _incident_or_404(iid)
    if inc["lifecycle_status"] != "HOSPITAL_SELECTED":
        flash("Select a hospital in the lifecycle before sending a pre-alert.", "danger")
        return redirect(url_for("incident_detail", iid=iid))
    try:
        hosp_id = _number(request.form.get("hospital_id"), "Hospital", 1, integer=True)
        eta = _number(request.form.get("eta_minutes", "15"), "ETA", 1, 240, integer=True)
        patients = _number(request.form.get("patient_count", "1"), "Patient count", 1, 100, integer=True)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("incident_detail", iid=iid))
    hospital = query("SELECT id,status FROM response_resources WHERE id=? AND agency_type='hospital' AND active=1", (hosp_id,), one=True)
    if not hospital:
        abort(404)
    if hospital["status"] not in {"Available", "Open"}:
        flash("The selected hospital is not currently available.", "danger")
        return redirect(url_for("incident_detail", iid=iid))
    mutate("INSERT INTO hospital_alerts(incident_id,hospital_resource_id,severity,eta_minutes,patient_count) VALUES(?,?,?,?,?)",
           (iid,hosp_id,inc["severity"],eta,patients))
    mutate("UPDATE incidents SET lifecycle_status='HOSPITAL_NOTIFIED',updated_at=? WHERE id=?",(datetime.now().isoformat(timespec="seconds"),iid))
    mutate("INSERT INTO lifecycle_events(incident_id,from_status,to_status,username,notes) VALUES(?,?,?,'HOSPITAL_NOTIFIED',?,'Hospital pre-alert sent')",(iid,"HOSPITAL_SELECTED","HOSPITAL_NOTIFIED",user["username"]))
    notify_hospital_resource(hosp_id, f"Pre-Alert — Incoming Patient",
                             f"Incident #{iid}, Severity {inc['severity']}, ETA {eta} min",
                             "HOSPITAL_ALERT", iid)
    _blockchain_record("HOSPITAL_NOTIFIED",f"incident_id={iid} eta={eta}min")
    audit(user["username"],"ALERT_HOSPITAL",f"incident={iid} eta={eta}min")
    flash("Hospital pre-alert sent.","success")
    return redirect(url_for("incident_detail",iid=iid))

@app.route("/hospital_alerts/<int:aid>/acknowledge", methods=["POST"])
@login_required
def acknowledge_alert(aid):
    user = current_user()
    if not has_permission(user["role"],"update_hospital_status"):
        flash("Access denied.","danger"); return redirect(url_for("dashboard"))
    alert = query("""SELECT ha.id FROM hospital_alerts ha
        JOIN response_resources rr ON rr.id=ha.hospital_resource_id
        JOIN hospital_registry hr ON lower(hr.name)=lower(rr.resource_name)
        WHERE ha.id=? AND hr.linked_user_id=? AND ha.alert_status='SENT'""", (aid, user["id"]), one=True)
    if not alert:
        abort(404)
    mutate("UPDATE hospital_alerts SET alert_status='ACKNOWLEDGED',acknowledged_by=? WHERE id=?",(user["username"],aid))
    audit(user["username"],"ACK_HOSPITAL_ALERT",f"id={aid}")
    flash("Alert acknowledged.","success")
    return redirect(url_for("dashboard"))

# ── Notifications ─────────────────────────────────────────────────────────────
@app.route("/api/notifications")
@login_required
def api_notifications():
    user = current_user()
    rows = query("SELECT * FROM notifications WHERE recipient_username=? ORDER BY id DESC LIMIT 20",(user["username"],))
    return jsonify(notifications=[dict(r) for r in rows],
                   unread=sum(1 for r in rows if not r["read"]))

@app.route("/api/notifications/<int:nid>/read", methods=["POST"])
@login_required
def mark_notif_read(nid):
    user = current_user()
    mutate("UPDATE notifications SET read=1 WHERE id=? AND recipient_username=?",(nid,user["username"]))
    return jsonify(ok=True)

@app.route("/api/notifications/read_all", methods=["POST"])
@login_required
def mark_all_read():
    user = current_user()
    mutate("UPDATE notifications SET read=1 WHERE recipient_username=?",(user["username"],))
    return jsonify(ok=True)

# ── Cameras ───────────────────────────────────────────────────────────────────
@app.route("/cameras")
@login_required
def cameras():
    if not has_permission(current_user()["role"],"manage_agencies") and current_user()["role"] not in ("admin","operator"):
        abort(403)
    rows = query("SELECT * FROM cameras ORDER BY status, camera_id")
    return render_template("cameras.html", user=current_user(), cameras=[dict(r) for r in rows])

@app.route("/cameras/add", methods=["POST"])
@permission_required("manage_agencies")
def add_camera():
    user = current_user()
    cid = request.form.get("camera_id","").strip()
    loc = request.form.get("location_text","").strip()
    lat = request.form.get("latitude","").strip()
    lng = request.form.get("longitude","").strip()
    url = request.form.get("stream_url","").strip()
    if not cid or len(cid) > 64:
        flash("A camera ID of at most 64 characters is required.","danger")
        return redirect(url_for("cameras"))
    try:
        lat, lng = _coordinates(lat, lng)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("cameras"))
    mutate("INSERT OR IGNORE INTO cameras(camera_id,location_text,latitude,longitude,stream_url) VALUES(?,?,?,?,?)",
           (cid,loc[:500],lat,lng,url[:1000]))
    audit(user["username"],"ADD_CAMERA",cid)
    flash(f"Camera {cid} added.","success")
    return redirect(url_for("cameras"))

@app.route("/cameras/<int:cid>/status", methods=["POST"])
@permission_required("manage_agencies")
def update_camera_status(cid):
    status = request.form.get("status","Online")
    if status not in CAMERA_STATUSES:
        flash("Invalid camera status.", "danger")
        return redirect(url_for("cameras"))
    if not query("SELECT id FROM cameras WHERE id=?", (cid,), one=True):
        abort(404)
    mutate("UPDATE cameras SET status=?,last_seen=? WHERE id=?",(status,datetime.now().isoformat(timespec="seconds"),cid))
    _sio_emit("resource_update",{"type":"camera","id":cid,"status":status})
    flash("Camera status updated.","success")
    return redirect(url_for("cameras"))

# ── Blockchain ────────────────────────────────────────────────────────────────
@app.route("/blockchain")
@login_required
def blockchain():
    if not has_permission(current_user()["role"],"view_blockchain"): abort(403)
    rows = query("SELECT * FROM blockchain_log ORDER BY block_height DESC LIMIT 100")
    chain = []
    for r in rows:
        d = dict(r)
        raw = r["event_type"]+r["event_data"]+r["prev_hash"]+r["created_at"]
        computed = hashlib.sha256(raw.encode()).hexdigest()
        d["valid"] = (computed == r["sha256_hash"])
        chain.append(d)
    return render_template("blockchain.html", user=current_user(), chain=chain)

# ── Registries ────────────────────────────────────────────────────────────────
@app.route("/ambulances")
@login_required
def ambulance_registry():
    user = current_user()
    if user["role"] not in {"admin", "operator", "responder"}:
        abort(403)
    if user["role"] == "responder":
        rows = query("SELECT * FROM ambulance_registry WHERE linked_user_id=?",(user["id"],))
    else:
        rows = query("SELECT * FROM ambulance_registry ORDER BY operational_status, unit_id")
    return render_template("ambulance_registry.html", user=user, ambulances=[dict(r) for r in rows],
                           has_permission=has_permission)

@app.route("/ambulances/add", methods=["POST"])
@permission_required("manage_registries")
def add_ambulance():
    user = current_user()
    f = request.form
    unit_id = f.get("unit_id", "").strip()
    if not unit_id or len(unit_id) > 64:
        flash("A valid ambulance unit ID is required.", "danger")
        return redirect(url_for("ambulance_registry"))
    try:
        crew_count = _number(f.get("crew_count", "2"), "Crew count", 1, 20, integer=True)
        latitude, longitude = _coordinates(f.get("base_latitude", ""), f.get("base_longitude", ""))
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("ambulance_registry"))
    mutate("""INSERT OR IGNORE INTO ambulance_registry
        (unit_id,registration_number,organization,contact_number,crew_count,capability,
         base_location,base_latitude,base_longitude,driver_name,paramedic_name)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (unit_id, f.get("registration_number","").strip(),
         f.get("organization","").strip(), f.get("contact_number","").strip(),
         crew_count, f.get("capability","Basic"),
         f.get("base_location","").strip(),
         latitude, longitude,
         f.get("driver_name","").strip(), f.get("paramedic_name","").strip()))
    ambulance = query("SELECT id,gps_resource_id FROM ambulance_registry WHERE unit_id=?", (unit_id,), one=True)
    if ambulance and not ambulance["gps_resource_id"]:
        resource_name = f"{unit_id} — {f.get('organization', '').strip() or 'Ambulance'}"
        mutate("""INSERT INTO response_resources
            (agency_type,resource_name,location_text,latitude,longitude,status,capability)
            VALUES('ambulance',?,?,?,?,'Available',?)""",
            (resource_name, f.get("base_location", "").strip(), latitude, longitude, f.get("capability", "Basic")))
        resource = query("SELECT id FROM response_resources WHERE resource_name=?", (resource_name,), one=True)
        mutate("UPDATE ambulance_registry SET gps_resource_id=? WHERE id=?", (resource["id"], ambulance["id"]))
    audit(user["username"],"ADD_AMBULANCE",f.get("unit_id",""))
    flash("Ambulance added to registry.","success")
    return redirect(url_for("ambulance_registry"))

@app.route("/ambulances/<int:aid>/status", methods=["POST"])
@login_required
def update_ambulance_status(aid):
    user = current_user()
    row = query("SELECT * FROM ambulance_registry WHERE id=?",(aid,),one=True)
    if not row: abort(404)
    can_manage = user["role"] in {"admin", "operator"}
    is_assigned_responder = user["role"] == "responder" and row["linked_user_id"] == user["id"]
    if not (can_manage or is_assigned_responder): abort(403)
    status = request.form.get("status","AVAILABLE")
    if status not in AMBULANCE_STATUSES:
        flash("Invalid ambulance status.","danger"); return redirect(url_for("ambulance_registry"))
    mutate("UPDATE ambulance_registry SET operational_status=? WHERE id=?",(status,aid))
    audit(user["username"],"UPDATE_AMB_STATUS",f"id={aid} status={status}")
    flash("Ambulance status updated.","success")
    return redirect(url_for("ambulance_registry"))

@app.route("/hospitals")
@login_required
def hospital_registry():
    user = current_user()
    if user["role"] not in {"admin", "operator", "hospital"}:
        abort(403)
    if user["role"] == "hospital":
        rows = query("SELECT * FROM hospital_registry WHERE linked_user_id=?",(user["id"],))
    else:
        rows = query("SELECT * FROM hospital_registry ORDER BY operating_status, name")
    return render_template("hospital_registry.html", user=user, hospitals=[dict(r) for r in rows],
                           has_permission=has_permission)

@app.route("/hospitals/add", methods=["POST"])
@permission_required("manage_registries")
def add_hospital():
    user = current_user()
    f = request.form
    code = f.get("hospital_code", "").strip()
    name = f.get("name", "").strip()
    if not code or not name or len(code) > 64 or len(name) > 200:
        flash("Hospital code and name are required.", "danger")
        return redirect(url_for("hospital_registry"))
    try:
        latitude, longitude = _coordinates(f.get("latitude", ""), f.get("longitude", ""))
        total_beds = _number(f.get("total_beds", "0"), "Total beds", 0, 100000, integer=True)
        available_beds = _number(f.get("available_beds", "0"), "Available beds", 0, total_beds, integer=True)
        icu_beds = _number(f.get("icu_beds", "0"), "ICU beds", 0, total_beds, integer=True)
        icu_available = _number(f.get("icu_available", "0"), "Available ICU beds", 0, icu_beds, integer=True)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("hospital_registry"))
    mutate("""INSERT OR IGNORE INTO hospital_registry
        (hospital_code,name,address,latitude,longitude,contact_number,emergency_contact,
         total_beds,available_beds,icu_beds,icu_available,trauma_center,capability_level)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (code, name, f.get("address","").strip(), latitude, longitude,
         f.get("contact_number","").strip(), f.get("emergency_contact","").strip(),
         total_beds, available_beds, icu_beds, icu_available,
         1 if f.get("trauma_center") else 0, f.get("capability_level","General")))
    # Keep dispatch recommendations and hospital pre-alerts connected to every
    # newly registered hospital, not only the seeded facilities.
    hospital = query("SELECT id FROM hospital_registry WHERE hospital_code=?", (code,), one=True)
    resource = query("SELECT id FROM response_resources WHERE agency_type='hospital' AND lower(resource_name)=lower(?)", (name,), one=True)
    if hospital and not resource:
        mutate("""INSERT INTO response_resources
            (agency_type,resource_name,location_text,latitude,longitude,status,capacity_note,capability)
            VALUES('hospital',?,?,?,?,?,?,?)""",
            (name, f.get("address", "").strip(), latitude, longitude, "Available",
             f"Beds: {available_beds}/{total_beds}; ICU: {icu_available}/{icu_beds}", f.get("capability_level", "General")))
    audit(user["username"],"ADD_HOSPITAL",f.get("hospital_code",""))
    flash("Hospital added to registry.","success")
    return redirect(url_for("hospital_registry"))

@app.route("/hospitals/<int:hid>/capacity", methods=["POST"])
@login_required
def update_hospital_capacity(hid):
    user = current_user()
    row = query("SELECT * FROM hospital_registry WHERE id=?",(hid,),one=True)
    if not row: abort(404)
    can_manage = user["role"] in {"admin", "operator"}
    is_linked_hospital = user["role"] == "hospital" and row["linked_user_id"] == user["id"]
    if not (can_manage or is_linked_hospital): abort(403)
    try:
        ab = _number(request.form.get("available_beds", "0"), "Available beds", 0, row["total_beds"], integer=True)
        ia = _number(request.form.get("icu_available", "0"), "Available ICU beds", 0, row["icu_beds"], integer=True)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("hospital_registry"))
    ops = request.form.get("operating_status","Open")
    if ops not in HOSPITAL_STATUSES:
        flash("Invalid hospital operating status.", "danger")
        return redirect(url_for("hospital_registry"))
    mutate("UPDATE hospital_registry SET available_beds=?,icu_available=?,operating_status=? WHERE id=?",(ab,ia,ops,hid))
    resource_status = "Available" if ops == "Open" else ("Busy" if ops == "Divert" else "Offline")
    mutate("""UPDATE response_resources SET status=?,capacity_note=?
        WHERE agency_type='hospital' AND lower(resource_name)=lower(?)""",
        (resource_status, f"Beds: {ab}/{row['total_beds']}; ICU: {ia}/{row['icu_beds']}", row["name"]))
    audit(user["username"],"UPDATE_HOSP_CAPACITY",f"id={hid} beds={ab} icu={ia}")
    _sio_emit("resource_update",{"type":"hospital","id":hid,"available_beds":ab})
    flash("Hospital capacity updated.","success")
    return redirect(url_for("hospital_registry"))

@app.route("/departments")
@login_required
def departments():
    user = current_user()
    if user["role"] not in {"admin", "operator", "police", "fire", "traffic"}:
        abort(403)
    rows = query("SELECT * FROM department_registry ORDER BY dept_type, name")
    by_type = {}
    for r in rows:
        t = r["dept_type"]
        by_type.setdefault(t, []).append(dict(r))
    return render_template("departments.html", user=user, by_type=by_type,
                           has_permission=has_permission)

@app.route("/departments/add", methods=["POST"])
@permission_required("manage_registries")
def add_department():
    user = current_user()
    f = request.form
    code = f.get("dept_code", "").strip()
    name = f.get("name", "").strip()
    if not code or not name or len(code) > 64 or len(name) > 200:
        flash("Department code and name are required.", "danger")
        return redirect(url_for("departments"))
    try:
        latitude, longitude = _coordinates(f.get("latitude", ""), f.get("longitude", ""))
        unit_count = _number(f.get("unit_count", "0"), "Unit count", 0, 100000, integer=True)
        available_units = _number(f.get("available_units", "0"), "Available units", 0, unit_count, integer=True)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("departments"))
    mutate("""INSERT OR IGNORE INTO department_registry
        (dept_code,dept_type,name,jurisdiction,address,latitude,longitude,
         contact_number,unit_count,available_units)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (code, f.get("dept_type","police"), name,
         f.get("jurisdiction","").strip(), f.get("address","").strip(),
         latitude, longitude, f.get("contact_number","").strip(), unit_count, available_units))
    audit(user["username"],"ADD_DEPARTMENT",f.get("dept_code",""))
    flash("Department added.","success")
    return redirect(url_for("departments"))

# ── Existing routes (reports, resources, audit, users, map) ──────────────────
@app.route("/incidents/<int:iid>/assign", methods=["POST"])
@permission_required("manage_resources")
def assign_incident_unit(iid):
    operator = current_user()
    incident = _incident_or_404(iid)
    if incident["lifecycle_status"] not in {"VERIFIED", "DISPATCHED"}:
        flash("Verify the incident before dispatching a unit.", "danger")
        return redirect(url_for("incident_detail", iid=iid))

    username = request.form.get("username", "").strip()
    target = query("SELECT id,username,role FROM users WHERE username=? AND active=1", (username,), one=True)
    if not target or target["role"] not in {"responder", "hospital", "police", "fire", "traffic"}:
        flash("Choose an active operational account.", "danger")
        return redirect(url_for("incident_detail", iid=iid))

    linked_registry = {
        "responder": (
            "SELECT id FROM ambulance_registry WHERE linked_user_id=? AND active=1",
            (target["id"],),
        ),
        "hospital": (
            "SELECT id FROM hospital_registry WHERE linked_user_id=? AND active=1",
            (target["id"],),
        ),
        "police": (
            "SELECT id FROM department_registry WHERE linked_user_id=? AND dept_type='police' AND active=1",
            (target["id"],),
        ),
        "fire": (
            "SELECT id FROM department_registry WHERE linked_user_id=? AND dept_type='fire' AND active=1",
            (target["id"],),
        ),
        "traffic": (
            "SELECT id FROM department_registry WHERE linked_user_id=? AND dept_type='traffic' AND active=1",
            (target["id"],),
        ),
    }
    registry_sql, registry_params = linked_registry[target["role"]]
    if not query(registry_sql, registry_params, one=True):
        flash("The selected account is not linked to an active operational registry record.", "danger")
        return redirect(url_for("incident_detail", iid=iid))

    resource_label = request.form.get("resource_label", "").strip()[:120]
    mutate("""INSERT OR IGNORE INTO incident_assignments
        (incident_id,assignment_role,assignee_user_id,resource_label,assigned_by)
        VALUES(?,?,?,?,?)""", (iid, target["role"], target["id"], resource_label, operator["id"]))
    field = {"responder": "ambulance_assigned", "police": "police_assigned",
             "traffic": "police_assigned", "fire": "fire_assigned"}.get(target["role"])
    if field:
        mutate(f"UPDATE incidents SET {field}=1,updated_at=? WHERE id=?",
               (datetime.now().isoformat(timespec="seconds"), iid))
    if incident["lifecycle_status"] == "VERIFIED":
        now = datetime.now().isoformat(timespec="seconds")
        mutate("UPDATE incidents SET lifecycle_status='DISPATCHED',updated_at=? WHERE id=?", (now, iid))
        mutate("""INSERT INTO lifecycle_events(incident_id,from_status,to_status,username,notes)
            VALUES(?,?,?,?,?)""", (iid, "VERIFIED", "DISPATCHED", operator["username"], "Unit-level dispatch created"))
        _blockchain_record("INCIDENT_DISPATCHED", f"id={iid} assignee={target['username']}")
    create_notification(target["role"], target["username"], f"Dispatch — Incident #{iid}",
                        f"You have been assigned. {resource_label}".strip(), "DISPATCH", iid)
    audit(operator["username"], "ASSIGN_INCIDENT_UNIT", f"incident={iid} user={username}")
    flash(f"{username} assigned to incident #{iid}.", "success")
    return redirect(url_for("incident_detail", iid=iid))

@app.route("/dispatch")
@permission_required("manage_resources")
def dispatch_console():
    incidents = query("""SELECT id,severity,location_text,lifecycle_status FROM incidents
        WHERE status != 'Closed' AND lifecycle_status IN ('VERIFIED','DISPATCHED')
        ORDER BY CASE severity WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END, updated_at DESC LIMIT 100""")
    staff = query("""SELECT username,role,agency_name FROM users
        WHERE active=1 AND role IN ('responder','hospital','police','fire','traffic') ORDER BY role,username""")
    return render_template("dispatch.html", user=current_user(), incidents=[dict(row) for row in incidents], staff=[dict(row) for row in staff])

@app.route("/driver-tracking")
@login_required
def driver_tracking():
    user = current_user()
    if user["role"] != "responder":
        abort(403)
    ambulance = query("SELECT * FROM ambulance_registry WHERE linked_user_id=? AND active=1", (user["id"],), one=True)
    if not ambulance:
        flash("Your account is not linked to an active ambulance unit.", "warning")
        return redirect(url_for("dashboard"))
    return render_template("driver_tracking.html", user=user, ambulance=dict(ambulance))

@app.route("/api/assignments/mine")
@login_required
def my_assignments_api():
    user = current_user()
    if user["role"] not in {"responder", "police", "fire", "traffic", "hospital"}:
        return jsonify(assignments=[])
    assignments = query("""SELECT ia.incident_id,ia.assignment_status,ia.resource_label,i.severity,i.location_text,i.lifecycle_status
        FROM incident_assignments ia JOIN incidents i ON i.id=ia.incident_id
        WHERE ia.assignee_user_id=? AND ia.assignment_role=?
          AND ia.assignment_status NOT IN ('CANCELLED','COMPLETED') AND i.status != 'Closed'
        ORDER BY i.updated_at DESC""", (user["id"], user["role"]))
    return jsonify(assignments=[dict(row) for row in assignments])

@app.route("/api/driver/location", methods=["POST"])
@login_required
def update_driver_location():
    user = current_user()
    if user["role"] != "responder":
        return jsonify(error="Only a linked ambulance responder can submit GPS."), 403
    ambulance = query("SELECT * FROM ambulance_registry WHERE linked_user_id=? AND active=1", (user["id"],), one=True)
    if not ambulance:
        return jsonify(error="No ambulance unit is linked to this account."), 403
    data = request.get_json(silent=True) or {}
    try:
        latitude, longitude = _coordinates(data.get("latitude"), data.get("longitude"))
        speed = _number(data.get("speed_kmh", 0), "Speed", 0, 250)
        heading = _number(data.get("heading", 0), "Heading", 0, 360)
        raw_accuracy = data.get("accuracy_meters")
        accuracy = (
            _number(raw_accuracy, "GPS accuracy", 0, 100000)
            if raw_accuracy not in (None, "")
            else None
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    resource_id = ambulance["gps_resource_id"] or ambulance["id"]
    incident_id = data.get("incident_id")
    if incident_id not in (None, ""):
        try:
            incident_id = _number(incident_id, "Incident", 1, integer=True)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        assigned = _is_user_assigned_to_incident(user["id"], incident_id, "responder")
        if not assigned:
            return jsonify(error="You are not assigned to this incident."), 403
    else:
        incident_id = None
    now = datetime.now().isoformat(timespec="seconds")
    mutate("""INSERT INTO ambulance_gps(resource_id,incident_id,latitude,longitude,accuracy_meters,speed_kmh,heading,status,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?)""", (resource_id, incident_id, latitude, longitude, accuracy, speed, heading,
                                        data.get("status", "EN_ROUTE")[:30], now))
    location_event = {"resource_id": resource_id, "incident_id": incident_id,
                      "latitude": latitude, "longitude": longitude, "speed_kmh": speed, "heading": heading}
    if incident_id:
        _emit_incident_event("ambulance_location", location_event, incident_id)
    else:
        _sio_emit("ambulance_location", location_event, room="role:admin")
        _sio_emit("ambulance_location", location_event, room="role:operator")
    return jsonify(ok=True, captured_at=now)

@app.route("/api/cameras/<int:cid>/detections", methods=["POST"])
@permission_required("review_incidents")
def ingest_camera_detection(cid):
    camera = query("SELECT * FROM cameras WHERE id=? AND status='Online'", (cid,), one=True)
    if not camera:
        return jsonify(error="Online camera not found."), 404
    data = request.get_json(silent=True) or {}
    event_id = str(data.get("event_id", "")).strip()[:100]
    if not event_id:
        return jsonify(error="event_id is required."), 400
    try:
        latitude, longitude = _coordinates(data.get("latitude", camera["latitude"]), data.get("longitude", camera["longitude"]))
        confidence = _number(data.get("confidence", 0.5), "Confidence", 0, 1)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if latitude is None:
        return jsonify(error="Camera or event coordinates are required."), 400
    existing = query("SELECT incident_id FROM camera_events WHERE camera_id=? AND external_event_id=?", (cid, event_id), one=True)
    if existing:
        return jsonify(ok=True, duplicate_event=True, incident_id=existing["incident_id"])
    observed_raw = str(data.get("observed_at", datetime.now().isoformat(timespec="seconds")))
    try:
        observed = datetime.fromisoformat(observed_raw)
    except ValueError:
        return jsonify(error="observed_at must be ISO-8601."), 400
    vehicle_type = str(data.get("vehicle_type", "Unknown"))[:80]
    incident_id, merged = _create_or_merge_incident("CAMERA", confidence, camera["location_text"], latitude, longitude,
                                                     vehicle_type, observed, json.dumps(data)[:10000])
    mutate("""INSERT INTO camera_events(camera_id,external_event_id,incident_id,observed_at,latitude,longitude,confidence,vehicle_type,payload)
        VALUES(?,?,?,?,?,?,?,?,?)""", (cid, event_id, incident_id, observed.isoformat(timespec="seconds"), latitude,
                                         longitude, confidence, vehicle_type, json.dumps(data)[:10000]))
    audit(current_user()["username"], "INGEST_CAMERA_DETECTION", f"camera={cid} incident={incident_id}")
    return jsonify(ok=True, incident_id=incident_id, merged_with_existing=merged), 201

@app.route("/api/integrations/cameras/<int:cid>/detections", methods=["POST"])
def ingest_camera_detection_service(cid):
    """Adapter endpoint for a future CV worker connected to an approved camera feed.

    Configure RESQLINK_CAMERA_INGEST_KEY and send it only in the
    X-ResQLink-Camera-Key header over HTTPS.  The worker submits detection
    metadata; it never receives portal credentials or database access.
    """
    expected_key = os.environ.get("RESQLINK_CAMERA_INGEST_KEY", "")
    supplied_key = request.headers.get("X-ResQLink-Camera-Key", "")
    if not expected_key or not hmac.compare_digest(expected_key, supplied_key):
        return jsonify(error="Camera integration is not authorized."), 403
    camera = query("SELECT * FROM cameras WHERE id=? AND status='Online'", (cid,), one=True)
    if not camera:
        return jsonify(error="Online camera not found."), 404
    data = request.get_json(silent=True) or {}
    event_id = str(data.get("event_id", "")).strip()[:100]
    if not event_id:
        return jsonify(error="event_id is required."), 400
    try:
        latitude, longitude = _coordinates(data.get("latitude", camera["latitude"]), data.get("longitude", camera["longitude"]))
        confidence = _number(data.get("confidence", 0.5), "Confidence", 0, 1)
        observed = datetime.fromisoformat(str(data.get("observed_at", datetime.now().isoformat(timespec="seconds"))))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if latitude is None:
        return jsonify(error="Camera or event coordinates are required."), 400
    existing = query("SELECT incident_id FROM camera_events WHERE camera_id=? AND external_event_id=?", (cid, event_id), one=True)
    if existing:
        return jsonify(ok=True, duplicate_event=True, incident_id=existing["incident_id"])
    vehicle_type = str(data.get("vehicle_type", "Unknown"))[:80]
    incident_id, merged = _create_or_merge_incident("CAMERA", confidence, camera["location_text"], latitude, longitude,
                                                     vehicle_type, observed, json.dumps(data)[:10000])
    mutate("""INSERT INTO camera_events(camera_id,external_event_id,incident_id,observed_at,latitude,longitude,confidence,vehicle_type,payload)
        VALUES(?,?,?,?,?,?,?,?,?)""", (cid, event_id, incident_id, observed.isoformat(timespec="seconds"), latitude,
                                         longitude, confidence, vehicle_type, json.dumps(data)[:10000]))
    return jsonify(ok=True, incident_id=incident_id, merged_with_existing=merged), 201

@app.route("/reports", methods=["GET","POST"])
@login_required
def reports():
    user = current_user()
    if request.method == "POST":
        if not has_permission(user["role"],"submit_report"):
            flash("Access denied.","danger"); return redirect(url_for("reports"))
        desc=request.form.get("description","").strip(); loc=request.form.get("location_text","").strip()
        if not desc or not loc: flash("Description and location required.","danger"); return redirect(url_for("reports"))
        lat=request.form.get("latitude","").strip(); lng=request.form.get("longitude","").strip()
        cont=request.form.get("contact","").strip(); inc_time=request.form.get("incident_time","")
        if len(desc) > 3000 or len(loc) > 500 or len(cont) > 100 or len(inc_time) > 64:
            flash("One or more report fields are too long.", "danger")
            return redirect(url_for("reports"))
        try:
            latitude, longitude = _coordinates(lat, lng)
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("reports"))
        fname=None; ev=request.files.get("evidence")
        if ev and ev.filename and allowed_file(ev.filename):
            ext=Path(secure_filename(ev.filename)).suffix.lower()
            fname=f"report_{secrets.token_hex(12)}{ext}"; ev.save(UPLOAD_DIR/fname)
        reported_at = inc_time or datetime.now().isoformat(timespec="seconds")
        db = get_db()
        try:
            report_id = db.execute("""INSERT INTO citizen_reports
                (reporter_username,description,incident_time,location_text,latitude,longitude,contact,evidence_filename)
                VALUES(?,?,?,?,?,?,?,?)""",
                (user["username"], desc, reported_at, loc, latitude, longitude, cont, fname)).lastrowid
            db.commit()
        finally:
            db.close()
        try:
            observed = datetime.fromisoformat(reported_at)
        except ValueError:
            observed = datetime.now()
        if latitude is not None:
            incident_id, merged = _create_or_merge_incident(
                "CITIZEN", 0.55, loc, latitude, longitude, "Unknown", observed,
                f"Citizen report #{report_id}: {desc[:500]}")
            mutate("UPDATE citizen_reports SET linked_incident_id=?,status=? WHERE id=?",
                   (incident_id, "Merged" if merged else "Awaiting Review", report_id))
            message = f"Report merged into incident #{incident_id}." if merged else f"Report created incident #{incident_id}."
        else:
            message = "Report submitted for operator review; add GPS coordinates to enable automatic matching."
        audit(user["username"],"SUBMIT_REPORT",loc)
        flash(message,"success"); return redirect(url_for("reports"))
    rows = query("SELECT * FROM citizen_reports ORDER BY id DESC LIMIT 200") if has_permission(user["role"],"view_reports") else query("SELECT * FROM citizen_reports WHERE reporter_username=? ORDER BY id DESC",(user["username"],))
    return render_template("reports.html", user=user, reports=[dict(r) for r in rows], has_permission=has_permission)

@app.route("/resources")
@login_required
def resources():
    if not has_permission(current_user()["role"], "manage_resources"):
        abort(403)
    rows = query("SELECT * FROM response_resources WHERE active=1 ORDER BY agency_type,resource_name")
    return render_template("resources.html", user=current_user(), resources=[dict(r) for r in rows], has_permission=has_permission)

@app.route("/resources/add", methods=["POST"])
@permission_required("manage_resources")
def add_resource():
    user=current_user(); atype=request.form.get("agency_type"); name=request.form.get("resource_name","").strip()
    if not name or atype not in ("ambulance","hospital","police","fire"):
        flash("Invalid data.","danger"); return redirect(url_for("resources"))
    mutate("INSERT INTO response_resources(agency_type,resource_name,location_text,capacity_note) VALUES(?,?,?,?)",
           (atype,name,request.form.get("location_text","").strip(),request.form.get("capacity_note","").strip()))
    audit(user["username"],"ADD_RESOURCE",f"{atype}:{name}"); flash(f"{name} added.","success")
    return redirect(url_for("resources"))

@app.route("/resources/<int:rid>/status", methods=["POST"])
@permission_required("manage_resources")
def update_resource_status(rid):
    status=request.form.get("status","Available")
    if status not in RESOURCE_STATUSES:
        flash("Invalid resource status.", "danger")
        return redirect(url_for("resources"))
    if not query("SELECT id FROM response_resources WHERE id=?", (rid,), one=True):
        abort(404)
    mutate("UPDATE response_resources SET status=? WHERE id=?",(status,rid))
    audit(current_user()["username"], "UPDATE_RESOURCE_STATUS", f"id={rid} status={status}")
    flash("Status updated.","success"); return redirect(url_for("resources"))

@app.route("/audit")
@permission_required("view_audit_log")
def audit_log():
    rows=query("SELECT * FROM audit_log ORDER BY id DESC LIMIT 500")
    return render_template("audit.html",user=current_user(),logs=[dict(r) for r in rows])

@app.route("/users")
@permission_required("manage_users")
def user_management():
    rows=query("SELECT id,username,role,agency_name,active,created_at FROM users ORDER BY id DESC")
    return render_template("users.html",user=current_user(),users=[dict(r) for r in rows],roles=ROLES)

@app.route("/onboarding")
@permission_required("manage_users")
def onboarding():
    return render_template("onboarding.html", user=current_user(), roles=ROLES)

@app.route("/users/add", methods=["POST"])
@permission_required("manage_users")
def add_user():
    user=current_user(); uname=request.form.get("username","").strip(); pwd=request.form.get("password","")
    role=request.form.get("role","public"); agency=request.form.get("agency_name","").strip()
    registry_code=request.form.get("registry_code","").strip().upper()
    if len(uname)<3 or len(pwd)<12 or role not in ROLES:
        flash("Invalid data. Password ≥12 chars.","danger"); return redirect(url_for("user_management"))
    if query("SELECT id FROM users WHERE username=?",(uname,),one=True):
        flash("Username exists.","danger"); return redirect(url_for("user_management"))
    registry_link = {
        "responder": ("ambulance_registry", "unit_id"),
        "hospital": ("hospital_registry", "hospital_code"),
        "police": ("department_registry", "dept_code"),
        "fire": ("department_registry", "dept_code"),
        "traffic": ("department_registry", "dept_code"),
    }.get(role)
    if registry_code:
        if not registry_link:
            flash("Only operational roles can be linked to a registry record.", "danger")
            return redirect(url_for("user_management"))
        table, code_col = registry_link
        record = query(f"SELECT id,linked_user_id FROM {table} WHERE {code_col}=? AND active=1", (registry_code,), one=True)
        if not record or record["linked_user_id"]:
            flash("That registry code is invalid or already linked to another account.", "danger")
            return redirect(url_for("user_management"))
    mutate("INSERT INTO users(username,password_hash,role,agency_name) VALUES(?,?,?,?)",
           (uname,bcrypt.hashpw(pwd.encode(),bcrypt.gensalt()),role,agency))
    new_user = query("SELECT id FROM users WHERE username=?", (uname,), one=True)
    if registry_code:
        table, code_col = registry_link
        mutate(f"UPDATE {table} SET linked_user_id=? WHERE {code_col}=?", (new_user["id"], registry_code))
    audit(user["username"],"CREATE_USER",f"{uname} role={role}"); flash(f"User {uname} created.","success")
    return redirect(url_for("user_management"))

@app.route("/users/<int:uid>/toggle", methods=["POST"])
@permission_required("manage_users")
def toggle_user(uid):
    user=current_user(); row=query("SELECT id,username,role,active FROM users WHERE id=?",(uid,),one=True)
    if not row: abort(404)
    if row["id"]==user["id"]: flash("Cannot deactivate own account.","danger"); return redirect(url_for("user_management"))
    if row["role"] == "admin" and row["active"] and query("SELECT COUNT(*) AS c FROM users WHERE role='admin' AND active=1", one=True)["c"] <= 1:
        flash("At least one active administrator is required.", "danger")
        return redirect(url_for("user_management"))
    ns=0 if row["active"] else 1
    mutate("UPDATE users SET active=? WHERE id=?",(ns,uid))
    audit(user["username"],"TOGGLE_USER",f"{row['username']} active={ns}")
    flash(f"User {'activated' if ns else 'deactivated'}.","success"); return redirect(url_for("user_management"))

# ── API Stats ─────────────────────────────────────────────────────────────────
@app.route("/api/stats")
@login_required
def api_stats():
    user = current_user()
    if not (has_permission(user["role"], "view_incidents") or has_permission(user["role"], "view_assigned_incidents")):
        return jsonify(error="Access denied"), 403
    return jsonify(get_stats())

@app.route("/api/incidents/chart")
@login_required
def api_incidents_chart():
    if not has_permission(current_user()["role"], "view_incidents"):
        return jsonify(error="Access denied"), 403
    sev=query("SELECT severity,COUNT(*) as cnt FROM incidents GROUP BY severity")
    st=query("SELECT status,COUNT(*) as cnt FROM incidents GROUP BY status")
    return jsonify(severity={r["severity"]:r["cnt"] for r in sev},status={r["status"]:r["cnt"] for r in st})

# ── Map routes ────────────────────────────────────────────────────────────────
@app.route("/map")
@login_required
def live_map():
    user=current_user()
    if not (has_permission(user["role"], "view_incidents") or has_permission(user["role"], "view_assigned_incidents")):
        abort(403)
    sql = "SELECT i.id,i.status,i.severity,i.vehicle_type,i.location_text,il.latitude,il.longitude,il.address FROM incidents i LEFT JOIN incident_locations il ON il.incident_id=i.id WHERE i.status NOT IN ('Closed')"
    params = []
    if user["role"] in {"responder", "police", "traffic", "fire"}:
        sql += " AND EXISTS (SELECT 1 FROM incident_assignments ia WHERE ia.incident_id=i.id AND ia.assignee_user_id=? AND ia.assignment_role=? AND ia.assignment_status NOT IN ('CANCELLED','COMPLETED'))"
        params = [user["id"], user["role"]]
    sql += " ORDER BY i.id DESC"
    incs=query(sql, params)
    ress = _map_resources_for_user(user)
    return render_template("live_map.html",user=user,incidents=[dict(r) for r in incs],
        resources=[dict(r) for r in ress],severity_colors=SEVERITY_COLORS)

@app.route("/incidents/<int:iid>/map")
@login_required
def incident_map(iid):
    user=current_user(); row=_incident_or_404(iid)
    if not _can_view_incident(user, row): abort(403)
    loc=query("SELECT * FROM incident_locations WHERE incident_id=?",(iid,),one=True)
    hosps=query("SELECT * FROM response_resources WHERE agency_type='hospital' AND latitude IS NOT NULL AND active=1")
    return render_template("incident_map.html",user=user,incident=dict(row),
        incident_loc=dict(loc) if loc else None,hospitals=[dict(r) for r in hosps],
        severity_colors=SEVERITY_COLORS)

def _map_resources_for_user(user):
    if user["role"] in {"admin", "operator"}:
        return query("""SELECT id,agency_type,resource_name,location_text,latitude,longitude,status
            FROM response_resources WHERE active=1 AND latitude IS NOT NULL""")
    if user["role"] == "responder":
        return query("""SELECT rr.id,rr.agency_type,rr.resource_name,rr.location_text,
                rr.latitude,rr.longitude,rr.status
            FROM response_resources rr
            JOIN ambulance_registry ar ON ar.gps_resource_id=rr.id
            WHERE rr.active=1 AND rr.latitude IS NOT NULL AND ar.linked_user_id=?""",
            (user["id"],))
    return []

@app.route("/api/map/live/<int:iid>")
@login_required
def api_map_live(iid):
    inc=_incident_or_404(iid)
    if not _can_view_incident(current_user(), inc):
        return jsonify(error="Access denied"),403
    loc=query("SELECT * FROM incident_locations WHERE incident_id=?",(iid,),one=True)
    has_incident_location = bool(loc)
    ilat=loc["latitude"] if loc else None
    ilng=loc["longitude"] if loc else None
    addr=(loc["address"] if loc else inc["location_text"] or "")
    user = current_user()
    if user["role"] in {"admin", "operator"}:
        ambs=query("""SELECT * FROM response_resources
            WHERE agency_type='ambulance' AND latitude IS NOT NULL AND active=1""")
    else:
        ambs=query("""SELECT DISTINCT rr.* FROM response_resources rr
            JOIN ambulance_registry ar ON ar.gps_resource_id=rr.id
            JOIN incident_assignments ia ON ia.assignee_user_id=ar.linked_user_id
              AND ia.assignment_role='responder'
            WHERE rr.agency_type='ambulance' AND rr.latitude IS NOT NULL AND rr.active=1
              AND ia.incident_id=? AND ia.assignment_status NOT IN ('CANCELLED','COMPLETED')""",
            (iid,))
    demo_mode = (
        os.environ.get("RESQLINK_DEMO_MODE") == "1"
        and DEPLOYMENT_ENV != "production"
    )
    try: created=_utc_datetime(inc["created_at"])
    except (TypeError, ValueError): created=_utc_datetime()
    t_sec=max(0,(_utc_datetime()-created).total_seconds())
    amb_out=[]
    for a in ambs:
        assignment = query(
            """SELECT ia.created_at,ia.assignment_status FROM incident_assignments ia
               JOIN ambulance_registry ar ON ar.linked_user_id=ia.assignee_user_id
                 AND ar.gps_resource_id=?
               WHERE ia.incident_id=? AND ia.assignment_role='responder'
                 AND ia.assignment_status NOT IN ('CANCELLED','COMPLETED')
               ORDER BY ia.id DESC LIMIT 1""",
            (a["id"], iid),
            one=True,
        )
        gps = query(
            """SELECT * FROM ambulance_gps WHERE resource_id=? AND incident_id=?
               ORDER BY id DESC LIMIT 1""",
            (a["id"], iid),
            one=True,
        )
        gps_age_seconds = None
        if gps:
            try:
                gps_time = _utc_datetime(gps["updated_at"])
                gps_age_seconds = max(0, int((_utc_datetime() - gps_time).total_seconds()))
            except (TypeError, ValueError):
                gps_age_seconds = None
        if gps and gps_age_seconds is not None and gps_age_seconds <= 90:
            sl, slg, spd, hdg, st = gps["latitude"], gps["longitude"], gps["speed_kmh"], gps["heading"], gps["status"]
            d = _haversine_km(sl, slg, ilat, ilng) if has_incident_location else None
            location_source = "device_gps"
            gps_updated_at = gps["updated_at"]
        elif gps:
            sl, slg, spd, hdg, st = gps["latitude"], gps["longitude"], 0, gps["heading"], "GPS_STALE"
            d = _haversine_km(sl, slg, ilat, ilng) if has_incident_location else None
            location_source = "stale_gps"
            gps_updated_at = gps["updated_at"]
        elif demo_mode and assignment and has_incident_location:
            try:
                assigned_at = _utc_datetime(assignment["created_at"])
                assignment_age = max(0, (_utc_datetime() - assigned_at).total_seconds())
            except (TypeError, ValueError):
                assignment_age = t_sec
            sl,slg,spd,hdg,st=_simulate_ambulance_pos(a["latitude"],a["longitude"],ilat,ilng,assignment_age)
            d=_haversine_km(sl,slg,ilat,ilng)
            location_source = "simulated_demo"
            gps_updated_at = None
        else:
            sl,slg,spd,hdg,st=a["latitude"],a["longitude"],0,0,"NO_LIVE_GPS"
            d=(
                _haversine_km(a["latitude"],a["longitude"],ilat,ilng)
                if has_incident_location else None
            )
            location_source = "registry"
            gps_updated_at = None
        eta=round(d,1) if d is not None and st=="EN_ROUTE" and location_source in {"device_gps","simulated_demo"} else None
        amb_out.append({"id":a["id"],"name":a["resource_name"],"base_lat":a["latitude"],"base_lng":a["longitude"],
            "sim_lat":round(sl,6),"sim_lng":round(slg,6),"speed_kmh":round(spd,1),"heading":round(hdg,1),
            "status":st if location_source in {"device_gps","simulated_demo","stale_gps"} else a["status"],
            "distance_km":round(d,2) if d is not None else None,
            "eta_min":eta,"eta_type":"straight_line_estimate" if eta is not None else None,
            "location_source":location_source,"gps_updated_at":gps_updated_at,
            "gps_age_seconds":gps_age_seconds,"accuracy_meters":gps["accuracy_meters"] if gps else None,
            "location_text":a["location_text"]})
    hosps=query("SELECT * FROM response_resources WHERE agency_type='hospital' AND latitude IS NOT NULL AND active=1")
    hosp_out=[{"id":h["id"],"name":h["resource_name"],"lat":h["latitude"],"lng":h["longitude"],
        "status":h["status"],"capacity_note":h["capacity_note"],
        "distance_km":round(_haversine_km(h["latitude"],h["longitude"],ilat,ilng),2) if has_incident_location else None,
        "location_text":h["location_text"]} for h in hosps]
    if has_incident_location:
        hosp_out.sort(key=lambda item: item["distance_km"])
    others=query("SELECT * FROM response_resources WHERE agency_type IN ('police','fire') AND latitude IS NOT NULL AND active=1")
    other_out=[{"id":o["id"],"name":o["resource_name"],"lat":o["latitude"],"lng":o["longitude"],
        "type":o["agency_type"],"status":o["status"],
        "distance_km":round(_haversine_km(o["latitude"],o["longitude"],ilat,ilng),2) if has_incident_location else None} for o in others]
    return jsonify(incident={"id":iid,"lat":ilat,"lng":ilng,"location_available":has_incident_location,
        "address":addr,"severity":inc["severity"],
        "status":inc["status"],"vehicle_type":inc["vehicle_type"] or "Unknown","vehicle_id":inc["vehicle_id"]},
        ambulances=amb_out,hospitals=hosp_out,others=other_out,
        timestamp=_utc_datetime().isoformat(timespec="seconds"))

@app.route("/api/map/global")
@login_required
def api_map_global():
    user = current_user()
    if not has_permission(user["role"], "view_incidents"):
        return jsonify(error="Access denied"), 403
    incs=query("SELECT i.id,i.status,i.severity,i.vehicle_type,i.location_text,i.ambulance_assigned,i.police_assigned,i.fire_assigned,il.latitude,il.longitude,il.address FROM incidents i LEFT JOIN incident_locations il ON il.incident_id=i.id WHERE i.status NOT IN ('Closed')")
    ress = _map_resources_for_user(user)
    return jsonify(incidents=[dict(r) for r in incs],resources=[dict(r) for r in ress],timestamp=datetime.now().isoformat(timespec="seconds"))

@app.route("/incidents/<int:iid>/set_location", methods=["POST"])
@login_required
def set_incident_location(iid):
    user=current_user()
    if not has_permission(user["role"],"review_incidents"): return jsonify(error="Access denied"),403
    _incident_or_404(iid)
    data=request.get_json(silent=True) or {}
    try:
        lat, lng = _coordinates(data.get("lat"), data.get("lng"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if lat is None:
        return jsonify(error="Latitude and longitude are required."), 400
    addr=str(data.get("address", "")).strip()[:500]
    db=get_db()
    db.execute("INSERT INTO incident_locations(incident_id,latitude,longitude,address) VALUES(?,?,?,?) ON CONFLICT(incident_id) DO UPDATE SET latitude=excluded.latitude,longitude=excluded.longitude,address=excluded.address",(iid,lat,lng,addr))
    db.commit(); db.close()
    audit(user["username"],"SET_INCIDENT_LOCATION",f"id={iid}")
    return jsonify(ok=True)

# ── Uploads & errors ──────────────────────────────────────────────────────────
@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    report = query("SELECT reporter_username FROM citizen_reports WHERE evidence_filename=?", (filename,), one=True)
    if not report:
        abort(404)
    user = current_user()
    if user["username"] != report["reporter_username"] and not has_permission(user["role"], "view_reports"):
        abort(403)
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@app.errorhandler(404)
def not_found(e): return render_template("error.html",code=404,message="Page not found."),404
@app.errorhandler(403)
def forbidden(e): return render_template("error.html",code=403,message="Access denied."),403
@app.errorhandler(500)
def server_error(e): return render_template("error.html",code=500,message="Internal server error."),500

# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    user = current_user()
    if not user:
        return False
    join_room(f"user:{user['username']}")
    join_room(f"role:{user['role']}")
@socketio.on("disconnect")
def on_disconnect(): pass

# ── Run ───────────────────────────────────────────────────────────────────────
from emergency_orchestration import register_emergency_orchestration
from cctv_upload_demo import register_cctv_upload_demo

register_emergency_orchestration(app, current_user, DB_PATH)
register_cctv_upload_demo(app, current_user, lambda: DB_PATH, add_evidence, audit)
print("  CCTV vehicle-evidence upload (no accident classification or dispatch): http://127.0.0.1:5000/cctv-demo/upload")


if __name__ == "__main__":
    initialize_database()
    start_replan_checker()
    print("\nResQLink AI — Complete Portal")
    print(f"  DB: {DB_PATH}")
    print(f"  URL: http://127.0.0.1:5000")
    if DEPLOYMENT_ENV != "production" and os.environ.get("RESQLINK_DEMO_MODE") == "1":
        print("\n  Demo accounts:")
        for u,p in [("admin","Admin@ResQLink1"),("operator1","Operator@ResQLink1"),
                    ("amb_crew1","AmbCrew@ResQLink1"),("hospital1","Hospital@ResQLink1"),
                    ("police1","Police@ResQLink1"),("fire1","Fire@ResQLink1"),
                    ("traffic1","Traffic@ResQLink1"),("public1","Public@ResQLink1")]:
            print(f"    {u:12s} / {p}")
    socketio.run(app, debug=DEPLOYMENT_ENV != "production" and os.environ.get("FLASK_DEBUG") == "1",
                 host=os.environ.get("RESQLINK_HOST", "127.0.0.1"),
                 port=int(os.environ.get("RESQLINK_PORT", "5000")))
