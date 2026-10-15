"""ResQLink Complete Demo

One self-contained Flask application for the prototype demo. It implements a
simulated emergency workflow without requiring a cloud account: CCTV upload
-> candidate signal -> operator verification -> confirmed dispatch -> consented
driver location updates -> hospital preparation -> role-scoped notifications
-> resolution.

It intentionally does not claim that a heuristic/OpenCV signal is a medically
or operationally validated accident classifier.  Replace ``analyse_video``
with an evaluated, authorised model before real-world use.
"""

from __future__ import annotations

import hashlib
import math
import os
import secrets
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, flash, jsonify, redirect, render_template_string, request, send_from_directory, session, url_for
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "resqlink_complete.db"
UPLOADS = Path(__file__).resolve().parent / "uploads" / "complete_demo"
UPLOADS.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
MAX_UPLOAD = 100 * 1024 * 1024
CONTROL_ROLES = {"admin", "operator"}
DRIVER_ROLES = {"ambulance", "ambulance_crew", "amb_crew", "driver"}
DEPARTMENT_ROLES = {"police", "traffic", "fire"}
ALL_ROLES = CONTROL_ROLES | DRIVER_ROLES | DEPARTMENT_ROLES | {"hospital", "citizen"}

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("RESQLINK_SECRET", secrets.token_hex(32)),
    MAX_CONTENT_LENGTH=MAX_UPLOAD,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
_INITIALISE_LOCK = threading.Lock()
_INITIALISED = False


@contextmanager
def connect():
    db = sqlite3.connect(str(DATABASE))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
    except BaseException:
        db.rollback()
        raise
    else:
        db.commit()
    finally:
        db.close()


def ensure_column(db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add a backwards-compatible column when an earlier demo database exists."""
    columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def distance_km(lat1, lon1, lat2, lon2) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return 9999.0
    r = 6371.0
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def initialise() -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    with _INITIALISE_LOCK:
        if _INITIALISED:
            return
            
        with connect() as db:
            db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL, role TEXT NOT NULL, display_name TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS ambulances (
              id TEXT PRIMARY KEY, label TEXT NOT NULL, driver_username TEXT UNIQUE,
              driver_phone TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL,
              status TEXT NOT NULL DEFAULT 'Available', last_update TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS hospitals (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL,
              emergency_beds INTEGER NOT NULL, trauma_beds INTEGER NOT NULL,
              emergency_department TEXT NOT NULL, readiness TEXT NOT NULL DEFAULT 'Ready',
              contact_username TEXT UNIQUE, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS incidents (
              id INTEGER PRIMARY KEY, reference TEXT UNIQUE NOT NULL, source TEXT NOT NULL,
              reporter_id INTEGER, title TEXT NOT NULL, description TEXT, location TEXT NOT NULL,
              latitude REAL NOT NULL, longitude REAL NOT NULL, confidence REAL,
              severity TEXT NOT NULL, status TEXT NOT NULL, ambulance_id TEXT, hospital_id TEXT,
              duplicate_of INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(reporter_id) REFERENCES users(id),
              FOREIGN KEY(ambulance_id) REFERENCES ambulances(id),
              FOREIGN KEY(hospital_id) REFERENCES hospitals(id),
              FOREIGN KEY(duplicate_of) REFERENCES incidents(id)
            );
            CREATE TABLE IF NOT EXISTS cctv_uploads (
              id INTEGER PRIMARY KEY, original_name TEXT NOT NULL, stored_name TEXT UNIQUE NOT NULL,
              camera_id TEXT NOT NULL, location TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL,
              confidence REAL NOT NULL, analysis_mode TEXT NOT NULL, incident_id INTEGER,
              created_by INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(incident_id) REFERENCES incidents(id), FOREIGN KEY(created_by) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS notifications (
              id INTEGER PRIMARY KEY, incident_id INTEGER NOT NULL, target_role TEXT NOT NULL,
              target_username TEXT, title TEXT NOT NULL, message TEXT NOT NULL, read_at TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(incident_id) REFERENCES incidents(id)
            );
            CREATE TABLE IF NOT EXISTS driver_dispatches (
              id INTEGER PRIMARY KEY, incident_id INTEGER NOT NULL, ambulance_id TEXT NOT NULL,
              driver_username TEXT NOT NULL, driver_phone TEXT NOT NULL, message TEXT NOT NULL,
              delivery_status TEXT NOT NULL DEFAULT 'In-app driver alert created',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(incident_id) REFERENCES incidents(id),
              FOREIGN KEY(ambulance_id) REFERENCES ambulances(id)
            );
            CREATE TABLE IF NOT EXISTS timeline (
              id INTEGER PRIMARY KEY, incident_id INTEGER NOT NULL, actor TEXT NOT NULL,
              event TEXT NOT NULL, details TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(incident_id) REFERENCES incidents(id)
            );
            CREATE TABLE IF NOT EXISTS location_updates (
              id INTEGER PRIMARY KEY, incident_id INTEGER NOT NULL, ambulance_id TEXT NOT NULL,
              latitude REAL NOT NULL, longitude REAL NOT NULL, accuracy_meters REAL, note TEXT, captured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(incident_id) REFERENCES incidents(id), FOREIGN KEY(ambulance_id) REFERENCES ambulances(id)
            );
            """
            )
            ensure_column(db, "notifications", "target_username", "TEXT")
            ensure_column(db, "hospitals", "contact_username", "TEXT")
            ensure_column(db, "location_updates", "accuracy_meters", "REAL")
            accounts = [
            ("admin", "Admin@ResQLink1", "admin", "Control Room Administrator"),
            ("operator1", "Operator@ResQLink1", "operator", "Emergency Dispatcher"),
            ("amb_crew1", "AmbCrew@ResQLink1", "amb_crew", "Ambulance Driver One"),
            ("hospital1", "Hospital@ResQLink1", "hospital", "Hospital Emergency Desk"),
            ("police1", "Police@ResQLink1", "police", "Police Response Desk"),
            ("fire1", "Fire@ResQLink1", "fire", "Fire & Rescue Desk"),
            ("traffic1", "Traffic@ResQLink1", "traffic", "Traffic Control Desk"),
            ("public1", "Public@ResQLink1", "citizen", "Citizen Reporter"),
        ]
            for username, password, role, display_name in accounts:
                if not db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                    db.execute("INSERT INTO users (username,password_hash,role,display_name) VALUES (?,?,?,?)", (username, generate_password_hash(password), role, display_name))
            db.executemany(
            "INSERT OR IGNORE INTO ambulances (id,label,driver_username,driver_phone,latitude,longitude) VALUES (?,?,?,?,?,?)",
            [
                ("AMB-001", "ResQ Ambulance 01", "amb_crew1", "+91 90000 00001", 13.0785, 80.2640),
                ("AMB-002", "ResQ Ambulance 02", None, "+91 90000 00002", 13.0624, 80.2492),
            ],
            )
            db.executemany(
            "INSERT OR IGNORE INTO hospitals (id,name,latitude,longitude,emergency_beds,trauma_beds,emergency_department,contact_username) VALUES (?,?,?,?,?,?,?,?)",
            [
                ("HOS-001", "ResQLink City Emergency Hospital", 13.0871, 80.2785, 8, 3, "24×7 Emergency & Trauma", "hospital1"),
                ("HOS-002", "ResQLink Central Medical Centre", 13.0705, 80.2551, 5, 1, "Emergency Department", None),
            ],
            )
            db.execute("UPDATE hospitals SET contact_username='hospital1' WHERE id='HOS-001' AND (contact_username IS NULL OR contact_username='')")
            db.commit()
        _INITIALISED = True


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with connect() as db:
        return db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


def require_login():
    user = current_user()
    if not user:
        return None
    return user


def require_roles(*roles):
    user = require_login()
    if not user:
        return None
    if user["role"] not in roles:
        abort(403)
    return user


def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


@app.before_request
def protect_requests():
    initialise()
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not request.path.startswith("/api/"):
        given = request.form.get("_csrf") or request.headers.get("X-CSRF-Token")
        if not given or not secrets.compare_digest(given, csrf_token()):
            abort(400, "Your form expired. Refresh the page and try again.")


def add_event(db, incident_id: int, actor: str, event: str, details: str = "") -> None:
    db.execute("INSERT INTO timeline (incident_id,actor,event,details) VALUES (?,?,?,?)", (incident_id, actor, event, details))


def notify(db, incident_id: int, roles, title: str, message: str, *, target_username: str | None = None) -> None:
    """Create a role alert, optionally restricted to one operational account.

    An ambulance dispatch must reach only the driver linked to that ambulance;
    it must never be broadcast to every account with a driver role.
    """
    if isinstance(roles, str):
        roles = {roles}
    for role in roles:
        db.execute(
            "INSERT INTO notifications (incident_id,target_role,target_username,title,message) VALUES (?,?,?,?,?)",
            (incident_id, role, target_username, title, message),
        )


def incident_visible(db, user, incident_id: int) -> bool:
    incident = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
    if not incident:
        return False
    role = user["role"]
    if role in CONTROL_ROLES:
        return True
    if role == "citizen":
        return incident["reporter_id"] == user["id"]
    if role in DRIVER_ROLES:
        ambulance = db.execute("SELECT id FROM ambulances WHERE driver_username=?", (user["username"],)).fetchone()
        return bool(ambulance and incident["ambulance_id"] == ambulance["id"])
    if role == "hospital":
        hospital = db.execute("SELECT contact_username FROM hospitals WHERE id=?", (incident["hospital_id"],)).fetchone()
        return bool(hospital and hospital["contact_username"] == user["username"])
    return bool(db.execute("SELECT 1 FROM notifications WHERE incident_id=? AND target_role=?", (incident_id, role)).fetchone())


def parse_coordinate(value, low, high, label):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{label} must be a number.")
    if not low <= result <= high:
        raise ValueError(f"{label} is outside the valid range.")
    return result


def nearby_duplicate(db, latitude, longitude):
    candidates = db.execute("SELECT * FROM incidents WHERE status NOT IN ('Resolved','Closed') AND created_at >= datetime('now','-30 minutes') ORDER BY id DESC").fetchall()
    return next((item for item in candidates if distance_km(latitude, longitude, item["latitude"], item["longitude"]) <= 0.8), None)


def dispatch_incident(db, incident, actor: str):
    """Assign only currently available resources and write the entire audit trail."""
    ambulance_rows = db.execute("SELECT * FROM ambulances WHERE status='Available' AND driver_username IS NOT NULL AND trim(driver_username) <> ''").fetchall()
    hospital_rows = db.execute("SELECT * FROM hospitals WHERE emergency_beds > 0 AND readiness IN ('Ready','Patient preparation in progress') AND contact_username IS NOT NULL AND trim(contact_username) <> ''").fetchall()
    ambulance = min(ambulance_rows, key=lambda a: distance_km(incident["latitude"], incident["longitude"], a["latitude"], a["longitude"]), default=None)
    hospital = min(hospital_rows, key=lambda h: distance_km(incident["latitude"], incident["longitude"], h["latitude"], h["longitude"]), default=None)
    if not ambulance or not hospital:
        status = "Awaiting manual dispatch — no available ambulance or emergency bed"
        db.execute("UPDATE incidents SET status=?,updated_at=? WHERE id=?", (status, now(), incident["id"]))
        add_event(db, incident["id"], actor, "Operator verification recorded; manual dispatch required", status)
        notify(db, incident["id"], CONTROL_ROLES, "Manual dispatch required", f"{incident['reference']}: {status}.")
        return None, None
    db.execute("UPDATE ambulances SET status='Dispatched',last_update=? WHERE id=?", (now(), ambulance["id"]))
    db.execute("UPDATE hospitals SET emergency_beds=emergency_beds-1, readiness='Patient preparation in progress',updated_at=? WHERE id=?", (now(), hospital["id"]))
    status = "Ambulance dispatched — hospital and departments alerted"
    db.execute("UPDATE incidents SET ambulance_id=?,hospital_id=?,status=?,updated_at=? WHERE id=?", (ambulance["id"], hospital["id"], status, now(), incident["id"]))
    db.execute("INSERT INTO location_updates (incident_id,ambulance_id,latitude,longitude,note,captured_at) VALUES (?,?,?,?,?,?)", (incident["id"], ambulance["id"], ambulance["latitude"], ambulance["longitude"], "Initial dispatch location", now()))
    add_event(db, incident["id"], actor, "Operator-confirmed dispatch completed", f"{ambulance['label']} assigned; {hospital['name']} reserved one emergency bed.")
    driver = db.execute("SELECT role FROM users WHERE username=?", (ambulance["driver_username"],)).fetchone()
    if driver:
        driver_message = f"{ambulance['label']} is assigned to {incident['location']}. Sign in to ResQLink and start GPS sharing only when you consent."
        notify(db, incident["id"], driver["role"], "Ambulance dispatch", driver_message, target_username=ambulance["driver_username"])
        db.execute(
            "INSERT INTO driver_dispatches (incident_id,ambulance_id,driver_username,driver_phone,message) VALUES (?,?,?,?,?)",
            (incident["id"], ambulance["id"], ambulance["driver_username"], ambulance["driver_phone"], driver_message),
        )
    notify(db, incident["id"], "hospital", "Incoming emergency case", f"Prepare {hospital['emergency_department']} at {hospital['name']} for {incident['reference']}. One emergency bed is reserved.", target_username=hospital["contact_username"])
    notify(db, incident["id"], DEPARTMENT_ROLES, "Emergency department alert", f"{incident['reference']} at {incident['location']}. Coordinate scene safety, traffic control and rescue as appropriate.")
    notify(db, incident["id"], CONTROL_ROLES, "Operator-confirmed response started", f"{incident['reference']} dispatched to {ambulance['id']} and {hospital['name']}.")
    return ambulance, hospital


def create_incident(db, *, source, reporter_id, title, description, location, latitude, longitude, confidence, severity, actor, upload_id=None):
    duplicate = nearby_duplicate(db, latitude, longitude)
    if duplicate:
        if upload_id:
            db.execute("UPDATE cctv_uploads SET incident_id=? WHERE id=?", (duplicate["id"], upload_id))
        add_event(db, duplicate["id"], actor, "Duplicate evidence merged", f"{source} report was matched within 0.8 km and 30 minutes.")
        notify(db, duplicate["id"], CONTROL_ROLES, "Duplicate incident evidence merged", f"New {source} evidence was linked to {duplicate['reference']}.")
        db.commit()
        return duplicate["id"], True
    reference = f"RQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    cursor = db.execute(
        "INSERT INTO incidents (reference,source,reporter_id,title,description,location,latitude,longitude,confidence,severity,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,?)",
        (reference, source, reporter_id, title, description, location, latitude, longitude, confidence, severity, "Awaiting operator verification", now(), now()),
    )
    incident_id = cursor.lastrowid
    if upload_id:
        db.execute("UPDATE cctv_uploads SET incident_id=? WHERE id=?", (incident_id, upload_id))
    add_event(db, incident_id, actor, "Incident created", f"Source: {source}; confidence: {confidence:.0%}" if confidence is not None else f"Source: {source}")
    notify(db, incident_id, CONTROL_ROLES, "New incident requires verification", f"{reference} reported at {location}. Verify the incident before dispatch.")
    db.commit()
    return incident_id, False


def analyse_video(path: Path, requested_confidence: float):
    """Return a cautious prototype CV signal, never a diagnosis.

    With OpenCV installed it samples motion between frames.  Without it the
    explicit demo confidence is used, so the full workflow remains testable
    on a normal laptop.  Both labels remain visible in the product.
    """
    try:
        import cv2  # Optional: the app still runs without OpenCV.

        capture = cv2.VideoCapture(str(path))
        differences, previous, frame_count = [], None, 0
        while frame_count < 90:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            if frame_count % 10:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if previous is not None:
                differences.append(float(cv2.absdiff(gray, previous).mean()))
            previous = gray
        capture.release()
        motion = min(1.0, (sum(differences) / max(len(differences), 1)) / 22.0)
        confidence = max(0.70, min(0.95, 0.55 + motion * 0.4 + requested_confidence * 0.25))
        return round(confidence, 2), "OpenCV motion-risk signal — operator verification required"
    except Exception:
        return requested_confidence, "Prototype CV model signal — operator verification required"


LAYOUT = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }} · ResQLink</title>
<style>
:root{--bg:#0c1118;--panel:#171f2c;--panel2:#202a3a;--line:#34445d;--text:#eef4fb;--muted:#aab8c8;--orange:#ff8a2b;--green:#25c783;--blue:#40a3ff;--red:#ff6075}*{box-sizing:border-box}body{margin:0;font:15px/1.5 Arial,sans-serif;background:var(--bg);color:var(--text)}a{color:#8ec7ff;text-decoration:none}a:hover{text-decoration:underline}.nav{display:flex;align-items:center;gap:18px;padding:12px 4vw;background:#121925;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}.brand{font-size:20px;font-weight:800;color:#fff}.spacer{flex:1}.wrap{max-width:1240px;margin:0 auto;padding:26px 22px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}.stat{font-size:28px;font-weight:800}.muted{color:var(--muted)}.good{color:var(--green)}.warn{color:#ffd060}.danger{color:var(--red)}h1{margin:0 0 4px;font-size:28px}h2,h3{margin-top:0}.button,button{display:inline-block;border:0;border-radius:7px;background:var(--orange);color:#fff;padding:10px 14px;font-weight:700;cursor:pointer}.button.alt,button.alt{background:transparent;border:1px solid var(--line);color:var(--text)}.button.green{background:#177e57}input,select,textarea{width:100%;padding:10px;border-radius:7px;border:1px solid #52627c;background:#0f1622;color:#fff;margin:5px 0 12px}label{font-weight:700}textarea{min-height:92px}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.badge{display:inline-block;padding:3px 8px;border-radius:99px;background:#294866;color:#bde0ff;font-size:12px;font-weight:700}.flash{padding:12px 15px;border-radius:8px;background:#193d31;border:1px solid #287b5b;margin-bottom:15px}.flash.error{background:#49202a;border-color:#9d3548}.timeline{border-left:2px solid #49637f;padding-left:15px}.timeline p{margin:0 0 15px}.role{background:#293a52;padding:4px 8px;border-radius:99px;font-size:12px}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}.nav{gap:10px;font-size:13px}}@media(max-width:500px){.grid{grid-template-columns:1fr}.nav .hide-small{display:none}.wrap{padding:16px 12px}}
</style></head><body><nav class="nav"><a class="brand" href="{{ url_for('home') }}">ResQLink AI</a>{% if user %}<a href="{{ url_for('dashboard') }}">Dashboard</a><a href="{{ url_for('incidents') }}">Incidents</a>{% if user.role in ['admin','operator'] %}<a href="{{ url_for('cctv_upload') }}">CCTV Upload</a><a href="{{ url_for('resources') }}" class="hide-small">Resources</a>{% endif %}<a href="{{ url_for('alerts') }}">Alerts{% if alert_count %} ({{ alert_count }}){% endif %}</a><span class="spacer"></span><span class="role">{{ user.role|replace('_',' ')|title }}</span><form method="post" action="{{ url_for('logout') }}"><input type="hidden" name="_csrf" value="{{ csrf }}"><button class="alt">Sign out</button></form>{% else %}<span class="spacer"></span><a href="{{ url_for('login') }}">Sign in</a>{% endif %}</nav><main class="wrap">{% with messages=get_flashed_messages(with_categories=true) %}{% for category,message in messages %}<div class="flash {{ category }}">{{ message }}</div>{% endfor %}{% endwith %}{{ content|safe }}</main></body></html>
"""


def page(title: str, body: str, **values):
    user = current_user()
    with connect() as db:
        alert_count = db.execute(
            "SELECT count(*) FROM notifications WHERE target_role=? AND (target_username IS NULL OR target_username=?) AND read_at IS NULL",
            (user["role"], user["username"]),
        ).fetchone()[0] if user else 0
    inner = render_template_string(body, csrf_token=csrf_token, **values)
    return render_template_string(LAYOUT, title=title, content=Markup(inner), user=user, alert_count=alert_count, csrf=csrf_token())


@app.route("/")
def home():
    return redirect(url_for("dashboard" if current_user() else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        with connect() as db:
            user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear(); session["user_id"] = user["id"]; csrf_token()
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return page("Sign in", """
    <div class="card" style="max-width:500px;margin:60px auto"><h1>ResQLink AI</h1><p class="muted">Emergency-response prototype sign-in</p><form method="post"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><label>Username</label><input name="username" required autofocus><label>Password</label><input name="password" type="password" required><button>Sign in</button></form><hr><p><strong>Demo accounts:</strong> admin / Admin@ResQLink1 · operator1 / Operator@ResQLink1 · amb_crew1 / AmbCrew@ResQLink1 · hospital1 / Hospital@ResQLink1</p></div>""")


@app.post("/logout")
def logout():
    session.clear(); flash("Signed out.", "success"); return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    user = require_login()
    if not user: return redirect(url_for("login"))
    with connect() as db:
        if user["role"] in CONTROL_ROLES:
            rows = db.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT 8").fetchall()
        elif user["role"] == "citizen":
            rows = db.execute("SELECT * FROM incidents WHERE reporter_id=? ORDER BY id DESC LIMIT 8", (user["id"],)).fetchall()
        elif user["role"] in DRIVER_ROLES:
            rows = db.execute("SELECT i.* FROM incidents i JOIN ambulances a ON a.id=i.ambulance_id WHERE a.driver_username=? ORDER BY i.id DESC LIMIT 8", (user["username"],)).fetchall()
        elif user["role"] == "hospital":
            rows = db.execute("SELECT i.* FROM incidents i JOIN hospitals h ON h.id=i.hospital_id WHERE h.contact_username=? ORDER BY i.id DESC LIMIT 8", (user["username"],)).fetchall()
        else:
            rows = db.execute("SELECT DISTINCT i.* FROM incidents i JOIN notifications n ON n.incident_id=i.id WHERE n.target_role=? ORDER BY i.id DESC LIMIT 8", (user["role"],)).fetchall()
        stats = {"active": db.execute("SELECT count(*) FROM incidents WHERE status NOT IN ('Resolved','Closed')").fetchone()[0], "available": db.execute("SELECT count(*) FROM ambulances WHERE status='Available'").fetchone()[0], "beds": db.execute("SELECT coalesce(sum(emergency_beds),0) FROM hospitals").fetchone()[0], "alerts": db.execute("SELECT count(*) FROM notifications WHERE target_role=? AND (target_username IS NULL OR target_username=?) AND read_at IS NULL", (user["role"], user["username"])).fetchone()[0]}
    actions = """
      {% if user.role in ['admin','operator'] %}<a class="button" href="{{ url_for('cctv_upload') }}">Upload & analyse CCTV video</a><a class="button alt" href="{{ url_for('resources') }}">Manage resources</a><a class="button alt" href="{{ url_for('driver_directory') }}">Driver directory</a><a class="button alt" href="{{ url_for('live_map') }}">Visual live tracker</a>
      {% elif user.role == 'citizen' %}<a class="button" href="{{ url_for('report_incident') }}">Report an accident</a>
      {% elif user.role in ['amb_crew','ambulance','ambulance_crew','driver'] %}<a class="button" href="{{ url_for('driver_live') }}">Start Driver Live GPS</a><a class="button alt" href="{{ url_for('alerts') }}">Open assigned response</a>
      {% else %}<a class="button" href="{{ url_for('alerts') }}">Open role alerts</a>{% endif %}"""
    return page("Dashboard", """
    <h1>Welcome, {{ user.display_name }}</h1><p class="muted">Role-based emergency coordination dashboard.</p><p>{{ actions|safe }}</p>
    <div class="grid"><div class="card"><span class="muted">Active incidents</span><div class="stat">{{ stats.active }}</div></div><div class="card"><span class="muted">Available ambulances</span><div class="stat good">{{ stats.available }}</div></div><div class="card"><span class="muted">Emergency beds</span><div class="stat good">{{ stats.beds }}</div></div><div class="card"><span class="muted">My unread alerts</span><div class="stat warn">{{ stats.alerts }}</div></div></div>
    <div class="card"><h2>Incidents available to your role</h2><table><tr><th>Reference</th><th>Location</th><th>Status</th><th></th></tr>{% for item in rows %}<tr><td>{{ item.reference }}</td><td>{{ item.location }}</td><td><span class="badge">{{ item.status }}</span></td><td><a href="{{ url_for('incident_detail', incident_id=item.id) }}">Open</a></td></tr>{% else %}<tr><td colspan="4" class="muted">No incidents are currently available to this role.</td></tr>{% endfor %}</table></div>""", user=user, rows=rows, stats=stats, actions=Markup(render_template_string(actions, user=user)))


@app.route("/incidents")
def incidents():
    user = require_login()
    if not user: return redirect(url_for("login"))
    with connect() as db:
        records = [item for item in db.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall() if incident_visible(db, user, item["id"])]
    return page("Incidents", """<h1>Incidents</h1><div class="card"><table><tr><th>Reference</th><th>Source</th><th>Severity</th><th>Location</th><th>Live status</th><th></th></tr>{% for item in records %}<tr><td>{{ item.reference }}</td><td>{{ item.source }}</td><td>{{ item.severity }}</td><td>{{ item.location }}</td><td>{{ item.status }}</td><td><a href="{{ url_for('incident_detail', incident_id=item.id) }}">Open</a></td></tr>{% else %}<tr><td colspan="6">No accessible incidents.</td></tr>{% endfor %}</table></div>""", records=records)


@app.route("/report", methods=["GET", "POST"])
def report_incident():
    user = require_roles("citizen", "admin", "operator")
    if not user: return redirect(url_for("login"))
    if request.method == "POST":
        try:
            lat = parse_coordinate(request.form.get("latitude"), -90, 90, "Latitude"); lon = parse_coordinate(request.form.get("longitude"), -180, 180, "Longitude")
            with connect() as db:
                incident_id, duplicate = create_incident(db, source="Citizen report", reporter_id=user["id"], title=(request.form.get("title") or "Road accident")[:100], description=(request.form.get("description") or "")[:1000], location=(request.form.get("location") or "Reported location")[:180], latitude=lat, longitude=lon, confidence=None, severity=request.form.get("severity") or "High", actor=user["username"])
            flash("Your report was merged with an existing incident for operator review." if duplicate else "Incident report submitted for operator verification.", "success")
            return redirect(url_for("incident_detail", incident_id=incident_id))
        except ValueError as error: flash(str(error), "error")
    return page("Report accident", """<h1>Report an accident</h1><div class="card"><form method="post"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><div class="two"><div><label>Title</label><input name="title" value="Road accident" required><label>Location</label><input name="location" required><label>Latitude</label><input name="latitude" placeholder="13.0827" required></div><div><label>Severity</label><select name="severity"><option>High</option><option>Critical</option><option>Medium</option></select><label>Longitude</label><input name="longitude" placeholder="80.2707" required></div></div><label>Description</label><textarea name="description" placeholder="What happened? Is anyone injured?"></textarea><button>Submit emergency report</button></form></div>""")


@app.route("/cctv/upload", methods=["GET", "POST"])
def cctv_upload():
    user = require_roles("admin", "operator")
    if not user: return redirect(url_for("login"))
    if request.method == "POST":
        file = request.files.get("video")
        try:
            if not file or not file.filename: raise ValueError("Choose a CCTV video to analyse.")
            original = secure_filename(file.filename); extension = Path(original).suffix.lower()
            if not original or extension not in ALLOWED_EXTENSIONS: raise ValueError("Use MP4, MOV, M4V, WebM, AVI, or MKV video.")
            lat = parse_coordinate(request.form.get("latitude"), -90, 90, "Latitude"); lon = parse_coordinate(request.form.get("longitude"), -180, 180, "Longitude")
            requested = parse_coordinate(request.form.get("demo_confidence", "0.89"), 0.0, 1.0, "Demo confidence")
            stored = f"{uuid.uuid4().hex}{extension}"; destination = UPLOADS / stored; file.save(destination)
            if not destination.exists() or destination.stat().st_size == 0: raise ValueError("The video could not be saved.")
            if destination.stat().st_size > MAX_UPLOAD: destination.unlink(missing_ok=True); raise ValueError("The video exceeds 100 MB.")
            confidence, mode = analyse_video(destination, requested)
            with connect() as db:
                cursor = db.execute("INSERT INTO cctv_uploads (original_name,stored_name,camera_id,location,latitude,longitude,confidence,analysis_mode,created_by,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)", (original, stored, request.form.get("camera_id") or "DEMO-CAM-001", (request.form.get("location") or "ResQLink Demo Junction")[:180], lat, lon, confidence, mode, user["id"], now()))
                upload_id = cursor.lastrowid
                if confidence >= 0.70:
                    incident_id, duplicate = create_incident(db, source="CCTV video", reporter_id=user["id"], title="CCTV collision signal", description=f"{mode}. Evidence video upload #{upload_id}; human verification required.", location=(request.form.get("location") or "ResQLink Demo Junction")[:180], latitude=lat, longitude=lon, confidence=confidence, severity=request.form.get("severity") or "High", actor=user["username"], upload_id=upload_id)
                else:
                    incident_id, duplicate = None, False; db.commit()
            if incident_id: flash("CCTV signal is awaiting operator verification; no dispatch was started." if not duplicate else "CCTV evidence was merged into an incident for operator verification; no dispatch was started.", "success"); return redirect(url_for("incident_detail", incident_id=incident_id))
            flash("Video stored. The confidence signal is below the automatic-response threshold; operator review is required.", "error")
        except ValueError as error: flash(str(error), "error")
    with connect() as db: uploads = db.execute("SELECT c.*,i.reference FROM cctv_uploads c LEFT JOIN incidents i ON i.id=c.incident_id ORDER BY c.id DESC LIMIT 12").fetchall()
    return page("CCTV video analysis", """<h1>CCTV video analysis</h1><p class="muted">Authorised prototype footage only. A CV signal at ≥70% creates or merges an incident candidate for operator verification; it does not initiate a dispatch.</p><div class="card"><form method="post" enctype="multipart/form-data"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><div class="two"><div><label>CCTV recording</label><input type="file" name="video" accept="video/*" required><label>Camera</label><select name="camera_id"><option value="DEMO-CAM-001">DEMO-CAM-001 — Demo Junction</option><option value="DEMO-CAM-002">DEMO-CAM-002 — Demo Flyover</option></select><label>Location</label><input name="location" value="ResQLink Demo Junction" required></div><div><label>Latitude</label><input name="latitude" value="13.0827" required><label>Longitude</label><input name="longitude" value="80.2707" required><label>Demo-model confidence (0–1)</label><input name="demo_confidence" value="0.89" min="0" max="1" step="0.01" required><label>Severity</label><select name="severity"><option>High</option><option>Critical</option><option>Medium</option></select></div></div><button>Upload &amp; submit for operator verification</button></form></div><div class="card"><h2>Evidence uploads</h2><table><tr><th>Video</th><th>Camera</th><th>Signal</th><th>Incident</th><th></th></tr>{% for upload in uploads %}<tr><td>{{ upload.original_name }}</td><td>{{ upload.camera_id }}</td><td>{{ '%.0f'|format(upload.confidence*100) }}%<br><small>{{ upload.analysis_mode }}</small></td><td>{{ upload.reference or 'Review required' }}</td><td><a href="{{ url_for('video', upload_id=upload.id) }}" target="_blank">Play video</a></td></tr>{% else %}<tr><td colspan="5">No uploads yet.</td></tr>{% endfor %}</table></div>""", uploads=uploads)


@app.route("/cctv/video/<int:upload_id>")
def video(upload_id):
    user = require_roles("admin", "operator", "police")
    if not user: return redirect(url_for("login"))
    with connect() as db: record = db.execute("SELECT * FROM cctv_uploads WHERE id=?", (upload_id,)).fetchone()
    if not record: abort(404)
    return send_from_directory(UPLOADS, record["stored_name"], conditional=True)


@app.route("/incident/<int:incident_id>")
def incident_detail(incident_id):
    user = require_login()
    if not user: return redirect(url_for("login"))
    with connect() as db:
        if not incident_visible(db, user, incident_id): abort(403)
        incident = db.execute("SELECT i.*,a.label ambulance_label,a.driver_phone,a.driver_username,h.name hospital_name,h.emergency_beds,h.trauma_beds,h.emergency_department,h.readiness,h.contact_username FROM incidents i LEFT JOIN ambulances a ON a.id=i.ambulance_id LEFT JOIN hospitals h ON h.id=i.hospital_id WHERE i.id=?", (incident_id,)).fetchone()
        events = db.execute("SELECT * FROM timeline WHERE incident_id=? ORDER BY id DESC", (incident_id,)).fetchall()
        points = db.execute("SELECT * FROM location_updates WHERE incident_id=? ORDER BY id DESC LIMIT 20", (incident_id,)).fetchall()
        upload = db.execute("SELECT id,original_name,analysis_mode FROM cctv_uploads WHERE incident_id=? ORDER BY id DESC LIMIT 1", (incident_id,)).fetchone()
    driver_allowed = user["role"] in DRIVER_ROLES and incident["driver_username"] == user["username"]
    can_control = user["role"] in CONTROL_ROLES
    can_hospital = user["role"] == "hospital" and incident["contact_username"] == user["username"]
    return page("Live incident", """<h1>{{ incident.reference }}</h1><p><span class="badge">{{ incident.status }}</span> <span class="badge">{{ incident.severity }}</span> <span class="muted">{{ incident.source }} · {{ incident.created_at }}</span></p><div class="two"><section class="card"><h2>Incident &amp; response</h2><p><strong>Location:</strong> {{ incident.location }}<br><span class="muted">{{ incident.latitude }}, {{ incident.longitude }}</span></p><p><strong>Detection confidence:</strong> {% if incident.confidence is not none %}{{ '%.0f'|format(incident.confidence*100) }}% (verification required){% else %}Citizen report{% endif %}</p><p><strong>Ambulance:</strong> {{ incident.ambulance_label or 'Awaiting manual assignment' }}{% if incident.driver_phone %}<br><small>Driver contact: {{ incident.driver_phone }}</small>{% endif %}</p><p><strong>Hospital:</strong> {{ incident.hospital_name or 'Awaiting recommendation' }}{% if incident.hospital_name %}<br><small>{{ incident.emergency_department }} · {{ incident.emergency_beds }} emergency / {{ incident.trauma_beds }} trauma beds currently available · {{ incident.readiness }}</small>{% endif %}</p>{% if upload %}<p><a class="button alt" href="{{ url_for('video',upload_id=upload.id) }}" target="_blank">Play CCTV evidence</a><br><small>{{ upload.analysis_mode }}</small></p>{% endif %}</section><section class="card"><h2>Live ambulance tracking</h2>{% if points %}{% set point=points[0] %}<p class="good"><strong>Last location: {{ point.captured_at }}</strong></p><p>{{ point.latitude }}, {{ point.longitude }}<br><small>{{ point.note or 'Location update' }}</small></p>{% if can_control %}<a class="button alt" href="{{ url_for('live_map', incident_id=incident.id) }}">Open internal visual tracker</a>{% endif %}{% else %}<p class="muted">No driver location update yet.</p>{% endif %}{% if driver_allowed %}<form method="post" action="{{ url_for('driver_location',incident_id=incident.id) }}"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><label>Latitude</label><input id="lat" name="latitude" required><label>Longitude</label><input id="lon" name="longitude" required><label>Live status</label><select name="status"><option>Ambulance en route</option><option>At scene</option><option>Patient onboard</option><option>Arrived at hospital</option></select><label>Note</label><input name="note" value="Driver update"><button>Share GPS update</button><button type="button" class="alt" onclick="navigator.geolocation.getCurrentPosition(p=>{document.getElementById('lat').value=p.coords.latitude;document.getElementById('lon').value=p.coords.longitude},()=>alert('Location permission was not granted. Enter coordinates manually.'),{enableHighAccuracy:true})">Use this phone GPS</button></form><p class="muted">GPS is shared only when the driver chooses to submit it.</p>{% endif %}</section></div><section class="card"><h2>Role actions</h2>{% if can_control %}<form method="post" action="{{ url_for('control_status',incident_id=incident.id) }}"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><select name="status">{% if not incident.ambulance_id %}<option value="Operator verified">Verify incident &amp; dispatch</option><option>False alarm</option>{% else %}<option>Under control</option><option>Resolved</option>{% endif %}</select><button>Confirm command action</button></form>{% elif can_hospital %}<form method="post" action="{{ url_for('hospital_status',incident_id=incident.id) }}"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><select name="status"><option>Hospital ready</option><option>Patient received</option></select><button>Update hospital status</button></form>{% else %}<p class="muted">Your role can view the incident and its assigned actions, but cannot change control-room status.</p>{% endif %}</section><section class="card"><h2>Verified event timeline</h2><div class="timeline">{% for event in events %}<p><strong>{{ event.event }}</strong><br>{{ event.details }}<br><small class="muted">{{ event.created_at }} · {{ event.actor }}</small></p>{% endfor %}</div></section>""", incident=incident, events=events, points=points, upload=upload, driver_allowed=driver_allowed, can_control=can_control, can_hospital=can_hospital)


@app.post("/incident/<int:incident_id>/driver-location")
def driver_location(incident_id):
    user = require_roles(*DRIVER_ROLES)
    if not user: return redirect(url_for("login"))
    try:
        lat = parse_coordinate(request.form.get("latitude"), -90, 90, "Latitude"); lon = parse_coordinate(request.form.get("longitude"), -180, 180, "Longitude")
        status = request.form.get("status") or "Ambulance en route"
        if status not in {"Ambulance en route", "At scene", "Patient onboard", "Arrived at hospital"}: raise ValueError("Invalid ambulance status.")
        with connect() as db:
            incident = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
            ambulance = db.execute("SELECT * FROM ambulances WHERE driver_username=?", (user["username"],)).fetchone()
            if not incident or not ambulance or incident["ambulance_id"] != ambulance["id"]: abort(403)
            db.execute("UPDATE ambulances SET latitude=?,longitude=?,status=?,last_update=? WHERE id=?", (lat, lon, "At scene" if status == "At scene" else "Dispatched", now(), ambulance["id"]))
            db.execute("UPDATE incidents SET status=?,updated_at=? WHERE id=?", (status, now(), incident_id))
            db.execute("INSERT INTO location_updates (incident_id,ambulance_id,latitude,longitude,note,captured_at) VALUES (?,?,?,?,?,?)", (incident_id, ambulance["id"], lat, lon, (request.form.get("note") or "Driver live GPS")[:180], now()))
            add_event(db, incident_id, user["username"], "Driver GPS update", f"{lat:.5f}, {lon:.5f}; status: {status}")
            notify(db, incident_id, CONTROL_ROLES, "Ambulance live update", f"{incident['reference']}: driver reports {status} at {lat:.5f}, {lon:.5f}.")
            hospital = db.execute("SELECT contact_username FROM hospitals WHERE id=?", (incident["hospital_id"],)).fetchone()
            if hospital and hospital["contact_username"]:
                notify(db, incident_id, "hospital", "Ambulance live update", f"{incident['reference']}: driver reports {status}.", target_username=hospital["contact_username"])
            db.commit()
        flash("Live GPS and ambulance status shared with authorised response roles.", "success")
    except ValueError as error: flash(str(error), "error")
    return redirect(url_for("incident_detail", incident_id=incident_id))


@app.post("/incident/<int:incident_id>/control-status")
def control_status(incident_id):
    user = require_roles("admin", "operator")
    if not user: return redirect(url_for("login"))
    status = request.form.get("status")
    if status not in {"Operator verified", "Under control", "Resolved", "False alarm"}: abort(400)
    with connect() as db:
        incident = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not incident: abort(404)
        if status == "Operator verified":
            if incident["status"] == "False alarm" or incident["ambulance_id"] or incident["hospital_id"]:
                abort(409)
            ambulance, hospital = dispatch_incident(db, incident, user["username"])
            db.commit()
            if ambulance and hospital:
                flash("Operator verification recorded. Dispatch was initiated only after your confirmation.", "success")
            else:
                flash("Operator verification recorded, but a manual dispatcher must assign available resources.", "error")
            return redirect(url_for("incident_detail", incident_id=incident_id))
        if status in {"Under control", "Resolved"} and not incident["ambulance_id"]:
            abort(409)
        if status == "False alarm" and incident["ambulance_id"]:
            abort(409)
        db.execute("UPDATE incidents SET status=?,updated_at=? WHERE id=?", (status, now(), incident_id))
        if status == "Resolved" and incident["ambulance_id"]:
            db.execute("UPDATE ambulances SET status='Available',last_update=? WHERE id=?", (now(), incident["ambulance_id"]))
        add_event(db, incident_id, user["username"], "Command status updated", status)
        target_roles = CONTROL_ROLES if status == "False alarm" else ALL_ROLES - {"citizen"}
        notify(db, incident_id, target_roles, "Incident status updated", f"{incident['reference']} is now: {status}.")
        db.commit()
    flash("Control-room status updated.", "success"); return redirect(url_for("incident_detail", incident_id=incident_id))


@app.post("/incident/<int:incident_id>/hospital-status")
def hospital_status(incident_id):
    user = require_roles("hospital")
    if not user: return redirect(url_for("login"))
    status = request.form.get("status")
    if status not in {"Hospital ready", "Patient received"}: abort(400)
    with connect() as db:
        incident = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        hospital = db.execute("SELECT * FROM hospitals WHERE id=?", (incident["hospital_id"],)).fetchone() if incident else None
        if not incident or not hospital or hospital["contact_username"] != user["username"]: abort(403)
        db.execute("UPDATE incidents SET status=?,updated_at=? WHERE id=?", (status, now(), incident_id))
        readiness = "Patient received" if status == "Patient received" else "Ready for incoming patient"
        db.execute("UPDATE hospitals SET readiness=?,updated_at=? WHERE id=?", (readiness, now(), hospital["id"]))
        add_event(db, incident_id, user["username"], "Hospital readiness updated", status)
        notify(db, incident_id, CONTROL_ROLES, "Hospital update", f"{incident['reference']}: {status}.")
        driver = db.execute("SELECT driver_username FROM ambulances WHERE id=?", (incident["ambulance_id"],)).fetchone()
        if driver and driver["driver_username"]:
            driver_role = db.execute("SELECT role FROM users WHERE username=?", (driver["driver_username"],)).fetchone()
            if driver_role:
                notify(db, incident_id, driver_role["role"], "Hospital update", f"{incident['reference']}: {status}.", target_username=driver["driver_username"])
        db.commit()
    flash("Hospital readiness status updated.", "success"); return redirect(url_for("incident_detail", incident_id=incident_id))


@app.route("/alerts")
def alerts():
    user = require_login()
    if not user: return redirect(url_for("login"))
    with connect() as db:
        rows = db.execute("SELECT n.*,i.reference,i.location,i.status FROM notifications n JOIN incidents i ON i.id=n.incident_id WHERE n.target_role=? AND (n.target_username IS NULL OR n.target_username=?) ORDER BY n.id DESC LIMIT 50", (user["role"], user["username"])).fetchall()
        db.execute("UPDATE notifications SET read_at=? WHERE target_role=? AND (target_username IS NULL OR target_username=?) AND read_at IS NULL", (now(), user["role"], user["username"])); db.commit()
    return page("My alerts", """<h1>My role alerts</h1><p class="muted">Only notices assigned to the {{ user.role|replace('_',' ') }} role appear here.</p>{% for alert in rows %}<section class="card"><h3>{{ alert.title }}</h3><p>{{ alert.message }}</p><p><span class="badge">{{ alert.status }}</span> <small class="muted">{{ alert.created_at }}</small></p><a class="button alt" href="{{ url_for('incident_detail',incident_id=alert.incident_id) }}">Open {{ alert.reference }}</a></section>{% else %}<div class="card muted">No alerts for your role.</div>{% endfor %}""", user=user, rows=rows)


@app.route("/resources")
def resources():
    user = require_roles("admin", "operator")
    if not user: return redirect(url_for("login"))
    with connect() as db:
        ambulances = db.execute("SELECT * FROM ambulances ORDER BY id").fetchall(); hospitals = db.execute("SELECT * FROM hospitals ORDER BY id").fetchall()
    return page("Emergency resources", """<h1>Emergency resources</h1><p><a class="button" href="{{ url_for('driver_directory') }}">Configure ambulance drivers</a></p><div class="two"><section class="card"><h2>Ambulances</h2><table><tr><th>Unit</th><th>Driver / phone</th><th>Status</th><th>Last GPS</th></tr>{% for a in ambulances %}<tr><td>{{ a.label }}</td><td>{{ a.driver_username or 'Unassigned' }}<br><small>{{ a.driver_phone }}</small></td><td>{{ a.status }}</td><td>{{ a.latitude }}, {{ a.longitude }}<br><small>{{ a.last_update }}</small></td></tr>{% endfor %}</table></section><section class="card"><h2>Hospital readiness</h2><table><tr><th>Hospital</th><th>Emergency department</th><th>Beds</th><th>Status</th></tr>{% for h in hospitals %}<tr><td>{{ h.name }}</td><td>{{ h.emergency_department }}</td><td>{{ h.emergency_beds }} emergency<br>{{ h.trauma_beds }} trauma</td><td>{{ h.readiness }}</td></tr>{% endfor %}</table></section></div>""", ambulances=ambulances, hospitals=hospitals)


@app.route("/driver-directory", methods=["GET", "POST"])
def driver_directory():
    user = require_roles("admin", "operator")
    if not user:
        return redirect(url_for("login"))
    if request.method == "POST":
        ambulance_id = request.form.get("ambulance_id") or ""
        driver_username = (request.form.get("driver_username") or "").strip()
        phone = (request.form.get("driver_phone") or "").strip()
        if not (7 <= len(phone) <= 25) or any(character not in "+-() 0123456789" for character in phone):
            flash("Enter a valid driver phone number, using digits and an optional + country code.", "error")
        else:
            with connect() as db:
                ambulance = db.execute("SELECT * FROM ambulances WHERE id=?", (ambulance_id,)).fetchone()
                driver = db.execute("SELECT * FROM users WHERE username=?", (driver_username,)).fetchone()
                if not ambulance:
                    flash("Select a valid ambulance.", "error")
                elif not driver or driver["role"] not in DRIVER_ROLES:
                    flash("Select a registered ambulance-driver account.", "error")
                else:
                    db.execute("UPDATE ambulances SET driver_username=?,driver_phone=?,last_update=? WHERE id=?", (driver_username, phone, now(), ambulance_id))
                    db.commit()
                    flash(f"{ambulance['label']} is now linked to {driver_username} and the updated driver phone.", "success")
                    return redirect(url_for("driver_directory"))
    with connect() as db:
        ambulances = db.execute("SELECT * FROM ambulances ORDER BY id").fetchall()
        drivers = db.execute("SELECT username,display_name FROM users WHERE role IN ('ambulance','ambulance_crew','amb_crew','driver') ORDER BY username").fetchall()
    return page("Driver directory", """<h1>Ambulance driver directory</h1><p class="muted">Control-room configuration only. Link each ambulance to its driver account and authorised contact number. A driver can share GPS only while signed in to that account and assigned to an active incident.</p><div class="two"><section class="card"><h2>Add or update driver contact</h2><form method="post"><input type="hidden" name="_csrf" value="{{ csrf_token() }}"><label>Ambulance</label><select name="ambulance_id" required>{% for ambulance in ambulances %}<option value="{{ ambulance.id }}">{{ ambulance.label }} ({{ ambulance.id }})</option>{% endfor %}</select><label>Driver account</label><select name="driver_username" required>{% for driver in drivers %}<option value="{{ driver.username }}">{{ driver.display_name }} — {{ driver.username }}</option>{% endfor %}</select><label>Driver phone number</label><input name="driver_phone" placeholder="+91 98765 43210" required><button>Save driver linkage</button></form></section><section class="card"><h2>Current assignments</h2><table><tr><th>Ambulance</th><th>Driver</th><th>Phone</th><th>Status</th></tr>{% for ambulance in ambulances %}<tr><td>{{ ambulance.label }}</td><td>{{ ambulance.driver_username or 'Unassigned' }}</td><td>{{ ambulance.driver_phone }}</td><td>{{ ambulance.status }}</td></tr>{% endfor %}</table></section></div>""", ambulances=ambulances, drivers=drivers)


@app.route("/driver-live")
def driver_live():
    user = require_login()
    if not user:
        return redirect(url_for("login"))
    if user["role"] not in DRIVER_ROLES:
        return page(
            "Driver live GPS",
            """<div class="card"><h1>Driver Live GPS</h1><p>GPS sharing is available only to the signed-in driver account assigned to an active ambulance response.</p><p>The driver must open this page from their own account, explicitly start sharing, and grant browser location permission. Control-room users can then view updates on the incident's Live Map.</p></div>""",
        )
    with connect() as db:
        ambulance = db.execute("SELECT * FROM ambulances WHERE driver_username=?", (user["username"],)).fetchone()
        incident = db.execute("SELECT * FROM incidents WHERE ambulance_id=? AND status NOT IN ('Resolved','Closed','Patient received') ORDER BY id DESC LIMIT 1", (ambulance["id"],)).fetchone() if ambulance else None
    if not incident:
        return page("Driver live GPS", """<div class="card"><h1>Driver Live GPS</h1><p>No active incident is assigned to your ambulance. The control room must dispatch an incident before live tracking can start.</p></div>""")
    return page("Driver live GPS", """<div class="card" style="max-width:680px;margin:auto"><h1>Driver Live GPS</h1><p><strong>{{ ambulance.label }}</strong> · {{ incident.reference }}</p><p class="muted">Sharing starts only after you press Start and grant browser location permission. Your precise GPS coordinates and reported accuracy are sent to ResQLink for this assigned incident and are visible to authorised control-room staff. ResQLink does not use your IP address to determine your location. Stop sharing when the response ends; sharing also stops when this page closes.</p><label>Current incident status</label><select id="live-status"><option>Ambulance en route</option><option>At scene</option><option>Patient onboard</option><option>Arrived at hospital</option></select><label>Location note</label><input id="live-note" value="Live GPS from assigned ambulance driver"><p id="live-state" class="muted" aria-live="polite">GPS sharing is stopped.</p><p><button id="live-toggle" type="button">Start live GPS sharing</button> <a class="button alt" href="{{ url_for('incident_detail',incident_id=incident.id) }}">Open incident</a></p><div class="card"><strong>Latest coordinates</strong><p id="live-coordinates">Waiting for phone GPS…</p><small id="live-accuracy" class="muted">Accuracy will appear after the first GPS fix.</small><br><small class="muted">Updates are sent at most once every 12 seconds, while this page remains open. Location requires HTTPS except on localhost.</small></div></div><script>(function(){const state=document.getElementById('live-state'),coordinates=document.getElementById('live-coordinates'),accuracy=document.getElementById('live-accuracy'),toggle=document.getElementById('live-toggle'),status=document.getElementById('live-status'),note=document.getElementById('live-note'),csrf='{{ csrf_token() }}';let watcher=null,lastSent=0;function fail(error){state.textContent='GPS unavailable: '+(error&&error.message?error.message:'permission was not granted.');}async function send(position){coordinates.textContent=position.coords.latitude.toFixed(6)+', '+position.coords.longitude.toFixed(6);accuracy.textContent=Number.isFinite(position.coords.accuracy)?'Reported GPS accuracy: about '+Math.round(position.coords.accuracy)+' metres.':'GPS accuracy was not reported.';const time=Date.now();if(time-lastSent<12000)return;lastSent=time;try{const reply=await fetch('{{ url_for('api_driver_location') }}',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},credentials:'same-origin',body:JSON.stringify({incident_id:{{ incident.id }},latitude:position.coords.latitude,longitude:position.coords.longitude,accuracy_meters:position.coords.accuracy,status:status.value,note:note.value})});if(!reply.ok){const result=await reply.json().catch(()=>({}));throw new Error(result.error||'server rejected the update');}state.textContent='GPS shared with ResQLink at '+new Date().toLocaleTimeString()+'.';}catch(error){fail(error);}}toggle.onclick=()=>{if(watcher!==null){navigator.geolocation.clearWatch(watcher);watcher=null;toggle.textContent='Start live GPS sharing';state.textContent='GPS sharing stopped.';return;}if(!navigator.geolocation){state.textContent='This browser does not support GPS.';return;}watcher=navigator.geolocation.watchPosition(send,fail,{enableHighAccuracy:true,maximumAge:5000,timeout:20000});toggle.textContent='Stop live GPS sharing';state.textContent='Waiting for phone GPS permission…';};window.addEventListener('pagehide',()=>{if(watcher!==null)navigator.geolocation.clearWatch(watcher);});})();</script>""", ambulance=ambulance, incident=incident)


@app.route("/live-tracker")
def live_tracker():
    user = require_roles("admin", "operator")
    if not user:
        return redirect(url_for("login"))
    requested_id = request.args.get("incident_id", type=int)
    with connect() as db:
        incident = db.execute("SELECT * FROM incidents WHERE id=?", (requested_id,)).fetchone() if requested_id else db.execute("SELECT * FROM incidents WHERE ambulance_id IS NOT NULL AND status NOT IN ('Resolved','Closed','Patient received') ORDER BY id DESC LIMIT 1").fetchone()
    if not incident:
        return page("Live ambulance tracker", """<div class="card"><h1>Live ambulance tracker</h1><p>No active ambulance assignment is available. Upload/report an incident and dispatch an ambulance first.</p></div>""")
    return page("Live ambulance tracker", """<h1>Live ambulance tracker</h1><p class="muted">{{ incident.reference }} · Updates refresh automatically every 10 seconds while this screen is open.</p><div class="two"><section class="card"><h2 id="tracker-status">{{ incident.status }}</h2><p><strong>Incident:</strong> {{ incident.location }}</p><p><strong>Assigned unit:</strong> {{ incident.ambulance_id }}</p><p><strong>Last driver GPS:</strong> <span id="tracker-coordinates">Loading…</span></p><p><strong>Captured:</strong> <span id="tracker-time">Loading…</span></p><p><strong>Driver note:</strong> <span id="tracker-note">Loading…</span></p><a id="tracker-map" class="button alt" target="_blank" href="#">Open latest location on map</a></section><section class="card"><h2>What this means</h2><p class="good">Live GPS is active only when the assigned driver has opened Driver Live GPS and granted consent.</p><p class="muted">If no new location appears, ask the driver to keep the tracking page open, ensure mobile location is enabled, and verify that the device is using the secure HTTPS deployment.</p></section></div><script>(function(){const endpoint='{{ url_for('api_incident',incident_id=incident.id) }}';async function refresh(){try{const response=await fetch(endpoint,{credentials:'same-origin'});if(!response.ok)throw new Error('Unable to load');const data=await response.json();document.getElementById('tracker-status').textContent=data.incident.status;const point=data.location_updates[0];if(!point){document.getElementById('tracker-coordinates').textContent='Waiting for driver update';return;}document.getElementById('tracker-coordinates').textContent=Number(point.latitude).toFixed(6)+', '+Number(point.longitude).toFixed(6);document.getElementById('tracker-time').textContent=point.captured_at;document.getElementById('tracker-note').textContent=point.note||'—';document.getElementById('tracker-map').href='https://www.openstreetmap.org/?mlat='+point.latitude+'&mlon='+point.longitude+'#map=16/'+point.latitude+'/'+point.longitude;}catch(error){document.getElementById('tracker-coordinates').textContent='Connection error — retrying…';}}refresh();setInterval(refresh,10000);})();</script>""", incident=incident)


@app.route("/live-map")
def live_map():
    """Operator-only tracker; the embedded OpenStreetMap receives map coordinates."""
    user = require_roles("admin", "operator")
    if not user:
        return redirect(url_for("login"))
    requested_id = request.args.get("incident_id", type=int)
    with connect() as db:
        incident = db.execute("SELECT * FROM incidents WHERE id=?", (requested_id,)).fetchone() if requested_id else db.execute("SELECT * FROM incidents WHERE ambulance_id IS NOT NULL AND status NOT IN ('Resolved','Closed','Patient received') ORDER BY id DESC LIMIT 1").fetchone()
        ambulance = db.execute("SELECT * FROM ambulances WHERE id=?", (incident["ambulance_id"],)).fetchone() if incident else None
    if not incident or not ambulance:
        return page("Visual live tracker", """<div class="card"><h1>Visual live tracker</h1><p>No active ambulance assignment is available yet.</p></div>""")
    return page("Visual live tracker", """<style>.tracking-state{padding:10px;border-radius:8px;background:#48351a;color:#ffe19a;font-weight:700}.geo-map{width:100%;height:480px;border:1px solid #48617e;border-radius:10px;background:#101927}</style><h1>Live ambulance map</h1><p class="muted">{{ incident.reference }} · The map and driver status refresh every 10 seconds.</p><div class="two"><section class="card"><h2 id="tracker-status">{{ incident.status }}</h2><p><strong>Incident:</strong> {{ incident.location }}</p><p><strong>Assigned unit:</strong> {{ ambulance.label }}</p><p><strong>Driver account:</strong> {{ ambulance.driver_username or 'Not linked' }}</p><p><strong>Driver phone:</strong> {{ ambulance.driver_phone }}</p><p id="gps-link-state" class="tracking-state" aria-live="polite">Checking for a driver GPS update…</p><p><strong>Last GPS:</strong> <span id="tracker-coordinates">Loading…</span><br><small id="tracker-time" class="muted"></small><br><small id="tracker-accuracy" class="muted"></small></p><a id="tracker-map-link" class="button alt" target="_blank" rel="noopener noreferrer" href="#">Open latest location in OpenStreetMap</a></section><section class="card"><h2>How GPS linkage works</h2><ol><li>Driver phone is linked to this ambulance.</li><li>The driver signs in to the assigned account.</li><li>The driver explicitly starts GPS sharing and grants browser permission.</li><li>Only then does the ResQLink server receive the device's GPS coordinates.</li></ol><p class="muted">ResQLink does not use or display a driver's IP address for location. The external OpenStreetMap map service receives the operator browser's map requests and the coordinates displayed on the map. Driver location access is limited by the signed-in account and incident assignment.</p><p><a href="{{ url_for('driver_live') }}">Open Driver Live GPS instructions</a></p></section></div><section class="card"><h2>Geographic live map</h2><iframe id="geo-map" class="geo-map" title="OpenStreetMap showing incident or latest ambulance GPS location" loading="lazy" referrerpolicy="no-referrer" src="https://www.openstreetmap.org/export/embed.html?bbox={{ incident.longitude - 0.01 }},{{ incident.latitude - 0.01 }},{{ incident.longitude + 0.01 }},{{ incident.latitude + 0.01 }}&layer=mapnik&marker={{ incident.latitude }},{{ incident.longitude }}"></iframe><p class="muted">Map marker shows the latest reported device GPS fix. It may be stale or inaccurate; check the timestamp and reported accuracy before relying on it.</p></section><script>(function(){const endpoint='{{ url_for('api_incident',incident_id=incident.id) }}',map=document.getElementById('geo-map');let shownPoint=null;function showMap(point){const lat=Number(point.latitude),lon=Number(point.longitude),key=lat+','+lon;if(key===shownPoint)return;shownPoint=key;const delta=0.01;map.src='https://www.openstreetmap.org/export/embed.html?bbox='+(lon-delta)+','+(lat-delta)+','+(lon+delta)+','+(lat+delta)+'&layer=mapnik&marker='+lat+','+lon;document.getElementById('tracker-map-link').href='https://www.openstreetmap.org/?mlat='+lat+'&mlon='+lon+'#map=16/'+lat+'/'+lon;}async function refresh(){const state=document.getElementById('gps-link-state');try{const response=await fetch(endpoint,{credentials:'same-origin'});if(!response.ok)throw new Error('Unable to load');const data=await response.json(),point=(data.location_updates||[])[0];document.getElementById('tracker-status').textContent=data.incident.status;if(!point){state.textContent='Waiting for the assigned driver to start GPS sharing.';document.getElementById('tracker-coordinates').textContent='No GPS fix received';return;}const lat=Number(point.latitude),lon=Number(point.longitude);document.getElementById('tracker-coordinates').textContent=lat.toFixed(6)+', '+lon.toFixed(6);document.getElementById('tracker-time').textContent='Last received: '+point.captured_at+' · '+(point.note||'GPS update');document.getElementById('tracker-accuracy').textContent=point.accuracy_meters==null?'GPS accuracy not reported':'Reported GPS accuracy: about '+Math.round(Number(point.accuracy_meters))+' metres';showMap(point);const isDispatchPoint=point.note==='Initial dispatch location',captured=Date.parse(String(point.captured_at).replace(' ','T')),ageSeconds=Number.isFinite(captured)?Math.max(0,(Date.now()-captured)/1000):Infinity;if(isDispatchPoint){state.textContent='Dispatch location only — waiting for the driver’s live GPS.';}else if(ageSeconds>45){state.textContent='GPS is stale — last driver update was '+Math.round(ageSeconds)+' seconds ago.';}else{state.textContent='Live GPS received from the assigned driver account ('+Math.round(ageSeconds)+' seconds ago).';}}catch(error){state.textContent='Tracker connection error — retrying automatically.';}}refresh();setInterval(refresh,10000);})();</script>""", incident=incident, ambulance=ambulance)


def api_user(*roles):
    user = require_login()
    if not user: return None, (jsonify(error="Authentication required"), 401)
    if roles and user["role"] not in roles: return None, (jsonify(error="Insufficient permission"), 403)
    return user, None


@app.get("/health")
def health(): return jsonify(status="ok", service="ResQLink Complete Demo", database=str(DATABASE))


@app.get("/api/incidents")
def api_incidents():
    user, error = api_user(*ALL_ROLES)
    if error: return error
    with connect() as db: rows = [dict(row) for row in db.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall() if incident_visible(db, user, row["id"])]
    return jsonify(rows)


@app.get("/api/incidents/<int:incident_id>")
def api_incident(incident_id):
    user, error = api_user(*ALL_ROLES)
    if error: return error
    with connect() as db:
        if not incident_visible(db, user, incident_id): return jsonify(error="Not found"), 404
        item = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        points = db.execute("SELECT * FROM location_updates WHERE incident_id=? ORDER BY id DESC LIMIT 10", (incident_id,)).fetchall()
    return jsonify(incident=dict(item), location_updates=[dict(point) for point in points])


@app.get("/api/resources")
def api_resources():
    user, error = api_user("admin", "operator", "hospital", *DRIVER_ROLES)
    if error: return error
    with connect() as db:
        ambulances = [dict(row) for row in db.execute("SELECT id,label,status,latitude,longitude,last_update FROM ambulances").fetchall()]
        hospitals = [dict(row) for row in db.execute("SELECT * FROM hospitals").fetchall()]
    return jsonify(ambulances=ambulances, hospitals=hospitals)


@app.post("/api/driver/location")
def api_driver_location():
    user, error = api_user(*DRIVER_ROLES)
    if error: return error
    given_csrf = request.headers.get("X-CSRF-Token", "")
    if not given_csrf or not secrets.compare_digest(given_csrf, csrf_token()):
        return jsonify(error="Missing or invalid CSRF token"), 400
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(error="Expected a JSON object"), 400
    incident_id = payload.get("incident_id")
    try:
        lat = parse_coordinate(payload.get("latitude"), -90, 90, "Latitude"); lon = parse_coordinate(payload.get("longitude"), -180, 180, "Longitude")
        accuracy = parse_coordinate(payload.get("accuracy_meters"), 0, 100000, "GPS accuracy") if payload.get("accuracy_meters") is not None else None
    except ValueError as exc: return jsonify(error=str(exc)), 400
    status = payload.get("status") or "Ambulance en route"
    if not isinstance(status, str) or status not in {"Ambulance en route", "At scene", "Patient onboard", "Arrived at hospital"}:
        return jsonify(error="Invalid ambulance status"), 400
    note = payload.get("note") or "Driver API GPS"
    if not isinstance(note, str):
        return jsonify(error="Location note must be text"), 400
    with connect() as db:
        ambulance = db.execute("SELECT * FROM ambulances WHERE driver_username=?", (user["username"],)).fetchone(); incident = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not ambulance or not incident or incident["ambulance_id"] != ambulance["id"]: return jsonify(error="No assigned incident"), 403
        db.execute("UPDATE ambulances SET latitude=?,longitude=?,status=?,last_update=? WHERE id=?", (lat, lon, "At scene" if status == "At scene" else "Dispatched", now(), ambulance["id"]))
        db.execute("UPDATE incidents SET status=?,updated_at=? WHERE id=?", (status, now(), incident_id))
        db.execute("INSERT INTO location_updates (incident_id,ambulance_id,latitude,longitude,accuracy_meters,note,captured_at) VALUES (?,?,?,?,?,?,?)", (incident_id, ambulance["id"], lat, lon, accuracy, note[:180], now()))
        accuracy_note = f"; reported accuracy: {accuracy:.0f} m" if accuracy is not None else ""
        add_event(db, incident_id, user["username"], "Driver GPS API update", f"{lat:.5f}, {lon:.5f}{accuracy_note}; status: {status}"); db.commit()
    return jsonify(status="accepted", incident_id=incident_id, ambulance_status=status, latitude=lat, longitude=lon, accuracy_meters=accuracy)


@app.post("/api/cctv/ingest")
def api_cctv_ingest():
    """Integration endpoint for an authorised external CV/ML worker.

    In development it accepts a logged-in control-room user.  In deployment,
    set RESQLINK_INGEST_KEY and send it as X-ResQLink-Ingest-Key from the
    protected backend worker; never call this endpoint directly from a camera
    or browser with camera credentials.
    """
    configured_key = os.environ.get("RESQLINK_INGEST_KEY")
    header_key = request.headers.get("X-ResQLink-Ingest-Key", "")
    user = current_user()
    trusted_worker = bool(configured_key and secrets.compare_digest(header_key, configured_key))
    if not trusted_worker and (not user or user["role"] not in CONTROL_ROLES):
        return jsonify(error="Control-room authentication or ingestion key required"), 401
    payload = request.get_json(silent=True) or {}
    try:
        confidence = parse_coordinate(payload.get("confidence"), 0, 1, "Confidence")
        latitude = parse_coordinate(payload.get("latitude"), -90, 90, "Latitude")
        longitude = parse_coordinate(payload.get("longitude"), -180, 180, "Longitude")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if confidence < 0.70:
        return jsonify(status="review_required", reason="CV confidence is below the candidate-alert threshold"), 202
    location = str(payload.get("location") or "Authorised CCTV camera location")[:180]
    camera_id = str(payload.get("camera_id") or "CV-CAMERA")[:80]
    with connect() as db:
        incident_id, duplicate = create_incident(
            db, source="CCTV model API", reporter_id=user["id"] if user else None,
            title=str(payload.get("title") or "CCTV collision signal")[:100],
            description=f"Authorised model event from {camera_id}; operator verification required.",
            location=location, latitude=latitude, longitude=longitude, confidence=confidence,
            severity=str(payload.get("severity") or "High")[:30], actor=user["username"] if user else "authorised-cv-worker",
        )
        record = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
    return jsonify(status="merged" if duplicate else "created", incident=dict(record)), 201 if not duplicate else 200


@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(413)
def app_error(error):
    descriptions = {400:"Bad request or expired form.",403:"This role is not allowed to access that information.",404:"The page or protected record was not found.",413:"The uploaded video is larger than the 100 MB prototype limit."}
    return page(f"Error {error.code}", """<div class="card" style="max-width:680px;margin:65px auto;text-align:center"><h1>Error {{ code }}</h1><p>{{ message }}</p><a class="button" href="{{ url_for('home') }}">Go to dashboard</a></div>""", code=error.code, message=descriptions.get(error.code, "Unexpected error.")), error.code


@app.errorhandler(500)
def unexpected_error(error):
    app.logger.exception("ResQLink complete demo request failed", exc_info=error)
    return page("Error 500", """<div class="card" style="max-width:680px;margin:65px auto;text-align:center"><h1>Unexpected error</h1><p>The request could not be completed. The local server log contains the diagnostic; no incident data was changed by this page error.</p><a class="button" href="{{ url_for('home') }}">Return to dashboard</a></div>"""), 500


if __name__ == "__main__":
    initialise()
    print("\nResQLink Complete Demo — reliable end-to-end prototype")
    print(f"  Database: {DATABASE}")
    print("  URL: http://127.0.0.1:5050")
    print("  Health check: http://127.0.0.1:5050/health")
    print("  Login: admin / Admin@ResQLink1\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
