import sqlite3
import hashlib
import secrets
from datetime import datetime
from pathlib import Path

import streamlit as st
import bcrypt

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "resqlink_incidents.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ROLES = {
    "admin": "Administrator",
    "operator": "Emergency Operator",
    "responder": "Emergency Responder",
    "hospital": "Hospital Staff",
    "police": "Police / Traffic",
    "fire": "Fire & Rescue",
    "public": "Public User",
}

PERMISSIONS = {
    "admin": {"manage_users", "view_incidents", "review_incidents", "manage_resources", "view_audit_log", "view_reports", "manage_agencies"},
    "operator": {"view_incidents", "review_incidents", "manage_resources", "view_reports", "manage_agencies"},
    "responder": {"view_assigned_incidents", "update_response_status"},
    "hospital": {"view_hospital_requests", "update_hospital_status"},
    "police": {"view_assigned_incidents", "update_response_status"},
    "fire": {"view_assigned_incidents", "update_response_status"},
    "public": {"submit_report", "view_own_reports"},
}


def connect():
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    return db


def initialize_database():
    with connect() as db:
        existing_users = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        if existing_users is None:
            db.execute("""CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash BLOB NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','operator','responder','hospital','police','fire','public')),
                agency_name TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
        elif "'police'" not in (existing_users["sql"] or "") or "'fire'" not in (existing_users["sql"] or ""):
            # Migrate the earlier auth.py users table safely while retaining existing accounts.
            old_columns = {row["name"] for row in db.execute("PRAGMA table_info(users)").fetchall()}
            db.execute("ALTER TABLE users RENAME TO users_before_portal_upgrade")
            db.execute("""CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash BLOB NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','operator','responder','hospital','police','fire','public')),
                agency_name TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            old_rows = db.execute("SELECT * FROM users_before_portal_upgrade").fetchall()
            for row in old_rows:
                keys = set(row.keys())
                db.execute("""INSERT INTO users(id,username,password_hash,role,agency_name,active,created_at)
                    VALUES(?,?,?,?,?,?,?)""", (
                    row["id"], row["username"], row["password_hash"], row["role"],
                    row["agency_name"] if "agency_name" in keys else "",
                    row["active"] if "active" in keys else 1,
                    row["created_at"] if "created_at" in keys else datetime.now().isoformat(sep=" ")
                ))
            db.execute("DROP TABLE users_before_portal_upgrade")
        db.execute("""CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS citizen_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reporter_username TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'Citizen Web',
            description TEXT NOT NULL,
            incident_time TEXT NOT NULL,
            location_text TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            contact TEXT,
            evidence_filename TEXT,
            status TEXT NOT NULL DEFAULT 'Submitted',
            linked_incident_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS response_resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agency_type TEXT NOT NULL CHECK(agency_type IN ('ambulance','hospital','police','fire')),
            resource_name TEXT NOT NULL,
            location_text TEXT NOT NULL DEFAULT '',
            latitude REAL,
            longitude REAL,
            status TEXT NOT NULL DEFAULT 'Available',
            capacity_note TEXT NOT NULL DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS response_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER,
            report_id INTEGER,
            username TEXT NOT NULL,
            agency_role TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        # Existing incident schema is deliberately retained. No incident rows are deleted or rewritten.


def audit(username, action, details=""):
    with connect() as db:
        db.execute("INSERT INTO audit_log(username, action, details) VALUES (?, ?, ?)",
                   (username, action, details))


def has_permission(role, permission):
    return permission in PERMISSIONS.get(role, set())


def create_user(username, password, role, agency_name=""):
    username = username.strip()
    if len(username) < 3:
        raise ValueError("Username must contain at least 3 characters.")
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if role not in ROLES:
        raise ValueError("Invalid role.")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    with connect() as db:
        db.execute("INSERT INTO users(username,password_hash,role,agency_name) VALUES(?,?,?,?)",
                   (username, hashed, role, agency_name.strip()))


def authenticate(username, password):
    with connect() as db:
        row = db.execute("SELECT id,username,password_hash,role,agency_name FROM users WHERE username=? AND active=1",
                         (username.strip(),)).fetchone()
    if not row:
        # Avoid revealing whether a username exists.
        return None
    stored = row["password_hash"]
    if isinstance(stored, str):
        stored = stored.encode("utf-8")
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), stored)
    except (ValueError, TypeError):
        return None
    if not ok:
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"],
            "agency_name": row["agency_name"], "role_name": ROLES[row["role"]]}


def can(role, permission):
    if not has_permission(role, permission):
        st.error("Access denied: your account does not have permission for this action.")
        return False
    return True


def table_count(table):
    with connect() as db:
        return db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def show_incidents():
    with connect() as db:
        rows = db.execute("""SELECT id, vehicle_id, frame, vehicle_type, movement_per_sec,
                             direction_change_angle, deceleration, status, severity,
                             ambulance_assigned, police_assigned, created_at
                             FROM incidents ORDER BY id DESC LIMIT 500""").fetchall()
    if rows:
        st.dataframe([dict(r) for r in rows], use_container_width=True, hide_index=True)
    else:
        st.info("No incident records yet. AI movement candidates are not confirmed accidents.")


def show_reports(username=None):
    with connect() as db:
        if username is None:
            rows = db.execute("SELECT * FROM citizen_reports ORDER BY id DESC LIMIT 500").fetchall()
        else:
            rows = db.execute("SELECT * FROM citizen_reports WHERE reporter_username=? ORDER BY id DESC",
                              (username,)).fetchall()
    if rows:
        st.dataframe([dict(r) for r in rows], use_container_width=True, hide_index=True)
    else:
        st.info("No reports found.")


def report_form(username):
    st.subheader("Submit an accident report")
    st.caption("This is a local demonstration form. Submitting here does not contact emergency services.")
    with st.form("citizen_report_form", clear_on_submit=True):
        description = st.text_area("What happened? *", max_chars=3000)
        incident_time = st.datetime_input("Approximate incident date and time", value=datetime.now())
        location_text = st.text_input("Location description / nearby landmark *")
        c1, c2 = st.columns(2)
        with c1:
            lat_text = st.text_input("Latitude (optional)", placeholder="e.g. 13.0827")
        with c2:
            lon_text = st.text_input("Longitude (optional)", placeholder="e.g. 80.2707")
        contact = st.text_input("Contact number (optional)")
        evidence = st.file_uploader("Photo or short video evidence (optional)", type=["png", "jpg", "jpeg", "mp4", "mov"])
        submitted = st.form_submit_button("Submit report")
    if submitted:
        if not description.strip() or not location_text.strip():
            st.error("Please provide the incident description and location.")
            return
        try:
            lat = float(lat_text) if lat_text.strip() else None
            lon = float(lon_text) if lon_text.strip() else None
            if lat is not None and not -90 <= lat <= 90:
                raise ValueError("Latitude must be between -90 and 90.")
            if lon is not None and not -180 <= lon <= 180:
                raise ValueError("Longitude must be between -180 and 180.")
            if (lat is None) != (lon is None):
                raise ValueError("Enter both latitude and longitude, or leave both blank.")
        except ValueError as e:
            st.error(str(e))
            return
        filename = None
        if evidence is not None:
            # Use a generated filename rather than trusting the uploaded name.
            suffix = Path(evidence.name).suffix.lower()
            filename = f"report_{secrets.token_hex(12)}{suffix}"
            (UPLOAD_DIR / filename).write_bytes(evidence.getvalue())
        with connect() as db:
            db.execute("""INSERT INTO citizen_reports
                (reporter_username, description, incident_time, location_text, latitude, longitude, contact, evidence_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (username, description.strip(), incident_time.isoformat(sep=" "),
                 location_text.strip(), lat, lon, contact.strip(), filename))
        audit(username, "citizen_report_submitted", f"Report submitted; evidence={'yes' if filename else 'no'}")
        st.success("Report submitted to the local demonstration database. No emergency service has been contacted.")


def admin_user_management():
    st.subheader("Create a user account")
    st.caption("Create strong unique passwords. Do not use real emergency-service credentials in this demo.")
    with st.form("create_user_form"):
        username = st.text_input("Username")
        password = st.text_input("Temporary password (minimum 12 characters)", type="password")
        role = st.selectbox("Role", list(ROLES.keys()), format_func=lambda r: ROLES[r])
        agency = st.text_input("Agency / organization name (optional)")
        submitted = st.form_submit_button("Create account")
    if submitted:
        try:
            create_user(username, password, role, agency)
            audit(st.session_state.user["username"], "user_created", f"username={username.strip()}, role={role}")
            st.success(f"Account created for {username.strip()} ({ROLES[role]}).")
        except sqlite3.IntegrityError:
            st.error("That username already exists.")
        except ValueError as e:
            st.error(str(e))
    with connect() as db:
        users = db.execute("SELECT id, username, role, agency_name, active, created_at FROM users ORDER BY id").fetchall()
    st.subheader("Accounts")
    if users:
        st.dataframe([dict(r) for r in users], use_container_width=True, hide_index=True)
        options = {f'{r["username"]} ({r["role"]})': r["id"] for r in users if r["username"] != st.session_state.user["username"]}
        if options:
            selected = st.selectbox("Account to enable / disable", list(options))
            if st.button("Toggle account active status"):
                with connect() as db:
                    current = db.execute("SELECT active FROM users WHERE id=?", (options[selected],)).fetchone()
                    db.execute("UPDATE users SET active=? WHERE id=?", (0 if current["active"] else 1, options[selected]))
                audit(st.session_state.user["username"], "user_active_toggled", selected)
                st.rerun()


def resource_management():
    st.subheader("Resource registry (demonstration records)")
    st.warning("These are manually entered demo resources. They are not connected to live ambulance fleets or hospital systems.")
    with st.form("resource_form"):
        agency_type = st.selectbox("Agency/resource type", ["ambulance", "hospital", "police", "fire"])
        resource_name = st.text_input("Resource / facility name")
        location_text = st.text_input("Location description")
        c1, c2 = st.columns(2)
        with c1:
            lat = st.text_input("Latitude (optional)")
        with c2:
            lon = st.text_input("Longitude (optional)")
        status = st.selectbox("Status", ["Available", "Busy", "Dispatched", "Offline", "Receiving patients", "At scene"])
        capacity_note = st.text_input("Capacity / notes (e.g. demo capacity; do not enter sensitive patient data)")
        submitted = st.form_submit_button("Add resource")
    if submitted:
        if not resource_name.strip():
            st.error("Enter a resource name.")
        else:
            try:
                latitude = float(lat) if lat.strip() else None
                longitude = float(lon) if lon.strip() else None
                if (latitude is None) != (longitude is None):
                    raise ValueError("Enter both coordinates or leave both blank.")
                if latitude is not None and not -90 <= latitude <= 90:
                    raise ValueError("Latitude must be between -90 and 90.")
                if longitude is not None and not -180 <= longitude <= 180:
                    raise ValueError("Longitude must be between -180 and 180.")
                with connect() as db:
                    db.execute("""INSERT INTO response_resources
                        (agency_type,resource_name,location_text,latitude,longitude,status,capacity_note)
                        VALUES(?,?,?,?,?,?,?)""",
                        (agency_type, resource_name.strip(), location_text.strip(), latitude, longitude, status, capacity_note.strip()))
                audit(st.session_state.user["username"], "resource_added", f"type={agency_type}, name={resource_name.strip()}")
                st.success("Demo resource added.")
            except ValueError as e:
                st.error(str(e))
    with connect() as db:
        rows = db.execute("SELECT * FROM response_resources WHERE active=1 ORDER BY agency_type, resource_name").fetchall()
    if rows:
        st.dataframe([dict(r) for r in rows], use_container_width=True, hide_index=True)
        coords = [{"name": r["resource_name"], "latitude": r["latitude"], "longitude": r["longitude"], "type": r["agency_type"], "status": r["status"]} for r in rows if r["latitude"] is not None and r["longitude"] is not None]
        if coords:
            st.subheader("Map of resources with supplied coordinates")
            st.map([{"lat": r["latitude"], "lon": r["longitude"]} for r in coords])
            st.caption("Map markers use the coordinates entered by the operator; blank coordinates are not guessed.")
    else:
        st.info("No resource records have been added.")


def response_update(role, username):
    st.subheader("Update response status")
    with connect() as db:
        incidents = db.execute("SELECT id, status, severity FROM incidents ORDER BY id DESC LIMIT 300").fetchall()
        reports = db.execute("SELECT id, status, location_text FROM citizen_reports ORDER BY id DESC LIMIT 300").fetchall()
    choices = [f"Incident #{r['id']} — {r['status']} / {r['severity']}" for r in incidents]
    report_choices = [f"Citizen report #{r['id']} — {r['status']} / {r['location_text']}" for r in reports]
    source = st.radio("Record type", ["Incident", "Citizen report"], horizontal=True)
    if source == "Incident" and choices:
        selected = st.selectbox("Select incident", choices)
        record_id = int(selected.split("#")[1].split(" ")[0])
        report_id = None
        incident_id = record_id
    elif source == "Citizen report" and report_choices:
        selected = st.selectbox("Select report", report_choices)
        report_id = int(selected.split("#")[1].split(" ")[0])
        incident_id = None
    else:
        st.info("No records of this type are available.")
        return
    statuses = ["Acknowledged", "En route", "On scene", "Transporting", "Arrived at hospital", "Completed", "Unable to respond", "Needs escalation"]
    status = st.selectbox("New response status", statuses)
    notes = st.text_area("Operational note (no sensitive patient details)")
    if st.button("Save response update"):
        with connect() as db:
            db.execute("""INSERT INTO response_updates(incident_id,report_id,username,agency_role,status,notes)
                          VALUES(?,?,?,?,?,?)""",
                       (incident_id, report_id, username, role, status, notes.strip()))
            if report_id is not None and role in ("operator", "admin"):
                db.execute("UPDATE citizen_reports SET status=? WHERE id=?", (status, report_id))
        audit(username, "response_status_updated", f"role={role}, record={source}, id={incident_id or report_id}, status={status}")
        st.success("Response update saved to the local database.")


def main():
    st.set_page_config(page_title="ResQLink AI", page_icon="🚨", layout="wide")
    initialize_database()

    st.title("🚨 ResQLink AI")
    st.caption("Smart Road Accident Detection & Emergency Response Coordination — local demonstration")

    with connect() as db:
        user_count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    if not st.session_state.get("user"):
        st.info("Sign in to open your role-specific portal.")
        if user_count == 0:
            st.warning("First-time setup: create the initial administrator account. There are no default credentials.")
            with st.form("initial_admin"):
                username = st.text_input("Administrator username")
                p1 = st.text_input("Password (minimum 12 characters)", type="password")
                p2 = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create administrator")
            if submitted:
                if p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        create_user(username, p1, "admin", "System Administration")
                        audit(username.strip(), "initial_admin_created", "Initial administrator account created")
                        st.success("Administrator created. Please sign in.")
                        st.rerun()
                    except (ValueError, sqlite3.IntegrityError) as e:
                        st.error(str(e))
            st.stop()

        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state.user = user
                audit(user["username"], "login_success", f"role={user['role']}")
                st.rerun()
            else:
                st.error("Login failed. Check your username and password or contact the administrator.")
        st.divider()
        st.subheader("Public accident reporting")
        st.write("You can create a Public User account to submit reports and view only your own reports.")
        with st.form("public_signup"):
            new_username = st.text_input("Choose username", key="public_new_user")
            new_password = st.text_input("Choose password (minimum 12 characters)", type="password", key="public_new_password")
            confirm = st.text_input("Confirm password", type="password", key="public_confirm_password")
            signup = st.form_submit_button("Create Public User account")
        if signup:
            if new_password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    create_user(new_username, new_password, "public")
                    audit(new_username.strip(), "public_account_created", "Public account self-registration")
                    st.success("Account created. Sign in above.")
                except sqlite3.IntegrityError:
                    st.error("That username already exists.")
                except ValueError as e:
                    st.error(str(e))
        st.stop()

    user = st.session_state.user
    role = user["role"]
    st.sidebar.success(f"Signed in: {user['username']}")
    st.sidebar.write(ROLES[role])
    if user.get("agency_name"):
        st.sidebar.caption(user["agency_name"])
    if st.sidebar.button("Log out"):
        audit(user["username"], "logout", "")
        del st.session_state["user"]
        st.rerun()

    if role == "public":
        tabs = st.tabs(["Report an accident", "My reports"])
        with tabs[0]:
            if can(role, "submit_report"):
                report_form(user["username"])
        with tabs[1]:
            if can(role, "view_own_reports"):
                show_reports(user["username"])
        return

    if role in ("responder", "police", "fire"):
        tabs = st.tabs(["Assigned-work view", "Response updates", "Map"])
        with tabs[0]:
            st.subheader(f"{ROLES[role]} portal")
            st.info("This demo currently shows records available in the local database. Per-agency assignment filtering and real-time notifications are not yet integrated.")
            if can(role, "view_assigned_incidents"):
                show_incidents()
        with tabs[1]:
            if can(role, "update_response_status"):
                response_update(role, user["username"])
        with tabs[2]:
            with connect() as db:
                rows = db.execute("SELECT latitude,longitude FROM response_resources WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND active=1").fetchall()
            if rows:
                st.map([{"lat": r["latitude"], "lon": r["longitude"]} for r in rows])
            else:
                st.info("No resource coordinates entered yet. The map will not invent locations.")
        return

    if role == "hospital":
        tabs = st.tabs(["Hospital requests", "Capacity / status", "Map"])
        with tabs[0]:
            st.subheader("Hospital emergency coordination")
            st.warning("This is a demo request view; it is not connected to hospital systems.")
            if can(role, "view_hospital_requests"):
                with connect() as db:
                    rows = db.execute("SELECT * FROM response_resources WHERE agency_type='hospital' AND active=1 ORDER BY id DESC").fetchall()
                if rows:
                    st.dataframe([dict(r) for r in rows], use_container_width=True, hide_index=True)
                else:
                    st.info("No demonstration hospital records. An administrator/operator can add one.")
        with tabs[1]:
            if can(role, "update_hospital_status"):
                st.write("Ask an administrator/operator to maintain the demo hospital registry in this initial build.")
                st.info("Direct editing and real hospital capacity integration are not yet connected.")
        with tabs[2]:
            with connect() as db:
                rows = db.execute("SELECT latitude,longitude FROM response_resources WHERE agency_type='hospital' AND latitude IS NOT NULL AND longitude IS NOT NULL AND active=1").fetchall()
            if rows:
                st.map([{"lat": r["latitude"], "lon": r["longitude"]} for r in rows])
            else:
                st.info("No hospital coordinates available.")
        return

    # Admin/operator control-center portal
    tabs = st.tabs(["Control center", "Citizen reports", "Resources & map", "Response updates", "User management", "Audit log"])
    with tabs[0]:
        st.subheader("Emergency Control Center")
        st.caption("AI movement detections are candidates for review, not confirmed accidents.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Incident records", table_count("incidents"))
        c2.metric("Citizen reports", table_count("citizen_reports"))
        c3.metric("Registered resources", table_count("response_resources"))
        if can(role, "view_incidents"):
            show_incidents()
    with tabs[1]:
        if can(role, "view_reports"):
            show_reports()
            if role == "operator" or role == "admin":
                with connect() as db:
                    pending = db.execute("SELECT id,status FROM citizen_reports WHERE status='Submitted' ORDER BY id").fetchall()
                if pending:
                    st.subheader("Review submitted report")
                    report_id = st.selectbox("Report ID", [r["id"] for r in pending])
                    new_status = st.selectbox("Review status", ["Under review", "Verified", "Rejected", "Duplicate"])
                    if st.button("Update report review status"):
                        with connect() as db:
                            db.execute("UPDATE citizen_reports SET status=? WHERE id=?", (new_status, report_id))
                        audit(user["username"], "citizen_report_reviewed", f"report_id={report_id}, status={new_status}")
                        st.success("Report review status updated.")
    with tabs[2]:
        if can(role, "manage_resources"):
            resource_management()
    with tabs[3]:
        if can(role, "manage_resources"):
            response_update(role, user["username"])
            with connect() as db:
                updates = db.execute("SELECT * FROM response_updates ORDER BY id DESC LIMIT 200").fetchall()
            if updates:
                st.subheader("Recent response updates")
                st.dataframe([dict(r) for r in updates], use_container_width=True, hide_index=True)
    with tabs[4]:
        if can(role, "manage_users"):
            admin_user_management()
        else:
            st.info("Only administrators can manage user accounts.")
    with tabs[5]:
        if can(role, "view_audit_log"):
            with connect() as db:
                rows = db.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 500").fetchall()
            if rows:
                st.dataframe([dict(r) for r in rows], use_container_width=True, hide_index=True)
            else:
                st.info("No audit events yet.")


if __name__ == "__main__":
    main()
