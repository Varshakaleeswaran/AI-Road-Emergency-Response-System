import sqlite3
import bcrypt
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "resqlink_incidents.db"

ROLES = {
    "admin": "Administrator",
    "operator": "Emergency Operator",
    "responder": "Emergency Responder",
    "hospital": "Hospital Staff",
    "public": "Public User",
}


def initialize_auth_database():
    """Create authentication tables without modifying existing incidents."""
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash BLOB NOT NULL,
                role TEXT NOT NULL CHECK (
                    role IN ('admin', 'operator', 'responder', 'hospital', 'public')
                ),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def create_user(username, password, role):
    """Create a user with a securely hashed password."""
    username = username.strip()

    if not username:
        raise ValueError("Username cannot be empty.")

    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")

    if role not in ROLES:
        raise ValueError("Invalid role.")

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
            """,
            (username, password_hash, role),
        )


def authenticate(username, password):
    """Return the active user's account details, or None if login fails."""
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row

        user = db.execute(
            """
            SELECT id, username, password_hash, role
            FROM users
            WHERE username = ? AND active = 1
            """,
            (username.strip(),),
        ).fetchone()

    if user is None:
        return None

    stored_hash = user["password_hash"]
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")

    try:
        password_matches = bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash
        )
    except (ValueError, TypeError):
        return None

    if not password_matches:
        return None

    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "role_name": ROLES[user["role"]],
    }


def has_permission(role, permission):
    """Check whether a role is allowed to perform an operation."""
    permissions = {
        "admin": {
            "manage_users",
            "view_incidents",
            "review_incidents",
            "manage_resources",
            "view_audit_log",
        },
        "operator": {
            "view_incidents",
            "review_incidents",
            "manage_resources",
        },
        "responder": {
            "view_assigned_incidents",
            "update_response_status",
        },
        "hospital": {
            "view_hospital_requests",
            "update_hospital_status",
        },
        "public": {
            "submit_report",
            "view_own_reports",
        },
    }

    return permission in permissions.get(role, set())


def record_audit(username, action, details=""):
    """Record an action in the audit log."""
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            INSERT INTO audit_log (username, action, details)
            VALUES (?, ?, ?)
            """,
            (username, action, details),
        )


if __name__ == "__main__":
    initialize_auth_database()
    print("Authentication tables initialized.")
    print("Existing incident records were not intentionally modified.")
