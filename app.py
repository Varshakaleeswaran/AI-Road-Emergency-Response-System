import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# --------------------------------------------------
# ResQLink AI - Incident Review Dashboard
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "suspicious_movement.csv"
DB_PATH = BASE_DIR / "resqlink_incidents.db"

STATUSES = ["New", "Under Review", "Verified", "Closed"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]


st.set_page_config(
    page_title="ResQLink AI",
    page_icon="🚑",
    layout="wide",
)

st.title("🚑 ResQLink AI")
st.subheader("Smart Road Accident Detection & Emergency Response")
st.caption(
    "Prototype dashboard | AI movement candidates require human verification. "
    "Emergency assignments are simulated."
)


# --------------------------------------------------
# Database setup
# --------------------------------------------------

def connect_db():
    return sqlite3.connect(DB_PATH, timeout=10)


def initialize_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_key TEXT NOT NULL UNIQUE,
                vehicle_id INTEGER,
                frame INTEGER,
                vehicle_type TEXT,
                movement_per_sec REAL,
                direction_change_angle REAL,
                deceleration REAL,
                status TEXT NOT NULL DEFAULT 'New',
                severity TEXT NOT NULL DEFAULT 'Medium',
                notes TEXT NOT NULL DEFAULT '',
                ambulance_assigned INTEGER NOT NULL DEFAULT 0,
                police_assigned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def get_incidents():
    with connect_db() as conn:
        return pd.read_sql_query(
            "SELECT * FROM incidents ORDER BY id DESC",
            conn,
        )


def get_incident(candidate_key):
    with connect_db() as conn:
        return conn.execute(
            "SELECT * FROM incidents WHERE candidate_key = ?",
            (candidate_key,),
        ).fetchone()


def create_incident(row, candidate_key):
    now = datetime.now().isoformat(timespec="seconds")

    with connect_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO incidents (
                candidate_key, vehicle_id, frame, vehicle_type,
                movement_per_sec, direction_change_angle, deceleration,
                status, severity, notes,
                ambulance_assigned, police_assigned,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'New', 'Medium', '',
                    0, 0, ?, ?)
        """, (
            candidate_key,
            int(row["vehicle_id"]),
            int(row["frame"]),
            str(row["stable_vehicle_type"]),
            float(row["movement_per_sec"]),
            float(row["direction_change_angle"]),
            float(row["deceleration"]),
            now,
            now,
        ))


def update_incident(
    incident_id,
    status=None,
    severity=None,
    notes=None,
    ambulance=None,
    police=None,
):
    updates = {}
    if status is not None:
        updates["status"] = status
    if severity is not None:
        updates["severity"] = severity
    if notes is not None:
        updates["notes"] = notes
    if ambulance is not None:
        updates["ambulance_assigned"] = int(ambulance)
    if police is not None:
        updates["police_assigned"] = int(police)

    if not updates:
        return

    updates["updated_at"] = datetime.now().isoformat(timespec="seconds")

    assignments = ", ".join(f"{column} = ?" for column in updates)
    values = list(updates.values()) + [incident_id]

    with connect_db() as conn:
        conn.execute(
            f"UPDATE incidents SET {assignments} WHERE id = ?",
            values,
        )


initialize_db()


# --------------------------------------------------
# Load existing AI output (read-only)
# --------------------------------------------------

if not CSV_PATH.exists():
    st.error(f"AI results file not found: {CSV_PATH}")
    st.info(
        "Place this app.py file beside your existing suspicious_movement.csv."
    )
    st.stop()

try:
    candidates = pd.read_csv(CSV_PATH)
except Exception as exc:
    st.error(f"Could not read the AI results CSV: {exc}")
    st.stop()

required_columns = [
    "frame",
    "vehicle_id",
    "stable_vehicle_type",
    "movement_per_sec",
    "direction_change_angle",
    "deceleration",
]

missing = [column for column in required_columns if column not in candidates]

if missing:
    st.error(f"The CSV is missing required columns: {missing}")
    st.stop()

if candidates.empty:
    st.warning("The AI results file contains no candidate records.")
    st.stop()

for column in required_columns:
    candidates[column] = pd.to_numeric(
        candidates[column], errors="coerce"
    ) if column != "stable_vehicle_type" else candidates[column]

candidates = candidates.dropna(subset=[
    "frame",
    "vehicle_id",
    "movement_per_sec",
]).copy()

if candidates.empty:
    st.error("No usable candidate records were found.")
    st.stop()

candidates["frame"] = candidates["frame"].astype(int)
candidates["vehicle_id"] = candidates["vehicle_id"].astype(int)

# Stable key: source row number + frame + vehicle ID.
# This identifies a candidate from this particular CSV.
candidates = candidates.reset_index(drop=True)
candidates["candidate_key"] = candidates.apply(
    lambda row: (
        f"suspicious_movement:{int(row.name)}:"
        f"{int(row['frame'])}:{int(row['vehicle_id'])}"
    ),
    axis=1,
)


# --------------------------------------------------
# Dashboard overview
# --------------------------------------------------

incidents = get_incidents()

total_candidates = len(candidates)
total_incidents = len(incidents)
under_review = (
    int((incidents["status"] == "Under Review").sum())
    if not incidents.empty else 0
)
verified = (
    int((incidents["status"] == "Verified").sum())
    if not incidents.empty else 0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Movement Candidates", total_candidates)
col2.metric("Saved Incidents", total_incidents)
col3.metric("Under Review", under_review)
col4.metric("Verified by Operator", verified)

st.divider()

tab_dashboard, tab_review, tab_history = st.tabs([
    "📊 AI Results",
    "🔎 Incident Review & Response",
    "📋 Incident History",
])


# --------------------------------------------------
# Tab 1: AI results
# --------------------------------------------------

with tab_dashboard:
    st.header("Existing AI Movement Results")
    st.write(
        "These records were produced by your existing analysis pipeline. "
        "They are candidate anomalies, not confirmed road accidents."
    )

    filter_type = st.selectbox(
        "Filter by vehicle type",
        ["All"] + sorted(
            candidates["stable_vehicle_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    display = candidates.copy()
    if filter_type != "All":
        display = display[
            display["stable_vehicle_type"].astype(str) == filter_type
        ]

    display_columns = [
        "frame",
        "vehicle_id",
        "stable_vehicle_type",
        "movement_per_sec",
        "direction_change_angle",
        "deceleration",
    ]

    st.dataframe(
        display[display_columns].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Movement is measured in pixels per second, not calibrated road speed."
    )


# --------------------------------------------------
# Tab 2: Incident review and simulated response
# --------------------------------------------------

with tab_review:
    st.header("Review a Candidate")

    choices = candidates.to_dict("records")
    choice_labels = {
        row["candidate_key"]: (
            f"Frame {row['frame']} | Vehicle {row['vehicle_id']} | "
            f"{row.get('stable_vehicle_type', 'Unknown')} | "
            f"{row['movement_per_sec']:.1f} px/s"
        )
        for row in choices
    }

    selected_key = st.selectbox(
        "Select an AI movement candidate",
        options=list(choice_labels.keys()),
        format_func=lambda key: choice_labels[key],
    )

    selected_row = candidates[
        candidates["candidate_key"] == selected_key
    ].iloc[0]

    left, right = st.columns(2)

    with left:
        st.subheader("AI Evidence")
        st.write(f"**Frame:** {int(selected_row['frame'])}")
        st.write(f"**Vehicle ID:** {int(selected_row['vehicle_id'])}")
        st.write(
            f"**Vehicle type:** {selected_row['stable_vehicle_type']}"
        )
        st.write(
            f"**Movement:** {selected_row['movement_per_sec']:.2f} pixels/sec"
        )
        st.write(
            "**Direction change:** "
            f"{selected_row['direction_change_angle']:.2f} degrees"
        )
        st.write(
            f"**Deceleration feature:** {selected_row['deceleration']:.2f}"
        )
        st.warning(
            "These features alone do not establish that a collision occurred."
        )

        if st.button("Create / Open Incident", type="primary"):
            create_incident(selected_row, selected_key)
            st.success("Incident record is available in the local database.")
            st.rerun()

    with right:
        st.subheader("Human Review")

        incident = get_incident(selected_key)

        if incident is None:
            st.info("Create an incident to enable status and response controls.")
        else:
            (
                incident_id, candidate_key, vehicle_id, frame, vehicle_type,
                movement, angle, deceleration, status, severity, notes,
                ambulance_assigned, police_assigned, created_at, updated_at
            ) = incident

            st.write(f"**Incident ID:** {incident_id}")
            st.write(f"**Created:** {created_at}")
            st.write(f"**Current status:** {status}")

            with st.form("incident_update_form"):
                new_status = st.selectbox(
                    "Review status",
                    STATUSES,
                    index=STATUSES.index(status),
                )

                new_severity = st.selectbox(
                    "Prototype severity assessment",
                    SEVERITIES,
                    index=SEVERITIES.index(severity),
                )

                new_notes = st.text_area(
                    "Operator notes",
                    value=notes,
                    help="Record observations and the reason for your decision.",
                )

                save = st.form_submit_button("Save Review")

            if save:
                update_incident(
                    incident_id,
                    status=new_status,
                    severity=new_severity,
                    notes=new_notes,
                )
                st.success("Review saved.")
                st.rerun()

            st.divider()
            st.subheader("Simulated Emergency Coordination")
            st.caption(
                "Demonstration only. No real emergency service is contacted."
            )

            with st.form("resource_assignment_form"):
                ambulance = st.checkbox(
                    "Assign simulated ambulance",
                    value=bool(ambulance_assigned),
                )
                police = st.checkbox(
                    "Assign simulated police unit",
                    value=bool(police_assigned),
                )
                assign = st.form_submit_button("Save Simulated Assignments")

            if assign:
                update_incident(
                    incident_id,
                    ambulance=ambulance,
                    police=police,
                )
                st.success("Simulated assignments saved.")
                st.rerun()

            st.write(
                "**Ambulance:** "
                + ("Assigned (simulation)" if ambulance_assigned else "Not assigned")
            )
            st.write(
                "**Police:** "
                + ("Assigned (simulation)" if police_assigned else "Not assigned")
            )


# --------------------------------------------------
# Tab 3: Incident history
# --------------------------------------------------

with tab_history:
    st.header("Saved Incident History")

    history = get_incidents()

    if history.empty:
        st.info("No incident records have been created yet.")
    else:
        st.dataframe(
            history[[
                "id",
                "frame",
                "vehicle_id",
                "vehicle_type",
                "status",
                "severity",
                "ambulance_assigned",
                "police_assigned",
                "created_at",
                "updated_at",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Records are stored locally in resqlink_incidents.db."
        )


st.divider()
st.caption(
    "ResQLink AI prototype | Human verification required | "
    "Resource assignments are simulated | Not for real emergency dispatch"
)
