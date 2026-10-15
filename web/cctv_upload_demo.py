"""Protected CCTV uploads and operator-reviewed YOLO/ByteTrack evidence."""

from __future__ import annotations

import os
import json
import logging
import queue
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from flask import (
    abort,
    flash,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from cctv_video_analysis import analyze_video_evidence


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
MAX_VIDEO_BYTES = 60 * 1024 * 1024
DEMO_CAMERAS = (
    {
        "camera_id": "DEMO-CAM-001",
        "location": "ResQLink Demo Junction",
        "latitude": 13.0827,
        "longitude": 80.2707,
    },
    {
        "camera_id": "DEMO-CAM-002",
        "location": "ResQLink Demo Flyover",
        "latitude": 13.0569,
        "longitude": 80.2425,
    },
)


PAGE = r"""
{% extends 'base.html' %}
{% block title %}CCTV Demo Video Upload | ResQLink AI{% endblock %}
{% block content %}
<style>
  .cctv-demo-page { max-width: 1180px; margin: 0 auto; }
  .cctv-demo-page .form-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem 1.25rem; align-items:start; }
  .cctv-demo-page .form-group { display:flex; flex-direction:column; gap:.45rem; }
  .cctv-demo-page .form-group label { font-weight:700; }
  .cctv-demo-page input, .cctv-demo-page select { min-height:42px; width:100%; box-sizing:border-box; }
  .cctv-demo-page small { color:var(--text-muted, #aeb7c4); line-height:1.4; }
  .cctv-demo-page .full-width { grid-column:1 / -1; }
  .cctv-demo-page .upload-help { margin:.25rem 0 1rem; padding:1rem; border-left:4px solid #f58220; background:rgba(245,130,32,.08); }
  .cctv-demo-page .required { color:#ef5350; }
  @media (max-width:720px) { .cctv-demo-page .form-grid { grid-template-columns:1fr; } }
</style>
<div class="cctv-demo-page">
<div class="page-header">
  <div>
    <h1><i class="fas fa-file-video"></i> CCTV Demo Video</h1>
    <p>Upload an authorised recording for vehicle-detection evidence and operator review.</p>
  </div>
  <a href="/cameras" class="btn btn-outline"><i class="fas fa-video"></i> Camera registry</a>
</div>

<div class="alert alert-info">
  <i class="fas fa-shield-alt"></i>
  <div><strong>CV evidence only:</strong> YOLO/ByteTrack reports vehicle detections and track motion; it does not classify accidents. Uploads do not create incidents, dispatch resources, or notify responders. A control-room operator reviews the video and may attach completed evidence to an incident.</div>
</div>

<div class="card" style="max-width: 980px;">
  <div class="card-header"><h3><i class="fas fa-upload"></i> Upload CCTV evidence</h3></div>
  <div class="card-body">
    <p class="upload-help"><strong>Review workflow:</strong> choose an authorised video and its camera/location. The CV worker extracts vehicle detections and tracked movement for an operator to review; it does not decide whether a crash occurred.</p>
    <form method="post" enctype="multipart/form-data" class="form-grid">
      <input type="hidden" name="_csrf_token" value="{{ csrf_token() }}">
      <div class="form-group full-width">
        <label for="video">CCTV recording <span class="required">*</span></label>
        <input id="video" name="video" type="file" accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,video/x-matroska" required>
        <small>MP4 is recommended. Maximum file size: 60 MB. Upload only footage you are authorised to use.</small>
      </div>
      <div class="form-group">
        <label for="camera_id">Registered camera <span class="required">*</span></label>
        <select id="camera_id" name="camera_id" required>
          {% if demo_cameras_enabled %}
          <option value="DEMO-CAM-001" data-location="ResQLink Demo Junction" data-latitude="13.0827" data-longitude="80.2707" selected>DEMO-CAM-001 — ResQLink Demo Junction (prototype)</option>
          <option value="DEMO-CAM-002" data-location="ResQLink Demo Flyover" data-latitude="13.0569" data-longitude="80.2425">DEMO-CAM-002 — ResQLink Demo Flyover (prototype)</option>
          {% endif %}
          {% for camera in cameras %}
          {% if demo_cameras_enabled or camera.camera_id not in ['DEMO-CAM-001', 'DEMO-CAM-002'] %}
          <option value="{{ camera.camera_id }}" data-location="{{ camera.location or '' }}" data-latitude="{{ camera.latitude if camera.latitude is not none else '' }}" data-longitude="{{ camera.longitude if camera.longitude is not none else '' }}">{{ camera.camera_id }}{% if camera.location %} — {{ camera.location }}{% endif %}</option>
          {% endif %}
          {% endfor %}
        </select>
        <small>Prototype camera options are available only in explicit demo mode. Production accepts registered cameras only.</small>
      </div>
      <div class="form-group">
        <label for="location">Detection location</label>
        <input id="location" name="location" maxlength="200" placeholder="Filled from the selected camera when available">
      </div>
      <div class="form-group">
        <label for="latitude">Latitude</label>
        <input id="latitude" name="latitude" type="number" step="any" min="-90" max="90" placeholder="e.g. 13.0827">
      </div>
      <div class="form-group">
        <label for="longitude">Longitude</label>
        <input id="longitude" name="longitude" type="number" step="any" min="-180" max="180" placeholder="e.g. 80.2707">
      </div>
      <div class="form-group">
        <label for="detected_at">Detection time</label>
        <input id="detected_at" name="detected_at" type="datetime-local" value="{{ now }}">
      </div>
      <div class="full-width" style="display:flex;justify-content:flex-end;gap:.75rem;">
        <a href="/cameras" class="btn btn-outline">Cancel</a>
        <button class="btn btn-primary" type="submit"><i class="fas fa-microchip"></i> Upload &amp; analyse vehicle evidence</button>
      </div>
    </form>
  </div>
</div>

<div class="card" style="margin-top:1.5rem;">
  <div class="card-header"><h3><i class="fas fa-history"></i> Recent CCTV demo uploads</h3></div>
  <div class="table-responsive">
    <table class="data-table">
      <thead><tr><th>Video</th><th>Camera</th><th>Observed</th><th>CV evidence</th><th>Incident link</th><th>Review status</th></tr></thead>
      <tbody>
      {% for upload in uploads %}
        <tr>
          <td><a href="{{ url_for('cctv_demo_video', upload_id=upload.id) }}" target="_blank" rel="noopener"><i class="fas fa-play-circle"></i> {{ upload.original_name }}</a></td>
          <td>{{ upload.camera_id }}<br><small>{{ upload.location or '—' }}</small></td>
          <td>{{ upload.detected_at }}</td>
          <td>
            {% if upload.analysis_data %}
              {{ upload.analysis_data.detections.vehicle_detection_observations if upload.analysis_data.detections.vehicle_detection_observations is defined else upload.analysis_data.detections.vehicle_detections }}
              detection observations across {{ upload.analysis_data.detections.detection_frames }} frames
              (a vehicle may appear in multiple frames);
              {{ upload.analysis_data.tracking.unique_tracks }} tracked vehicles
              <details><summary>Details</summary>
                <small>
                  {{ upload.analysis_data.detections.vehicle_types }}<br>
                  Mean vehicle detector score:
                  {{ upload.analysis_data.detections.mean_vehicle_detector_confidence or '—' }}<br>
                  Mean track path: {{ upload.analysis_data.tracking.mean_track_path_length_pixels or '—' }} px<br>
                  Max track speed: {{ upload.analysis_data.tracking.maximum_track_speed_pixels_per_second or '—' }} px/s<br>
                  Accident classification: not performed
                </small>
              </details>
            {% elif upload.analysis_error %}
              <small class="text-danger">{{ upload.analysis_error }}</small>
            {% elif upload.analysis_status in ['Queued for CV evidence analysis', 'Analysing video with YOLO/ByteTrack'] %}
              <span class="text-muted">Waiting for CV worker</span>
            {% else %}
              <span class="text-muted">No CV summary; legacy demo signal is not model evidence.</span>
            {% endif %}
          </td>
          <td>
            {% if upload.response_case_id %}
              <a href="{{ url_for('response_case', case_id=upload.response_case_id) }}">Legacy case #{{ upload.response_case_id }}</a>
            {% elif upload.linked_incident_id %}Incident #{{ upload.linked_incident_id }}
            {% elif upload.analysis_data %}
              <form method="post" action="{{ url_for('cctv_demo_link_upload', upload_id=upload.id) }}">
                <input type="hidden" name="_csrf_token" value="{{ csrf_token() }}">
                <select name="incident_id" required>
                  <option value="">Select incident</option>
                  {% for incident in incidents %}
                  <option value="{{ incident.id }}">#{{ incident.id }} — {{ incident.label }}</option>
                  {% endfor %}
                </select>
                <button class="btn btn-sm btn-outline" type="submit">Attach evidence</button>
              </form>
            {% else %}<span class="text-muted">Not linked</span>{% endif %}
          </td>
          <td><span class="badge badge-warning">{{ upload.analysis_status }}</span>{% if upload.duplicate_of %}<br><small>Similar-time upload #{{ upload.duplicate_of }}; not auto-merged</small>{% endif %}</td>
        </tr>
      {% else %}
        <tr><td colspan="6" class="text-center text-muted">No CCTV video has been uploaded yet.</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>
 </div>
{% endblock %}
{% block extra_js %}
<script>
const cameraSelect = document.getElementById('camera_id');
const locationInput = document.getElementById('location');
const latitudeInput = document.getElementById('latitude');
const longitudeInput = document.getElementById('longitude');
function applyCameraDetails() {
  if (!cameraSelect) return;
  const chosen = cameraSelect.options[cameraSelect.selectedIndex];
  if (!chosen || !chosen.value) return;
  if (locationInput && chosen.dataset.location) locationInput.value = chosen.dataset.location;
  if (latitudeInput && chosen.dataset.latitude) latitudeInput.value = chosen.dataset.latitude;
  if (longitudeInput && chosen.dataset.longitude) longitudeInput.value = chosen.dataset.longitude;
}
if (cameraSelect) {
  cameraSelect.addEventListener('change', applyCameraDetails);
  applyCameraDetails();
}
</script>
{% endblock %}
"""


def _upload_directory(configured_path=None) -> Path:
    configured_path = configured_path or os.environ.get("RESQLINK_CCTV_UPLOAD_DIR")
    folder = (
        Path(configured_path)
        if configured_path
        else Path(__file__).resolve().parent / "uploads" / "cctv_demo"
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _open_database(db_path):
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


@contextmanager
def _database_connection(db_path):
    connection = _open_database(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _ensure_upload_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS cctv_video_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL UNIQUE,
            location TEXT,
            latitude REAL,
            longitude REAL,
            detected_at TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            analysis_status TEXT NOT NULL DEFAULT 'Queued for CV evidence analysis',
            linked_incident_id INTEGER,
            duplicate_of INTEGER,
            created_by INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    upload_columns = {row[1] for row in connection.execute("PRAGMA table_info(cctv_video_uploads)")}
    migrations = {
        "response_case_id": "INTEGER",
        "analysis_summary": "TEXT",
        "analysis_error": "TEXT",
        "analysis_started_at": "TEXT",
        "analysis_notified_at": "TEXT",
    }
    for name, declaration in migrations.items():
        if name not in upload_columns:
            connection.execute(
                f"ALTER TABLE cctv_video_uploads ADD COLUMN {name} {declaration}"
            )


_ANALYSIS_QUEUE = queue.Queue()
_ANALYSIS_WORKER_LOCK = threading.Lock()
_ANALYSIS_WORKER_STARTED = False
_LOGGER = logging.getLogger(__name__)


def _process_analysis_job(db_path, upload_id, video_path, model_path):
    with _database_connection(db_path) as connection:
        claimed = connection.execute(
            """
            UPDATE cctv_video_uploads
            SET analysis_status = 'Analysing video with YOLO/ByteTrack',
                analysis_error = NULL, analysis_started_at = ?
            WHERE id = ? AND analysis_status = 'Queued for CV evidence analysis'
            """,
            (datetime.now().isoformat(timespec="seconds"), upload_id),
        ).rowcount
    if not claimed:
        return

    try:
        summary = analyze_video_evidence(video_path, model_path)
    except Exception as error:
        _LOGGER.exception("CV evidence analysis failed for upload %s", upload_id)
        with _database_connection(db_path) as connection:
            connection.execute(
                """
                UPDATE cctv_video_uploads
                SET analysis_status = 'CV analysis failed',
                    analysis_error = ?
                WHERE id = ?
                """,
                (str(error)[:1000], upload_id),
            )
            _notify_control_room(
                connection,
                upload_id,
                "CV_ANALYSIS_FAILED",
                f"CCTV vehicle-evidence analysis failed for upload #{upload_id}. "
                "Review the upload page for the reported error.",
            )
        return

    with _database_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE cctv_video_uploads
            SET analysis_status = 'CV evidence ready — operator review required',
                analysis_summary = ?, analysis_error = NULL
            WHERE id = ?
            """,
            (json.dumps(summary, sort_keys=True), upload_id),
        )
        _notify_control_room(
            connection,
            upload_id,
            "CV_EVIDENCE_READY",
            f"Vehicle-detection evidence is ready for operator review for upload #{upload_id}. "
            "No accident was classified and no dispatch was triggered.",
        )


def _notify_control_room(connection, upload_id, notification_type, body):
    """Persist one in-portal notification per active control-room user and upload."""
    required_tables = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('users', 'notifications')"
        )
    }
    if required_tables != {"users", "notifications"}:
        _LOGGER.warning(
            "Cannot notify control room for CCTV upload %s: portal notification tables are unavailable",
            upload_id,
        )
        return

    upload = connection.execute(
        "SELECT analysis_notified_at FROM cctv_video_uploads WHERE id = ?",
        (upload_id,),
    ).fetchone()
    if not upload or upload["analysis_notified_at"]:
        return

    recipients = connection.execute(
        "SELECT id, username, role FROM users "
        "WHERE active = 1 AND role IN ('admin', 'operator')"
    ).fetchall()
    if not recipients:
        _LOGGER.warning(
            "CCTV upload %s completed without an active admin or operator to notify",
            upload_id,
        )
        return
    now = datetime.now().isoformat(timespec="seconds")
    for recipient in recipients:
        connection.execute(
            """
            INSERT INTO notifications
                (recipient_role, recipient_username, title, body, notif_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                recipient["role"],
                recipient["username"],
                f"CCTV upload #{upload_id}: "
                f"{'evidence ready' if notification_type == 'CV_EVIDENCE_READY' else 'analysis failed'}",
                body,
                notification_type,
            ),
        )
    connection.execute(
        "UPDATE cctv_video_uploads SET analysis_notified_at = ? WHERE id = ?",
        (now, upload_id),
    )


def _analysis_worker():
    while True:
        job = _ANALYSIS_QUEUE.get()
        db_path, upload_id, video_path, model_path = job
        try:
            _process_analysis_job(db_path, upload_id, video_path, model_path)
        except Exception as error:
            _LOGGER.exception("CV worker could not process upload %s", upload_id)
            try:
                with _database_connection(db_path) as connection:
                    connection.execute(
                        """
                        UPDATE cctv_video_uploads
                        SET analysis_status = 'CV analysis failed',
                            analysis_error = ?
                        WHERE id = ?
                        """,
                        (str(error)[:1000], upload_id),
                    )
            except sqlite3.Error:
                _LOGGER.exception("Could not persist CV worker failure for upload %s", upload_id)
        finally:
            _ANALYSIS_QUEUE.task_done()


def _queue_analysis(db_path, upload_id, video_path, model_path):
    global _ANALYSIS_WORKER_STARTED
    with _ANALYSIS_WORKER_LOCK:
        if not _ANALYSIS_WORKER_STARTED:
            threading.Thread(target=_analysis_worker, daemon=True, name="resqlink-cv-worker").start()
            _ANALYSIS_WORKER_STARTED = True
    _ANALYSIS_QUEUE.put((Path(db_path), upload_id, Path(video_path), Path(model_path)))


def _camera_rows(connection, *, allow_demo=False):
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL UNIQUE,
                location TEXT,
                stream_url TEXT,
                latitude REAL,
                longitude REAL,
                status TEXT DEFAULT 'Online',
                last_seen TEXT
            )
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(cameras)")}
        if "camera_id" not in columns:
            return DEMO_CAMERAS
        location = (
            "location" if "location" in columns
            else "location_text" if "location_text" in columns
            else "NULL"
        )
        latitude = "latitude" if "latitude" in columns else "NULL"
        longitude = "longitude" if "longitude" in columns else "NULL"
        rows = connection.execute(
            f"SELECT camera_id, {location} AS location, {latitude} AS latitude, {longitude} AS longitude FROM cameras ORDER BY camera_id"
        ).fetchall()
        rows = [row for row in rows if str(row["camera_id"] or "").strip()]
        if not allow_demo:
            rows = [
                row for row in rows
                if row["camera_id"] not in {camera["camera_id"] for camera in DEMO_CAMERAS}
            ]
        if rows:
            return rows

        if not allow_demo:
            return []

        # A new prototype needs usable data on first launch.  These are local
        # demo records, not live roadside-camera connections, and can be
        # replaced from Camera Management at any time.
        insertable = {
            "camera_id": None,
            "location": None,
            "latitude": None,
            "longitude": None,
            "stream_url": "demo://authorised-upload",
            "status": "Online",
        }
        for camera in DEMO_CAMERAS:
            values = dict(insertable)
            values.update(camera)
            fields = [field for field in values if field in columns]
            connection.execute(
                f"INSERT OR IGNORE INTO cameras ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
                [values[field] for field in fields],
            )
        connection.commit()
        return connection.execute(
            f"SELECT camera_id, {location} AS location, {latitude} AS latitude, {longitude} AS longitude FROM cameras ORDER BY camera_id"
        ).fetchall()
    except sqlite3.Error:
        # Do not turn registry errors into fake operational camera choices.
        return DEMO_CAMERAS if allow_demo else []


def _incident_rows(connection):
    """Read incidents defensively so this feature also works with older DBs."""
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(incidents)")}
        if "id" not in columns:
            return []
        label_column = next((name for name in ("title", "incident_id", "description", "location") if name in columns), None)
        status_column = "status" if "status" in columns else None
        selected = ["id"] + ([label_column] if label_column else []) + ([status_column] if status_column else [])
        rows = connection.execute(
            f"SELECT {', '.join(selected)} FROM incidents ORDER BY id DESC LIMIT 30"
        ).fetchall()
        return [
            {"id": row["id"], "label": row[label_column] if label_column and row[label_column] else (row[status_column] if status_column else "Incident")}
            for row in rows
        ]
    except sqlite3.Error:
        return []


def _number(value, low, high, label):
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.")
    if not low <= parsed <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return parsed


def register_cctv_upload_demo(app, current_user, db_path, add_evidence, audit):
    """Attach demo routes without coupling them to the main portal schema."""

    def _database_path():
        return Path(db_path() if callable(db_path) else db_path)

    def _dispatch_analysis(db_file, upload_id, video_path, model_path):
        if (
            app.config.get("CCTV_ANALYSIS_ENABLED", True)
            and app.config.get("CCTV_INLINE_WORKER_ENABLED", True)
        ):
            _queue_analysis(db_file, upload_id, video_path, model_path)

    def _storage_path():
        return _upload_directory(app.config.get("CCTV_UPLOAD_DIR"))

    def _authorised_user():
        user = current_user()
        if not user:
            return None
        return user if user["role"] in {"admin", "operator"} else False

    def _load_page(connection):
        _ensure_upload_table(connection)
        demo_cameras_enabled = bool(
            app.config.get("CCTV_DEMO_CAMERAS_ENABLED", False)
        )
        rows = connection.execute(
            "SELECT * FROM cctv_video_uploads ORDER BY id DESC LIMIT 25"
        ).fetchall()
        uploads = []
        for row in rows:
            upload = dict(row)
            try:
                upload["analysis_data"] = (
                    json.loads(upload["analysis_summary"])
                    if upload.get("analysis_summary") else None
                )
            except (TypeError, json.JSONDecodeError):
                upload["analysis_data"] = None
            uploads.append(upload)
        queued = connection.execute(
            "SELECT id, stored_name FROM cctv_video_uploads "
            "WHERE analysis_status = 'Queued for CV evidence analysis'"
        ).fetchall()
        root = Path(__file__).resolve().parent.parent
        model_path = Path(os.environ.get("RESQLINK_YOLO_MODEL", root / "yolo26n.pt"))
        if not model_path.is_absolute():
            model_path = root / model_path
        for row in queued:
            _dispatch_analysis(
                _database_path(),
                row["id"],
                _storage_path() / row["stored_name"],
                model_path,
            )
        return render_template_string(
            PAGE,
            cameras=_camera_rows(connection, allow_demo=demo_cameras_enabled),
            incidents=_incident_rows(connection),
            demo_cameras_enabled=demo_cameras_enabled,
            uploads=uploads,
            now=datetime.now().strftime("%Y-%m-%dT%H:%M"),
        )

    @app.route("/cctv-demo/upload", methods=["GET", "POST"])
    def cctv_demo_upload():
        if not app.config.get("CCTV_ANALYSIS_ENABLED", True):
            abort(
                503,
                description="CCTV video analysis is unavailable on this serverless demo deployment.",
            )
        user = _authorised_user()
        if user is None:
            return redirect(url_for("login", next=request.path))
        if user is False:
            flash("Only control-room administrators and operators can upload CCTV demo footage.", "error")
            return redirect(url_for("dashboard"))

        database_path = _database_path()
        with _database_connection(database_path) as connection:
            _ensure_upload_table(connection)
            if request.method == "POST":
                video = request.files.get("video")
                if not video or not video.filename:
                    flash("Choose a CCTV video before uploading.", "error")
                    return _load_page(connection)

                original_name = secure_filename(video.filename)
                extension = Path(original_name).suffix.lower()
                if not original_name or extension not in ALLOWED_VIDEO_EXTENSIONS:
                    flash("Use an MP4, MOV, WebM, AVI, or MKV CCTV video.", "error")
                    return _load_page(connection)
                if request.content_length and request.content_length > MAX_VIDEO_BYTES + 1024 * 1024:
                    flash("The video is larger than the 60 MB limit.", "error")
                    return _load_page(connection)

                camera_id = (request.form.get("camera_id") or "").strip()
                demo_cameras_enabled = bool(
                    app.config.get("CCTV_DEMO_CAMERAS_ENABLED", False)
                )
                camera_ids = {
                    row["camera_id"]
                    for row in _camera_rows(connection, allow_demo=demo_cameras_enabled)
                }
                if demo_cameras_enabled:
                    camera_ids.update(camera["camera_id"] for camera in DEMO_CAMERAS)
                if camera_id not in camera_ids:
                    flash("Select a registered CCTV camera.", "error")
                    return _load_page(connection)
                try:
                    latitude = _number(request.form.get("latitude"), -90, 90, "Latitude")
                    longitude = _number(request.form.get("longitude"), -180, 180, "Longitude")
                except ValueError as error:
                    flash(str(error), "error")
                    return _load_page(connection)

                stored_name = f"{uuid.uuid4().hex}{extension}"
                destination = _storage_path() / stored_name
                video.save(destination)
                if not destination.exists() or destination.stat().st_size == 0:
                    destination.unlink(missing_ok=True)
                    flash("The video could not be stored. Please try again.", "error")
                    return _load_page(connection)
                if destination.stat().st_size > MAX_VIDEO_BYTES:
                    destination.unlink(missing_ok=True)
                    flash("The video is larger than the 60 MB limit.", "error")
                    return _load_page(connection)

                duplicate = connection.execute(
                    """
                    SELECT id FROM cctv_video_uploads
                    WHERE camera_id = ? AND created_at >= datetime('now', '-20 minutes')
                    ORDER BY id DESC LIMIT 1
                    """,
                    (camera_id,),
                ).fetchone()
                detected_at = (request.form.get("detected_at") or "").strip() or datetime.now().strftime("%Y-%m-%dT%H:%M")
                location = (request.form.get("location") or "").strip()[:200]
                cursor = connection.execute(
                    """
                    INSERT INTO cctv_video_uploads
                    (camera_id, original_name, stored_name, location, latitude, longitude,
                     detected_at, confidence, analysis_status, linked_incident_id,
                     duplicate_of, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        camera_id, original_name, stored_name, location, latitude, longitude,
                        detected_at, 0.0, "Queued for CV evidence analysis", None,
                        duplicate["id"] if duplicate else None, user["id"],
                    ),
                )
                connection.commit()
                root = Path(__file__).resolve().parent.parent
                model_path = Path(os.environ.get("RESQLINK_YOLO_MODEL", root / "yolo26n.pt"))
                if not model_path.is_absolute():
                    model_path = root / model_path
                _dispatch_analysis(
                    database_path,
                    cursor.lastrowid,
                    destination,
                    model_path,
                )
                flash(
                    "CCTV video queued for vehicle-evidence analysis. No incident or dispatch was created.",
                    "success",
                )
                return redirect(url_for("cctv_demo_upload"))
            return _load_page(connection)

    @app.route("/cctv-demo/upload/<int:upload_id>/link", methods=["POST"])
    def cctv_demo_link_upload(upload_id):
        if not app.config.get("CCTV_ANALYSIS_ENABLED", True):
            abort(
                503,
                description="CCTV video analysis is unavailable on this serverless demo deployment.",
            )
        user = _authorised_user()
        if user is None:
            return redirect(url_for("login", next=request.path))
        if user is False:
            abort(403)

        try:
            incident_id = int(request.form.get("incident_id", ""))
        except (TypeError, ValueError):
            flash("Choose a valid incident before attaching evidence.", "error")
            return redirect(url_for("cctv_demo_upload"))

        with _database_connection(_database_path()) as connection:
            _ensure_upload_table(connection)
            upload = connection.execute(
                "SELECT * FROM cctv_video_uploads WHERE id = ?", (upload_id,)
            ).fetchone()
            incident = connection.execute(
                "SELECT id FROM incidents WHERE id = ?", (incident_id,)
            ).fetchone()
            if not upload or not incident:
                abort(404)
            if not upload["analysis_summary"]:
                flash(
                    "Wait for vehicle-evidence analysis to finish before linking the video.",
                    "error",
                )
                return redirect(url_for("cctv_demo_upload"))
            if upload["linked_incident_id"]:
                if upload["linked_incident_id"] == incident_id:
                    flash("This video is already attached to that incident.", "info")
                else:
                    flash("This video is already linked to another incident; it was not moved.", "error")
                return redirect(url_for("cctv_demo_upload"))

            summary = json.loads(upload["analysis_summary"])
            raw_data = json.dumps({
                "cctv_upload_id": upload_id,
                "original_name": upload["original_name"],
                "video_url": url_for("cctv_demo_video", upload_id=upload_id),
                "analysis": summary,
            }, sort_keys=True)
            add_evidence(
                incident_id,
                "CAMERA",
                0.0,
                upload["location"] or "",
                raw_data,
                db=connection,
            )
            connection.execute(
                "UPDATE cctv_video_uploads SET linked_incident_id = ? WHERE id = ?",
                (incident_id, upload_id),
            )
            connection.commit()

        audit(
            user["username"],
            "ATTACH_CCTV_EVIDENCE",
            f"upload={upload_id} incident={incident_id}",
        )
        flash(f"CV evidence attached to incident #{incident_id}; no dispatch was triggered.", "success")
        return redirect(url_for("cctv_demo_upload"))

    @app.route("/cctv-demo/video/<int:upload_id>")
    def cctv_demo_video(upload_id):
        if not app.config.get("CCTV_ANALYSIS_ENABLED", True):
            abort(
                503,
                description="CCTV video analysis is unavailable on this serverless demo deployment.",
            )
        user = _authorised_user()
        if user is None:
            return redirect(url_for("login", next=request.path))
        if user is False:
            abort(403)
        with _database_connection(_database_path()) as connection:
            _ensure_upload_table(connection)
            upload = connection.execute(
                "SELECT stored_name FROM cctv_video_uploads WHERE id = ?", (upload_id,)
            ).fetchone()
        if not upload:
            abort(404)
        return send_from_directory(_storage_path(), upload["stored_name"], conditional=True)
