"""Supervised, database-backed worker for recorded CCTV evidence analysis."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from cctv_upload_demo import (
    _database_connection,
    _ensure_upload_table,
    _process_analysis_job,
)


LOGGER = logging.getLogger("resqlink.cctv_worker")
ANALYSING_STATUS = "Analysing video with YOLO/ByteTrack"
QUEUED_STATUS = "Queued for CV evidence analysis"
LEASE_SECONDS = 3600


def recover_expired_jobs(db_path: Path, lease_seconds: int = LEASE_SECONDS) -> int:
    """Requeue analysis rows left running beyond the maximum worker lease."""
    cutoff = (datetime.now() - timedelta(seconds=lease_seconds)).isoformat(
        timespec="seconds"
    )
    with _database_connection(db_path) as connection:
        _ensure_upload_table(connection)
        updated = connection.execute(
            """
            UPDATE cctv_video_uploads
            SET analysis_status = ?, analysis_error = ?,
                analysis_started_at = NULL
            WHERE analysis_status = ?
              AND (analysis_started_at IS NULL OR analysis_started_at < ?)
            """,
            (
                QUEUED_STATUS,
                "Recovered after the CCTV worker lease expired.",
                ANALYSING_STATUS,
                cutoff,
            ),
        )
        return updated.rowcount


def process_pending_once(
    db_path: Path,
    upload_dir: Path,
    model_path: Path,
) -> bool:
    """Claim and process one persisted queued upload, if any are available."""
    with _database_connection(db_path) as connection:
        _ensure_upload_table(connection)
        row = connection.execute(
            """
            SELECT id, stored_name
            FROM cctv_video_uploads
            WHERE analysis_status = ?
            ORDER BY id
            LIMIT 1
            """,
            (QUEUED_STATUS,),
        ).fetchone()
    if row is None:
        return False

    stored_name = row["stored_name"]
    if not stored_name or Path(stored_name).name != stored_name:
        with _database_connection(db_path) as connection:
            connection.execute(
                """
                UPDATE cctv_video_uploads
                SET analysis_status = 'CV analysis failed',
                    analysis_error = ?
                WHERE id = ? AND analysis_status = ?
                """,
                ("Stored video filename is invalid.", row["id"], QUEUED_STATUS),
            )
        LOGGER.error("Rejected invalid stored filename for upload %s", row["id"])
        return True

    _process_analysis_job(
        db_path,
        row["id"],
        upload_dir / stored_name,
        model_path,
    )
    return True


def run_worker(
    db_path: Path,
    upload_dir: Path,
    model_path: Path,
    poll_seconds: float = 2.0,
    lease_seconds: int = LEASE_SECONDS,
    stop_event: threading.Event | None = None,
) -> None:
    """Poll durable upload rows until stopped by the service manager."""
    db_path = Path(db_path).resolve()
    upload_dir = Path(upload_dir).resolve()
    model_path = Path(model_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Portal database not found: {db_path}")
    if not upload_dir.is_dir():
        raise NotADirectoryError(f"CCTV upload directory not found: {upload_dir}")
    if poll_seconds <= 0 or lease_seconds <= 1800:
        raise ValueError("Polling must be positive and the lease must exceed 1800 seconds.")

    _ensure_upload_table_for_worker(db_path)
    stopper = stop_event or threading.Event()
    LOGGER.info("CCTV worker polling %s for persisted uploads", db_path)
    while not stopper.is_set():
        try:
            recovered = recover_expired_jobs(db_path, lease_seconds)
            if recovered:
                LOGGER.warning("Requeued %s expired CCTV analysis job(s)", recovered)
            processed = process_pending_once(db_path, upload_dir, model_path)
        except (OSError, sqlite3.Error):
            LOGGER.exception("CCTV worker could not read or update its queue")
            processed = False
        if not processed:
            stopper.wait(poll_seconds)


def _ensure_upload_table_for_worker(db_path: Path) -> None:
    with _database_connection(db_path) as connection:
        _ensure_upload_table(connection)


def _argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=os.environ.get("RESQLINK_DATABASE_PATH"),
        help="Portal SQLite path (defaults to RESQLINK_DATABASE_PATH).",
    )
    parser.add_argument(
        "--upload-dir",
        default=os.environ.get(
            "RESQLINK_CCTV_UPLOAD_DIR",
            str(Path(__file__).resolve().parent / "uploads" / "cctv_demo"),
        ),
        help="Protected CCTV storage directory.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RESQLINK_YOLO_MODEL", str(root / "yolo26n.pt")),
        help="YOLO checkpoint path.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    return parser


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("RESQLINK_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _argument_parser()
    args = parser.parse_args()
    if not args.database:
        parser.error("set RESQLINK_DATABASE_PATH or provide --database")

    stop_event = threading.Event()

    def request_stop(*signal_args):
        signum = signal_args[0] if signal_args else "unknown"
        LOGGER.info("Received signal %s; stopping CCTV worker", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_worker(
        Path(args.database),
        Path(args.upload_dir),
        Path(args.model),
        args.poll_seconds,
        args.lease_seconds,
        stop_event,
    )


if __name__ == "__main__":
    main()
