"""Run the Render portal and its single SQLite-backed CCTV worker together."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from flask_app import initialize_database, start_replan_checker


WEB_DIR = Path(__file__).resolve().parent


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    database_path = Path(os.environ["RESQLINK_DATABASE_PATH"]).expanduser()
    upload_dir = Path(os.environ["RESQLINK_CCTV_UPLOAD_DIR"]).expanduser()
    model_path = Path(os.environ["RESQLINK_YOLO_MODEL"]).expanduser()
    port = os.environ.get("PORT")
    if not port:
        raise RuntimeError("Render must provide the PORT environment variable.")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_database()
    start_replan_checker()

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    processes = []
    try:
        processes.append(
            subprocess.Popen(
                [sys.executable, str(WEB_DIR / "cctv_worker.py")],
                cwd=WEB_DIR,
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    "gunicorn",
                    "--worker-class",
                    "gthread",
                    "--workers",
                    "1",
                    "--threads",
                    "100",
                    "--bind",
                    f"0.0.0.0:{port}",
                    "--timeout",
                    "0",
                    "flask_app:app",
                ],
                cwd=WEB_DIR,
            )
        )

        while not stop_requested:
            for process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"Managed service process exited unexpectedly with code {exit_code}."
                    )
            time.sleep(1)
    finally:
        _terminate(processes)


if __name__ == "__main__":
    main()
