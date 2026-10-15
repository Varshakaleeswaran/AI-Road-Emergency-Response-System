"""Vercel entrypoint for a temporary, non-persistent ResQLink demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path


WEB_DIR = Path(__file__).resolve().parent / "web"
sys.path.insert(0, str(WEB_DIR))

os.environ.setdefault("RESQLINK_ENV", "development")
os.environ.setdefault("RESQLINK_DEMO_MODE", "1")
os.environ.setdefault("RESQLINK_DATABASE_PATH", "/tmp/resqlink-demo.db")
os.environ.setdefault("RESQLINK_UPLOAD_DIR", "/tmp/resqlink-report-uploads")
os.environ.setdefault("RESQLINK_CCTV_UPLOAD_DIR", "/tmp/resqlink-cctv-uploads")

if not os.environ.get("RESQLINK_SECRET_KEY"):
    raise RuntimeError(
        "Set RESQLINK_SECRET_KEY in the Vercel project environment variables."
    )

from flask_app import app, initialize_database  # noqa: E402

app.config["CCTV_ANALYSIS_ENABLED"] = False
app.config["CCTV_INLINE_WORKER_ENABLED"] = False
initialize_database()
