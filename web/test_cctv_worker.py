import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from cctv_upload_demo import _database_connection, _ensure_upload_table
from cctv_worker import process_pending_once, recover_expired_jobs


class TestCCTVWorker(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "portal.db"
        self.upload_dir = self.root / "uploads"
        self.upload_dir.mkdir()
        with _database_connection(self.database) as connection:
            _ensure_upload_table(connection)

    def tearDown(self):
        self.temporary.cleanup()

    def _insert_upload(self, stored_name, status, started_at=None):
        with _database_connection(self.database) as connection:
            cursor = connection.execute(
                """
                INSERT INTO cctv_video_uploads
                (camera_id, original_name, stored_name, detected_at,
                 analysis_status, analysis_started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "CAM-TEST",
                    "sample.mp4",
                    stored_name,
                    "2026-10-13T00:00",
                    status,
                    started_at,
                ),
            )
            return cursor.lastrowid

    def test_pending_upload_is_processed_and_result_persisted(self):
        upload_id = self._insert_upload(
            "stored-video.mp4", "Queued for CV evidence analysis"
        )
        summary = {
            "decision_scope": "vehicle evidence only; accident classification not performed",
            "incident_created": False,
            "dispatch_triggered": False,
        }

        with patch("cctv_upload_demo.analyze_video_evidence", return_value=summary):
            processed = process_pending_once(
                self.database,
                self.upload_dir,
                self.root / "model.pt",
            )

        self.assertTrue(processed)
        with _database_connection(self.database) as connection:
            upload = connection.execute(
                "SELECT analysis_status, analysis_summary FROM cctv_video_uploads WHERE id=?",
                (upload_id,),
            ).fetchone()
        self.assertEqual(
            upload["analysis_status"],
            "CV evidence ready — operator review required",
        )
        self.assertEqual(json.loads(upload["analysis_summary"]), summary)

    def test_expired_running_job_is_requeued_but_recent_job_is_preserved(self):
        now = datetime.now()
        expired_id = self._insert_upload(
            "expired.mp4",
            "Analysing video with YOLO/ByteTrack",
            (now - timedelta(hours=2)).isoformat(timespec="seconds"),
        )
        active_id = self._insert_upload(
            "active.mp4",
            "Analysing video with YOLO/ByteTrack",
            now.isoformat(timespec="seconds"),
        )

        recovered = recover_expired_jobs(self.database, lease_seconds=3600)

        self.assertEqual(recovered, 1)
        with _database_connection(self.database) as connection:
            expired = connection.execute(
                "SELECT analysis_status, analysis_error, analysis_started_at "
                "FROM cctv_video_uploads WHERE id=?",
                (expired_id,),
            ).fetchone()
            active = connection.execute(
                "SELECT analysis_status FROM cctv_video_uploads WHERE id=?",
                (active_id,),
            ).fetchone()
        self.assertEqual(expired["analysis_status"], "Queued for CV evidence analysis")
        self.assertIn("lease expired", expired["analysis_error"])
        self.assertIsNone(expired["analysis_started_at"])
        self.assertEqual(active["analysis_status"], "Analysing video with YOLO/ByteTrack")

    def test_invalid_persisted_video_path_is_failed_without_processing(self):
        upload_id = self._insert_upload(
            "..\\outside.mp4", "Queued for CV evidence analysis"
        )

        with patch("cctv_upload_demo.analyze_video_evidence") as analyze:
            processed = process_pending_once(
                self.database,
                self.upload_dir,
                self.root / "model.pt",
            )

        self.assertTrue(processed)
        analyze.assert_not_called()
        with _database_connection(self.database) as connection:
            status = connection.execute(
                "SELECT analysis_status, analysis_error FROM cctv_video_uploads WHERE id=?",
                (upload_id,),
            ).fetchone()
        self.assertEqual(status["analysis_status"], "CV analysis failed")
        self.assertEqual(status["analysis_error"], "Stored video filename is invalid.")


if __name__ == "__main__":
    unittest.main()
