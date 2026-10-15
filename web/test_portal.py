import os
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flask_app
import emergency_orchestration
import cctv_upload_demo

class TestResQLinkPortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.previous_demo_mode = os.environ.get("RESQLINK_DEMO_MODE")
        os.environ["RESQLINK_DEMO_MODE"] = "1"
        cls.previous_cctv_upload_dir = flask_app.app.config.get("CCTV_UPLOAD_DIR")
        cls.previous_demo_cameras_enabled = flask_app.app.config.get(
            "CCTV_DEMO_CAMERAS_ENABLED"
        )
        flask_app.app.config["CCTV_DEMO_CAMERAS_ENABLED"] = True
        flask_app.DB_PATH = Path(cls.temp_dir.name) / "portal-test.db"
        flask_app.UPLOAD_DIR = Path(cls.temp_dir.name) / "uploads"
        flask_app.UPLOAD_DIR.mkdir()
        flask_app.initialize_database()
        flask_app.app.config['TESTING'] = True
        cls.client = flask_app.app.test_client()
        flask_app.mutate("""INSERT INTO incidents
            (candidate_key,vehicle_id,frame,vehicle_type,status,severity,lifecycle_status)
            VALUES(?,?,?,?,?,?,?)""",
            ("test-incident-1", 1, 1, "car", "New", "Medium", "REPORTED"))

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        if cls.previous_demo_mode is None:
            os.environ.pop("RESQLINK_DEMO_MODE", None)
        else:
            os.environ["RESQLINK_DEMO_MODE"] = cls.previous_demo_mode
        if cls.previous_cctv_upload_dir is None:
            flask_app.app.config.pop("CCTV_UPLOAD_DIR", None)
        else:
            flask_app.app.config["CCTV_UPLOAD_DIR"] = cls.previous_cctv_upload_dir
        if cls.previous_demo_cameras_enabled is None:
            flask_app.app.config.pop("CCTV_DEMO_CAMERAS_ENABLED", None)
        else:
            flask_app.app.config["CCTV_DEMO_CAMERAS_ENABLED"] = (
                cls.previous_demo_cameras_enabled
            )

    def csrf_token(self, path='/login'):
        self.client.get(path)
        with self.client.session_transaction() as sess:
            return sess['_csrf_token']

    def set_user(self, username, role, agency_name='HQ'):
        user = flask_app.query("SELECT id FROM users WHERE username=?", (username,), one=True)
        with self.client.session_transaction() as sess:
            sess['user'] = {'id': user['id'], 'username': username, 'role': role,
                            'agency_name': agency_name, 'role_name': flask_app.ROLES[role]}
            sess['_csrf_token'] = 'test-csrf-token'

    def test_01_landing_page(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'ResQLink AI', resp.data)
        landing_copy = b" ".join(resp.data.split())
        self.assertIn(b"vehicle-detection and tracking evidence only", landing_copy)
        self.assertIn(b"they do not verify accidents or dispatch resources", landing_copy)
        self.assertNotIn(b"AI-Powered Accident Detection", landing_copy)

    def test_health_probes_distinguish_process_and_database_readiness(self):
        live = self.client.get("/health/live")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json, {"status": "ok"})

        ready = self.client.get("/health/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json, {"status": "ready"})

        previous_db_path = flask_app.DB_PATH
        try:
            flask_app.DB_PATH = Path(self.temp_dir.name) / "uninitialized.db"
            not_ready = self.client.get("/health/ready")
            self.assertEqual(not_ready.status_code, 503)
            self.assertEqual(not_ready.json, {"status": "not_ready"})
        finally:
            flask_app.DB_PATH = previous_db_path

    def test_02_login_and_dashboard(self):
        # Authenticate as admin
        resp = self.client.post('/login', data={
            'username': 'admin',
            'password': 'Admin@ResQLink1',
            '_csrf_token': self.csrf_token(),
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'System Overview', resp.data)

    def test_03_incidents_page(self):
        self.set_user('admin', 'admin')
        resp = self.client.get('/incidents')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Incident Records', resp.data)

    def test_04_incident_detail_and_map(self):
        self.set_user('admin', 'admin')
        
        inc = flask_app.query("SELECT id FROM incidents LIMIT 1", one=True)
        if inc:
            iid = inc['id']
            # Detail page
            resp = self.client.get(f'/incidents/{iid}')
            self.assertEqual(resp.status_code, 200)
            self.assertIn(b'AI Evidence', resp.data)
            
            # Incident map page
            resp_map = self.client.get(f'/incidents/{iid}/map')
            self.assertEqual(resp_map.status_code, 200)

            # API recommendation endpoint
            resp_rec = self.client.get(f'/incidents/{iid}/recommend')
            self.assertEqual(resp_rec.status_code, 200)

    def test_05_live_map(self):
        self.set_user('admin', 'admin')
        resp = self.client.get('/map')
        self.assertEqual(resp.status_code, 200)

    def test_06_blockchain_ledger(self):
        self.set_user('admin', 'admin')
        resp = self.client.get('/blockchain')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Blockchain', resp.data)

    def test_07_registries(self):
        self.set_user('admin', 'admin')
        self.assertEqual(self.client.get('/ambulances').status_code, 200)
        self.assertEqual(self.client.get('/hospitals').status_code, 200)
        self.assertEqual(self.client.get('/departments').status_code, 200)
        self.assertEqual(self.client.get('/cameras').status_code, 200)
        self.assertEqual(self.client.get('/resources').status_code, 200)
        self.assertEqual(self.client.get('/users').status_code, 200)

    def test_08_csrf_is_required_for_state_changes(self):
        response = self.client.post('/login', data={
            'username': 'admin', 'password': 'Admin@ResQLink1'
        })
        self.assertEqual(response.status_code, 400)

    def test_09_public_registration_cannot_escalate_role(self):
        with self.client.session_transaction() as sess:
            sess.clear()
        token = self.csrf_token('/register')
        response = self.client.post('/register', data={
            '_csrf_token': token, 'username': 'new_citizen',
            'password': 'A-safe-password1', 'confirm_password': 'A-safe-password1',
            'role': 'admin', 'agency_name': 'Untrusted Agency'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        account = flask_app.query("SELECT role FROM users WHERE username='new_citizen'", one=True)
        self.assertEqual(account['role'], 'public')

    def test_10_public_users_cannot_read_operational_data(self):
        self.set_user('public1', 'public', '')
        self.assertEqual(self.client.get('/incidents').status_code, 403)
        self.assertEqual(self.client.get('/map').status_code, 403)
        self.assertEqual(self.client.get('/resources').status_code, 403)

    def test_11_lifecycle_rejects_state_skips(self):
        self.set_user('admin', 'admin')
        incident = flask_app.query("SELECT id FROM incidents WHERE candidate_key='test-incident-1'", one=True)
        response = self.client.post(
            f"/incidents/{incident['id']}/lifecycle",
            data={'_csrf_token': 'test-csrf-token', 'to_state': 'CLOSED'},
        )
        self.assertEqual(response.status_code, 302)
        current = flask_app.query("SELECT lifecycle_status FROM incidents WHERE id=?", (incident['id'],), one=True)
        self.assertEqual(current['lifecycle_status'], 'REPORTED')

    def test_12_hospital_staff_can_only_update_linked_hospital(self):
        self.set_user('hospital1', 'hospital', 'Apollo Hospital')
        own = flask_app.query("SELECT id,total_beds,icu_beds FROM hospital_registry WHERE hospital_code='HOSP-001'", one=True)
        other = flask_app.query("SELECT id FROM hospital_registry WHERE hospital_code='HOSP-002'", one=True)
        denied = self.client.post(f"/hospitals/{other['id']}/capacity", data={
            '_csrf_token': 'test-csrf-token', 'available_beds': 1,
            'icu_available': 1, 'operating_status': 'Open'
        })
        self.assertEqual(denied.status_code, 403)
        allowed = self.client.post(f"/hospitals/{own['id']}/capacity", data={
            '_csrf_token': 'test-csrf-token', 'available_beds': own['total_beds'],
            'icu_available': own['icu_beds'], 'operating_status': 'Open'
        })
        self.assertEqual(allowed.status_code, 302)

    def test_13_citizen_reports_with_matching_gps_merge_into_one_incident(self):
        self.set_user('public1', 'public', '')
        report = {
            '_csrf_token': 'test-csrf-token', 'description': 'Two vehicles collided near the signal.',
            'location_text': 'Test Junction', 'latitude': '13.0500', 'longitude': '80.2500',
            'incident_time': '', 'contact': ''
        }
        self.assertEqual(self.client.post('/reports', data=report).status_code, 302)
        report['description'] = 'Second witness confirms the collision.'
        self.assertEqual(self.client.post('/reports', data=report).status_code, 302)
        rows = flask_app.query("SELECT linked_incident_id,status FROM citizen_reports WHERE reporter_username='public1' ORDER BY id DESC LIMIT 2")
        self.assertEqual(rows[0]['linked_incident_id'], rows[1]['linked_incident_id'])
        self.assertEqual(rows[0]['status'], 'Merged')
        self.assertEqual(flask_app.query(
            "SELECT COUNT(*) AS count FROM evidence WHERE incident_id=? AND source='CITIZEN'",
            (rows[0]['linked_incident_id'],), one=True,
        )["count"], 2)

    def test_14_linked_driver_can_submit_live_gps(self):
        self.set_user('amb_crew1', 'responder', 'CMCH Ambulance')
        response = self.client.post('/api/driver/location', json={
            'latitude': 13.0814, 'longitude': 80.2735, 'accuracy_meters': 8.5,
            'speed_kmh': 35, 'heading': 90
        }, headers={'X-CSRF-Token': 'test-csrf-token'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()['ok'])
        latest_gps = flask_app.query(
            'SELECT * FROM ambulance_gps ORDER BY id DESC LIMIT 1', one=True
        )
        self.assertIsNotNone(latest_gps)
        self.assertEqual(latest_gps['accuracy_meters'], 8.5)

    def test_14b_incident_map_reports_real_stale_simulated_and_static_sources(self):
        self.set_user('admin', 'admin')
        incident = flask_app.query(
            "SELECT id FROM incidents WHERE candidate_key='test-incident-1'", one=True
        )
        missing_location = self.client.get(
            f"/api/map/live/{incident['id']}"
        ).get_json()
        self.assertIsNone(missing_location['incident']['lat'])
        self.assertFalse(missing_location['incident']['location_available'])
        self.assertTrue(all(
            item['distance_km'] is None and item['eta_min'] is None
            for item in missing_location['ambulances']
        ))
        flask_app.mutate("""INSERT OR REPLACE INTO incident_locations
            (incident_id,latitude,longitude,address) VALUES(?,?,?,?)""",
            (incident['id'], 13.0827, 80.2707, 'Test incident location'))
        ambulance = flask_app.query("""SELECT rr.id,ar.linked_user_id FROM response_resources rr
            JOIN ambulance_registry ar ON ar.gps_resource_id=rr.id
            WHERE rr.agency_type='ambulance' AND rr.active=1 ORDER BY rr.id LIMIT 1""", one=True)
        self.assertIsNotNone(ambulance)
        assignment = flask_app.mutate("""INSERT OR IGNORE INTO incident_assignments
            (incident_id,assignment_role,assignee_user_id,resource_label)
            VALUES(?,'responder',?,'AMB-001')""",
            (incident['id'], ambulance['linked_user_id']))
        simulated = self.client.get(f"/api/map/live/{incident['id']}").get_json()
        unit = next(item for item in simulated['ambulances'] if item['id'] == ambulance['id'])
        self.assertEqual(unit['location_source'], 'simulated_demo')

        self.set_user('amb_crew1', 'responder', 'CMCH Ambulance')
        responder_map = self.client.get(f"/api/map/live/{incident['id']}").get_json()
        self.assertEqual([item['id'] for item in responder_map['ambulances']], [ambulance['id']])
        self.set_user('admin', 'admin')

        flask_app.mutate("""INSERT INTO ambulance_gps
            (resource_id,incident_id,latitude,longitude,accuracy_meters,status)
            VALUES(?,?,?,?,?,'EN_ROUTE')""",
            (ambulance['id'], incident['id'], 13.08, 80.27, 6.0))
        device = self.client.get(f"/api/map/live/{incident['id']}").get_json()
        unit = next(item for item in device['ambulances'] if item['id'] == ambulance['id'])
        self.assertEqual(unit['location_source'], 'device_gps')
        self.assertEqual(unit['accuracy_meters'], 6.0)

        flask_app.mutate("""UPDATE ambulance_gps SET updated_at=datetime('now','-5 minutes')
            WHERE resource_id=? AND incident_id=?""", (ambulance['id'], incident['id']))
        stale = self.client.get(f"/api/map/live/{incident['id']}").get_json()
        unit = next(item for item in stale['ambulances'] if item['id'] == ambulance['id'])
        self.assertEqual(unit['location_source'], 'stale_gps')
        self.assertEqual(unit['status'], 'GPS_STALE')
        self.assertIsNone(unit['eta_min'])

        previous_demo_mode = os.environ.get('RESQLINK_DEMO_MODE')
        try:
            os.environ['RESQLINK_DEMO_MODE'] = '0'
            flask_app.mutate(
                "DELETE FROM ambulance_gps WHERE resource_id=? AND incident_id=?",
                (ambulance['id'], incident['id']),
            )
            static = self.client.get(f"/api/map/live/{incident['id']}").get_json()
            unit = next(item for item in static['ambulances'] if item['id'] == ambulance['id'])
            self.assertEqual(unit['location_source'], 'registry')
            self.assertIsNone(unit['eta_min'])
        finally:
            if previous_demo_mode is None:
                os.environ.pop('RESQLINK_DEMO_MODE', None)
            else:
                os.environ['RESQLINK_DEMO_MODE'] = previous_demo_mode
            flask_app.mutate(
                "DELETE FROM incident_assignments WHERE incident_id=? AND assignee_user_id=?",
                (incident['id'], ambulance['linked_user_id']),
            )
            flask_app.mutate(
                "DELETE FROM incident_locations WHERE incident_id=?",
                (incident['id'],),
            )

    def test_15_dispatch_is_scoped_to_the_selected_operational_account(self):
        self.set_user('admin', 'admin')
        incident = flask_app.query("SELECT id FROM incidents WHERE candidate_key='test-incident-1'", one=True)
        flask_app.mutate(
            "UPDATE incidents SET lifecycle_status='VERIFIED' WHERE id=?",
            (incident["id"],),
        )
        response = self.client.post(f"/incidents/{incident['id']}/assign", data={
            '_csrf_token': 'test-csrf-token', 'username': 'amb_crew1', 'resource_label': 'AMB-001'
        })
        self.assertEqual(response.status_code, 302)
        self.set_user('amb_crew1', 'responder', 'CMCH Ambulance')
        self.assertEqual(self.client.get(f"/incidents/{incident['id']}").status_code, 200)

    def test_16_production_initialization_has_no_demo_accounts_or_resources(self):
        original_db_path = flask_app.DB_PATH
        original_demo_mode = os.environ.get("RESQLINK_DEMO_MODE")
        try:
            os.environ["RESQLINK_DEMO_MODE"] = "0"
            flask_app.DB_PATH = Path(self.temp_dir.name) / "production-test.db"
            flask_app.initialize_database()
            self.assertIsNone(flask_app.query(
                "SELECT id FROM users WHERE username='admin'", one=True
            ))
            self.assertEqual(flask_app.query(
                "SELECT COUNT(*) AS count FROM hospital_registry", one=True
            )["count"], 0)
            self.assertEqual(flask_app.query(
                "SELECT COUNT(*) AS count FROM cameras", one=True
            )["count"], 0)
            result = flask_app.app.test_cli_runner().invoke(
                args=["create-admin"],
                input="coimbatore-admin\nVeryStrongAdminPassword2026!\n"
                      "VeryStrongAdminPassword2026!\n",
            )
            self.assertEqual(result.exit_code, 0, result.output)
            admin = flask_app.query(
                "SELECT role,password_hash FROM users WHERE username='coimbatore-admin'",
                one=True,
            )
            self.assertEqual(admin["role"], "admin")
            self.assertTrue(flask_app.bcrypt.checkpw(
                b"VeryStrongAdminPassword2026!", admin["password_hash"]
            ))
            backup_path = Path(self.temp_dir.name) / "production-test-backup.db"
            backup_result = flask_app.app.test_cli_runner().invoke(
                args=["backup-database", str(backup_path)]
            )
            self.assertEqual(backup_result.exit_code, 0, backup_result.output)
            backup_db = sqlite3.connect(backup_path)
            try:
                self.assertEqual(
                    backup_db.execute("PRAGMA integrity_check").fetchone()[0],
                    "ok",
                )
                self.assertEqual(
                    backup_db.execute(
                        "SELECT COUNT(*) FROM users WHERE username='coimbatore-admin'"
                    ).fetchone()[0],
                    1,
                )
            finally:
                backup_db.close()
            no_overwrite = flask_app.app.test_cli_runner().invoke(
                args=["backup-database", str(backup_path)]
            )
            self.assertNotEqual(no_overwrite.exit_code, 0)
            self.assertIn("already exists", no_overwrite.output)
        finally:
            flask_app.DB_PATH = original_db_path
            if original_demo_mode is None:
                os.environ.pop("RESQLINK_DEMO_MODE", None)
            else:
                os.environ["RESQLINK_DEMO_MODE"] = original_demo_mode

    def test_17_cctv_signal_waits_for_operator_verification_before_dispatch(self):
        db = flask_app.get_db()
        try:
            cctv_upload_demo._ensure_upload_table(db)
            db.execute(
                """INSERT INTO cctv_video_uploads
                (camera_id,original_name,stored_name,detected_at,analysis_status)
                VALUES(?,?,?,?,?)""",
                ("CAM-TEST", "test.mp4", "test.mp4", "2026-10-13T00:00", "Analysis complete"),
            )
            upload_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            db.commit()
        finally:
            db.close()
        case = emergency_orchestration.process_cctv_detection(
            flask_app.DB_PATH, upload_id, "CAM-TEST", "Test junction",
            13.0500, 80.2500, 0.91,
        )
        self.assertEqual(case["status"], "Awaiting operator verification")
        self.assertIsNone(case["ambulance_id"])
        self.assertIsNone(case["hospital_id"])
        self.assertEqual(flask_app.query(
            "SELECT COUNT(*) AS count FROM response_notifications "
            "WHERE case_id=? AND target_role NOT IN ('admin','operator')",
            (case["id"],), one=True,
        )["count"], 0)
        beds_before = flask_app.query(
            "SELECT SUM(emergency_beds) AS beds FROM response_hospitals", one=True
        )["beds"]

        response_app = flask_app.Flask("response-verification-test")
        response_app.secret_key = "test-only-secret"
        emergency_orchestration.register_emergency_orchestration(
            response_app, flask_app.current_user, flask_app.DB_PATH
        )
        response_client = response_app.test_client()
        with response_client.session_transaction() as session:
            session["user"] = {"username": "public1", "role": "public"}
        denied = response_client.post(
            f"/response-case/{case['id']}/verify-dispatch",
        )
        self.assertEqual(denied.status_code, 403)

        with response_client.session_transaction() as session:
            session["user"] = {"username": "operator1", "role": "operator"}
        verified = response_client.post(
            f"/response-case/{case['id']}/verify-dispatch",
        )
        self.assertEqual(verified.status_code, 302)
        dispatched_case = flask_app.query(
            "SELECT status,ambulance_id,hospital_id FROM response_cases WHERE id=?",
            (case["id"],), one=True,
        )
        self.assertEqual(
            dispatched_case["status"],
            "Ambulance assigned — prototype in-app alerts queued (not external dispatch)",
        )
        self.assertIsNotNone(dispatched_case["ambulance_id"])
        self.assertIsNotNone(dispatched_case["hospital_id"])
        beds_after = flask_app.query(
            "SELECT SUM(emergency_beds) AS beds FROM response_hospitals", one=True
        )["beds"]
        self.assertEqual(beds_before - beds_after, 1)

        response_client.post(
            f"/response-case/{case['id']}/verify-dispatch",
        )
        beds_after_retry = flask_app.query(
            "SELECT SUM(emergency_beds) AS beds FROM response_hospitals", one=True
        )["beds"]
        self.assertEqual(beds_after, beds_after_retry)

    def test_17b_response_simulation_and_device_gps_are_separate_and_account_scoped(self):
        import emergency_orchestration as orchestration

        db_path = Path(self.temp_dir.name) / "response-simulation-test.db"
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE users(username TEXT PRIMARY KEY, role TEXT NOT NULL)")
            db.executemany(
                "INSERT INTO users(username,role) VALUES(?,?)",
                [("amb_crew1", "responder"), ("hospital1", "hospital")],
            )
        with orchestration._connect(db_path) as db:
            orchestration.ensure_response_schema(db)
            cursor = db.execute(
                """INSERT INTO response_cases
                   (case_number,location,latitude,longitude,confidence,status,analysis_state)
                   VALUES('SIM-TEST-1','Test junction',13.05,80.25,0.91,
                          'Awaiting operator verification','Test only')"""
            )
            case_id = cursor.lastrowid
        dispatched, status = orchestration.dispatch_verified_case(
            db_path, case_id, "operator1"
        )
        self.assertTrue(dispatched)
        self.assertIn("not external dispatch", status)

        response_app = flask_app.Flask("response-simulation-test",template_folder=str(Path(__file__).resolve().parent / "templates"),)
        response_app.secret_key = "simulation-test-secret"
        response_app.jinja_env.globals["csrf_token"] = lambda: "test-csrf-token"
        response_app.context_processor(
            lambda: {"user": flask_app.current_user()}
        )
        orchestration.register_emergency_orchestration(
            response_app, flask_app.current_user, db_path
        )
        for endpoint in (
            "logout", "dashboard", "incidents", "new_incident", "live_map",
            "reports", "dispatch_console", "resources", "driver_tracking",
            "ambulance_registry", "hospital_registry", "departments", "cameras",
            "onboarding", "user_management", "audit_log", "blockchain",
            "cctv_demo_upload",
        ):
            response_app.add_url_rule(
                f"/test/{endpoint}", endpoint=endpoint, view_func=lambda: ""
            )
        client = response_app.test_client()
        with client.session_transaction() as session:
            session["user"] = {
                "username": "amb_crew1", "role": "responder",
                "role_name": "Ambulance Responder",
            }
        driver_page = client.get(f"/response-case/{case_id}")
        self.assertEqual(driver_page.status_code, 200)
        self.assertIn(b"Start consented phone GPS", driver_page.data)
        self.assertIn(b"contact only; not a tracking method", driver_page.data)
        self.assertIn(b"watchPosition", driver_page.data)
        gps = client.post(
            f"/response-case/{case_id}/gps",
            json={
                "latitude": 13.051,
                "longitude": 80.251,
                "accuracy_meters": 12.5,
                "note": "Consent-based device GPS",
            },
            headers={"X-CSRF-Token": "test-csrf-token"},
        )
        self.assertEqual(gps.status_code, 200)
        self.assertEqual(gps.json["accuracy_meters"], 12.5)

        with client.session_transaction() as session:
            session["user"] = {
                "username": "unassigned_driver", "role": "responder",
                "role_name": "Ambulance Responder",
            }
            session["_csrf_token"] = "test-csrf-token"
        denied_gps = client.post(
            f"/response-case/{case_id}/gps",
            json={"latitude": 13.05, "longitude": 80.25},
            headers={"X-CSRF-Token": "test-csrf-token"},
        )
        self.assertEqual(denied_gps.status_code, 403)

        with orchestration._connect(db_path) as db:
            driver_alerts = db.execute(
                "SELECT target_username FROM response_notifications WHERE target_role='responder'"
            ).fetchall()
            self.assertEqual([row["target_username"] for row in driver_alerts], ["amb_crew1"])
            initial_real_locations = db.execute(
                "SELECT COUNT(*) FROM response_location_updates"
            ).fetchone()[0]
            initial_case_status = db.execute(
                "SELECT status FROM response_cases WHERE id=?", (case_id,)
            ).fetchone()["status"]
            initial_beds = db.execute(
                "SELECT SUM(emergency_beds) FROM response_hospitals"
            ).fetchone()[0]
            initial_ambulance = dict(db.execute(
                "SELECT * FROM response_ambulances WHERE ambulance_id='AMB-DEMO-01'"
            ).fetchone())

        with client.session_transaction() as session:
            session["user"] = {
                "username": "operator1", "role": "operator",
                "role_name": "Emergency Operator",
            }
            session["_csrf_token"] = "test-csrf-token"
        started = client.post(
            f"/response-case/{case_id}/simulation/start",
            data={"_csrf_token": "test-csrf-token", "scenario": "standard"},
            follow_redirects=True,
        )
        self.assertEqual(started.status_code, 200)
        self.assertIn(b"Response simulation started.", started.data)
        with orchestration._connect(db_path) as db:
            run_key = db.execute(
                "SELECT run_key FROM response_simulation_runs WHERE case_id=?",
                (case_id,),
            ).fetchone()["run_key"]

        result = None
        for _ in orchestration.SIMULATION_SCENARIOS["standard"]:
            result = client.post(
                f"/response-simulation/{run_key}/advance",
                headers={"X-CSRF-Token": "test-csrf-token"},
            )
            self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["state"], "Completed")
        self.assertEqual(result.json["progress_percent"], 100)
        self.assertTrue(any(event["latitude"] is not None for event in result.json["events"]))

        with orchestration._connect(db_path) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM response_location_updates").fetchone()[0],
                initial_real_locations,
            )
            self.assertEqual(
                db.execute("SELECT status FROM response_cases WHERE id=?", (case_id,)).fetchone()["status"],
                initial_case_status,
            )
            self.assertEqual(
                db.execute("SELECT SUM(emergency_beds) FROM response_hospitals").fetchone()[0],
                initial_beds,
            )
            self.assertEqual(
                dict(db.execute(
                    "SELECT * FROM response_ambulances WHERE ambulance_id='AMB-DEMO-01'"
                ).fetchone()),
                initial_ambulance,
            )
            simulation_messages = [
                row["message"] for row in db.execute(
                    """SELECT message FROM response_simulation_events
                       WHERE run_key=? ORDER BY step""", (run_key,)
                )
            ]
            self.assertTrue(any("SIMULATION STARTED" in message for message in simulation_messages))

        for scenario, expected_message in (
            ("driver_no_ack", "acknowledgement timeout"),
            ("gps_lost", "GPS signal lost"),
            ("hospital_unavailable", "no capacity"),
        ):
            response = client.post(
                f"/response-case/{case_id}/simulation/start",
                data={"_csrf_token": "test-csrf-token", "scenario": scenario},
                follow_redirects=True,
            )
            self.assertEqual(response.status_code, 200)
            with orchestration._connect(db_path) as db:
                run_key = db.execute(
                    """SELECT run_key FROM response_simulation_runs
                       WHERE case_id=? AND scenario=? ORDER BY id DESC LIMIT 1""",
                    (case_id, scenario),
                ).fetchone()["run_key"]
            scenario_result = None
            for _ in orchestration.SIMULATION_SCENARIOS[scenario]:
                scenario_result = client.post(
                    f"/response-simulation/{run_key}/advance",
                    headers={"X-CSRF-Token": "test-csrf-token"},
                )
                self.assertEqual(scenario_result.status_code, 200)
            self.assertEqual(scenario_result.json["state"], "Completed")
            self.assertTrue(any(
                expected_message.lower() in event["message"].lower()
                for event in scenario_result.json["events"]
            ), f"{scenario} events: {scenario_result.json['events']}")

        with orchestration._connect(db_path) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM response_location_updates").fetchone()[0],
                initial_real_locations,
            )
            self.assertEqual(
                db.execute("SELECT status FROM response_cases WHERE id=?", (case_id,)).fetchone()["status"],
                initial_case_status,
            )
            self.assertEqual(
                db.execute("SELECT SUM(emergency_beds) FROM response_hospitals").fetchone()[0],
                initial_beds,
            )
            self.assertEqual(
                dict(db.execute(
                    "SELECT * FROM response_ambulances WHERE ambulance_id='AMB-DEMO-01'"
                ).fetchone()),
                initial_ambulance,
            )

    def test_18_complete_portal_dispatch_requires_explicit_operator_confirmation(self):
        import resqlink_complete as complete

        old_database = complete.DATABASE
        old_uploads = complete.UPLOADS
        old_initialised = complete._INITIALISED
        try:
            complete.DATABASE = Path(self.temp_dir.name) / "complete-test.db"
            complete.UPLOADS = Path(self.temp_dir.name) / "complete-uploads"
            complete.UPLOADS.mkdir()
            complete._INITIALISED = False
            complete.initialise()

            with complete.connect() as db:
                admin = db.execute(
                    "SELECT id FROM users WHERE username='admin'"
                ).fetchone()
                incident_id, _ = complete.create_incident(
                    db, source="CCTV video", reporter_id=admin["id"],
                    title="Collision candidate", description="Operator review test",
                    location="Coimbatore test junction", latitude=11.0168,
                    longitude=76.9558, confidence=0.91, severity="High",
                    actor="admin",
                )
                incident = db.execute(
                    "SELECT status,ambulance_id,hospital_id FROM incidents WHERE id=?",
                    (incident_id,),
                ).fetchone()
                self.assertEqual(incident["status"], "Awaiting operator verification")
                self.assertIsNone(incident["ambulance_id"])
                self.assertIsNone(incident["hospital_id"])
                self.assertEqual(db.execute(
                    "SELECT COUNT(*) FROM ambulances WHERE status='Dispatched'"
                ).fetchone()[0], 0)
                beds_before = db.execute(
                    "SELECT SUM(emergency_beds) FROM hospitals"
                ).fetchone()[0]

            client = complete.app.test_client()
            login_page = client.get("/login")
            self.assertEqual(login_page.status_code, 200)
            with client.session_transaction() as session:
                csrf = session["csrf"]
            login = client.post("/login", data={
                "username": "admin", "password": "Admin@ResQLink1",
                "_csrf": csrf,
            })
            self.assertEqual(login.status_code, 302)
            with client.session_transaction() as session:
                csrf = session["csrf"]
            confirmation = client.post(
                f"/incident/{incident_id}/control-status",
                data={"_csrf": csrf, "status": "Operator verified"},
            )
            self.assertEqual(confirmation.status_code, 302)

            with complete.connect() as db:
                incident = db.execute(
                    "SELECT status,ambulance_id,hospital_id FROM incidents WHERE id=?",
                    (incident_id,),
                ).fetchone()
                self.assertEqual(
                    incident["status"],
                    "Ambulance dispatched — hospital and departments alerted",
                )
                self.assertIsNotNone(incident["ambulance_id"])
                self.assertIsNotNone(incident["hospital_id"])
                beds_after = db.execute(
                    "SELECT SUM(emergency_beds) FROM hospitals"
                ).fetchone()[0]
                self.assertEqual(beds_before - beds_after, 1)
        finally:
            complete.DATABASE = old_database
            complete.UPLOADS = old_uploads
            complete._INITIALISED = old_initialised

    def test_18b_complete_portal_driver_gps_is_consented_assigned_and_csrf_protected(self):
        import resqlink_complete as complete

        old_database = complete.DATABASE
        old_uploads = complete.UPLOADS
        old_initialised = complete._INITIALISED
        try:
            complete.DATABASE = Path(self.temp_dir.name) / "complete-gps-test.db"
            complete.UPLOADS = Path(self.temp_dir.name) / "complete-gps-uploads"
            complete.UPLOADS.mkdir()
            complete._INITIALISED = False
            complete.initialise()
            with complete.connect() as db:
                admin = db.execute(
                    "SELECT id FROM users WHERE username='admin'"
                ).fetchone()
                incident_id, _ = complete.create_incident(
                    db, source="Citizen report", reporter_id=admin["id"],
                    title="GPS tracking test", description="Test assignment",
                    location="GPS test junction", latitude=11.0168,
                    longitude=76.9558, confidence=None, severity="Medium",
                    actor="admin",
                )
                unassigned_id, _ = complete.create_incident(
                    db, source="Citizen report", reporter_id=admin["id"],
                    title="Unassigned GPS test", description="No unit assigned",
                    location="Other test junction", latitude=12.5,
                    longitude=77.5, confidence=None, severity="Medium",
                    actor="admin",
                )
                db.execute(
                    "UPDATE incidents SET ambulance_id='AMB-001',status=? WHERE id=?",
                    ("Ambulance dispatched — test", incident_id),
                )

            driver_client = complete.app.test_client()
            driver_client.get("/login")
            with driver_client.session_transaction() as session:
                csrf = session["csrf"]
            login = driver_client.post("/login", data={
                "username": "amb_crew1", "password": "AmbCrew@ResQLink1",
                "_csrf": csrf,
            })
            self.assertEqual(login.status_code, 302)
            driver_page = driver_client.get("/driver-live")
            self.assertEqual(driver_page.status_code, 200)
            self.assertIn(b"does not use your IP address", driver_page.data)
            with driver_client.session_transaction() as session:
                csrf = session["csrf"]

            payload = {
                "incident_id": incident_id,
                "latitude": 11.02,
                "longitude": 76.96,
                "accuracy_meters": 18.5,
                "status": "Ambulance en route",
            }
            no_csrf = driver_client.post(
                "/api/driver/location", json=payload
            )
            self.assertEqual(no_csrf.status_code, 400)
            bad_accuracy = driver_client.post(
                "/api/driver/location",
                json={**payload, "accuracy_meters": 100001},
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(bad_accuracy.status_code, 400)
            unassigned = driver_client.post(
                "/api/driver/location",
                json={**payload, "incident_id": unassigned_id},
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(unassigned.status_code, 403)

            accepted = driver_client.post(
                "/api/driver/location",
                json=payload,
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(accepted.status_code, 200)
            self.assertEqual(accepted.json["accuracy_meters"], 18.5)
            with complete.connect() as db:
                saved = db.execute(
                    "SELECT latitude,longitude,accuracy_meters FROM location_updates "
                    "WHERE incident_id=? ORDER BY id DESC LIMIT 1",
                    (incident_id,),
                ).fetchone()
                self.assertEqual(saved["accuracy_meters"], 18.5)
                self.assertEqual(saved["latitude"], 11.02)
                self.assertEqual(saved["longitude"], 76.96)

            operator_client = complete.app.test_client()
            operator_client.get("/login")
            with operator_client.session_transaction() as session:
                csrf = session["csrf"]
            operator_login = operator_client.post("/login", data={
                "username": "admin", "password": "Admin@ResQLink1",
                "_csrf": csrf,
            })
            self.assertEqual(operator_login.status_code, 302)
            live_map = operator_client.get(f"/live-map?incident_id={incident_id}")
            self.assertEqual(live_map.status_code, 200)
            self.assertIn(b"openstreetmap.org/export/embed.html", live_map.data)
            self.assertIn(b"reported accuracy", live_map.data)
            self.assertIn(b"does not use or display a driver", live_map.data)
            gps_instructions = operator_client.get("/driver-live")
            self.assertEqual(gps_instructions.status_code, 200)
            self.assertIn(b"available only to the signed-in driver account", gps_instructions.data)
        finally:
            complete.DATABASE = old_database
            complete.UPLOADS = old_uploads
            complete._INITIALISED = old_initialised

    def test_19_cctv_upload_runs_as_vehicle_evidence_not_an_incident(self):
        self.set_user('admin', 'admin')
        upload_dir = Path(self.temp_dir.name) / "cctv-upload"
        flask_app.app.config["CCTV_UPLOAD_DIR"] = upload_dir
        incident_count = flask_app.query(
            "SELECT COUNT(*) AS count FROM incidents", one=True
        )["count"]
        response_case_count = flask_app.query(
            "SELECT COUNT(*) AS count FROM response_cases", one=True
        )["count"]
        with patch.object(cctv_upload_demo, "_queue_analysis") as queue_analysis:
            response = self.client.post(
                "/cctv-demo/upload",
                data={
                    "_csrf_token": "test-csrf-token",
                    "camera_id": "DEMO-CAM-001",
                    "location": "Test junction",
                    "latitude": "11.0168",
                    "longitude": "76.9558",
                    "video": (io.BytesIO(b"video bytes"), "review.mp4"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 302)
        queue_analysis.assert_called_once()
        upload = flask_app.query(
            "SELECT id,analysis_status,confidence,linked_incident_id "
            "FROM cctv_video_uploads ORDER BY id DESC LIMIT 1",
            one=True,
        )
        self.assertEqual(upload["analysis_status"], "Queued for CV evidence analysis")
        self.assertEqual(upload["confidence"], 0.0)
        self.assertIsNone(upload["linked_incident_id"])
        self.assertEqual(
            flask_app.query("SELECT COUNT(*) AS count FROM incidents", one=True)["count"],
            incident_count,
        )
        self.assertEqual(
            flask_app.query(
                "SELECT COUNT(*) AS count FROM response_cases", one=True
            )["count"],
            response_case_count,
        )

        with patch.object(cctv_upload_demo, "_queue_analysis"):
            page = self.client.get("/cctv-demo/upload")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"CV evidence only", page.data)
        self.assertIn(b"does not classify accidents", page.data)

    def test_cctv_analysis_routes_can_be_disabled_for_serverless_deployments(self):
        previous_setting = flask_app.app.config.get("CCTV_ANALYSIS_ENABLED", True)
        flask_app.app.config["CCTV_ANALYSIS_ENABLED"] = False
        try:
            response = self.client.get("/cctv-demo/upload")
        finally:
            flask_app.app.config["CCTV_ANALYSIS_ENABLED"] = previous_setting

        self.assertEqual(response.status_code, 503)
        self.assertIn(b"unavailable on this serverless demo deployment", response.data)

    def test_production_cctv_upload_rejects_demo_camera(self):
        self.set_user("admin", "admin")
        upload_dir = Path(self.temp_dir.name) / "production-cctv-upload"
        flask_app.app.config["CCTV_UPLOAD_DIR"] = upload_dir
        flask_app.app.config["CCTV_DEMO_CAMERAS_ENABLED"] = False
        previous_inline_worker = flask_app.app.config.get("CCTV_INLINE_WORKER_ENABLED")
        flask_app.app.config["CCTV_INLINE_WORKER_ENABLED"] = False
        try:
            with patch.object(cctv_upload_demo, "_queue_analysis"):
                response = self.client.post(
                    "/cctv-demo/upload",
                    data={
                        "_csrf_token": "test-csrf-token",
                        "camera_id": "DEMO-CAM-001",
                        "video": (io.BytesIO(b"video bytes"), "production-reject.mp4"),
                    },
                    content_type="multipart/form-data",
                )
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Select a registered CCTV camera.", response.data)
            self.assertIsNone(
                flask_app.query(
                    "SELECT id FROM cctv_video_uploads WHERE original_name=?",
                    ("production-reject.mp4",),
                    one=True,
                )
            )
            self.assertFalse((upload_dir / "production-reject.mp4").exists())

            flask_app.mutate(
                """INSERT OR REPLACE INTO cameras
                (camera_id,location_text,latitude,longitude,status)
                VALUES(?,?,?,?,?)""",
                ("CAM-PROD-TEST", "Registered test camera", 11.0, 76.0, "Online"),
            )
            with patch.object(cctv_upload_demo, "_queue_analysis") as queue_analysis:
                accepted = self.client.post(
                    "/cctv-demo/upload",
                    data={
                        "_csrf_token": "test-csrf-token",
                        "camera_id": "CAM-PROD-TEST",
                        "video": (io.BytesIO(b"video bytes"), "production-queued.mp4"),
                    },
                    content_type="multipart/form-data",
                )
            self.assertEqual(accepted.status_code, 302)
            queue_analysis.assert_not_called()
            queued = flask_app.query(
                "SELECT stored_name,analysis_status FROM cctv_video_uploads "
                "WHERE original_name=?",
                ("production-queued.mp4",),
                one=True,
            )
            self.assertEqual(
                queued["analysis_status"], "Queued for CV evidence analysis"
            )
            self.assertTrue((upload_dir / queued["stored_name"]).is_file())
        finally:
            flask_app.app.config["CCTV_DEMO_CAMERAS_ENABLED"] = True
            if previous_inline_worker is None:
                flask_app.app.config.pop("CCTV_INLINE_WORKER_ENABLED", None)
            else:
                flask_app.app.config["CCTV_INLINE_WORKER_ENABLED"] = (
                    previous_inline_worker
                )

    def test_20_operator_can_attach_completed_cv_evidence_without_dispatch(self):
        self.set_user('admin', 'admin')
        upload_dir = Path(self.temp_dir.name) / "cctv-link-upload"
        flask_app.app.config["CCTV_UPLOAD_DIR"] = upload_dir
        with patch.object(cctv_upload_demo, "_queue_analysis"):
            response = self.client.post(
                "/cctv-demo/upload",
                data={
                    "_csrf_token": "test-csrf-token",
                    "camera_id": "DEMO-CAM-001",
                    "location": "Test junction",
                    "latitude": "11.0168",
                    "longitude": "76.9558",
                    "video": (io.BytesIO(b"video bytes"), "review-link.mp4"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 302)
        upload_id = flask_app.query(
            "SELECT id FROM cctv_video_uploads WHERE original_name='review-link.mp4'",
            one=True,
        )["id"]
        incident_id = flask_app.query(
            "SELECT id FROM incidents WHERE candidate_key='test-incident-1'",
            one=True,
        )["id"]
        assignments_before = flask_app.query(
            "SELECT COUNT(*) AS count FROM incident_assignments WHERE incident_id=?",
            (incident_id,),
            one=True,
        )["count"]
        summary = {
            "pipeline": "YOLO vehicle detection + ByteTrack",
            "decision_scope": "vehicle evidence only; accident classification not performed",
            "tracking": {
                "unique_tracks": 1,
                "tracked_observations": 2,
                "tracked_frames": 2,
                "mean_track_path_length_pixels": 12.0,
                "maximum_track_speed_pixels_per_second": 8.0,
            },
            "detections": {
                "vehicle_detections": 2,
                "detection_frames": 2,
                "vehicle_types": {"car": 2},
                "mean_vehicle_detector_confidence": 0.8,
            },
            "incident_created": False,
            "dispatch_triggered": False,
        }
        with patch.object(
            cctv_upload_demo, "analyze_video_evidence", return_value=summary
        ):
            cctv_upload_demo._process_analysis_job(
                flask_app.DB_PATH,
                upload_id,
                upload_dir / "review-link.mp4",
                Path("yolo26n.pt"),
            )
        for username in ("admin", "operator1"):
            notification = flask_app.query(
                "SELECT notif_type,body FROM notifications "
                "WHERE recipient_username=? AND title=?",
                (username, f"CCTV upload #{upload_id}: evidence ready"),
                one=True,
            )
            self.assertEqual(notification["notif_type"], "CV_EVIDENCE_READY")
            self.assertIn("No accident was classified", notification["body"])
        self.assertIsNone(flask_app.query(
            "SELECT id FROM notifications WHERE recipient_username='amb_crew1' "
            "AND title=?",
            (f"CCTV upload #{upload_id}: evidence ready",),
            one=True,
        ))
        response = self.client.post(
            f"/cctv-demo/upload/{upload_id}/link",
            data={
                "_csrf_token": "test-csrf-token",
                "incident_id": str(incident_id),
            },
        )
        self.assertEqual(response.status_code, 302)
        attached = flask_app.query(
            "SELECT source,confidence,raw_data FROM evidence "
            "WHERE incident_id=? ORDER BY id DESC LIMIT 1",
            (incident_id,),
            one=True,
        )
        self.assertEqual(attached["source"], "CAMERA")
        self.assertEqual(attached["confidence"], 0.0)
        self.assertIn("accident classification not performed", attached["raw_data"])
        upload = flask_app.query(
            "SELECT linked_incident_id FROM cctv_video_uploads WHERE id=?",
            (upload_id,),
            one=True,
        )
        self.assertEqual(upload["linked_incident_id"], incident_id)
        self.assertEqual(
            flask_app.query(
                "SELECT COUNT(*) AS count FROM incident_assignments WHERE incident_id=?",
                (incident_id,),
                one=True,
            )["count"],
            assignments_before,
        )

    def test_cctv_analysis_failure_notifies_control_room(self):
        self.set_user("operator1", "operator")
        flask_app.mutate(
            """INSERT INTO cctv_video_uploads
            (camera_id,original_name,stored_name,detected_at,created_by)
            VALUES(?,?,?,?,?)""",
            ("DEMO-CAM-001", "broken.mp4", "broken-job.mp4", "2026-10-14T12:00", 1),
        )
        upload_id = flask_app.query(
            "SELECT id FROM cctv_video_uploads WHERE stored_name='broken-job.mp4'",
            one=True,
        )["id"]
        with patch.object(
            cctv_upload_demo, "analyze_video_evidence",
            side_effect=RuntimeError("model weights unavailable"),
        ):
            cctv_upload_demo._process_analysis_job(
                flask_app.DB_PATH,
                upload_id,
                Path(self.temp_dir.name) / "broken-job.mp4",
                Path("missing-model.pt"),
            )

        upload = flask_app.query(
            "SELECT analysis_status,analysis_error FROM cctv_video_uploads WHERE id=?",
            (upload_id,),
            one=True,
        )
        self.assertEqual(upload["analysis_status"], "CV analysis failed")
        self.assertIn("model weights unavailable", upload["analysis_error"])
        notification = flask_app.query(
            "SELECT notif_type FROM notifications WHERE recipient_username='operator1' "
            "AND title=?",
            (f"CCTV upload #{upload_id}: analysis failed",),
            one=True,
        )
        self.assertEqual(notification["notif_type"], "CV_ANALYSIS_FAILED")

    def test_response_alerts_page_renders_for_signed_in_roles(self):
        for username, role, agency in (
            ("admin", "admin", "ResQLink HQ"),
            ("amb_crew1", "responder", "CMCH Ambulance"),
            ("hospital1", "hospital", "Apollo Hospital"),
            ("police1", "police", "Chennai Police"),
            ("fire1", "fire", "Chennai Fire"),
            ("traffic1", "traffic", "Chennai Traffic"),
        ):
            with self.subTest(role=role):
                self.set_user(username, role, agency)
                response = self.client.get("/response-alerts")
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"My Emergency Alerts", response.data)

    def test_dispatch_requires_verified_incident_and_notifies_selected_linked_unit(self):
        self.set_user("admin", "admin")
        flask_app.mutate(
            """INSERT INTO incidents(candidate_key,vehicle_id,frame,vehicle_type)
            VALUES(?,?,?,?)""",
            ("dispatch-verification-test", 5, 10, "car"),
        )
        incident = flask_app.query(
            "SELECT id FROM incidents WHERE candidate_key='dispatch-verification-test'",
            one=True,
        )
        incident_id = incident["id"]
        assignment_count = flask_app.query(
            "SELECT COUNT(*) AS count FROM incident_assignments WHERE incident_id=?",
            (incident_id,),
            one=True,
        )["count"]

        response = self.client.post(
            f"/incidents/{incident_id}/assign",
            data={
                "_csrf_token": "test-csrf-token",
                "username": "amb_crew1",
                "resource_label": "dispatch integration test",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            flask_app.query(
                "SELECT COUNT(*) AS count FROM incident_assignments WHERE incident_id=?",
                (incident_id,),
                one=True,
            )["count"],
            assignment_count,
        )

        flask_app.mutate(
            "UPDATE incidents SET lifecycle_status='VERIFIED' WHERE id=?",
            (incident_id,),
        )
        response = self.client.post(
            f"/incidents/{incident_id}/accept_recommendation",
            data={"_csrf_token": "test-csrf-token"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            flask_app.query(
                "SELECT lifecycle_status FROM incidents WHERE id=?",
                (incident_id,),
                one=True,
            )["lifecycle_status"],
            "VERIFIED",
        )

        response = self.client.post(
            f"/incidents/{incident_id}/assign",
            data={
                "_csrf_token": "test-csrf-token",
                "username": "amb_crew1",
                "resource_label": "dispatch integration test",
            },
        )
        self.assertEqual(response.status_code, 302)
        assignment = flask_app.query(
            """SELECT u.username,ia.assignment_role FROM incident_assignments ia
            JOIN users u ON u.id=ia.assignee_user_id
            WHERE ia.incident_id=? AND ia.resource_label=?""",
            (incident_id, "dispatch integration test"),
            one=True,
        )
        self.assertEqual(assignment["username"], "amb_crew1")
        self.assertEqual(assignment["assignment_role"], "responder")
        self.assertIsNotNone(flask_app.query(
            """SELECT id FROM notifications WHERE recipient_username='amb_crew1'
            AND incident_id=? AND body LIKE '%dispatch integration test%'""",
            (incident_id,),
            one=True,
        ))
        self.assertIsNone(flask_app.query(
            """SELECT id FROM notifications WHERE recipient_username='police1'
            AND incident_id=? AND body LIKE '%dispatch integration test%'""",
            (incident_id,),
            one=True,
        ))

if __name__ == '__main__':
    unittest.main()
