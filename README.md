# ResQLink AI — Smart Road Accident Detection & Emergency Coordination System

ResQLink AI is a road-safety prototype combining a multi-role Flask portal with computer-vision vehicle detection and tracking. CCTV video analysis is evidence-only: it does not decide whether an accident occurred or start a dispatch.

---

## 🌟 Key Features

1. **Computer-Vision Evidence Extraction**
   - YOLO vehicle detection and ByteTrack tracking from recorded video (`yolo26n.pt` by default).
   - The protected CCTV upload runs the existing tracking pipeline and records per-frame vehicle detection observations and pixel-space track-motion summaries for operator review. A vehicle can be counted on multiple frames; these observations are not unique vehicle counts.
   - Detection and tracking use the same 640-pixel inference size and 0.10 confidence threshold so the two evidence outputs are comparable.
   - Accident classification is not performed by the CCTV upload; detector confidence is not represented as accident confidence.

2. **Full-Stack Emergency Web Portal (`/web`)**
   - **Role-Based Access Control (RBAC):** Admin, Emergency Operator, Responder, Hospital Staff, Police, Fire & Rescue, Traffic Control, and Public citizens.
   - **Interactive Maps:** Leaflet maps display incidents, registered resources, and available location data.
   - **Resource Recommendations:** Candidate hospitals and responder units are ranked using local registry data and distance/capability scores; this is not a live emergency-service integration.
   - **Re-planning:** Operators can record and review re-planning events; live traffic and service availability must be verified before operational use.
   - **Hospital Pre-Alerts:** Demo in-portal notifications are not connected to external hospital systems.
   - **Cryptographic Audit Trail (Blockchain Ledger):** SHA-256 tamper-evident chaining of all critical actions and lifecycle state changes.
   - **Citizen Crowdsourcing Portal:** File eyewitness reports with GPS tags and file attachments (images/video).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Installed packages:
  ```bash
  pip install -r web/requirements.txt
  ```

### 2. Launching the Web Portal
```bash
cd web
```

For a local demo with sample accounts and Chennai sample resources, explicitly enable demo mode:

```powershell
$env:RESQLINK_DEMO_MODE = "1"
python flask_app.py
```
Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

For a non-demo installation, leave `RESQLINK_DEMO_MODE` unset. Create the initial administrator interactively from the `web` directory, then start the portal with a persistent secret:

```powershell
$env:RESQLINK_ENV = "production"
$env:RESQLINK_DATABASE_PATH = "D:\ResQLink\resqlink-prod.db"
$env:RESQLINK_SECRET_KEY = "replace-with-a-long-random-secret"
flask --app flask_app create-admin
python flask_app.py
```

Production mode requires an explicit database path and a secret of at least 32 characters, disables demo users/sample Chennai resources, disables Flask debug mode, and marks session cookies secure by default. Use a new, access-controlled database path rather than the existing demo database. Serve it behind HTTPS; only set `RESQLINK_COOKIE_SECURE=0` for isolated local testing over plain HTTP. The production bootstrap does not insert built-in demo credentials or sample emergency resources. Register and verify real Coimbatore agencies and units before using dispatch workflows.

Prototype CCTV camera choices are available only when `RESQLINK_DEMO_MODE=1` outside production. Production uploads must select a camera already present in the registered camera table; missing or incompatible camera registries do not fall back to demo locations.

These bootstrap safeguards are not a production-readiness certification: this app still uses SQLite and its built-in Flask-SocketIO runner, and the CCTV upload runs vehicle detection/tracking only, not accident classification. Do not expose it to real dispatch operations until the remaining deployment, AI validation, security, and agency-integration work is complete.

Service checks: `GET /health/live` returns process liveness; `GET /health/ready` returns ready only when the portal SQLite database passes a quick integrity check and required core tables exist. These endpoints expose only a status and do not certify AI, agency, or dispatch readiness.

### Render deployment

The repository includes a Render Blueprint in [`render.yaml`](./render.yaml). In Render, create a new Blueprint and select this repository. The Blueprint creates one paid web service with a persistent disk; the portal and its single CCTV worker run as supervised processes in that same service because Render disks are not shared between separate services. SQLite and stored report/CCTV uploads persist under `/var/data`. The build installs the optional AI dependencies as well as the portal requirements.

After the service is created:

1. Open the service Shell and create the first administrator with `cd web && flask --app flask_app create-admin`. Production mode does not seed demo accounts, sample resources, or cameras. Register and verify operational accounts and resources before using the portal.
2. The YOLO checkpoint is intentionally not stored in Git. Before using CCTV analysis, place the approved `yolo26n.pt` checkpoint at `/var/data/models/yolo26n.pt` (or update `RESQLINK_YOLO_MODEL` to its persistent-disk path). Until the file is present, CCTV analysis jobs will fail explicitly; the portal itself can still run.
3. Keep the generated `RESQLINK_SECRET_KEY` secret. Back up the SQLite database and uploaded files separately. A Render persistent disk is attached to this single service and is not a substitute for backups.

This deployment is a single-instance prototype: SQLite and the worker are not horizontally scalable, and this setup does not certify the accident-classification pipeline or emergency dispatch workflow for operational use.

### Vercel temporary demo

The repository also includes a Vercel Flask entrypoint for a limited preview. Import this repository on the `main` branch, leave the **Root Directory** at the repository root, select the detected **Flask** framework preset, and use the default install command. Leave **Build Command** and **Output Directory** unset. Set a private `RESQLINK_SECRET_KEY` environment variable with at least 32 characters before deploying. The root `pyproject.toml` contains the Python project metadata and Vercel entrypoint configuration required by Vercel's `uv` build.

This preview enables demo accounts and sample Chennai data. It uses temporary `/tmp` SQLite and upload storage, which may disappear or differ between function instances and deployments. CCTV upload and analysis routes are disabled because Vercel does not run the persistent CCTV worker. Do not use this preview for sensitive reports, reliable incident records, CCTV processing, or real dispatch. The sample `admin` / `Admin@ResQLink1` credentials are for a private UI walkthrough only; do not share the deployment while those credentials are enabled. Use the Render persistent-disk deployment above for the complete portal and worker.

Set `RESQLINK_SECRET_KEY` in Vercel's project environment variables; never commit the secret to this repository.

Create a consistent, integrity-checked database backup from the `web` directory; the destination directory must exist and an existing backup is never overwritten:

```powershell
flask --app flask_app backup-database "D:\ResQLink\backups\resqlink-2026-10-14.db"
```

Back up protected CCTV uploads separately, encrypt backups at rest, restrict access, and regularly test restoring a backup. The command backs up only the SQLite database.

### Recorded-video CV evidence

The web application's base requirements do not install its optional model dependencies. From the `web` directory, install them with:

```powershell
pip install -r requirements-ai.txt
```

These optional dependencies also include scikit-learn for the repository's offline classifier experiments. To reproduce the selected temporal model's grouped, video-level cross-validation from the repository root:

```powershell
python evaluate_selected_temporal_model.py
```

This reports out-of-fold window metrics; it does not train a production model or connect predictions to incident creation or dispatch. The separate `stage1_accident_classifier.joblib` experiment distinguishes single-vehicle from multi-vehicle accident types, not accident from non-accident. Treat all classifier results as exploratory until independently labeled event intervals and verified normal footage support held-out evaluation.

The CCTV upload queues the existing `track_vehicles.py` YOLO/ByteTrack pipeline. In development/demo mode it uses a local background thread. In production, uploads remain persisted as queued database rows for the separately supervised worker; the web process does not launch the model.

Start one worker process alongside the portal, using the same database and storage paths:

```powershell
$env:RESQLINK_ENV = "production"
$env:RESQLINK_DATABASE_PATH = "D:\ResQLink\resqlink-prod.db"
$env:RESQLINK_CCTV_UPLOAD_DIR = "D:\ResQLink\protected-cctv"
$env:RESQLINK_YOLO_MODEL = "D:\ResQLink\models\yolo26n.pt"
python web/cctv_worker.py
```

The worker polls the persisted upload rows, claims queued jobs atomically, records failures, and requeues work left running beyond its one-hour lease. Run one supervised worker instance for this SQLite deployment; this is restart-recoverable but is not a horizontally scalable job broker. Set `RESQLINK_YOLO_MODEL` to an available checkpoint if it is not the repository's `yolo26n.pt`; set `RESQLINK_CCTV_UPLOAD_DIR` to protected storage outside the source tree. When analysis finishes or fails, active control-room accounts receive a persisted in-portal notification; the portal refreshes notifications periodically. The upload page shows detection and track-motion summaries, and analysis errors are displayed to control-room users. After reviewing the footage, an operator can attach completed CV evidence to an existing incident; this action does not dispatch resources.

Dispatch recommendations are advisory. The Dispatch Console accepts assignments only after an incident reaches `VERIFIED` and only to an active operational account linked to an active registry record; assignment notifications go to the selected account, not every user in that role. Hospital pre-alerts are a separate in-portal workflow. These notifications are not external SMS, phone, or agency-system integrations.

This pipeline reports vehicle evidence only. The current classifier artifacts do not provide a validated accident-versus-normal decision for new footage, so the upload intentionally does not label a crash, create an incident candidate, or infer accident probability. The in-process worker is for local development/demo use; the production worker is a separately supervised, single-instance SQLite consumer, not a horizontally scalable job broker.

### Ambulance location and response simulation

The main portal's `/map` is a registered-resource snapshot, not a live fleet tracker. An incident-specific map at `/incidents/<id>/map` distinguishes:

- **Assigned driver phone GPS:** the signed-in responder's browser location, sent only after the driver starts sharing and grants location permission. On the driver's **Driver GPS** page, select the assigned incident before sharing to associate fixes with that incident. HTTPS is required on phones. Reported browser accuracy and the last-fix age are shown; fixes older than 90 seconds are marked stale and are not given an ETA.
- **Simulated demo movement:** deterministic prototype movement for an assigned demo ambulance, only when `RESQLINK_DEMO_MODE=1` outside production. It is labelled as simulated and is never written into the actual GPS history.
- **Stale device fix:** the last consented location remains visible with its age, but is explicitly marked stale; no arrival estimate is shown.
- **Registered base location:** a static registry coordinate, not evidence of a dispatched or moving vehicle.

The incident map refreshes its data every five seconds. Any ETA shown is only a straight-line distance estimate (assuming 60 km/h), not a traffic-aware arrival prediction. A phone number is contact information only; the application does not locate a driver from their phone number or IP address. Public map tiles and optional route geometry/address lookup use third-party OpenStreetMap/CARTO/OSRM services and therefore disclose the map area/coordinates to those services.

The response-case simulator separately demonstrates a normal handover, driver acknowledgement timeout, GPS loss, and hospital unavailability/manual diversion. These scenarios are illustrative only: portal notifications are not external calls, SMS, radio messages, or hospital-system alerts. No simulated state represents an actual emergency response or replaces an operator's verification and approved agency procedures.

Public registration creates a citizen account only. An administrator must create and verify emergency-service accounts.

## Standalone complete-demo UI (`web/resqlink_complete.py`)

This separate, self-contained prototype runs on port 5050 and uses its own `resqlink_complete.db`; it is not the main portal on port 5000:

```powershell
Set-Location web
python resqlink_complete.py
```

Open `http://127.0.0.1:5050`. In this local demo, the sample control-room account is `admin` / `Admin@ResQLink1`, and the sample ambulance-driver account is `amb_crew1` / `AmbCrew@ResQLink1`. These are demo credentials only; do not deploy them.

After an operator assigns an incident to the ambulance, the driver signs in on the phone, opens **Driver Live GPS**, and explicitly starts sharing and grants browser location permission. Control-room users can review the moving GPS marker at **Live Map**. The driver can stop sharing or close the page. GPS requires HTTPS in a real deployment (localhost is treated as a secure context by browsers). The app uses the signed-in driver session and assigned incident to authorize updates; it does not use the driver's IP address to determine location. The embedded OpenStreetMap service receives map requests from the operator's browser, including the location displayed on the map. This standalone workflow is a demo, not a certified emergency dispatch service.

## Connected prototype workflow

The web portal uses RBAC throughout the UI and API. Citizens submit their own reports; an administrator onboards verified operational accounts and links them to one registry record (`AMB-001`, `HOSP-001`, `POLICE-001`, etc.). Operators dispatch those exact accounts to an incident, so ambulance, police, fire, traffic, and hospital users only receive their own assignment/alert data.

- Use **Admin → Onboard Staff** to create a verified operational account and link it to an ambulance, hospital, or department code.
- Citizen reports with GPS are automatically matched to an open incident within 500 metres and 30 minutes; original reports remain separately recorded as evidence.
- A linked ambulance responder opens **Driver GPS** on the driver's phone, grants browser location permission, chooses the assigned incident, and starts sharing. Authorized command staff see the recorded positions on the incident map.
- Hospitals update beds, ICU availability, and operating state from their own dashboard. Capacity is synchronised into the dispatch resource pool and pre-alerts are delivered only to the selected hospital.
- Camera records are managed by admin/operator accounts. The prototype accepts authorized CV detection metadata and supports later connection of approved roadside-camera workers without storing their stream passwords in the portal.

### CCTV / CV adapter

Set a dedicated secret for the CV worker and serve the production portal over HTTPS:

```powershell
$env:RESQLINK_CAMERA_INGEST_KEY = "replace-with-a-long-random-camera-worker-secret"
python flask_app.py
```

The existing CV/ML pipeline can submit a JSON Lines stream through the adapter:

```powershell
python cctv_event_adapter.py --portal https://your-portal.example --camera-id 1 --api-key $env:RESQLINK_CAMERA_INGEST_KEY < detections.jsonl
```

Each event must include `event_id`, `observed_at` (ISO-8601), and `confidence` (0–1); it can also include `latitude`, `longitude`, `vehicle_type`, and a safe evidence reference. This separate adapter endpoint creates a candidate for operator review. Do not send raw vehicle detections as accident events; use it only for externally validated candidate events. The recorded-video upload flow described above is evidence-only.

For a driver phone to share browser GPS, deploy the portal through HTTPS on a host reachable by the phone. Modern mobile browsers allow geolocation on `localhost` for local desktop testing, but normally require HTTPS for a LAN or internet URL. Set `RESQLINK_HOST=0.0.0.0` only behind a firewall/reverse proxy; do not expose the Flask development server directly to the public internet.

### 3. Demo Credentials
| Role | Username | Password |
|---|---|---|
| **Administrator** | `admin` | `Admin@ResQLink1` |
| **Emergency Operator** | `operator1` | `Operator@ResQLink1` |
| **Ambulance Responder** | `amb_crew1` | `AmbCrew@ResQLink1` |
| **Hospital Staff** | `hospital1` | `Hospital@ResQLink1` |
| **Police Officer** | `police1` | `Police@ResQLink1` |
| **Fire & Rescue** | `fire1` | `Fire@ResQLink1` |
| **Traffic Control** | `traffic1` | `Traffic@ResQLink1` |
| **Citizen (Public)** | `public1` | `Public@ResQLink1` |

---

## 🧪 Testing & Validation

Run the automated integration test suite:
```bash
cd web
python test_portal.py
```
The focused CV evidence and evaluation tests can be run with:
```powershell
python -m unittest test_cctv_video_analysis test_cv_evaluation
```

### Structured CV evaluation

`cv_evaluation.py` evaluates annotated detector boxes, tracked vehicle identities, and accident-event intervals without changing the live CCTV pipeline. It does not tune thresholds, classify uploaded videos, create incidents, or dispatch resources. Run it from the `web` directory, supplying box inputs, event inputs, or both:

```powershell
python cv_evaluation.py `
  --ground-truth-boxes detections-ground-truth.csv `
  --predicted-boxes detections-predictions.csv `
  --ground-truth-events accident-events-ground-truth.csv `
  --predicted-events accident-event-intervals.csv `
  --video-manifest reviewed-videos.csv `
  --output evaluation-results.json
```

Box CSVs require `video,frame,class,x1,y1,x2,y2`; prediction rows also accept `confidence` and `track_id`, and ground-truth rows accept `track_id`. Frames are 1-based; box coordinates are pixel-space `x1,y1,x2,y2` in the original video. For detection precision/recall, annotate all visible target vehicles on the evaluated frames, not only the accident vehicle. Ground-truth track IDs must remain stable for each vehicle throughout a clip. The evaluator reports missed boxes, unmatched predictions, class errors, overlapping prediction pairs, track-observation coverage, adjacent-frame ID switches, and track fragments.

Event CSVs require `video,start_frame,end_frame` with inclusive frame bounds. Ground truth may also include `event_frame`; predictions may include `decision_frame` to calculate detection delay. Each predicted row must represent one consolidated alert interval, not one overlapping model window. The optional manifest requires `video,frame_count,fps,event_free_verified`. When the flag is true, include `verified_by` and `verified_at`; `event_free_intervals` may list reviewed inclusive frame ranges separated by semicolons (for example `100-250;400-600`). If no ranges are supplied, the entire clip is treated as reviewed. The false-alarm rate is calculated only over verified non-event exposure, excluding labeled accident intervals; without verified negative exposure, the hours and rate are unavailable. Choose and record the model threshold and alert-consolidation policy using training/validation data before evaluating held-out videos.

The existing positive-frame vehicle labels and window-level model benchmark do not supply complete frame-by-frame vehicle identities or fully reviewed normal-video intervals. They therefore cannot establish detection precision, tracking ID-switch/fragmentation rates, or a trustworthy false-alarms-per-hour figure. Keep the current 640-pixel inference and 0.10 detector-confidence configuration unchanged until annotations identify a measurable failure mode. Operator verification and dispatch confirmation remain mandatory.

### Tracking and classifier evaluation status

Run the exploratory tracker comparison from the `web` directory:

```powershell
python compare_labeled_video_trackers.py --context-frames 30
```

It compares the default ByteTrack config against three single-parameter variants on all 25 accident clips, using the annotated impact frame plus surrounding context. Detector inference remains fixed at 640 pixels and confidence 0.10. The report measures positive-frame detection/track association and how often the event-frame-associated tracker ID reappears in nearby frames. That nearby-ID count is only a continuity proxy: the current labels do not contain persistent identities, exhaustive vehicle boxes, or reviewed occlusion intervals, so this experiment cannot establish ID switches, fragmentation, or false-track rates. The experiment does not update the production tracker configuration.

On the current set, the impact vehicle reaches IoU ≥0.3 on 21/25 clips and receives a track ID on 20/25; only 4/7 targets in the smallest box-area quartile and 3/4 targets in the poor/very-poor quality subset receive IDs. The three single-setting variants produced the same event-frame association and local-ID observations as baseline. This does not prove equivalent tracking generally; it means these limited labels show no basis to change the current settings. Detailed per-video results are in [tracker_config_event_frame_comparison.csv](./data/ACCIDENT/tracker_config_event_frame_comparison.csv) and [tracker_config_event_frame_comparison.json](./data/ACCIDENT/tracker_config_event_frame_comparison.json).

Recreate the saved out-of-fold window/event audit with:

```powershell
python evaluate_window_classifier_oof.py
```

The current 512-window dataset contains 21 accident-event windows (4.1%) and 491 negative windows. Four of the 25 metadata impact frames are not represented by a positive classifier window: three videos have no feature windows, and one impact frame falls in a gap between generated windows. Its grouped ExtraTrees fold F1 scores vary from about 0.068 to 0.571; pooled window F1 is about 0.245, while the reported unweighted mean of fold F1 is about 0.394. The audit consolidates adjacent positive windows into candidate alert episodes and calculates point-impact event recall/delay, but remains a limited offline diagnostic. A positive label currently means only that a 50-frame window contains the annotated impact frame; the other windows in accident clips are labeled `non_event` without annotated event start/end intervals. The mixed-resolution dataset also uses pixel-space motion/distance summaries, so feature scale and within-window timing need review before retraining.

After adjacent positive windows are merged, ExtraTrees matches 13/25 annotated impact frames (point-event precision 0.25, recall 0.52, F1 0.338) with a mean decision delay of 3.09 seconds for matched events. The same audit finds 39 unmatched alert episodes. These figures are exploratory: the ground truth is an impact frame, not the full event interval, and episode merging is not a validated alert policy. See [window_classifier_oof_event_audit.json](./data/ACCIDENT/window_classifier_oof_event_audit.json) and [window_classifier_oof_alarm_episodes.csv](./data/ACCIDENT/window_classifier_oof_alarm_episodes.csv).

Using the metadata's visual-quality groups, ExtraTrees detected 13/19 clear-quality impact points (F1 0.703), but detected 0/6 poor/very-poor impact points. These small, event-positive-only groups do not replace hard-negative evaluation; they show that current results do not generalize to difficult footage.

The recorded normal clips have known durations in [normal_traffic_evaluation_manifest.csv](./data/ACCIDENT/normal_traffic_evaluation_manifest.csv), but all are currently marked `event_free_verified=false`. ExtraTrees produced 32 candidate alarm episodes across these clips (31 in `normal_02`, one in `normal_01`); these must not be reported as false alarms per hour until footage has been independently reviewed and the manifest is updated. No validated accident-event model is connected to the operator incident workflow. Keep human verification and dispatch approval mandatory.

---

## 📁 Project Structure

```
road_accident_AI_1/
├── web/                           # Flask Web Application & Socket.IO Portal
│   ├── flask_app.py               # Main Flask application & background replan engine
│   ├── test_portal.py             # Integration test suite
│   ├── cv_evaluation.py           # Annotation-based detector/tracker/event metrics
│   ├── compare_labeled_video_trackers.py  # Positive-frame ByteTrack comparison
│   ├── evaluate_window_classifier_oof.py  # Saved grouped-CV error audit
│   ├── requirements.txt           # Python dependencies
│   ├── templates/                 # Jinja2 templates (role dashboards, maps, detail)
│   └── static/                    # Custom CSS design system, fonts, and JS
├── resqlink_incidents.db          # SQLite persistent database & blockchain ledger
├── track_vehicles.py              # YOLO vehicle tracking pipeline
├── movement_analysis.py           # Kinematics & sudden movement anomaly detector
├── stage1_inference.py            # Machine learning accident classifier
└── yolo26n.pt / yolo26s.pt        # Computer vision neural network weights
```
