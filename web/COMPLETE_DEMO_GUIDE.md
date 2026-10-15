# ResQLink Complete Demo

Start the stable complete prototype by double-clicking `START_COMPLETE_DEMO.bat`, or run:

```powershell
cd C:\Users\varsha kaleeswaran\Downloads\road_accident_AI_1\web
python resqlink_complete.py
```

Open `http://127.0.0.1:5050` — not port 5000.  This application uses its own
database, `resqlink_complete.db`, so it is unaffected by the earlier portal's
routes and test data.

## Demo accounts

| Role | Username | Password |
| --- | --- | --- |
| Control room | `admin` | `Admin@ResQLink1` |
| Dispatcher | `operator1` | `Operator@ResQLink1` |
| Ambulance driver | `amb_crew1` | `AmbCrew@ResQLink1` |
| Hospital desk | `hospital1` | `Hospital@ResQLink1` |
| Police | `police1` | `Police@ResQLink1` |
| Fire and rescue | `fire1` | `Fire@ResQLink1` |
| Traffic control | `traffic1` | `Traffic@ResQLink1` |
| Citizen | `public1` | `Public@ResQLink1` |

## End-to-end CCTV incident

1. Sign in as `admin` and select **CCTV Upload**.
2. Upload an authorised MP4, keep the confidence at `0.89`, and select
   **Upload & submit for operator verification**.
3. The prototype creates or merges an incident candidate for operator review.
   It does **not** dispatch a unit, reserve a hospital bed, or notify response
   teams automatically.
4. As `admin` or `operator1`, open the incident, review the footage and
   location, then explicitly choose **Verify incident & dispatch** only if the
   event is genuine. This confirmation starts the demo dispatch workflow. Use
   **False alarm** to close an unverified false positive.
5. Open the incident to view the evidence, resource assignment, hospital
   readiness, timeline, and live location history.
6. In a separate browser profile/window, sign in as `amb_crew1` and select
   **Start Driver Live GPS**. The driver explicitly starts or stops continuous
   location sharing and the page sends an update at most once every 12 seconds.
   The browser must grant location permission. The control room can configure
   the driver's phone number and the ambulance assignment from **Driver
   directory** before the demo.
   In the administrator dashboard, select **Visual live tracker** to watch a
   local visual of the ambulance marker, route trail, updated coordinates and
   status automatically every ten seconds. The local visual does not send
   sensitive location data to a third-party map provider.
7. Sign in as `hospital1`, open its alert, and select **Hospital ready** then
   **Patient received**.
8. Police, traffic and fire accounts each see only their own alert and the
   incident information necessary to act.
9. Finally, sign in as `admin`/`operator1` and set the incident to
   **Resolved**. The ambulance becomes available again and every operational
   role receives the status update.

## Safety boundary

This is a local prototype. Its video function records a clearly-labelled CV
signal (optional OpenCV motion analysis if OpenCV is installed), its hospital
availability is local demo data, and its alerts are in-app notifications.  A
real deployment needs an evaluated accident model, authorised CCTV access,
consented mobile GPS, verified hospital-bed APIs, and an approved SMS/push
provider.

## Testing GPS from an actual phone

Browser GPS works on `localhost` during a laptop demo. For an actual driver
phone, the phone must reach the application over **HTTPS** (for example via a
company-approved HTTPS tunnel or deployed domain) and the driver must consent
to sharing their location. Browsers normally block GPS on a plain `http://LAN-
IP` address. Do not collect a driver's location without a clear operational
need, consent, retention policy, and authorised access controls.
