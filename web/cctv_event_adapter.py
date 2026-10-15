"""Publish validated CV/ML accident detections to the ResQLink portal.

This is deliberately an adapter, not a camera-password store.  Run the model
worker close to an approved CCTV source, then feed one JSON event per line to
this program.  The portal receives metadata and an optional evidence reference;
the camera credentials and raw stream remain outside the web application.
"""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def publish(portal_url, camera_id, api_key, event):
    required = {"event_id", "observed_at", "confidence"}
    missing = sorted(required.difference(event))
    if missing:
        raise ValueError(f"Missing required event fields: {', '.join(missing)}")
    request = Request(
        f"{portal_url.rstrip('/')}/api/integrations/cameras/{camera_id}/detections",
        data=json.dumps(event).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-ResQLink-Camera-Key": api_key},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Forward CV accident events to ResQLink")
    parser.add_argument("--portal", required=True, help="HTTPS portal base URL")
    parser.add_argument("--camera-id", required=True, type=int, help="ResQLink camera record ID")
    parser.add_argument("--api-key", required=True, help="RESQLINK_CAMERA_INGEST_KEY value")
    args = parser.parse_args()

    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            result = publish(args.portal, args.camera_id, args.api_key, json.loads(line))
            print(json.dumps({"line": line_number, **result}), flush=True)
        except (ValueError, json.JSONDecodeError, HTTPError, URLError) as exc:
            print(json.dumps({"line": line_number, "error": str(exc)}), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
