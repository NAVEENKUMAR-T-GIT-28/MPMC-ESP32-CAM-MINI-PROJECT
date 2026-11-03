# Wildlife Conflict AI Monitor — Updated Dashboard

This update keeps the existing Flask + SQLite + YOLO classification backend and the manual image upload workflow, while replacing the dashboard UI and setting the camera stream default to the ESP32 SoftAP address.

## Project layout

```text
wildlife_conflict_ai_updated/
├── README.md
└── server/
    ├── app.py
    ├── requirements.txt
    ├── run_server.bat
    ├── yolov8n-cls.pt       # copy your existing weights here
    ├── templates/index.html
    ├── static/style.css
    ├── uploads/             # created/used by Flask
    ├── results/             # reserved output folder
    └── events.db            # created automatically
```

## 1. Put your model in place

Copy your existing `yolov8n-cls.pt` into the `server` directory. Do not download a second copy if you already have the weights.

## 2. Start Flask

Double-click `server/run_server.bat`, or run manually:

```powershell
cd server
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## 3. Connect to the ESP32 AP

1. Connect the laptop Wi-Fi to `Wildlife_Monitor`.
2. Configure the laptop Wi-Fi IPv4 address as `192.168.4.2`, subnet mask `255.255.255.0`, gateway/DNS blank (matching the AP setup already used).
3. Confirm Flask is running on the laptop and Windows Firewall permits inbound TCP 5000 on this private project network.
4. In the dashboard, use `http://192.168.4.1/stream` and press **Start monitoring**.

The stream URL is editable. If your firmware uses a different address/path, enter that exact URL.

## 4. Important firmware requirement

The live preview/inference worker pulls an MJPEG HTTP stream. Your ESP32 firmware must expose `GET /stream` and continuously send multipart JPEG frames. A firmware build that only captures a still image after PIR and POSTs it to `http://192.168.4.2:5000/api/predict` does **not** provide a live stream endpoint. Those two paths are separate:

- **PIR still-image path:** ESP32 POSTs one JPEG to `/api/predict`; backend classifies it and saves an event. This already matches the successful HTTP 200 response in your test.
- **Live-stream path:** laptop GETs `http://192.168.4.1/stream`; backend samples frames and classifies at the selected interval.

For both functions simultaneously, the ESP32 firmware must run the camera stream server while also handling PIR-triggered captures/uploads. Do not start a second camera driver or initialize the camera twice.

## 5. Verify the network and API

On the laptop:

```powershell
Test-NetConnection 192.168.4.1 -Port 80
Invoke-RestMethod http://127.0.0.1:5000/health
```

Then open `http://192.168.4.1/stream` from the laptop browser. If it does not show a changing MJPEG stream, fix the ESP32 firmware/stream endpoint before starting AI monitoring.

## 6. What this update fixes

- Replaces the old stale `172.16.143.195` stream URL with the SoftAP default `192.168.4.1`.
- Modern responsive dashboard and clearer server/camera connection states.
- Live preview remains proxied through Flask at `/camera/stream` (no browser mixed-origin/CORS dependency on the ESP32).
- Camera start/stop and status controls remain connected to the existing Flask API.
- Manual image upload is preserved and still uses `POST /api/predict`.
- Latest result, confidence bars, event count and history remain available.
- Error messages from the backend are shown in the UI instead of silently failing.

## Model limitation

Your current `yolov8n-cls.pt` is a generic ImageNet classification model. It may return unrelated labels (for example, `nematode`) for wildlife scenes. It is not a species detector and does not assess conflict/threat. A suitable wildlife model requires a validated wildlife dataset and matching inference code. Do not interpret its current labels as confirmed wildlife detections.
