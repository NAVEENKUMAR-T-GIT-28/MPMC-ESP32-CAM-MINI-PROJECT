from __future__ import annotations

import io
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths and application setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
DB_PATH = BASE_DIR / "events.db"
MODEL_PATH = BASE_DIR / "yolov8n-cls.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# YOLO model
# ---------------------------------------------------------------------------

model = None
model_error = None
model_lock = threading.Lock()

try:
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        app.logger.info("Loaded model: %s", MODEL_PATH)
    else:
        model_error = (
            f"Missing model weights: {MODEL_PATH}. "
            "Copy yolov8n-cls.pt into the server folder."
        )
        app.logger.error(model_error)
except Exception as exc:
    model_error = f"Model failed to load: {exc}"
    app.logger.exception("Model initialization failed")


# ---------------------------------------------------------------------------
# Camera worker state
# ---------------------------------------------------------------------------

camera_lock = threading.RLock()
camera_stop = threading.Event()
camera_thread: threading.Thread | None = None

camera_state = {
    "running": False,
    "connected": False,
    "stream_url": "",
    "interval_seconds": 5,
    "started_at": None,
    "last_frame_at": None,
    "last_prediction_at": None,
    "last_error": None,
    "latest_event": None,
    "frame_jpeg": None,
    "frame_sequence": 0,
}


# ---------------------------------------------------------------------------
# Database and image helpers
# ---------------------------------------------------------------------------

def db():
    return sqlite3.connect(DB_PATH, timeout=15)


def init_db():
    with db() as con:
        con.row_factory = sqlite3.Row
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                top_predictions TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )


def local_timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_image(raw: bytes):
    if not raw:
        raise ValueError("No image bytes received.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The request did not contain a valid image.") from exc

    if image.format not in {"JPEG", "PNG", "WEBP"}:
        raise ValueError("Supported image formats are JPEG, PNG and WebP.")

    return image.convert("RGB")


def run_prediction(image: Image.Image):
    if model is None:
        raise RuntimeError(model_error or "The YOLO model is unavailable.")

    # A single model instance is shared by uploads and the camera worker.
    with model_lock:
        result = model.predict(source=image, verbose=False)[0]

    top = []
    if result.probs is not None:
        for class_id, score in zip(result.probs.top5, result.probs.top5conf):
            top.append(
                {
                    "label": result.names[int(class_id)],
                    "confidence": round(float(score), 6),
                }
            )

    if not top:
        raise RuntimeError(
            "The model returned no classification probabilities. "
            "Check that yolov8n-cls.pt is a classification model."
        )

    return top


def save_event(image: Image.Image, filename: str, source: str):
    top = run_prediction(image)

    event_id = uuid.uuid4().hex
    image_name = f"{event_id}.jpg"
    image.save(UPLOAD_DIR / image_name, "JPEG", quality=90)

    timestamp = local_timestamp()
    safe_filename = Path(filename or "capture.jpg").name[:200]
    safe_source = (source or "unknown")[:80]

    with db() as con:
        cur = con.execute(
            """
            INSERT INTO events
                (event_id, created_at, filename, image_name, predicted_class,
                 confidence, top_predictions, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                timestamp,
                safe_filename,
                image_name,
                top[0]["label"],
                top[0]["confidence"],
                json.dumps(top),
                safe_source,
            ),
        )
        row_id = cur.lastrowid

    return {
        "id": row_id,
        "event_id": event_id,
        "created_at": timestamp,
        "filename": safe_filename,
        "image_url": f"/uploads/{image_name}",
        "predicted_class": top[0]["label"],
        "confidence": top[0]["confidence"],
        "top_predictions": top,
        "source": safe_source,
    }


def list_events(limit=50):
    with db() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    output = []
    for row in rows:
        output.append(
            {
                "id": row["id"],
                "event_id": row["event_id"],
                "created_at": row["created_at"],
                "filename": row["filename"],
                "image_url": f"/uploads/{row['image_name']}",
                "predicted_class": row["predicted_class"],
                "confidence": row["confidence"],
                "top_predictions": json.loads(row["top_predictions"]),
                "source": row["source"],
            }
        )
    return output


# ---------------------------------------------------------------------------
# Camera state helpers
# ---------------------------------------------------------------------------

def update_camera_state(**values):
    with camera_lock:
        camera_state.update(values)


def public_camera_state():
    with camera_lock:
        public_state = {
            key: value
            for key, value in camera_state.items()
            if key not in {"frame_jpeg", "frame_sequence"}
        }
        public_state["has_frame"] = camera_state["frame_jpeg"] is not None
        public_state["frame_sequence"] = camera_state["frame_sequence"]
        return public_state


def extract_jpeg_frames(buffer: bytearray):
    """
    Yield complete JPEG frames from a byte stream.

    ESP32-CAM sends multipart MJPEG. JPEG start/end markers make frame
    extraction independent of the exact multipart boundary string.
    """
    frames = []

    while True:
        start = buffer.find(b"\xff\xd8")  # JPEG Start Of Image
        if start < 0:
            # Keep a possible partial marker at the end of the buffer.
            if len(buffer) > 1:
                del buffer[:-1]
            break

        if start > 0:
            del buffer[:start]

        end = buffer.find(b"\xff\xd9", 2)  # JPEG End Of Image
        if end < 0:
            break

        frames.append(bytes(buffer[: end + 2]))
        del buffer[: end + 2]

    # Protect against a malformed stream filling memory indefinitely.
    if len(buffer) > 2 * 1024 * 1024:
        buffer.clear()

    return frames


# ---------------------------------------------------------------------------
# ESP32 MJPEG reader + laptop YOLO inference
# ---------------------------------------------------------------------------

def camera_worker(stream_url: str, interval_seconds: int):
    """
    Read the ESP32 MJPEG stream with requests (not cv2.VideoCapture),
    show a preview frame, and classify one frame every interval_seconds.
    """
    app.logger.info("Camera worker starting: %s", stream_url)
    last_prediction = 0.0
    last_preview_update = 0.0

    try:
        while not camera_stop.is_set():
            try:
                app.logger.info("Connecting to ESP32 stream: %s", stream_url)

                with requests.get(
                    stream_url,
                    stream=True,
                    timeout=(5, 15),
                    headers={"Cache-Control": "no-cache"},
                ) as response:
                    response.raise_for_status()
                    app.logger.info(
                        "ESP32 HTTP stream connected (status %s).",
                        response.status_code,
                    )

                    buffer = bytearray()

                    for chunk in response.iter_content(chunk_size=4096):
                        if camera_stop.is_set():
                            break
                        if not chunk:
                            continue

                        buffer.extend(chunk)

                        for jpeg_bytes in extract_jpeg_frames(buffer):
                            if camera_stop.is_set():
                                break

                            try:
                                image = parse_image(jpeg_bytes)
                            except ValueError as exc:
                                app.logger.warning("Bad JPEG frame: %s", exc)
                                continue

                            now = time.monotonic()

                            # Update dashboard preview at most twice per second.
                            if now - last_preview_update >= 0.5:
                                preview = image.copy()
                                preview.thumbnail((800, 600))
                                preview_buffer = io.BytesIO()
                                preview.save(
                                    preview_buffer,
                                    format="JPEG",
                                    quality=75,
                                )

                                with camera_lock:
                                    camera_state["connected"] = True
                                    camera_state["last_frame_at"] = local_timestamp()
                                    camera_state["frame_jpeg"] = (
                                        preview_buffer.getvalue()
                                    )
                                    camera_state["frame_sequence"] += 1
                                    camera_state["last_error"] = None

                                last_preview_update = now

                            # Sample the stream instead of running YOLO on
                            # every camera frame.
                            if now - last_prediction >= interval_seconds:
                                last_prediction = now

                                event = save_event(
                                    image,
                                    filename=f"esp32_frame_{int(time.time())}.jpg",
                                    source="esp32-cam-live-stream",
                                )

                                with camera_lock:
                                    camera_state["last_prediction_at"] = (
                                        event["created_at"]
                                    )
                                    camera_state["latest_event"] = event

                                app.logger.info(
                                    "YOLO prediction: %s (%.2f%%)",
                                    event["predicted_class"],
                                    event["confidence"] * 100,
                                )

                    if not camera_stop.is_set():
                        raise RuntimeError(
                            "ESP32 stream ended. Reconnecting in 3 seconds."
                        )

            except requests.RequestException as exc:
                app.logger.warning("ESP32 HTTP connection error: %s", exc)
                update_camera_state(connected=False, last_error=str(exc))
                if not camera_stop.is_set():
                    camera_stop.wait(3)

            except Exception as exc:
                app.logger.exception("Camera worker error: %s", exc)
                update_camera_state(connected=False, last_error=str(exc))
                if not camera_stop.is_set():
                    camera_stop.wait(3)

    finally:
        update_camera_state(
            running=False,
            connected=False,
            frame_jpeg=None,
        )
        app.logger.info("Camera worker stopped.")


# ---------------------------------------------------------------------------
# Dashboard and health
# ---------------------------------------------------------------------------

@app.get("/")
def dashboard():
    return render_template(
        "index.html",
        model_ready=model is not None,
        model_error=model_error,
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if model is not None else "model_unavailable",
            "model_loaded": model is not None,
            "model_file": MODEL_PATH.name,
            "model_error": model_error,
            "api_endpoint": "/api/predict",
            "camera_reader": "requests-mjpeg-jpeg-parser",
            "requests_version": requests.__version__,
        }
    ), (200 if model is not None else 503)


# ---------------------------------------------------------------------------
# Image upload / direct ESP32 image endpoint
# ---------------------------------------------------------------------------

@app.post("/api/predict")
def predict():
    """Accept multipart browser uploads or raw JPEG bytes from ESP32."""
    if model is None:
        return jsonify(
            {"success": False, "error": model_error or "Model unavailable"}
        ), 503

    try:
        if "image" in request.files:
            upload = request.files["image"]
            raw = upload.read()
            filename = upload.filename or "upload.jpg"
            source = request.form.get("source", "web-upload")
        else:
            raw = request.get_data(cache=False)
            filename = request.headers.get("X-Filename", "esp32-capture.jpg")
            source = request.headers.get("X-Device", "esp32-cam")

        if len(raw) > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify(
                {"success": False, "error": "Image exceeds 10 MB limit."}
            ), 413

        image = parse_image(raw)
        event = save_event(image, filename, source)

        app.logger.info(
            "Event %s: %s (%.2f%%) source=%s",
            event["event_id"],
            event["predicted_class"],
            event["confidence"] * 100,
            source,
        )
        return jsonify({"success": True, "event": event}), 200

    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception:
        app.logger.exception("Unhandled inference error")
        return jsonify(
            {"success": False, "error": "Inference failed. Check server terminal."}
        ), 500


# ---------------------------------------------------------------------------
# Camera start / stop / status
# ---------------------------------------------------------------------------

@app.get("/api/camera/status")
def camera_status():
    return jsonify(public_camera_state())


@app.post("/api/camera/start")
def camera_start():
    global camera_thread

    if model is None:
        return jsonify(
            {"success": False, "error": model_error or "YOLO model unavailable"}
        ), 503

    payload = request.get_json(silent=True) or {}
    stream_url = str(payload.get("stream_url", "")).strip()
    parsed = urlparse(stream_url)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Enter a valid ESP32 stream URL, for example "
                    "http://172.16.143.195/stream"
                ),
            }
        ), 400

    try:
        interval_seconds = int(payload.get("interval_seconds", 5))
    except (TypeError, ValueError):
        interval_seconds = 5

    interval_seconds = max(2, min(interval_seconds, 60))

    with camera_lock:
        if camera_thread is not None and camera_thread.is_alive():
            return jsonify(
                {
                    "success": False,
                    "error": "Camera monitoring is already running.",
                }
            ), 409

        camera_stop.clear()
        camera_state.update(
            {
                "running": True,
                "connected": False,
                "stream_url": stream_url,
                "interval_seconds": interval_seconds,
                "started_at": local_timestamp(),
                "last_frame_at": None,
                "last_prediction_at": None,
                "last_error": None,
                "latest_event": None,
                "frame_jpeg": None,
                "frame_sequence": 0,
            }
        )

        camera_thread = threading.Thread(
            target=camera_worker,
            args=(stream_url, interval_seconds),
            name="esp32-camera-worker",
            daemon=True,
        )
        camera_thread.start()

    app.logger.info(
        "Started camera monitoring at %s (YOLO every %ss)",
        stream_url,
        interval_seconds,
    )
    return jsonify(
        {
            "success": True,
            "message": "Camera monitoring started.",
            "camera": public_camera_state(),
        }
    ), 202


@app.post("/api/camera/stop")
def camera_stop_route():
    camera_stop.set()
    update_camera_state(running=False)
    return jsonify({"success": True, "message": "Camera stop requested."}), 200


# ---------------------------------------------------------------------------
# Local MJPEG proxy consumed by the dashboard
# ---------------------------------------------------------------------------

@app.get("/camera/stream")
def dashboard_camera_stream():
    """Proxy the latest worker frames to the dashboard as MJPEG."""

    @stream_with_context
    def generate():
        last_sequence = -1
        idle_after_stop = 0

        while True:
            with camera_lock:
                jpeg = camera_state["frame_jpeg"]
                sequence = camera_state["frame_sequence"]
                running = camera_state["running"]

            if jpeg is not None and sequence != last_sequence:
                last_sequence = sequence
                idle_after_stop = 0

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
            else:
                time.sleep(0.05)
                if not running:
                    idle_after_stop += 1
                    if idle_after_stop >= 20:
                        break

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Event history and uploaded images
# ---------------------------------------------------------------------------

@app.get("/api/events")
def events():
    try:
        limit = max(1, min(int(request.args.get("limit", "50")), 200))
    except ValueError:
        limit = 50

    return jsonify({"events": list_events(limit)})


@app.post("/api/events/clear")
def clear_events():
    try:
        with db() as con:
            con.execute("DELETE FROM events")
            con.commit()

        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and file_path.name != ".gitkeep":
                try:
                    file_path.unlink()
                except OSError:
                    app.logger.warning("Could not remove %s", file_path)

        app.logger.info("Database and uploaded records cleared successfully.")
        return jsonify(
            {"success": True, "message": "All detection records cleared."}
        ), 200

    except Exception as exc:
        app.logger.exception("Failed to clear database")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(
        {"success": False, "error": "Image exceeds 10 MB limit."}
    ), 413


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

init_db()

if __name__ == "__main__":
    # LAN binding is useful for device integration. Keep this development
    # server on a trusted private network; do not expose it to the internet.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
