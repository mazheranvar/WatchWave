"""
WatchWave — FastAPI Backend Server v3.0
Mazher Anvar | BSc Cyber Security | UWL | ID: 32146631

New endpoints in v3:
  GET  /devices       - Device fingerprint profiles
  GET  /alerts/export - Download alerts as CSV
  GET  /health        - System health check
  GET  /performance   - Performance metrics from log
"""

import json, os, csv, io, asyncio
from datetime import datetime
from typing import List
from collections import Counter

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI(
    title="WatchWave API",
    description="AI-Driven Wi-Fi Intrusion Monitoring System v3.0",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

LOG_FILE         = "intrusion_log.json"
FINGERPRINT_FILE = "device_fingerprints.json"
PERFORMANCE_FILE = "performance_log.json"

# ── WebSocket Manager ──────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# ── Helpers ────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_alerts():
    data = load_json(LOG_FILE)
    return data if isinstance(data, list) else []

last_seen_count = 0

async def watch_log_and_broadcast():
    global last_seen_count
    while True:
        await asyncio.sleep(2)
        alerts = load_alerts()
        if len(alerts) > last_seen_count:
            for alert in alerts[last_seen_count:]:
                await manager.broadcast({"event": "new_alert", "data": alert})
            last_seen_count = len(alerts)

@app.on_event("startup")
async def startup_event():
    global last_seen_count
    last_seen_count = len(load_alerts())
    asyncio.create_task(watch_log_and_broadcast())
    print("[API] WatchWave API v3.0 started.")

# ── REST Endpoints ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "WatchWave API", "version": "3.0.0",
        "status": "running", "author": "Mazher Anvar",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health")
def health():
    """System health check — uptime, file status, client count."""
    alerts = load_alerts()
    fps    = load_json(FINGERPRINT_FILE) or {}
    perf   = load_json(PERFORMANCE_FILE) or {}
    return {
        "status":          "healthy",
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_alerts":    len(alerts),
        "devices_profiled":len(fps),
        "ws_clients":      len(manager.active),
        "log_file_exists": os.path.exists(LOG_FILE),
        "fp_file_exists":  os.path.exists(FINGERPRINT_FILE),
        "avg_latency_ms":  perf.get("stats", {}).get("avg_latency_ms"),
        "avg_cpu_pct":     perf.get("stats", {}).get("avg_cpu_pct"),
        "avg_mem_mb":      perf.get("stats", {}).get("avg_mem_mb"),
    }

@app.get("/alerts")
def get_all_alerts():
    alerts = load_alerts()
    return {"total": len(alerts), "alerts": alerts}

@app.get("/alerts/latest")
def get_latest_alerts(limit: int = 10):
    alerts = load_alerts()
    return {"total": len(alerts), "alerts": alerts[-limit:][::-1]}

@app.get("/alerts/export")
def export_alerts_csv():
    """Download all alerts as a CSV file."""
    alerts = load_alerts()
    if not alerts:
        return {"error": "No alerts to export"}
    output  = io.StringIO()
    headers = ["timestamp","mac","attack_type","severity","confidence",
               "latency_ms","fp_flagged","fp_reason"]
    writer  = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(alerts)
    output.seek(0)
    filename = f"watchwave_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/stats")
def get_stats():
    alerts = load_alerts()
    if not alerts:
        return {"total_alerts": 0, "attack_breakdown": {},
                "severity_breakdown": {}, "top_macs": [], "last_alert": None}
    attack_counts   = Counter(a["attack_type"] for a in alerts)
    severity_counts = Counter(a["severity"]    for a in alerts)
    mac_counts      = Counter(a["mac"]         for a in alerts)
    return {
        "total_alerts":       len(alerts),
        "attack_breakdown":   dict(attack_counts),
        "severity_breakdown": dict(severity_counts),
        "top_macs":           [{"mac": m, "count": c} for m, c in mac_counts.most_common(5)],
        "last_alert":         alerts[-1] if alerts else None,
        "first_seen":         alerts[0]["timestamp"]  if alerts else None,
        "last_seen":          alerts[-1]["timestamp"] if alerts else None,
    }

@app.get("/status")
def get_status():
    alerts  = load_alerts()
    active  = False
    last_ts = None
    if alerts:
        last_ts = alerts[-1]["timestamp"]
        try:
            diff   = (datetime.now() - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
            active = diff < 60
        except Exception:
            pass
    return {
        "detector_active": active, "total_alerts": len(alerts),
        "last_alert_time": last_ts, "log_file": LOG_FILE,
        "ws_clients": len(manager.active),
    }

@app.get("/devices")
def get_devices():
    """Return device fingerprint profiles."""
    fps = load_json(FINGERPRINT_FILE) or {}
    known   = sum(1 for d in fps.values() if d.get("is_known"))
    flagged = sum(1 for d in fps.values() if d.get("flagged_count", 0) > 0)
    return {
        "total_devices":   len(fps),
        "known_devices":   known,
        "learning_devices":len(fps) - known,
        "flagged_devices": flagged,
        "devices":         fps,
    }

@app.get("/performance")
def get_performance():
    """Return system performance metrics."""
    perf = load_json(PERFORMANCE_FILE)
    if not perf:
        return {"error": "No performance data yet — run the detector first"}
    return perf

# ── WebSocket ──────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await manager.connect(ws)
    alerts = load_alerts()
    if alerts:
        await ws.send_json({"event": "history", "data": alerts[-5:][::-1]})
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"event": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  WatchWave API Server v3.0")
    print("  Mazher Anvar | BSc Cyber Security | UWL")
    print("="*55)
    print("  API:        http://localhost:8000")
    print("  Health:     http://localhost:8000/health")
    print("  Devices:    http://localhost:8000/devices")
    print("  Export CSV: http://localhost:8000/alerts/export")
    print("  Docs:       http://localhost:8000/docs")
    print("  WebSocket:  ws://localhost:8000/ws/live")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


# ── /compare endpoint (add this to api_server.py) ─────────
@app.get("/compare")
def get_model_comparison():
    """
    Returns RF vs SVM model comparison results.
    Based on model_comparison.py evaluation run.
    """
    return {
        "comparison": {
            "random_forest": {
                "accuracy":    1.00,
                "precision":   1.00,
                "recall":      1.00,
                "f1_score":    1.00,
                "cv_accuracy": 1.00,
                "cv_std":      0.00,
                "train_time_s":  0.40,
                "infer_time_ms": 0.0223,
                "selected":    True,
                "reason": "Equal accuracy with better explainability via feature importance"
            },
            "svm_rbf": {
                "accuracy":    1.00,
                "precision":   1.00,
                "recall":      1.00,
                "f1_score":    1.00,
                "cv_accuracy": 1.00,
                "cv_std":      0.00,
                "train_time_s":  0.10,
                "infer_time_ms": 0.0047,
                "selected":    False,
                "reason": "Faster inference but lacks feature importance interpretability"
            }
        },
        "winner": "Random Forest",
        "justification": (
            "Both models achieved 100% accuracy. Random Forest was selected "
            "as the primary model for WatchWave due to its feature importance "
            "scores, which provide explainability for security analysts, and "
            "its robustness to noisy real-world data. Although SVM demonstrated "
            "faster inference (0.0047ms vs 0.0223ms per sample), both speeds "
            "are well within real-time detection requirements."
        ),
        "dataset": {
            "total_samples":    10000,
            "samples_per_class": 2000,
            "train_split":      0.75,
            "test_split":       0.25,
            "classes":          5,
            "features":         8
        }
    }
