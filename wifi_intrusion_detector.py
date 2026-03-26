"""
WatchWave — AI-Driven Wi-Fi Intrusion Monitoring System v4.0
Author: Mazher Anvar Edassery Sadath
Student ID: 32146631 | BSc (Hons) Cyber Security | UWL
Supervisor: Dr. Terry Jacob | Module: CP6UA46O

v4.0 Additions:
  - Email alerts for HIGH severity attacks via Gmail
  - Alert cooldown to prevent email spam
"""

import time
import random
import threading
import logging
import json
import os
import smtplib
import psutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from plyer import notification as plyer_notify
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

try:
    from scapy.all import sniff, Dot11, Dot11Deauth, Dot11Beacon, Dot11ProbeReq
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
CONFIG = {
    "mode":               "simulation",
    "interface":          "wlan0mon",
    "log_file":           "intrusion_log.json",
    "fingerprint_file":   "device_fingerprints.json",
    "performance_file":   "performance_log.json",
    "alert_cooldown_sec": 10,
    "window_size":        10,
    "model_path":         "rf_model.pkl",
    "scaler_path":        "scaler.pkl",
    "fp_learn_windows":   5,
    "fp_anomaly_thresh":  2.5,
}

# ── Email Configuration ────────────────────────────────────
EMAIL_CONFIG = {
    "enabled":       True,
    "sender":        "aitrymail2@gmail.com",
    "app_password":  "jaun cvgu bkbb wxok",
    "recipient":     "aitrymail2@gmail.com",
    "smtp_server":   "smtp.gmail.com",
    "smtp_port":     587,
    "cooldown_sec":  60,   # minimum seconds between emails for same MAC
    "only_high":     True, # only email HIGH severity alerts
}

ATTACK_LABELS = {
    0: "Normal",
    1: "Deauth Flood",
    2: "Rogue Access Point",
    3: "MAC Spoofing",
    4: "Probe Request Flood",
}

SEVERITY = {0: "INFO", 1: "HIGH", 2: "HIGH", 3: "MEDIUM", 4: "MEDIUM"}

# ══════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ids_runtime.log")]
)
logger = logging.getLogger("WiFi-IDS")

FEATURE_NAMES = [
    "deauth_ratio", "beacon_ratio", "probe_ratio", "unique_mac_ratio",
    "avg_rssi", "rssi_variance", "conn_ratio", "frame_rate"
]

# ══════════════════════════════════════════════════════════════
# EMAIL ALERT ENGINE
# ══════════════════════════════════════════════════════════════
class EmailAlertEngine:
    def __init__(self):
        self.last_sent = {}
        self.sent_count = 0
        self.failed_count = 0

    def _should_send(self, mac):
        now = time.time()
        if mac in self.last_sent:
            if now - self.last_sent[mac] < EMAIL_CONFIG["cooldown_sec"]:
                return False
        self.last_sent[mac] = now
        return True

    def send(self, mac, attack_name, severity, confidence, timestamp, features):
        if not EMAIL_CONFIG["enabled"]:
            return
        if EMAIL_CONFIG["only_high"] and severity != "HIGH":
            return
        if not self._should_send(mac):
            return

        def _send_thread():
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"🚨 WatchWave Alert: {attack_name} Detected [{severity}]"
                msg["From"]    = EMAIL_CONFIG["sender"]
                msg["To"]      = EMAIL_CONFIG["recipient"]

                conf_pct = f"{confidence*100:.1f}%"

                html = f"""
<html><body style="font-family:Arial,sans-serif;background:#06090f;color:#dce8f8;padding:24px;">
  <div style="max-width:600px;margin:0 auto;background:#0c1220;border:1px solid #162035;padding:24px;">
    <h1 style="color:#00e5ff;font-size:22px;margin:0 0 4px;">WatchWave</h1>
    <p style="color:#3d5070;font-size:11px;margin:0 0 20px;">
      AI-Driven Wi-Fi Intrusion Monitoring System
    </p>
    <div style="border-left:4px solid #ff3d71;padding:16px;background:#070b14;margin-bottom:20px;">
      <h2 style="color:#ff3d71;margin:0 0 12px;font-size:16px;">
        🚨 INTRUSION ALERT [{severity}]
      </h2>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <tr><td style="color:#3d5070;padding:6px 0;width:140px;">Attack Type</td>
            <td style="color:#dce8f8;font-weight:bold;">{attack_name}</td></tr>
        <tr><td style="color:#3d5070;padding:6px 0;">MAC Address</td>
            <td style="color:#00e5ff;font-family:monospace;">{mac}</td></tr>
        <tr><td style="color:#3d5070;padding:6px 0;">Severity</td>
            <td style="color:#ff3d71;font-weight:bold;">{severity}</td></tr>
        <tr><td style="color:#3d5070;padding:6px 0;">Confidence</td>
            <td style="color:#dce8f8;">{conf_pct}</td></tr>
        <tr><td style="color:#3d5070;padding:6px 0;">Timestamp</td>
            <td style="color:#dce8f8;">{timestamp}</td></tr>
      </table>
    </div>
    <div style="background:#070b14;padding:14px;font-size:11px;color:#3d5070;margin-bottom:20px;">
      <strong style="color:#dce8f8;">Feature Values:</strong><br/><br/>
      deauth_ratio={features[0]:.3f} &nbsp;|&nbsp;
      probe_ratio={features[2]:.3f} &nbsp;|&nbsp;
      beacon_ratio={features[1]:.3f}<br/>
      avg_rssi={features[4]:.1f} &nbsp;|&nbsp;
      rssi_variance={features[5]:.2f} &nbsp;|&nbsp;
      frame_rate={features[7]:.1f}
    </div>
    <p style="color:#3d5070;font-size:10px;margin:0;">
      Mazher Anvar &nbsp;|&nbsp; BSc Cyber Security &nbsp;|&nbsp;
      University of West London &nbsp;|&nbsp; ID: 32146631
    </p>
  </div>
</body></html>"""

                msg.attach(MIMEText(html, "html"))

                with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                    server.starttls()
                    server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["app_password"])
                    server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["recipient"], msg.as_string())

                self.sent_count += 1
                logger.info(f"[EMAIL] Alert sent → {EMAIL_CONFIG['recipient']} | {attack_name} | {mac}")

            except Exception as e:
                self.failed_count += 1
                logger.error(f"[EMAIL] Failed to send: {e}")

        threading.Thread(target=_send_thread, daemon=True).start()

# ══════════════════════════════════════════════════════════════
# DEVICE FINGERPRINTING ENGINE
# ══════════════════════════════════════════════════════════════
class DeviceFingerprinter:
    def __init__(self):
        self.profiles = defaultdict(lambda: {
            "rssi_values":       [],
            "frame_types":       defaultdict(int),
            "first_seen":        None,
            "last_seen":         None,
            "observation_count": 0,
            "is_known":          False,
            "flagged_count":     0,
        })
        self._lock = threading.Lock()

    def update(self, mac, rssi, frame_type):
        with self._lock:
            p   = self.profiles[mac]
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if p["first_seen"] is None:
                p["first_seen"] = now
            p["last_seen"] = now
            p["rssi_values"].append(rssi)
            if len(p["rssi_values"]) > 100:
                p["rssi_values"].pop(0)
            p["frame_types"][frame_type] += 1

    def record_window(self, mac):
        with self._lock:
            p = self.profiles[mac]
            p["observation_count"] += 1
            if p["observation_count"] >= CONFIG["fp_learn_windows"]:
                p["is_known"] = True

    def check_anomaly(self, mac, current_rssi, current_frame_type):
        with self._lock:
            p = self.profiles[mac]
            if not p["is_known"] or len(p["rssi_values"]) < 5:
                return False, None
            rssi_arr  = np.array(p["rssi_values"][:-1])
            rssi_mean = rssi_arr.mean()
            rssi_std  = rssi_arr.std() if rssi_arr.std() > 0 else 1.0
            z_score   = abs(current_rssi - rssi_mean) / rssi_std
            if z_score > CONFIG["fp_anomaly_thresh"]:
                p["flagged_count"] += 1
                return True, (
                    f"RSSI anomaly — current={current_rssi:.1f}dBm "
                    f"vs profile mean={rssi_mean:.1f}±{rssi_std:.1f}dBm (z={z_score:.2f})"
                )
            total            = sum(p["frame_types"].values()) or 1
            expected_ratio   = p["frame_types"].get(current_frame_type, 0) / total
            if expected_ratio < 0.02 and current_frame_type in ("deauth",):
                p["flagged_count"] += 1
                return True, (
                    f"Unexpected frame type '{current_frame_type}' "
                    f"from previously normal device (ratio={expected_ratio:.3f})"
                )
            return False, None

    def get_summary(self):
        with self._lock:
            return {
                mac: {
                    "first_seen":        p["first_seen"],
                    "last_seen":         p["last_seen"],
                    "observation_count": p["observation_count"],
                    "is_known":          p["is_known"],
                    "avg_rssi":          round(float(np.mean(p["rssi_values"])), 2) if p["rssi_values"] else None,
                    "flagged_count":     p["flagged_count"],
                    "top_frame_type":    max(p["frame_types"], key=p["frame_types"].get) if p["frame_types"] else None,
                }
                for mac, p in self.profiles.items()
            }

    def save(self):
        try:
            with open(CONFIG["fingerprint_file"], "w") as f:
                json.dump(self.get_summary(), f, indent=2)
        except Exception as e:
            logger.error(f"Fingerprint save failed: {e}")

# ══════════════════════════════════════════════════════════════
# PERFORMANCE MONITOR
# ══════════════════════════════════════════════════════════════
class PerformanceMonitor:
    def __init__(self):
        self.records        = []
        self.false_positives = 0
        self.true_positives  = 0
        self.normal_count    = 0
        self._lock           = threading.Lock()

    def record(self, latency_ms, label, cpu_pct, mem_mb):
        with self._lock:
            self.records.append({
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latency_ms": round(latency_ms, 2),
                "label":      ATTACK_LABELS[label],
                "cpu_pct":    round(cpu_pct, 1),
                "mem_mb":     round(mem_mb, 1),
            })
            if label == 0:
                self.normal_count   += 1
            else:
                self.true_positives += 1

    def get_stats(self):
        with self._lock:
            if not self.records:
                return {}
            latencies = [r["latency_ms"] for r in self.records]
            cpus      = [r["cpu_pct"]    for r in self.records]
            mems      = [r["mem_mb"]     for r in self.records]
            return {
                "total_detections":  len(self.records),
                "attack_detections": self.true_positives,
                "normal_detections": self.normal_count,
                "false_positives":   self.false_positives,
                "avg_latency_ms":    round(float(np.mean(latencies)), 2),
                "min_latency_ms":    round(float(np.min(latencies)),  2),
                "max_latency_ms":    round(float(np.max(latencies)),  2),
                "avg_cpu_pct":       round(float(np.mean(cpus)), 1),
                "avg_mem_mb":        round(float(np.mean(mems)), 1),
            }

    def save(self):
        try:
            with open(CONFIG["performance_file"], "w") as f:
                json.dump({"stats": self.get_stats(), "records": self.records[-100:]}, f, indent=2)
        except Exception as e:
            logger.error(f"Performance save failed: {e}")

# ══════════════════════════════════════════════════════════════
# TRAINING DATA
# ══════════════════════════════════════════════════════════════
def generate_training_data(n_samples=10000):
    X, y = [], []
    profiles = {
        0: dict(dr=(0.00,0.02), br=(0.40,0.60), pr=(0.05,0.15), mr=(0.05,0.15),
                rssi=(55,75),   rv=(2,8),        cr=(0.01,0.05), fps=(8,25)),
        1: dict(dr=(0.55,0.85), br=(0.05,0.15), pr=(0.01,0.05), mr=(0.01,0.05),
                rssi=(60,80),   rv=(1,5),        cr=(0.01,0.03), fps=(60,120)),
        2: dict(dr=(0.00,0.02), br=(0.70,0.90), pr=(0.02,0.08), mr=(0.01,0.03),
                rssi=(75,95),   rv=(1,3),        cr=(0.10,0.20), fps=(35,65)),
        3: dict(dr=(0.00,0.02), br=(0.30,0.50), pr=(0.02,0.08), mr=(0.01,0.02),
                rssi=(45,65),   rv=(20,45),      cr=(0.10,0.20), fps=(15,35)),
        4: dict(dr=(0.00,0.02), br=(0.05,0.15), pr=(0.60,0.85), mr=(0.15,0.35),
                rssi=(40,62),   rv=(5,18),       cr=(0.01,0.04), fps=(50,100)),
    }
    per_class = n_samples // len(profiles)
    for label, p in profiles.items():
        for _ in range(per_class):
            noise = random.uniform(0.95, 1.05)
            row = [
                min(1.0, max(0.0, random.uniform(*p["dr"]) * noise)),
                min(1.0, max(0.0, random.uniform(*p["br"]) * noise)),
                min(1.0, max(0.0, random.uniform(*p["pr"]) * noise)),
                min(1.0, max(0.0, random.uniform(*p["mr"]) * noise)),
                random.uniform(*p["rssi"]),
                random.uniform(*p["rv"]),
                min(1.0, max(0.0, random.uniform(*p["cr"]) * noise)),
                random.uniform(*p["fps"]),
            ]
            X.append(row); y.append(label)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return np.array(X), np.array(y)

def load_or_train_model():
    if os.path.exists(CONFIG["model_path"]) and os.path.exists(CONFIG["scaler_path"]):
        logger.info("Loading existing model...")
        return joblib.load(CONFIG["model_path"]), joblib.load(CONFIG["scaler_path"])
    logger.info("Training model...")
    X, y   = generate_training_data()
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    clf    = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2,
                                    random_state=42, class_weight="balanced", n_jobs=-1)
    clf.fit(Xs, y)
    joblib.dump(clf, CONFIG["model_path"])
    joblib.dump(scaler, CONFIG["scaler_path"])
    return clf, scaler

# ══════════════════════════════════════════════════════════════
# ALERT ENGINE
# ══════════════════════════════════════════════════════════════
class AlertEngine:
    def __init__(self, email_engine):
        self.last_alert   = {}
        self.log_entries  = []
        self.email_engine = email_engine

    def _should_alert(self, mac):
        now = time.time()
        if mac in self.last_alert and now - self.last_alert[mac] < CONFIG["alert_cooldown_sec"]:
            return False
        self.last_alert[mac] = now
        return True

    def send_alert(self, mac, label, confidence, features,
                   latency_ms=0, fp_anomaly=False, fp_reason=None):
        if not self._should_alert(mac):
            return
        attack_name = ATTACK_LABELS[label]
        severity    = SEVERITY[label]
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conf_pct    = f"{confidence*100:.1f}%"

        print("\n" + "═"*65)
        print(f"  🚨  INTRUSION ALERT [{severity}]")
        print(f"  Attack Type  : {attack_name}")
        print(f"  MAC Address  : {mac}")
        print(f"  Confidence   : {conf_pct}")
        print(f"  Latency      : {latency_ms:.1f}ms")
        print(f"  Timestamp    : {timestamp}")
        if fp_anomaly:
            print(f"  🔍 FP FLAG   : {fp_reason}")
        print(f"  deauth={features[0]:.2f} | probe={features[2]:.2f} | "
              f"beacon={features[1]:.2f} | rssi={features[4]:.1f}")
        print("═"*65 + "\n")

        if PLYER_AVAILABLE:
            try:
                plyer_notify.notify(
                    title=f"⚠️ WatchWave: {attack_name}",
                    message=f"Severity: {severity}\nMAC: {mac}\nConf: {conf_pct}",
                    app_name="WatchWave IDS", timeout=8,
                )
            except Exception:
                pass

        # Send email for HIGH severity
        self.email_engine.send(mac, attack_name, severity, confidence, timestamp, features)

        entry = {
            "timestamp":   timestamp, "mac": mac,
            "attack_type": attack_name, "severity": severity,
            "confidence":  round(confidence, 4), "latency_ms": round(latency_ms, 2),
            "fp_flagged":  fp_anomaly, "fp_reason": fp_reason,
            "features":    dict(zip(FEATURE_NAMES, [round(float(f), 4) for f in features]))
        }
        self.log_entries.append(entry)
        try:
            with open(CONFIG["log_file"], "w") as f:
                json.dump(self.log_entries, f, indent=2)
        except Exception as e:
            logger.error(f"Log write failed: {e}")

    def send_fingerprint_alert(self, mac, reason):
        if not self._should_alert(f"fp_{mac}"):
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "─"*65)
        print(f"  🔍  DEVICE FINGERPRINT ANOMALY")
        print(f"  MAC: {mac} | Reason: {reason}")
        print(f"  Time: {timestamp}")
        print("─"*65 + "\n")
        entry = {
            "timestamp": timestamp, "mac": mac,
            "attack_type": "Device Fingerprint Anomaly", "severity": "MEDIUM",
            "confidence": 0.0, "latency_ms": 0,
            "fp_flagged": True, "fp_reason": reason, "features": {}
        }
        self.log_entries.append(entry)
        try:
            with open(CONFIG["log_file"], "w") as f:
                json.dump(self.log_entries, f, indent=2)
        except Exception as e:
            logger.error(f"Log write failed: {e}")

    def log_normal(self, mac, features, latency_ms):
        logger.info(f"[NORMAL] MAC={mac} | fps={features[7]:.1f} | latency={latency_ms:.1f}ms")

# ══════════════════════════════════════════════════════════════
# PACKET WINDOW BUFFER
# ══════════════════════════════════════════════════════════════
class PacketWindowBuffer:
    def __init__(self, window_sec=10):
        self.window_sec = window_sec
        self.frames     = deque()
        self._lock      = threading.Lock()

    def add_frame(self, frame):
        with self._lock:
            self.frames.append((time.time(), frame))

    def extract_features(self):
        with self._lock:
            now    = time.time()
            cutoff = now - self.window_sec
            while self.frames and self.frames[0][0] < cutoff:
                self.frames.popleft()
            if not self.frames:
                return None
            window = list(self.frames)

        total   = len(window)
        elapsed = max(window[-1][0] - window[0][0], 0.1)
        deauth  = sum(1 for _, f in window if f["type"] == "deauth")
        beacon  = sum(1 for _, f in window if f["type"] == "beacon")
        probe   = sum(1 for _, f in window if f["type"] == "probe")
        conn    = sum(1 for _, f in window if f["type"] == "assoc")
        macs    = set(f["mac"] for _, f in window)
        rssi_v  = [f["rssi"] for _, f in window]

        features = np.array([
            deauth / total, beacon / total, probe / total,
            len(macs) / total, float(np.mean(rssi_v)), float(np.var(rssi_v)),
            conn / total, total / elapsed,
        ])
        mac_counts   = defaultdict(int)
        for _, f in window:
            mac_counts[f["mac"]] += 1
        dominant_mac = max(mac_counts, key=mac_counts.get)
        return features, dominant_mac, window

# ══════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════
KNOWN_MACS  = [f"AA:BB:CC:DD:EE:{i:02X}" for i in range(1, 6)]
ATTACK_MACS = {
    "deauth": "DE:AD:BE:EF:00:01",
    "rogue":  "R0:GU:EA:PP:00:01",
    "spoof":  KNOWN_MACS[0],
    "probe":  "PR:0B:E0:00:00:01",
}

def normal_pkt():
    return {"type": random.choice(["beacon","beacon","beacon","data","assoc"]),
            "mac": random.choice(KNOWN_MACS), "rssi": random.uniform(55, 75)}

def attack_pkt(attack):
    return {
        "deauth": {"type":"deauth","mac":ATTACK_MACS["deauth"],"rssi":random.uniform(60,80)},
        "rogue":  {"type":"beacon","mac":ATTACK_MACS["rogue"], "rssi":random.uniform(80,96)},
        "spoof":  {"type":"assoc", "mac":ATTACK_MACS["spoof"], "rssi":random.uniform(30,50)},
        "probe":  {"type":"probe", "mac":f"PR:{random.randint(0,99):02X}:E0:{random.randint(0,99):02X}:00:01",
                   "rssi":random.uniform(40,62)},
    }.get(attack, normal_pkt())

def run_simulation(buffer):
    logger.info("Simulation started.")
    attacks = ["deauth", "rogue", "spoof", "probe"]
    cycle   = 0
    while True:
        cycle += 1
        if cycle % 3 == 0:
            atk = random.choice(attacks)
            dur = random.randint(10, 18)
            logger.info(f"[SIM] ⚠️  Injecting: {atk.upper()} ({dur}s)")
            end = time.time() + dur
            while time.time() < end:
                pkt = attack_pkt(atk) if random.random() < 0.95 else normal_pkt()
                buffer.add_frame(pkt)
                time.sleep(random.uniform(0.005, 0.03))
        else:
            dur = random.randint(12, 22)
            logger.info(f"[SIM] Normal traffic ({dur}s)...")
            end = time.time() + dur
            while time.time() < end:
                buffer.add_frame(normal_pkt())
                time.sleep(random.uniform(0.08, 0.25))

# ══════════════════════════════════════════════════════════════
# DETECTION LOOP
# ══════════════════════════════════════════════════════════════
def detection_loop(clf, scaler, buffer, alert_engine, fingerprinter, perf_monitor):
    logger.info("Detection loop running. Analysing every 5 seconds...")
    process = psutil.Process(os.getpid())
    while True:
        time.sleep(5)
        window_start = time.time()
        result       = buffer.extract_features()
        if result is None:
            continue
        features, mac, window = result

        for _, frame in window:
            fingerprinter.update(frame["mac"], frame["rssi"], frame["type"])
        fingerprinter.record_window(mac)
        fingerprinter.save()

        fp_anomaly, fp_reason = fingerprinter.check_anomaly(
            mac, current_rssi=features[4],
            current_frame_type="deauth" if features[0] > 0.3 else "beacon"
        )

        features_scaled = scaler.transform([features])
        label           = clf.predict(features_scaled)[0]
        proba           = clf.predict_proba(features_scaled)[0]
        confidence      = proba[label]
        latency_ms      = (time.time() - window_start) * 1000
        cpu_pct         = psutil.cpu_percent(interval=None)
        mem_mb          = process.memory_info().rss / (1024 * 1024)

        perf_monitor.record(latency_ms, label, cpu_pct, mem_mb)
        perf_monitor.save()

        if label == 0:
            alert_engine.log_normal(mac, features, latency_ms)
            if fp_anomaly:
                alert_engine.send_fingerprint_alert(mac, fp_reason)
        else:
            alert_engine.send_alert(mac, label, confidence, features,
                                     latency_ms=latency_ms,
                                     fp_anomaly=fp_anomaly,
                                     fp_reason=fp_reason)

        stats = perf_monitor.get_stats()
        if stats.get("total_detections", 0) % 10 == 0 and stats.get("total_detections", 0) > 0:
            logger.info(
                f"[PERF] avg_latency={stats['avg_latency_ms']}ms | "
                f"cpu={stats['avg_cpu_pct']}% | mem={stats['avg_mem_mb']}MB"
            )

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("  WatchWave — AI Wi-Fi Intrusion Monitoring System v4.0")
    print("  Mazher Anvar | BSc Cyber Security | UWL | ID: 32146631")
    print(f"  Mode: {CONFIG['mode'].upper()}")
    print(f"  Email alerts: {'ENABLED' if EMAIL_CONFIG['enabled'] else 'DISABLED'}")
    print("="*65 + "\n")

    clf, scaler    = load_or_train_model()
    buffer         = PacketWindowBuffer(window_sec=CONFIG["window_size"])
    email_engine   = EmailAlertEngine()
    alert_engine   = AlertEngine(email_engine)
    fingerprinter  = DeviceFingerprinter()
    perf_monitor   = PerformanceMonitor()

    src = threading.Thread(target=run_simulation, args=(buffer,), daemon=True)
    src.start()
    logger.info("Packet source started (simulation mode).")

    try:
        detection_loop(clf, scaler, buffer, alert_engine, fingerprinter, perf_monitor)
    except KeyboardInterrupt:
        stats = perf_monitor.get_stats()
        print(f"\n{'='*65}")
        print("  SESSION SUMMARY")
        print(f"  Total Detections : {stats.get('total_detections', 0)}")
        print(f"  Attacks Detected : {stats.get('attack_detections', 0)}")
        print(f"  Avg Latency      : {stats.get('avg_latency_ms', 0)}ms")
        print(f"  Avg CPU Usage    : {stats.get('avg_cpu_pct', 0)}%")
        print(f"  Avg Memory       : {stats.get('avg_mem_mb', 0)}MB")
        print(f"  Emails Sent      : {email_engine.sent_count}")
        print(f"  Devices Profiled : {len(fingerprinter.get_summary())}")
        print(f"{'='*65}")

if __name__ == "__main__":
    main()
