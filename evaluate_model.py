"""
Model Evaluation Script v2.0
AI-Driven Wi-Fi Intrusion Monitoring System
Mazher Anvar | BSc Cyber Security | UWL

Uses ratio-based features matching the v2 detector exactly.
Expected confidence: 85-99%
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import random

ATTACK_LABELS = {0:"Normal", 1:"Deauth Flood", 2:"Rogue AP", 3:"MAC Spoof", 4:"Probe Flood"}
FEATURE_NAMES = ["deauth_ratio","beacon_ratio","probe_ratio","unique_mac_ratio",
                 "avg_rssi","rssi_variance","conn_ratio","frame_rate"]

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
            X.append(row)
            y.append(label)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return np.array(X), np.array(y)

def evaluate():
    print("Generating ratio-based dataset and training v2 model...")
    X, y = generate_training_data(n_samples=10000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=2,
        random_state=42, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    label_names = [ATTACK_LABELS[i] for i in sorted(ATTACK_LABELS)]
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=label_names))
    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring="accuracy")
    print(f"Overall Accuracy   : {acc*100:.2f}%")
    print(f"5-Fold CV Accuracy : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("WatchWave v2.0 — Model Evaluation", fontsize=13, fontweight="bold")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].tick_params(axis='x', rotation=20)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    axes[1].bar(range(len(FEATURE_NAMES)), importances[indices],
                color="steelblue", edgecolor="white")
    axes[1].set_xticks(range(len(FEATURE_NAMES)))
    axes[1].set_xticklabels([FEATURE_NAMES[i] for i in indices], rotation=30, ha="right")
    axes[1].set_title("Feature Importance")
    axes[1].set_ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → evaluation_results.png")
    plt.show()
    joblib.dump(clf, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model saved → rf_model.pkl / scaler.pkl")
    print("\nNow run: python3 wifi_intrusion_detector.py")

if __name__ == "__main__":
    evaluate()
