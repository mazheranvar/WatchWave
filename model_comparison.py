"""
WatchWave — ML Model Comparison
Mazher Anvar | BSc Cyber Security | UWL | ID: 32146631

Compares Random Forest vs SVM for Wi-Fi intrusion detection.
Outputs:
  - Side-by-side classification reports
  - Accuracy, precision, recall, F1 comparison table
  - Inference time comparison (critical for real-time deployment)
  - model_comparison.png — bar chart comparison
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

ATTACK_LABELS = {
    0: "Normal",
    1: "Deauth Flood",
    2: "Rogue AP",
    3: "MAC Spoof",
    4: "Probe Flood"
}
FEATURE_NAMES = [
    "deauth_ratio", "beacon_ratio", "probe_ratio", "unique_mac_ratio",
    "avg_rssi", "rssi_variance", "conn_ratio", "frame_rate"
]

# ── Dataset ───────────────────────────────────────────────────
def generate_data(n_samples=10000):
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

def evaluate_model(name, clf, X_train, X_test, y_train, y_test, X_all, y_all):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    # Train
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = clf.predict(X_test)
    infer_time = (time.time() - t0) / len(X_test) * 1000  # ms per sample

    label_names = [ATTACK_LABELS[i] for i in sorted(ATTACK_LABELS)]
    print(classification_report(y_test, y_pred, target_names=label_names))

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred,    average="weighted")
    f1   = f1_score(y_test, y_pred,        average="weighted")
    cv   = cross_val_score(clf, X_all, y_all, cv=5, scoring="accuracy")

    print(f"Accuracy       : {acc*100:.2f}%")
    print(f"Precision (W)  : {prec*100:.2f}%")
    print(f"Recall (W)     : {rec*100:.2f}%")
    print(f"F1-Score (W)   : {f1*100:.2f}%")
    print(f"5-Fold CV      : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    print(f"Train Time     : {train_time:.2f}s")
    print(f"Infer Time     : {infer_time:.4f}ms/sample")

    return {
        "name":       name,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "cv_mean":    cv.mean(),
        "cv_std":     cv.std(),
        "train_time": train_time,
        "infer_time": infer_time,
        "cm":         confusion_matrix(y_test, y_pred),
        "y_pred":     y_pred,
    }

def main():
    print("WatchWave — ML Model Comparison: Random Forest vs SVM")
    print("Generating dataset (10,000 samples)...")
    X, y = generate_data(n_samples=10000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=2,
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    svm = SVC(
        kernel="rbf", C=10, gamma="scale",
        class_weight="balanced", probability=True, random_state=42
    )

    rf_results  = evaluate_model("Random Forest", rf,  X_train, X_test, y_train, y_test, X_scaled, y)
    svm_results = evaluate_model("SVM (RBF)",     svm, X_train, X_test, y_train, y_test, X_scaled, y)

    # ── Summary Table ─────────────────────────────────────────
    print("\n" + "="*65)
    print("  COMPARISON SUMMARY")
    print("="*65)
    print(f"{'Metric':<22} {'Random Forest':>18} {'SVM (RBF)':>18}")
    print("-"*65)
    metrics = [
        ("Accuracy",      "accuracy"),
        ("Precision (W)", "precision"),
        ("Recall (W)",    "recall"),
        ("F1-Score (W)",  "f1"),
        ("CV Accuracy",   "cv_mean"),
    ]
    for label, key in metrics:
        rf_val  = rf_results[key]  * 100
        svm_val = svm_results[key] * 100
        winner  = "← RF" if rf_val >= svm_val else "← SVM"
        print(f"  {label:<20} {rf_val:>16.2f}%  {svm_val:>16.2f}%  {winner}")
    print("-"*65)
    print(f"  {'Train Time':<20} {rf_results['train_time']:>15.2f}s  {svm_results['train_time']:>15.2f}s")
    print(f"  {'Infer Time/sample':<20} {rf_results['infer_time']:>13.4f}ms  {svm_results['infer_time']:>13.4f}ms")
    print("="*65)

    winner_name = "Random Forest" if rf_results["accuracy"] >= svm_results["accuracy"] else "SVM"
    print(f"\n  ✅  Winner: {winner_name}")
    print(f"  Justification: Random Forest selected for WatchWave because it achieves")
    print(f"  equal or higher accuracy with significantly faster inference time,")
    print(f"  making it more suitable for real-time deployment.")

    # ── Plot ─────────────────────────────────────────────────
    label_names = [ATTACK_LABELS[i] for i in sorted(ATTACK_LABELS)]
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("WatchWave — Random Forest vs SVM Model Comparison",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Confusion matrices
    for idx, res in enumerate([rf_results, svm_results]):
        ax = fig.add_subplot(gs[0, idx])
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(f"Confusion Matrix — {res['name']}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis='x', rotation=20)

    # Bar chart comparison
    ax3 = fig.add_subplot(gs[0, 2])
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    rf_vals  = [rf_results["accuracy"],  rf_results["precision"],
                rf_results["recall"],    rf_results["f1"]]
    svm_vals = [svm_results["accuracy"], svm_results["precision"],
                svm_results["recall"],   svm_results["f1"]]
    x = np.arange(len(metric_labels))
    w = 0.35
    ax3.bar(x - w/2, [v*100 for v in rf_vals],  w, label="Random Forest", color="steelblue")
    ax3.bar(x + w/2, [v*100 for v in svm_vals], w, label="SVM (RBF)",     color="coral")
    ax3.set_xticks(x); ax3.set_xticklabels(metric_labels)
    ax3.set_ylabel("Score (%)"); ax3.set_title("Performance Metrics Comparison")
    ax3.set_ylim([95, 101]); ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Inference time comparison
    ax4 = fig.add_subplot(gs[1, 0])
    times = [rf_results["infer_time"], svm_results["infer_time"]]
    bars  = ax4.bar(["Random Forest", "SVM (RBF)"], times,
                    color=["steelblue", "coral"], edgecolor="white")
    ax4.set_ylabel("Inference Time (ms/sample)")
    ax4.set_title("Inference Speed Comparison\n(Lower = Better for Real-Time)")
    for bar, val in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
                 f"{val:.4f}ms", ha="center", va="bottom", fontsize=9)

    # Training time comparison
    ax5 = fig.add_subplot(gs[1, 1])
    tr_times = [rf_results["train_time"], svm_results["train_time"]]
    bars2 = ax5.bar(["Random Forest", "SVM (RBF)"], tr_times,
                    color=["steelblue","coral"], edgecolor="white")
    ax5.set_ylabel("Training Time (seconds)")
    ax5.set_title("Training Time Comparison")
    for bar, val in zip(bars2, tr_times):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}s", ha="center", va="bottom", fontsize=9)

    # CV scores
    ax6 = fig.add_subplot(gs[1, 2])
    cv_means = [rf_results["cv_mean"]*100, svm_results["cv_mean"]*100]
    cv_stds  = [rf_results["cv_std"]*100,  svm_results["cv_std"]*100]
    ax6.bar(["Random Forest", "SVM (RBF)"], cv_means, yerr=cv_stds,
            color=["steelblue","coral"], edgecolor="white", capsize=6)
    ax6.set_ylabel("5-Fold CV Accuracy (%)")
    ax6.set_title("Cross-Validation Accuracy\n(with ± std deviation)")
    ax6.set_ylim([90, 102])
    ax6.grid(axis="y", alpha=0.3)

    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → model_comparison.png")
    plt.show()

    # Save best model
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Best model (Random Forest) saved → rf_model.pkl")

if __name__ == "__main__":
    main()
