import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Configure headless matplotlib backend before importing pyplot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "xgboost_hybrid")
FUSED_BASE = os.path.join(PROJECT_ROOT, "processed_data", "fused_features")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "model_evaluation")

MODELS_CONFIG = [
    ("highway_2", "full", "Final model tested on Highway 2"),
    ("highway_7", "full", "Final model tested on Highway 7"),
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return float(numerator / denominator) if denominator > 0 else default


def load_model_and_metadata(highway: str, variant: str) -> Tuple[xgb.XGBClassifier, Dict]:
    """Loads the trained XGBoost model and its associated metadata JSON."""
    model_path = os.path.join(MODELS_DIR, f"xgb_{highway}_{variant}.json")
    meta_path = os.path.join(MODELS_DIR, f"metadata_{highway}_{variant}.json")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Model metadata not found at: {meta_path}")

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return clf, meta


def load_test_data(highway: str, variant: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads test features (X) and ground truth labels (y) from verified NPZ archives."""
    test_dir = os.path.join(FUSED_BASE, highway, "test")
    npz_path = os.path.join(test_dir, "fused_arrays.npz")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Test feature archive not found at: {npz_path}")

    with np.load(npz_path) as npz:
        if variant == "full":
            X = npz["X_full"]
            expected_dim = 71
        elif variant == "ablation":
            X = npz["X_ablation"]
            expected_dim = 68
        else:
            raise ValueError(f"Unknown variant '{variant}'. Expected 'full' or 'ablation'.")

        y = npz["y"].astype(int)
        window_ids = npz["window_ids"] if "window_ids" in npz else []

    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch for {highway}_{variant}: "
            f"Expected {expected_dim} features, found {X.shape[1]}."
        )
    if len(X) != len(y):
        raise ValueError(f"Sample count mismatch: {len(X)} feature rows vs {len(y)} labels.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError(f"Test feature matrix for {highway}_{variant} contains NaN or Infinite values.")
    if np.isnan(y).any():
        raise ValueError(f"Test labels for {highway}_{variant} contain NaN values.")

    return X, y, window_ids


def evaluate_single_model(highway: str, variant: str, title: str) -> Dict:
    """
    Performs full evaluation of one model variant on the held-out test split.
    Calculates comprehensive performance metrics and confusion matrix values.
    """
    clf, meta = load_model_and_metadata(highway, variant)
    X_test, y_test, _ = load_test_data(highway, variant)

    n_samples = len(y_test)
    n_features = X_test.shape[1]

    # Extract verified threshold from metadata (selected on validation split)
    threshold = meta.get("selected_threshold", 0.50)
    thresh_source = "Validation-Selected Optimal Threshold"

    # Predict continuous probabilities and binary labels
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix counts
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Core performance metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    spec = safe_div(tn, tn + fp)
    fpr = safe_div(fp, tn + fp)
    fnr = safe_div(fn, tp + fn)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Area Under Curve metrics
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    # Classification report
    clf_report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Benign (Normal)", "DoS Attack"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return {
        "title": title,
        "experiment_key": f"{highway}_{variant}",
        "highway": highway,
        "variant": variant,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "threshold": float(threshold),
        "threshold_source": thresh_source,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        "classification_report_dict": clf_report_dict,
    }


def plot_confusion_matrix(cm_dict: Dict, highway: str, variant: str, threshold: float, output_path: str) -> None:
    """Saves confusion matrix heatmap plot to disk silently."""
    if not PLOTTING_AVAILABLE:
        return

    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    cm_data = np.array([[tn, fp], [fn, tp]])
    total = np.sum(cm_data)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    labels = np.array([
        [f"True Negative\n{tn:,}\n({tn/total*100:.2f}%)", f"False Positive\n{fp:,}\n({fp/total*100:.2f}%)"],
        [f"False Negative\n{fn:,}\n({fn/total*100:.2f}%)", f"True Positive\n{tp:,}\n({tp/total*100:.2f}%)"],
    ])

    sns.heatmap(
        cm_data,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold", "linespacing": 1.4},
        linewidths=1.5,
        linecolor="white",
    )

    hw_title = "Highway 2 (250 Vehicles)" if highway == "highway_2" else "Highway 7 (500 Vehicles)"
    var_title = "Full Hybrid" if variant == "full" else "Autonomous Ablation"

    ax.set_title(
        f"Confusion Matrix: {hw_title} — {var_title}\n(Decision Threshold T* = {threshold:.2f})",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Class (Ground Truth)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xticklabels(["Benign Traffic (0)", "DoS Attack (1)"], fontsize=10)
    ax.set_yticklabels(["Benign Traffic (0)", "DoS Attack (1)"], fontsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_evaluation_results(results_list: List[Dict], output_dir: str) -> None:
    """Exports summary results to CSV, JSON, and saves confusion matrix plots silently."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).isoformat()

    summary_rows = []
    for r in results_list:
        m = r["metrics"]
        cm = r["confusion_matrix"]
        summary_rows.append({
            "evaluation_timestamp": timestamp_str,
            "experiment_key": r["experiment_key"],
            "model_title": r["title"],
            "highway": r["highway"],
            "variant": r["variant"],
            "feature_count": r["n_features"],
            "test_sample_count": r["n_samples"],
            "decision_threshold": r["threshold"],
            "threshold_source": r["threshold_source"],
            "accuracy": round(m["accuracy"], 6),
            "balanced_accuracy": round(m["balanced_accuracy"], 6),
            "precision": round(m["precision"], 6),
            "recall_detection_rate": round(m["recall"], 6),
            "specificity": round(m["specificity"], 6),
            "false_positive_rate": round(m["false_positive_rate"], 6),
            "false_negative_rate": round(m["false_negative_rate"], 6),
            "f1_score": round(m["f1_score"], 6),
            "roc_auc": round(m["roc_auc"], 6),
            "pr_auc": round(m["pr_auc"], 6),
            "true_negatives_tn": cm["tn"],
            "false_positives_fp": cm["fp"],
            "false_negatives_fn": cm["fn"],
            "true_positives_tp": cm["tp"],
        })

    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, "evaluation_summary.json")
    json_payload = {
        "evaluation_timestamp": timestamp_str,
        "results": {r["experiment_key"]: r for r in results_list},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    if PLOTTING_AVAILABLE:
        for r in results_list:
            plot_path = os.path.join(output_dir, f"confusion_matrix_{r['highway']}_{r['variant']}.png")
            plot_confusion_matrix(
                cm_dict=r["confusion_matrix"],
                highway=r["highway"],
                variant=r["variant"],
                threshold=r["threshold"],
                output_path=plot_path,
            )


def print_clean_model_report(res: Dict) -> None:
    """
    Prints strictly the requested minimalist format:
    Model Name

    Precision = ...%
    Recall = ...%
    F1-score = ...%
    Accuracy = ...%
    ROC-AUC = ...
    PR-AUC = ...
    Specificity = ...%
    False Positive Rate = ...%
    False Negative Rate = ...%

    Confusion Matrix
                     Predicted Normal    Predicted DoS
    Actual Normal               11770              135
    Actual DoS                    121            10017
    """
    m = res["metrics"]
    cm = res["confusion_matrix"]

    print(f"{res['title']}\n")
    print(f"Precision = {m['precision']*100:.2f}%")
    print(f"Recall = {m['recall']*100:.2f}%")
    print(f"F1-score = {m['f1_score']*100:.2f}%")
    print(f"Accuracy = {m['accuracy']*100:.2f}%")
    print(f"ROC-AUC = {m['roc_auc']:.4f}")
    print(f"PR-AUC = {m['pr_auc']:.4f}")
    print(f"Specificity = {m['specificity']*100:.2f}%")
    print(f"False Positive Rate = {m['false_positive_rate']*100:.2f}%")
    print(f"False Negative Rate = {m['false_negative_rate']*100:.2f}%\n")
    print("Confusion Matrix")
    print("                 Predicted Normal    Predicted DoS")
    print(f"{'Actual Normal':<17s}{cm['tn']:>12d}{cm['fp']:>17d}")
    print(f"{'Actual DoS':<17s}{cm['fn']:>12d}{cm['tp']:>17d}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure clean UTF-8 rendering on all terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Clean model evaluation runner for 5G-V2X DoS Detection."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save evaluation summaries and plots silently",
    )
    args = parser.parse_args()

    results_list = []
    for hw, var, title in MODELS_CONFIG:
        res = evaluate_single_model(highway=hw, variant=var, title=title)
        results_list.append(res)

    # Print clean format for each model
    for i, res in enumerate(results_list):
        if i > 0:
            print()
        print_clean_model_report(res)

    # Save artifacts silently in background
    save_evaluation_results(results_list, args.output_dir)


if __name__ == "__main__":
    main()
