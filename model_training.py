"""
=============================================================
  model_training.py
  Fake Website Detection System
  Role: Train multiple classifiers, evaluate, pick the best,
        save model + scaler + feature importance plot.
=============================================================
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       confusion_matrix, classification_report,
                                       roc_auc_score)

from data_preprocessing  import load_or_generate_dataset, split_and_scale
from feature_extraction  import extract_features_dataframe

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Palette (cyberpunk-ish, matches the UI)
# ─────────────────────────────────────────────────────────────
CLR_LEGIT = "#00e5ff"   # cyan
CLR_FAKE  = "#ff1744"   # red
CLR_BG    = "#0d0f14"
CLR_CARD  = "#151821"
CLR_TEXT  = "#e0e6f0"
CLR_GRID  = "#1e2330"


# ─────────────────────────────────────────────────────────────
# 1. Candidate models
# ─────────────────────────────────────────────────────────────
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1,
        max_depth=5, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=1.0
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", probability=True, random_state=42
    ),
}


# ─────────────────────────────────────────────────────────────
# 2. Evaluation helper
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score (y_test, y_pred),           4),
        "Precision": round(precision_score(y_test, y_pred),           4),
        "Recall":    round(recall_score   (y_test, y_pred),           4),
        "F1 Score":  round(f1_score       (y_test, y_pred),           4),
        "ROC-AUC":   round(roc_auc_score  (y_test, y_proba),          4),
    }
    return metrics


# ─────────────────────────────────────────────────────────────
# 3. Plotting helpers
# ─────────────────────────────────────────────────────────────
def _base_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_CARD)
    ax.tick_params(colors=CLR_TEXT, labelsize=9)
    ax.xaxis.label.set_color(CLR_TEXT)
    ax.yaxis.label.set_color(CLR_TEXT)
    ax.title.set_color(CLR_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(CLR_GRID)
    ax.grid(color=CLR_GRID, linestyle="--", linewidth=0.5)
    return fig, ax


def plot_feature_importance(model, feature_names: list, path: str):
    """Bar chart of feature importances (Random Forest / GBM)."""
    if not hasattr(model, "feature_importances_"):
        print("[Plot] Model has no feature_importances_ – skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(15, len(feature_names))
    idx         = indices[:top_n]

    fig, ax = _base_fig(figsize=(11, 6))
    colors  = [CLR_LEGIT if importances[i] > np.median(importances) else CLR_FAKE
               for i in idx]
    bars    = ax.barh([feature_names[i] for i in idx][::-1],
                      importances[idx][::-1],
                      color=colors[::-1], edgecolor="none", height=0.65)

    # Value labels
    for bar, val in zip(bars, importances[idx][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=8, color=CLR_TEXT)

    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("🔍  Top Feature Importances", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, importances[idx].max() * 1.18)

    leg = [mpatches.Patch(color=CLR_LEGIT, label="High importance"),
           mpatches.Patch(color=CLR_FAKE,  label="Lower importance")]
    ax.legend(handles=leg, facecolor=CLR_CARD,
              labelcolor=CLR_TEXT, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Feature importance saved → {path}")


def plot_confusion_matrix(model, X_test, y_test, path: str):
    """Styled confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    labels = ["Fake", "Legitimate"]

    fig, ax = _base_fig(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest",
                   cmap=plt.cm.Blues, vmin=0, vmax=cm.max())

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=CLR_TEXT)

    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="True label",
           title="Confusion Matrix")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}",
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i,j] > thresh else CLR_TEXT,
                    fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Confusion matrix saved → {path}")


def plot_model_comparison(results: list, path: str):
    """Grouped bar chart comparing all candidate models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    names   = [r["Model"] for r in results]
    x       = np.arange(len(metrics))
    width   = 0.18

    fig, ax = _base_fig(figsize=(12, 5))
    palette = [CLR_LEGIT, CLR_FAKE, "#ffab00", "#ce93d8"]

    for i, (result, color) in enumerate(zip(results, palette)):
        vals = [result[m] for m in metrics]
        ax.bar(x + i * width, vals, width,
               label=result["Model"], color=color,
               alpha=0.85, edgecolor="none")

    ax.set_xticks(x + width * (len(results)-1) / 2)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("📊  Model Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=CLR_CARD, labelcolor=CLR_TEXT, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Model comparison saved → {path}")


# ─────────────────────────────────────────────────────────────
# 4. Main training pipeline
# ─────────────────────────────────────────────────────────────
def train(csv_path: str = None, dns_check: bool = False):
    """
    End-to-end training pipeline.
    1. Load dataset
    2. Extract features
    3. Scale
    4. Train all candidate models
    5. Evaluate and pick best by F1
    6. Save best model + plots + metrics report
    """
    print("\n" + "="*60)
    print("  FAKE WEBSITE DETECTION — TRAINING PIPELINE")
    print("="*60 + "\n")

    # ── Step 1: Load data ─────────────────────────────────
    df = load_or_generate_dataset(csv_path)

    # ── Step 2: Feature extraction ────────────────────────
    print("\n[Step 2] Extracting features …")
    X = extract_features_dataframe(df["url"], dns_check=dns_check)
    y = df["label"].reset_index(drop=True)
    feature_names = list(X.columns)

    # ── Step 3: Split & scale ─────────────────────────────
    print("\n[Step 3] Splitting & scaling …")
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    # ── Step 4 & 5: Train + evaluate ──────────────────────
    print("\n[Step 4] Training candidate models …\n")
    results   = []
    trained   = {}
    for name, clf in MODELS.items():
        print(f"  → Training {name} …", end="", flush=True)
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test, name)
        results.append(metrics)
        trained[name] = clf
        print(f"  Accuracy={metrics['Accuracy']:.4f}  "
              f"F1={metrics['F1 Score']:.4f}  "
              f"AUC={metrics['ROC-AUC']:.4f}")

    # ── Summary table ─────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "-"*60)
    print(results_df.to_string())
    print("-"*60)

    # ── Pick best model by F1 Score ───────────────────────
    best_name  = results_df["F1 Score"].idxmax()
    best_model = trained[best_name]
    print(f"\n✅  Best model: {best_name}  "
          f"(F1={results_df.loc[best_name,'F1 Score']:.4f})")

    # ── Classification report ─────────────────────────────
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred,
                                   target_names=["Fake","Legitimate"])
    print(f"\nClassification Report ({best_name}):\n{report}")

    # ── Step 6: Save model & artifacts ───────────────────
    model_path = "models/best_model.pkl"
    meta_path  = "models/model_meta.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    meta = {"feature_names": feature_names,
            "model_name":    best_name,
            "metrics":       results_df.loc[best_name].to_dict()}
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n[Save] Model  → {model_path}")
    print(f"[Save] Meta   → {meta_path}")

    # ── Plots ─────────────────────────────────────────────
    plot_feature_importance(best_model, feature_names,
                            "reports/feature_importance.png")
    plot_confusion_matrix  (best_model, X_test, y_test,
                            "reports/confusion_matrix.png")
    plot_model_comparison  (results, "reports/model_comparison.png")

    # Save metrics CSV
    results_df.to_csv("reports/model_metrics.csv")
    print("[Save] Metrics → reports/model_metrics.csv")

    print("\n🎉  Training complete!\n")
    return best_model, feature_names, results_df


if __name__ == "__main__":
    train()
