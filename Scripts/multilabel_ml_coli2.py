"""
Multi-Label / Multiclass-Multioutput Machine Learning Pipeline
==============================================================
Designed to be run from a Jupyter notebook via:
    %run -i multilabel_ml.py

Uses the variables in the notebook namespace:
    X_train, X_test  — feature matrices (DataFrame)
    y_train, y_test                       — label matrices   (DataFrame)

Feature names    → inferred from X_train.columns
Antibiotic names → inferred from y_train.columns 

Covers:
  1. Data ingestion from notebook namespace
  2. Base model benchmarking (5 algorithms)
  3. Hyper-parameter tuning (RandomizedSearchCV)
  4. Feature importance analysis
  5. Per-antibiotic report for the best model
  6. Full results export to ./Outputs/
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score,
    classification_report, jaccard_score,
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR   = "/home/kyomukama/Documents/MSBT/Sem2/ML/ML_group_work/Outputs/coli_ml_results2"
os.makedirs(OUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INGEST FROM NOTEBOOK NAMESPACE
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(arr):
    return arr.values if isinstance(arr, pd.DataFrame) else np.asarray(arr)

def _get_names(arr, prefix):
    if isinstance(arr, pd.DataFrame):
        return list(arr.columns)
    n = np.asarray(arr).shape[1] if np.asarray(arr).ndim > 1 else 1
    return [f"{prefix}_{i+1}" for i in range(n)]

_ns      = globals()   # %run -i executes in the caller's global scope
_missing = [v for v in ["X_train","X_test","y_train","y_test"] if v not in _ns]

if _missing:
    raise NameError(
        f"\n[multilabel_ml.py] Missing variables in notebook: {_missing}\n"
        "Define X_train, X_test, y_train, y_test first, then:\n"
        "    %run -i multilabel_ml.py"
    )

_feature_names = _get_names(_ns["X_train"], "feature")
_target_names  = _get_names(_ns["y_train"], "antibiotic")

_X_train = _to_numpy(_ns["X_train"])
_X_test  = _to_numpy(_ns["X_test"])
_y_train = _to_numpy(_ns["y_train"])
_y_test  = _to_numpy(_ns["y_test"])

print("=" * 65)
print("  MULTI-LABEL ML PIPELINE  —  using notebook variables")
print("=" * 65)
print(f"  X_train : {_X_train.shape}    X_test : {_X_test.shape}")
print(f"  y_train : {_y_train.shape}    y_test : {_y_test.shape}")
print(f"  Antibiotics ({len(_target_names)}): {_target_names}")
print(f"  Features    ({len(_feature_names)}): {_feature_names[:5]}{'…' if len(_feature_names)>5 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    return {
        "model":           name,
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss":    hamming_loss(y_true, y_pred),
        "f1_micro":        f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "jaccard_micro":   jaccard_score(y_true, y_pred, average="micro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BASE MODELS
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "ExtraTreeClassifier":    ExtraTreeClassifier(random_state=RANDOM_STATE),
    "ExtraTreesClassifier":   ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "KNeighborsClassifier":   KNeighborsClassifier(n_jobs=-1),
}


def run_base_models(X_tr, X_te, y_tr, y_te):
    results, trained = [], {}
    for name, clf in MODELS.items():
        print(f"  Fitting {name} …", end=" ", flush=True)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        m = evaluate(name, y_te, y_pred)
        results.append(m)
        trained[name] = clf
        print(f"F1-micro={m['f1_micro']:.4f}  hamming={m['hamming_loss']:.4f}")
    return pd.DataFrame(results).set_index("model"), trained


def plot_base_results(df):
    metrics = ["subset_accuracy", "hamming_loss", "f1_micro", "f1_macro", "jaccard_micro"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    fig.suptitle("Base Model Comparison", fontsize=14, fontweight="bold")
    palette = sns.color_palette("Set2", len(df))

    for ax, metric in zip(axes, metrics):
        vals = df[metric]
        bars = ax.barh(df.index, vals, color=palette)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlim(0, max(vals) * 1.18)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"01_base_model_comparison_{TIMESTAMP}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HYPER-PARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "DecisionTreeClassifier": {
        "max_depth":         [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", None],
    },
    "ExtraTreeClassifier": {
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", None],
    },
    "ExtraTreesClassifier": {
        "n_estimators":      [50, 100, 200, 300],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features":      ["sqrt", "log2"],
    },
    "RandomForestClassifier": {
        "n_estimators":      [50, 100, 200, 300],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features":      ["sqrt", "log2"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan", "minkowski"],
        "p":           [1, 2],
    },
}


def tune_models(X_tr, y_tr, base_trained):
    tuned, records = {}, []
    for name, clf in base_trained.items():
        print(f"  Tuning {name} …", end=" ", flush=True)
        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=PARAM_GRIDS[name],
            n_iter=20, scoring="f1_micro",
            cv=3, refit=True, n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_tr, y_tr)
        print(f"CV F1-micro={search.best_score_:.4f}  params={search.best_params_}")
        tuned[name] = search.best_estimator_
        records.append({
            "model":       name,
            "cv_f1_micro": search.best_score_,
            "best_params": json.dumps(search.best_params_),
        })
    return tuned, pd.DataFrame(records).set_index("model")


def eval_tuned(tuned, X_te, y_te):
    return pd.DataFrame(
        [evaluate(n, y_te, clf.predict(X_te)) for n, clf in tuned.items()]
    ).set_index("model")


def plot_base_vs_tuned(df_base, df_tuned):
    metric = "f1_micro"
    models = df_base.index.tolist()
    x, w   = np.arange(len(models)), 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, df_base[metric],  w, label="Base",  color="#4C72B0")
    ax.bar(x + w/2, df_tuned[metric], w, label="Tuned", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("F1-micro")
    ax.set_title("Base vs Tuned — F1-micro")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, (b, t) in enumerate(zip(df_base[metric], df_tuned[metric])):
        ax.text(i - w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, t + 0.01, f"{t:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"02_base_vs_tuned_{TIMESTAMP}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(tuned, feature_names):
    tree_models = {k: v for k, v in tuned.items() if hasattr(v, "feature_importances_")}
    if not tree_models:
        print("  No tree models with feature_importances_.")
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 7))
    if n == 1:
        axes = [axes]
    fig.suptitle("Feature Importance — Tuned Tree Models", fontsize=14, fontweight="bold")

    for ax, (name, clf) in zip(axes, tree_models.items()):
        importances = clf.feature_importances_
        top_n = min(20, len(feature_names))
        idx   = np.argsort(importances)[::-1][:top_n]
        ax.barh(
            [feature_names[i] for i in idx[::-1]],
            importances[idx[::-1]],
            color=sns.color_palette("viridis", top_n),
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Importance")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"03_feature_importance_{TIMESTAMP}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def save_importance_csv(tuned, feature_names):
    tree_models = {k: v for k, v in tuned.items() if hasattr(v, "feature_importances_")}
    rows = [
        {"model": name, "feature": feat, "importance": imp}
        for name, clf in tree_models.items()
        for feat, imp in zip(feature_names, clf.feature_importances_)
    ]
    df   = pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])
    path = os.path.join(OUT_DIR, f"feature_importances_{TIMESTAMP}.csv")
    df.to_csv(path, index=False)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PER-ANTIBIOTIC REPORT  (best model)
# ─────────────────────────────────────────────────────────────────────────────

def per_label_report(tuned, X_te, y_te, target_names):
    best_name, best_score, best_pred = None, -1, None
    for name, clf in tuned.items():
        y_pred = clf.predict(X_te)
        s = f1_score(y_te, y_pred, average="micro", zero_division=0)
        if s > best_score:
            best_score, best_name, best_pred = s, name, y_pred

    print(f"\n  ★  Best model : {best_name}  (F1-micro = {best_score:.4f})")

    report = classification_report(y_te, best_pred,
                                   target_names=target_names, zero_division=0)
    path = os.path.join(OUT_DIR, f"best_model_report_{TIMESTAMP}.txt")
    with open(path, "w") as f:
        f.write(f"Best Model : {best_name}\n")
        f.write(f"F1-micro   : {best_score:.4f}\n\n")
        f.write(report)
    print(f"  [saved] {path}")

    f1_per  = f1_score(y_te, best_pred, average=None, zero_division=0)
    fig_w   = max(10, len(target_names) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))
    sns.heatmap(
        f1_per.reshape(1, -1),
        annot=True, fmt=".3f", cmap="YlGn",
        xticklabels=target_names, yticklabels=[best_name],
        ax=ax, vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title(f"Per-Antibiotic F1 Score — {best_name}", fontweight="bold")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, f"04_per_antibiotic_f1_{TIMESTAMP}.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SUMMARY CSVs
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(df_base, df_tuned, tune_df):
    df_b = df_base.copy();  df_b["stage"]  = "base"
    df_t = df_tuned.copy(); df_t["stage"] = "tuned"
    summary = pd.concat([df_b, df_t])
    p1 = os.path.join(OUT_DIR, f"model_results_summary_{TIMESTAMP}.csv")
    p2 = os.path.join(OUT_DIR, f"tuning_best_params_{TIMESTAMP}.csv")
    summary.to_csv(p1)
    tune_df.to_csv(p2)
    print(f"  [saved] {p1}")
    print(f"  [saved] {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/4] Base model benchmarking …")
df_base, base_trained = run_base_models(_X_train, _X_test, _y_train, _y_test)
print("\n  Base Results:\n", df_base.to_string())
plot_base_results(df_base)

print("\n[2/4] Hyper-parameter tuning …")
tuned_models, tune_df = tune_models(_X_train, _y_train, base_trained)
df_tuned = eval_tuned(tuned_models, _X_test, _y_test)
print("\n  Tuned Results:\n", df_tuned.to_string())
plot_base_vs_tuned(df_base, df_tuned)

print("\n[3/4] Feature importance …")
plot_feature_importance(tuned_models, _feature_names)
save_importance_csv(tuned_models, _feature_names)

print("\n[4/4] Per-antibiotic report & summary …")
per_label_report(tuned_models, _X_test, _y_test, _target_names)
save_summary(df_base, df_tuned, tune_df)

print("\n" + "=" * 65)
print(f"  Done. All outputs saved to ./{OUT_DIR}/")
print("=" * 65)