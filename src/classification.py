from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    recall_score, accuracy_score, cohen_kappa_score
)
from sklearn.cluster import KMeans


from Regression import (
    HORIZONS, START_DATE, END_DATE, VALIDATION_START,
    load_dataset_from_csv, apply_best_model_on_dataset,
    embargo_rows_from_horizon
)


# Project output paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "classification"
CSV_DIR = OUTPUT_ROOT / "csv"
FIG_DIR = OUTPUT_ROOT / "figures"
for d in (CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = OUTPUT_ROOT

METRICS_VAL_CSV = CSV_DIR / "metrics_validation.csv"
VAL_SIGNALS_CSV = CSV_DIR / "validation_signals.csv"
TEST_SIGNALS_CSV = CSV_DIR / "test_signals.csv"


INCLUDE_KMEANS: bool = True
# K-means configuration
KMEANS_K: int = 5
KMEANS_Q_LOW: float = 0.01
KMEANS_Q_HIGH: float = 0.99
KMEANS_RANDOM_STATE: int = 42


GRID_STEP: float = 0.01
LOW_MIN: float  = 0.05
HIGH_MAX: float = 0.95


PLOT_ROC: bool = True


EXCLUDED_ROC_HORIZONS = {60}


ENABLE_CUSTOM_BACKTEST: bool = False
CUSTOM_VALID_START = pd.Timestamp("2017-01-01")
CUSTOM_VALID_END   = pd.Timestamp("2018-12-31")
CUSTOM_TEST_START  = pd.Timestamp("2019-01-01")
CUSTOM_TEST_END    = pd.Timestamp("2023-12-31")


def _global_paths(run_tag: str | None = None) -> Tuple[Path, Path]:
    if run_tag:
        run_dir = OUTPUT_DIR / f"run_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = OUTPUT_DIR
    mdl = run_dir / "classif_logit_GLOBAL.joblib"
    meta = run_dir / "classif_logit_GLOBAL.meta.json"
    return mdl, meta

def _fig_path(name: str, run_tag: str | None = None, ext: str = "png") -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return FIG_DIR / f"{name}{suffix}.{ext}"


def _iv_mean(df: pd.DataFrame, h: int) -> pd.Series:
    c, p = f"call_iv_{h}", f"put_iv_{h}"
    if {c, p}.issubset(df.columns):
        return (df[c].astype(float) + df[p].astype(float)) * 0.5
    alt = f"iv_atm{h}"
    if alt in df.columns:
        return df[alt].astype(float)
    raise KeyError(f"IV ATM manquante pour H={h} (attendus: {c},{p} ou {alt}).")

def _rv_past(df: pd.DataFrame, h: int) -> pd.Series:
    col = f"RV{h}_past"
    if col not in df.columns:
        raise KeyError(f"{col} absent du CSV (requis pour la feature 'margin').")
    return df[col].astype(float).rename(col)


# Reuse fitted clusters
_KMEANS_CACHE: Dict[int, Dict[str, object]] = {}

def _features_vrp_skew_conv(data: pd.DataFrame, h: int) -> pd.DataFrame:
    pred_lr = apply_best_model_on_dataset(data, h).astype(float).rename("x_pred_logratio")
    iv = _iv_mean(data, h).astype(float).rename("x_iv")
    rvp = _rv_past(data, h).astype(float)
    vrp_pred = (np.exp(pred_lr) * rvp - iv).rename("VRP_pred")
    sk = f"skew_slope25_{h}"
    bf = f"bf25_{h}"
    if sk not in data.columns or bf not in data.columns:
        raise KeyError(f"Colonnes manquantes pour H={h}: {sk} et/ou {bf}")
    out = pd.DataFrame(index=pred_lr.index)
    out["date"] = data.loc[pred_lr.index, "date"]
    out["VRP_pred"] = vrp_pred.astype(float)
    out["skew"] = data.loc[pred_lr.index, sk].astype(float)
    out["convexity"] = data.loc[pred_lr.index, bf].astype(float)
    return out.dropna()

def _ensure_kmeans_and_get_distances(data: pd.DataFrame, h: int) -> pd.DataFrame:
    feats3 = ["VRP_pred", "skew", "convexity"]
    F = _features_vrp_skew_conv(data, h)
    if h not in _KMEANS_CACHE:
        base_mask = (F["date"] < VALIDATION_START)
        idx_train_all = np.flatnonzero(base_mask.values)
        gap = embargo_rows_from_horizon(h)
        idx_train = idx_train_all[:-gap] if idx_train_all.size > gap else idx_train_all
        train_mask = pd.Series(False, index=F.index)
        if idx_train.size:
            train_mask.iloc[idx_train] = True
        clip_bounds: Dict[str, Tuple[float, float]] = {}
        F_clip = F.copy()
        for c in feats3:
            q1, q99 = F.loc[train_mask, c].quantile([KMEANS_Q_LOW, KMEANS_Q_HIGH])
            clip_bounds[c] = (float(q1), float(q99))
            F_clip[c] = F[c].clip(lower=q1, upper=q99)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(F_clip.loc[train_mask, feats3].values)
        kmeans = KMeans(n_clusters=KMEANS_K, n_init=50, random_state=KMEANS_RANDOM_STATE).fit(X_train)
        _KMEANS_CACHE[h] = {"clip_bounds": clip_bounds, "scaler": scaler, "kmeans": kmeans}
    obj = _KMEANS_CACHE[h]
    clip_bounds = obj["clip_bounds"]
    scaler: StandardScaler = obj["scaler"]
    kmeans: KMeans = obj["kmeans"]
    F_clip_all = F.copy()
    for c in feats3:
        lo, hi = clip_bounds[c]
        F_clip_all[c] = F_clip_all[c].clip(lower=lo, upper=hi)
    X_all = scaler.transform(F_clip_all[feats3].values)
    dists_all  = kmeans.transform(X_all)
    D = pd.DataFrame(index=F.index)
    D["km_dmin"]  = dists_all.min(axis=1)
    for j in range(KMEANS_K):
        D[f"km_d{j+1}"] = dists_all[:, j]
    return D

# Build classifier features
def build_features_logit(df: pd.DataFrame, h: int, pooled: bool = True) -> pd.DataFrame:
    pred_lr = apply_best_model_on_dataset(df, h).astype(float).rename("x_pred_logratio")
    iv = _iv_mean(df, h).astype(float).rename("x_iv")
    rvp = _rv_past(df, h).astype(float)
    margin = (np.exp(pred_lr) * rvp - iv).rename("x_margin")
    X = pd.concat([pred_lr, iv, margin], axis=1)
    if INCLUDE_KMEANS:
        try:
            D = _ensure_kmeans_and_get_distances(df, h)
            X = X.join(D, how="inner")
        except KeyError as e:
            print(f"[KMEANS désactivé pour H={h}] {e}")
    if pooled:
        for hh in HORIZONS:
            X[f"H_{hh}"] = 1.0 if hh == h else 0.0
    return X.dropna()

# Binary volatility target
def build_binary_target(df: pd.DataFrame, h: int) -> pd.Series:
    rv = df[f"RV{h}"].astype(float)
    iv = _iv_mean(df, h)
    return (rv - iv > 0.0).astype(int)


# Saved classifier metadata
@dataclass
class GlobalClfMeta:
    model_name: str
    feat_names: List[str]
    thresholds: Tuple[float, float]
    val_error_decisions: float
    val_coverage: float
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    horizons: List[int]
    created_at: str

def _save_artifact(clf, feat_names: List[str],
                   thresholds: Tuple[float, float],
                   err: float, cov: float,
                   train_start: pd.Timestamp, train_end: pd.Timestamp,
                   val_start: pd.Timestamp, val_end: pd.Timestamp,
                   run_tag: str | None = None) -> None:
    mdl, meta = _global_paths(run_tag)
    joblib.dump({"model": clf, "feat_order": list(feat_names),
                 "best_low": float(thresholds[0]), "best_high": float(thresholds[1])}, mdl)
    m = GlobalClfMeta(
        model_name="logit_calibrated",
        feat_names=list(feat_names),
        thresholds=(float(thresholds[0]), float(thresholds[1])),
        val_error_decisions=float(err),
        val_coverage=float(cov),
        train_start=str(train_start.date()) if pd.notna(train_start) else "",
        train_end=str(train_end.date()) if pd.notna(train_end) else "",
        val_start=str(val_start.date()) if pd.notna(val_start) else "",
        val_end=str(val_end.date()) if pd.notna(val_end) else "",
        horizons=list(HORIZONS),
        created_at=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(asdict(m), f, ensure_ascii=False, indent=2)

def _load_artifact(run_tag: str | None = None):
    mdl, meta = _global_paths(run_tag)
    if not mdl.exists() or not meta.exists():
        raise FileNotFoundError("Modèle global introuvable. Entraîne d’abord.")
    payload = joblib.load(mdl)
    with open(meta, "r", encoding="utf-8") as f:
        meta_j = json.load(f)
    return payload, meta_j


def _make_logit(interactions=False, penalty="l2", C=1.0, class_weight=None) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if interactions:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)))
    steps.append(("logit", LogisticRegression(
        max_iter=3000,
        solver="liblinear" if penalty == "l1" else "lbfgs",
        penalty=penalty,
        C=C,
        class_weight=class_weight
    )))
    return Pipeline(steps)

# Calibrate predicted probabilities
def _fit_and_calibrate(X_tr: np.ndarray, y_tr: np.ndarray, base: Pipeline) -> CalibratedClassifierCV:
    base.fit(X_tr, y_tr)
    n_pos = int(np.sum(y_tr == 1)); n_neg = int(np.sum(y_tr == 0))
    n_splits = max(2, min(5, n_pos, n_neg))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    def _new_cal(method: str):
        try:
            return CalibratedClassifierCV(estimator=base, method=method, cv=cv)
        except TypeError:
            return CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)
    cal = _new_cal("isotonic")
    try:
        cal.fit(X_tr, y_tr)
    except ValueError:
        cal = _new_cal("sigmoid"); cal.fit(X_tr, y_tr)
    return cal


def _collect_pool_between(data: pd.DataFrame,
                          valid_start: pd.Timestamp,
                          valid_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    mask_tr = data["date"] < valid_start
    mask_va = (data["date"] >= valid_start) & (data["date"] <= valid_end)
    X_tr_list, y_tr_list, X_va_list, y_va_list = [], [], [], []
    for h in HORIZONS:
        X_all = build_features_logit(data, h, pooled=True)
        y_all = build_binary_target(data, h).astype(int)
        train_idx = data.index[mask_tr]
        gap = embargo_rows_from_horizon(h)
        if len(train_idx) > gap:
            train_idx = train_idx[:-gap]
        train_idx = X_all.index.intersection(train_idx, sort=False)
        valid_idx = X_all.index.intersection(data.index[mask_va], sort=False)
        X_tr = X_all.loc[train_idx]
        y_tr = y_all.loc[train_idx]
        X_va = X_all.loc[valid_idx]
        y_va = y_all.loc[valid_idx]
        X_tr_list.append(X_tr)
        y_tr_list.append(y_tr)
        X_va_list.append(X_va)
        y_va_list.append(y_va)
    X_train = pd.concat(X_tr_list, axis=0, ignore_index=False)
    y_train = pd.concat(y_tr_list, axis=0).astype(int)
    X_valid = pd.concat(X_va_list, axis=0, ignore_index=False)
    y_valid = pd.concat(y_va_list, axis=0).astype(int)
    m = (~X_train.isna().any(axis=1)) & (y_train.notna())
    X_train, y_train = X_train.loc[m], y_train.loc[m]
    m = (~X_valid.isna().any(axis=1)) & (y_valid.notna())
    X_valid, y_valid = X_valid.loc[m], y_valid.loc[m]
    return X_train, y_train, X_valid, y_valid


def _decision_error(p: np.ndarray, y_bin: np.ndarray, low: float, high: float) -> Tuple[float, float]:
    y_true = np.where(y_bin == 1, 1, -1)
    y_pred = np.zeros_like(p, dtype=int)
    y_pred[p < low] = -1
    y_pred[p >= high] = 1
    mask = (y_pred != 0)
    cov = mask.mean()
    if mask.sum() == 0:
        return float("nan"), 0.0
    err = float(np.mean(y_pred[mask] != y_true[mask]))
    return err, float(cov)

# Optimize no-trade thresholds
def search_best_thresholds_asym(p: np.ndarray, y: np.ndarray,
                                step: float = GRID_STEP,
                                low_min: float = LOW_MIN,
                                high_max: float = HIGH_MAX) -> Tuple[float, float, float, float]:
    lows = np.arange(low_min, 0.50, step)
    highs = np.arange(0.51, high_max + 1e-12, step)
    best = (np.inf, 0.0, 0.4, 0.6, 0.5)
    for lo in lows:
        for hi in highs:
            if lo >= hi:
                continue
            err, cov = _decision_error(p, y, lo, hi)
            if not np.isfinite(err):
                continue
            center = abs((lo + hi) / 2.0 - 0.5)
            better = (err < best[0]) or (np.isclose(err, best[0]) and (cov > best[1] or (np.isclose(cov, best[1]) and center < best[4])))
            if better:
                best = (err, cov, float(lo), float(hi), center)
    return best[0], best[1], best[2], best[3]

def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size >= 2 else float("nan")
    except Exception:
        return float("nan")

def _compute_binary_metrics_on_decided(y_true_bin: np.ndarray, p: np.ndarray, low: float, high: float) -> Dict[str, float]:

    y_pred = np.full_like(p, fill_value=-1, dtype=int)
    y_pred[p < low] = 0
    y_pred[p >= high] = 1
    mask = (y_pred != -1)
    coverage = float(mask.mean())
    if mask.sum() == 0:
        return {
            "coverage": 0.0, "accuracy": float("nan"), "f1": float("nan"), "kappa": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "n_pred_zero": int((y_pred == -1).sum()), "n_pred_pos": 0, "n_pred_neg": 0
        }
    yt = y_true_bin[mask]
    yp = y_pred[mask]
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp) if np.unique(yt).size > 1 else float("nan")
    kap = cohen_kappa_score(yt, yp) if np.unique(yt).size > 1 else float("nan")
    sens = recall_score(yt, yp, pos_label=1) if np.unique(yt).size > 1 else float("nan")
    spec = recall_score(yt, yp, pos_label=0) if np.unique(yt).size > 1 else float("nan")
    return {
        "coverage": coverage,
        "accuracy": float(acc),
        "f1": float(f1),
        "kappa": float(kap),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "n_pred_neg": int((y_pred == 0).sum()),
        "n_pred_zero": int((y_pred == -1).sum()),
        "n_pred_pos": int((y_pred == 1).sum()),
    }

def _metrics_table_by_horizon(X_va: pd.DataFrame, y_va: np.ndarray, p_va: np.ndarray, low: float, high: float) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        col = f"H_{h}"
        if col not in X_va.columns:
            continue
        m = (X_va[col].values == 1.0)
        if not m.any():
            continue
        auc = _roc_auc_safe(y_va[m], p_va[m])
        mbin = _compute_binary_metrics_on_decided(y_va[m], p_va[m], low, high)
        rows.append({
            "Horizon": h, "Low": low, "High": high,
            "AUC": auc,
            "Sensitivity (TPR)": mbin["sensitivity"],
            "Specificity (TNR)": mbin["specificity"],
            "Accuracy": mbin["accuracy"],
            "F1_score": mbin["f1"],
            "Kappa": mbin["kappa"],
            "Coverage": mbin["coverage"],
            "n_val": int(m.sum()),
            "n_pred_neg": mbin["n_pred_neg"],
            "n_pred_zero": mbin["n_pred_zero"],
            "n_pred_pos": mbin["n_pred_pos"],
        })
    return pd.DataFrame(rows)

def _plot_roc_curves(X_va: pd.DataFrame, y_va: np.ndarray, scores: np.ndarray,
                     run_tag: str | None = None) -> None:
    plt.figure(figsize=(7.2, 5.8))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    plotted = 0

    for i, h in enumerate(HORIZONS):
        if h in EXCLUDED_ROC_HORIZONS:
            continue
        col = f"H_{h}"
        if col not in X_va.columns:
            continue
        m = (X_va[col].values == 1.0)
        if not m.any() or np.unique(y_va[m]).size < 2:
            continue

        fpr, tpr, _ = roc_curve(y_va[m], scores[m])
        spec = 1.0 - fpr
        order = np.argsort(-spec)
        spec = spec[order]; tpr = tpr[order]

        plt.step(spec, tpr, where="post", lw=2.2, label=f"H={h}",
                 color=colors[i % len(colors)])
        plotted += 1

    if plotted == 0:
        print("[ROC] Pas assez de classes positives/négatives pour tracer les courbes.")
        plt.close()
        return


    plt.plot([1, 0], [0, 1], ls="--", lw=1, color="grey", label="Random")

    plt.xlabel("Specificity (1 − FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("ROC — Validation (par horizon)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    ax = plt.gca()
    ax.set_xlim(1.0, 0.0)

    fig_path = _fig_path("roc_validation_by_horizon", run_tag)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"[Figure ROC] -> {fig_path.resolve()}")


# Train one pooled classifier
def train_pooled_classifier_between(data: pd.DataFrame,
                                    valid_start: pd.Timestamp,
                                    valid_end: pd.Timestamp,
                                    run_tag: str | None = None) -> Dict[str, float]:
    X_tr, y_tr, X_va, y_va = _collect_pool_between(data, valid_start, valid_end)
    feat_order = list(X_tr.columns)

    candidates = [
        dict(interactions=False, penalty="l2", C=1.0, class_weight=None),
        dict(interactions=True,  penalty="l2", C=1.0, class_weight=None),
        dict(interactions=True,  penalty="l1", C=0.5, class_weight="balanced"),
    ]

    best_tuple = None
    for cfg in candidates:
        base = _make_logit(**cfg)
        clf = _fit_and_calibrate(X_tr.values, y_tr.values, base)
        p_va = clf.predict_proba(X_va.values)[:, 1]
        err, cov, low, high = search_best_thresholds_asym(p_va, y_va.values, step=GRID_STEP)
        if not np.isfinite(err):
            continue
        if (best_tuple is None) or (err < best_tuple[0]) or (np.isclose(err, best_tuple[0]) and cov > best_tuple[1]):
            best_tuple = (err, cov, low, high, clf, cfg)

    if best_tuple is None:
        raise RuntimeError("Échec entraînement: aucune config viable.")

    err, cov, low, high, best_clf, best_cfg = best_tuple


    p_va_best = best_clf.predict_proba(X_va.values)[:, 1]
    auc_global = _roc_auc_safe(y_va.values, p_va_best)


    train_start = data.loc[data["date"] < valid_start, "date"].min()
    max_gap = max(embargo_rows_from_horizon(h) for h in HORIZONS)
    idx_train = np.flatnonzero((data["date"] < valid_start).values)
    if idx_train.size > max_gap:
        idx_train = idx_train[:-max_gap]
    train_end = data["date"].iloc[idx_train[-1]] if idx_train.size else pd.NaT

    _save_artifact(best_clf, feat_order, (low, high), err, cov,
                   train_start, train_end, valid_start, valid_end, run_tag=run_tag)


    tbl = _metrics_table_by_horizon(X_va, y_va.values, p_va_best, low, high)
    out_csv = METRICS_VAL_CSV if run_tag is None else (CSV_DIR / f"metrics_validation_{run_tag}.csv")
    tbl.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[Metrics CSV] -> {out_csv.resolve()}")


    if PLOT_ROC:
        _plot_roc_curves(X_va, y_va.values, p_va_best, run_tag=run_tag)


    print("\n=== Meilleur modèle GLOBAL (pooled, calibré) ===")
    print(f"Seuils [low, high] = [{low:.2f}, {high:.2f}]  |  err_décisions={100*err:.2f}%  |  couverture={100*cov:.1f}%")
    print(f"Config: {best_cfg}")
    print("\n=== AUC ROC (validation) ===")
    print(f"Global AUC={auc_global:.4f}")
    for h in HORIZONS:
        col = f"H_{h}"
        if col in X_va.columns:
            m = (X_va[col].values == 1.0)
            if m.any():
                auc_h = _roc_auc_safe(y_va.values[m], p_va_best[m])
                n_h  = int(m.sum())
                print(f"  H={h:2d}  | AUC={auc_h:.4f} | n_val={n_h}")

    return {
        "err_decisions": float(err),
        "coverage": float(cov),
        "low": float(low),
        "high": float(high),
        "auc_global": float(auc_global)
    }


def predict_proba_global(df: pd.DataFrame, horizon: int, run_tag: str | None = None) -> pd.Series:
    payload, _ = _load_artifact(run_tag)
    clf = payload["model"]
    feat_order = payload["feat_order"]
    X = build_features_logit(df, horizon, pooled=True).astype(float)
    X = X.reindex(columns=feat_order, fill_value=0.0)
    p = clf.predict_proba(X.values)[:, 1]
    return pd.Series(index=X.index, data=p, name=f"proba_pos_GLOBAL_H{horizon}")

def apply_saved_classifier(df: pd.DataFrame, horizon: int, run_tag: str | None = None) -> pd.Series:
    payload, _ = _load_artifact(run_tag)
    low = float(payload.get("best_low", 0.40))
    high = float(payload.get("best_high", 0.60))
    p = predict_proba_global(df, horizon, run_tag=run_tag)
    yhat = np.zeros_like(p.values, dtype=int)
    yhat[p.values < low] = -1
    yhat[p.values >= high] = +1
    return pd.Series(index=p.index, data=yhat, name=f"yhat3c_global_H{horizon}")

# Return probabilities and trade signal
def predict_with_details(df: pd.DataFrame, horizon: int,
                         low: Optional[float] = None, high: Optional[float] = None,
                         run_tag: str | None = None) -> pd.DataFrame:
    payload, _ = _load_artifact(run_tag)
    low_g = float(payload.get("best_low", 0.40))
    high_g = float(payload.get("best_high", 0.60))
    if low is None:  low  = low_g
    if high is None: high = high_g
    p = predict_proba_global(df, horizon, run_tag=run_tag)
    p_buy = p.values
    p_sell = 1.0 - p_buy
    decision = np.zeros_like(p_buy, dtype=int)
    decision[p_buy < low]  = -1
    decision[p_buy >= high] =  1
    out = pd.DataFrame({
        "p_buy": p_buy,
        "p_sell": p_sell,
        "decision": decision,
    }, index=p.index)
    out.name = f"inference_H{horizon}"
    return out


def plot_decision_map_best(data: pd.DataFrame, horizon: int, grid: int = 300, run_tag: str | None = None) -> None:
    payload, _ = _load_artifact(run_tag)
    clf = payload["model"]
    feat_order = payload["feat_order"]
    low, high = float(payload["best_low"]), float(payload["best_high"])

    mask_va = (data["date"] >= VALIDATION_START) & (data["date"] <= END_DATE)
    Xval = build_features_logit(data, horizon, pooled=True).loc[mask_va]
    if Xval.empty:
        print(f"[H={horizon}] Pas de données validation après dropna.")
        return

    z = (data[f"RV{horizon}"].astype(float) - _iv_mean(data, horizon)).loc[Xval.index].values
    y_sign = np.where(z < 0.0, -1, +1)

    f1, f2 = "x_pred_logratio", "x_iv"
    x1min, x1max = np.quantile(Xval[f1], [0.01, 0.99])
    x2min, x2max = np.quantile(Xval[f2], [0.01, 0.99])
    gx1, gx2 = np.meshgrid(np.linspace(x1min, x1max, grid),
                           np.linspace(x2min, x2max, grid))
    rv_past_med = float(np.median(_rv_past(data, horizon).loc[Xval.index].values))

    skew_col = f"skew_slope25_{horizon}"
    conv_col = f"bf25_{horizon}"
    skew_med = float(data.loc[Xval.index, skew_col].median()) if skew_col in data.columns else 0.0
    conv_med = float(data.loc[Xval.index, conv_col].median()) if conv_col in data.columns else 0.0

    grid_df = pd.DataFrame({
        f1: gx1.ravel(),
        f2: gx2.ravel(),
        "x_margin": (np.exp(gx1.ravel()) * rv_past_med - gx2.ravel()),
    })
    for hh in HORIZONS:
        grid_df[f"H_{hh}"] = 1.0 if hh == horizon else 0.0

    if INCLUDE_KMEANS and (skew_col in data.columns) and (conv_col in data.columns):
        try:
            _ = _ensure_kmeans_and_get_distances(data, horizon)
            obj = _KMEANS_CACHE[horizon]
            clip_bounds = obj["clip_bounds"]; scaler = obj["scaler"]; kmeans = obj["kmeans"]
            Fg = pd.DataFrame({
                "VRP_pred": grid_df["x_margin"].values,
                "skew": skew_med,
                "convexity": conv_med,
            })
            for c in ["VRP_pred", "skew", "convexity"]:
                lo, hi = clip_bounds[c]; Fg[c] = np.clip(Fg[c].values, lo, hi)
            Xk = scaler.transform(Fg[["VRP_pred","skew","convexity"]].values)
            dists = kmeans.transform(Xk)
            grid_df["km_dmin"] = dists.min(axis=1)
            for j in range(kmeans.n_clusters):
                grid_df[f"km_d{j+1}"] = dists[:, j]
        except Exception as e:
            print(f"[plot] KMEANS indisponible pour H={horizon} -> coupe sans km_* ({e})")

    grid_df = grid_df.reindex(columns=feat_order, fill_value=0.0)
    pgrid = clf.predict_proba(grid_df.values)[:, 1].reshape(gx1.shape)

    Yg = np.zeros_like(pgrid, dtype=int)
    Yg[pgrid <  low] = -1
    Yg[pgrid >= high] = +1

    plt.figure(figsize=(6.9, 5.6))
    plt.pcolormesh(gx1, gx2, Yg, shading="auto", alpha=0.35)
    neg = (y_sign == -1); pos = (y_sign == +1)
    plt.scatter(Xval.loc[neg, f1], Xval.loc[neg, f2], s=14, alpha=0.9, label="z<0")
    plt.scatter(Xval.loc[pos, f1], Xval.loc[pos, f2], s=14, alpha=0.9, label="z≥0")
    plt.title(f"Carte de décision — H={horizon}j\nSeuils [{low:.2f}, {high:.2f}] (modèle global calibré)")
    plt.xlabel(f"pred_logratio_RV{horizon}")
    plt.ylabel(f"iv_atm{horizon}")
    plt.legend(loc="best"); plt.grid(True, linewidth=0.5, alpha=0.4)

    fig_path = _fig_path(f"decision_map_H{horizon}", run_tag)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"[Decision map H={horizon}] -> {fig_path.resolve()}")


def _build_signals_for_period(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                              run_tag: str | None = None) -> pd.DataFrame:
    mask = (data["date"] >= start) & (data["date"] <= end)
    idx = data.index[mask]
    records: List[dict] = []
    payload, _ = _load_artifact(run_tag)
    lo_g = float(payload.get("best_low", 0.40))
    hi_g = float(payload.get("best_high", 0.60))
    for h in HORIZONS:
        if len(idx) == 0:
            continue
        details = predict_with_details(data.loc[idx], horizon=h, low=lo_g, high=hi_g, run_tag=run_tag)
        for ix, row in details.iterrows():
            records.append({
                "date": data.loc[ix, "date"],
                "horizon": h,
                "p_buy": float(row["p_buy"]),
                "p_sell": float(row["p_sell"]),
                "decision": int(row["decision"]),
                "low": float(lo_g),
                "high": float(hi_g),
            })
    df_sig = pd.DataFrame.from_records(records)
    if not df_sig.empty:
        df_sig.sort_values(["date", "horizon"], inplace=True)
    return df_sig


def main_default_validation():
    data = load_dataset_from_csv()
    _ = train_pooled_classifier_between(data, VALIDATION_START, END_DATE)

    for h in [10, 30, 60]:
        if h in HORIZONS:
            plot_decision_map_best(data, horizon=h)
    val_df = _build_signals_for_period(data, VALIDATION_START, END_DATE)
    if not val_df.empty:
        val_df.to_csv(VAL_SIGNALS_CSV, index=False, float_format="%.6f")
        print(f"[Validation signals] -> {VAL_SIGNALS_CSV.resolve()}")

def main_custom_backtest():
    data = load_dataset_from_csv()
    run_tag = f"{CUSTOM_VALID_START.date()}_{CUSTOM_VALID_END.date()}"
    _ = train_pooled_classifier_between(data, CUSTOM_VALID_START, CUSTOM_VALID_END, run_tag=run_tag)
    for h in [10, 30, 60]:
        if h in HORIZONS:
            plot_decision_map_best(data, horizon=h, run_tag=run_tag)
    test_df = _build_signals_for_period(data, CUSTOM_TEST_START, CUSTOM_TEST_END, run_tag=run_tag)
    if not test_df.empty:
        test_df.to_csv(TEST_SIGNALS_CSV, index=False, float_format="%.6f")
        print(f"[Test signals] -> {TEST_SIGNALS_CSV.resolve()}")


if __name__ == "__main__":
    if ENABLE_CUSTOM_BACKTEST:
        main_custom_backtest()
    else:
        main_default_validation()

