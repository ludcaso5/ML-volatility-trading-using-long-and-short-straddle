from __future__ import annotations

import os
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Sequence, Tuple, Callable

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression

import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
import json
import joblib


# Project paths
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CSV_PATH: Path = DATA_DIR / "spx_feature_panel_20100901_20230901.csv"
MODEL_STORE_DIR: Path = RESULTS_DIR / "models" / "regression"
REGRESSION_FIGURE_PATH: Path = RESULTS_DIR / "figures" / "regression_validation.png"
MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
REGRESSION_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY: Dict[Tuple[int, int], "ModelArtifact"] = {}

HORIZONS: List[int] = [10, 30, 60]


START_DATE: datetime = datetime(2010, 12, 1)
END_DATE: datetime = datetime(2018, 9, 1)
VALIDATION_START: datetime = datetime(2016, 9, 1)


# Reproducible thread limits
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def embargo_rows_from_horizon(h: int, rows_per_week: int = 1, bdays_per_week: int = 5) -> int:


    return max(1, math.ceil(h / bdays_per_week) * rows_per_week)


def r2_vs_persistence(y_true: pd.Series | np.ndarray,
                      y_pred: pd.Series | np.ndarray,
                      baseline_value: float = 0.0) -> float:


    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_base = np.mean((y_true - baseline_value) ** 2)
    return np.nan if not np.isfinite(mse_base) or mse_base <= 0 else 1.0 - mse_model / mse_base


# Saved model metadata
@dataclass
class ModelArtifact:


    horizon: int
    model_name: str
    estimator: object
    features: List[str]
    base_cols: List[str]
    best_params: Dict
    k_star: int
    mse: float
    r2: float
    train_start: Optional[pd.Timestamp]
    train_end: Optional[pd.Timestamp]
    val_start: Optional[pd.Timestamp]
    val_end: Optional[pd.Timestamp]
    created_at: str
    lib_versions: Dict[str, str]
    rank: int = 1


def _artifact_paths(h: int, rank: int = 1) -> Tuple[Path, Path]:


    model_path = MODEL_STORE_DIR / f"spx_vol_H{h}_r{rank}.joblib"
    meta_path = MODEL_STORE_DIR / f"spx_vol_H{h}_r{rank}.meta.json"
    return model_path, meta_path


# Persist fitted models
def save_artifact(art: ModelArtifact) -> None:
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = _artifact_paths(art.horizon, art.rank)
    joblib.dump(art.estimator, model_path)

    meta = asdict(art).copy()
    meta.pop("estimator", None)
    for k in ["train_start", "train_end", "val_start", "val_end"]:
        if isinstance(meta.get(k), pd.Timestamp):
            meta[k] = meta[k].strftime("%Y-%m-%d")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    MODEL_REGISTRY[(art.horizon, art.rank)] = art


def load_artifact(h: int, rank: int = 1) -> Optional[ModelArtifact]:


    key = (h, rank)
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]

    model_path, meta_path = _artifact_paths(h, rank)
    legacy_model_path = MODEL_STORE_DIR / f"spx_vol_H{h}.joblib"
    legacy_meta_path = MODEL_STORE_DIR / f"spx_vol_H{h}.meta.json"


    if not (model_path.exists() and meta_path.exists()):
        if rank != 1 or not (legacy_model_path.exists() and legacy_meta_path.exists()):
            return None
        model_path, meta_path = legacy_model_path, legacy_meta_path

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    estimator = joblib.load(model_path)

    art = ModelArtifact(
        horizon=int(meta["horizon"]),
        model_name=meta["model_name"],
        estimator=estimator,
        features=list(meta["features"]),
        base_cols=list(meta["base_cols"]),
        best_params=dict(meta["best_params"]),
        k_star=int(meta["k_star"]),
        mse=float(meta["mse"]),
        r2=float(meta["r2"]),
        train_start=pd.to_datetime(meta["train_start"]) if meta.get("train_start") else None,
        train_end=pd.to_datetime(meta["train_end"]) if meta.get("train_end") else None,
        val_start=pd.to_datetime(meta["val_start"]) if meta.get("val_start") else None,
        val_end=pd.to_datetime(meta["val_end"]) if meta.get("val_end") else None,
        created_at=meta["created_at"],
        lib_versions=dict(meta.get("lib_versions", {})),
        rank=int(meta.get("rank", 1)),
    )
    MODEL_REGISTRY[(h, art.rank)] = art
    return art


def get_best_model(h: int, rank: int = 1) -> Optional[ModelArtifact]:


    art = MODEL_REGISTRY.get((h, rank))
    if art is not None:
        return art
    return load_artifact(h, rank=rank)


def predict_with_best_model(df: pd.DataFrame, horizon: int, rank: int = 1) -> pd.Series:


    art = get_best_model(horizon, rank=rank)
    if art is None:
        raise RuntimeError(f"Aucun modèle sauvegardé pour H={horizon} (rank={rank}). Entraîner et sauvegarder d’abord.")

    X_base = df.reindex(columns=art.base_cols, copy=False)
    X_sel = X_base[art.features]
    y_pred = art.estimator.predict(X_sel)
    return pd.Series(y_pred, index=df.index, name=f"pred_logratio_RV{horizon}_r{rank}")


# Load the prepared feature panel
def load_dataset_from_csv(path: Path = CSV_PATH,
                          start: datetime | None = START_DATE,
                          end: datetime | None = END_DATE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Build it with Dataset_builder.py or place it in data/."
        )
    df = pd.read_csv(path)

    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

    if start is not None:
        df = df[df['date'] >= start]
    if end is not None:
        df = df[df['date'] <= end]
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:


    Xn = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    keep = [c for c in Xn.columns if Xn[c].notna().all() and Xn[c].nunique() > 1]
    return Xn[keep]


# mRMR-style feature selection
def mrmr_lite_select(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        k: int,
        corr_th: float = 0.85,
        mi_kwargs: Optional[Dict] = None,
) -> List[str]:


    if mi_kwargs is None:
        mi_kwargs = {"n_neighbors": 3, "random_state": 0}

    cols = list(X_train.columns)


    Xz = (X_train - X_train.mean()) / X_train.std(ddof=0)
    Xz = Xz.replace([np.inf, -np.inf], 0.0).fillna(0.0)


    mi = mutual_info_regression(Xz.values, y_train.values, **mi_kwargs)
    order = np.argsort(mi)[::-1]
    ranked = [cols[i] for i in order]


    selected: List[str] = []
    for f in ranked:
        if len(selected) >= k:
            break
        if not selected:
            selected.append(f)
            continue
        max_abs_corr = max(abs(Xz[f].corr(Xz[s])) for s in selected)
        if max_abs_corr < corr_th:
            selected.append(f)


    if len(selected) < k:
        for f in ranked:
            if f not in selected:
                selected.append(f)
                if len(selected) >= k:
                    break
    return selected[:k]


def build_mrmr_feature_sets(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        ks: Sequence[int],
        corr_th: float = 0.85,
        mi_kwargs: Optional[Dict] = None,
) -> Dict[int, List[str]]:


    ks = sorted(set(int(k) for k in ks if k > 0))
    out: Dict[int, List[str]] = {}
    for k in ks:
        k_eff = min(k, X_train.shape[1])
        out[k] = mrmr_lite_select(X_train, y_train, k_eff, corr_th=corr_th, mi_kwargs=mi_kwargs)
    return out


# Compact hyperparameter search
SMALL_GRIDS = {
    "ridge": {
        "ridge__alpha": [50, 500, 750, ], },
    "rf": {
        "n_estimators": [400],
        "max_depth": [3, 8],
        "min_samples_leaf": [2, 75], },
    "svr": {
        "svr__C": [0.1, 0.5, 1],
        "svr__gamma": [0.01, 0.001],
        "svr__epsilon": [0.01, 0.05, 0.1, ], },
    "cat": {
        "n_estimators": [300, 500],
        "depth": [9, 10],
        "learning_rate": [0.5, 0.8],
        "l2_leaf_reg": [5],
        "subsample": [0.85],
        "rsm": [0.7, 1]}, }


# Train by forecast horizon
def train_models_for_horizons(data: pd.DataFrame) -> Dict[int, Dict[str, object]]:


    results: Dict[int, Dict[str, object]] = {}


    feature_cols = [
        c for c in data.columns
        if c not in (["date"]
                     + [f"RV{h}" for h in HORIZONS]
                     + [f"dRV{h}" for h in HORIZONS]
                     + [f"logratio_RV{h}" for h in HORIZONS])
           and not c.startswith("call_price_")
           and not c.startswith("put_price_")
           and not c.startswith("VRP_")
    ]


    model_definitions: List[Tuple[str, Callable[..., object], Dict]] = [
        ("ridge", lambda **kw: make_pipeline(StandardScaler(), Ridge(**kw)), {"alpha": 1.0}),
        ("rf", RandomForestRegressor, {"n_estimators": 200, "max_depth": None, "random_state": 0, "n_jobs": 1}),
        ("svr", lambda **kw: make_pipeline(StandardScaler(), SVR(kernel="rbf", cache_size=1000, **kw)),
         {"C": 10.0, "gamma": "scale", "epsilon": 0.05}),
        ("cat", CatBoostRegressor,
         {"loss_function": "RMSE", "random_state": 0, "verbose": 0, "thread_count": -1, "allow_writing_files": False}),
    ]


    K_LIST = [10, 20, 30, 50]

    validation_start_date = VALIDATION_START

    for horizon in HORIZONS:
        target_col = f"logratio_RV{horizon}"
        if target_col not in data.columns:
            continue


        y = data[target_col].astype(float)
        X = data[feature_cols].copy()


        mask_train = data["date"] < validation_start_date
        mask_validation = (data["date"] >= validation_start_date) & (data["date"] <= END_DATE)


        X_train_raw = X.loc[mask_train].copy()
        y_train_raw = y.loc[mask_train].copy()
        X_validation_raw = X.loc[mask_validation].copy()
        y_validation = y.loc[mask_validation].copy()


        y0 = np.zeros_like(y_validation, dtype=float)
        base_mse = mean_squared_error(y_validation, y0)
        print(f"  Baseline persistance H={horizon:2d} | MSE={base_mse:.6f}")


                # Embargo prevents target overlap
        gap_rows = embargo_rows_from_horizon(horizon)
        if len(X_train_raw) > gap_rows:
            X_train_raw = X_train_raw.iloc[:-gap_rows]
            y_train_raw = y_train_raw.iloc[:-gap_rows]


        X_train = _clean_features(X_train_raw).astype(float)
        if X_train.shape[1] == 0:
            results[horizon] = {"_note": "Aucune variable exploitable après nettoyage."}
            continue

        X_validation = X_validation_raw[X_train.columns].astype(float)
        base_cols = list(X_train.columns)
        results[horizon] = {}


        ks_eff = [k for k in K_LIST if k <= X_train.shape[1]] or [min(50, X_train.shape[1])]
        mrmr_sets = build_mrmr_feature_sets(X_train, y_train_raw, ks_eff, corr_th=0.85)


        for name, cls, defaults in model_definitions:
            best_record = None
            grid = SMALL_GRIDS.get(name, {})
            param_combos = list(ParameterGrid(grid)) if grid else [dict()]

            for k in ks_eff:
                selected = mrmr_sets[k]
                for combo in param_combos:

                    estimator = cls(**defaults)
                    if combo:
                        try:
                            estimator.set_params(**combo)
                        except Exception:
                            estimator = cls(**{**defaults, **combo})

                    estimator.fit(X_train[selected], y_train_raw)

                    preds = estimator.predict(X_validation[selected])
                    mse = mean_squared_error(y_validation, preds)
                    r2v = r2_vs_persistence(y_validation, preds)
                    rec = (mse, r2v, estimator, selected, combo, k)
                    if (best_record is None) or (mse < best_record[0]):
                        best_record = rec

            if best_record is not None:
                mse, r2v, model, selected, best_params, k_star = best_record
                results[horizon][name] = {
                    "model": model,
                    "features": selected,
                    "mse": mse,
                    "r2": r2v,
                    "best_params": best_params,
                    "base_cols": base_cols,
                    "k_star": k_star,
                }


        mask_train_base = (data["date"] < validation_start_date)
        idx_train = np.flatnonzero(mask_train_base.values)
        if idx_train.size > gap_rows:
            idx_train = idx_train[:-gap_rows]
        n_train = idx_train.size
        train_start = data["date"].iloc[idx_train[0]] if n_train else pd.NaT
        train_end = data["date"].iloc[idx_train[-1]] if n_train else pd.NaT
        val_mask = (data["date"] >= validation_start_date) & (data["date"] <= END_DATE)
        val_start = data.loc[val_mask, "date"].min()
        val_end = data.loc[val_mask, "date"].max()

        if results[horizon]:
            candidates = []
            for fam, payload in results[horizon].items():
                if "mse" in payload:
                    candidates.append((fam, payload))
            candidates_sorted = sorted(candidates, key=lambda kv: kv[1].get("mse", float("inf")))
            top_k = candidates_sorted[:2]
            lib_versions = {
                "sklearn": __import__("sklearn").__version__,
                "catboost": __import__("catboost").__version__,
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            }
            for rank, (best_name, best_payload) in enumerate(top_k, start=1):
                art = ModelArtifact(
                    horizon=horizon,
                    model_name=best_name,
                    estimator=best_payload["model"],
                    features=list(best_payload["features"]),
                    base_cols=list(best_payload["base_cols"]),
                    best_params=dict(best_payload.get("best_params", {})),
                    k_star=int(best_payload.get("k_star", len(best_payload["features"]))),
                    mse=float(best_payload["mse"]),
                    r2=float(best_payload["r2"]),
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    lib_versions=lib_versions,
                    rank=rank,
                )
                save_artifact(art)

    return results


# Run validation and save outputs
def main() -> Dict[int, Dict[str, object]]:


    data = load_dataset_from_csv()


    if data.empty or not any(col.startswith("RV") for col in data.columns):
        print("Pas de données disponibles (colonnes RV manquantes). Vérifiez le contenu du CSV.")
        return {}


    feature_cols = [
        c for c in data.columns
        if c not in (["date"]
                     + [f"RV{h}" for h in HORIZONS]
                     + [f"dRV{h}" for h in HORIZONS]
                     + [f"logratio_RV{h}" for h in HORIZONS])
           and not c.startswith("call_price_")
           and not c.startswith("put_price_")
           and not c.startswith("VRP_")
    ]
    na_ratio_all = data[feature_cols].isna().mean().sort_values(ascending=False)
    bad_all = na_ratio_all[na_ratio_all > 0.10]
    print("\n=== Contrôle des données manquantes (>10%) ===")
    print(f"Nombre de variables : {len(feature_cols)}")
    if bad_all.empty:
        print("  OK : aucune variable n’a plus de 10% de NaN.")
    else:
        print(f"  {len(bad_all)} variable(s) avec >10% de NaN (top 50) :")
        for name, ratio in bad_all.head(50).items():
            print(f"    - {name:32s} {ratio * 100:5.1f}% manquants")


    results = train_models_for_horizons(data)


    validation_start_date = VALIDATION_START
    print("\n=== Jeu de données & fenêtres ===")
    print(f"Période dataset : {data['date'].min():%Y-%m-%d} → {data['date'].max():%Y-%m-%d} | N={len(data)}")
    print(f"Split: TRAIN < {validation_start_date:%Y-%m-%d}  |  VALIDATION ≥ {validation_start_date:%Y-%m-%d}")
    for h in HORIZONS:
        gap_rows = embargo_rows_from_horizon(h)
        mask_train_base = (data['date'] < validation_start_date)
        idx_train = np.flatnonzero(mask_train_base.values)
        if idx_train.size > gap_rows:
            idx_train = idx_train[:-gap_rows]
        n_train = idx_train.size
        n_val = int((data['date'] >= validation_start_date).sum())
        train_start = data['date'].iloc[idx_train[0]] if n_train else pd.NaT
        train_end = data['date'].iloc[idx_train[-1]] if n_train else pd.NaT
        val_start = data.loc[data['date'] >= validation_start_date, 'date'].min()
        val_end = data.loc[data['date'] >= validation_start_date, 'date'].max()
        print(f"  H={h:2d}j | Train: {train_start:%Y-%m-%d} → {train_end:%Y-%m-%d} (n={n_train}) | "
              f"Validation: {val_start:%Y-%m-%d} → {val_end:%Y-%m-%d} (n={n_val}) | Embargo={gap_rows}")


    print("\n=== Résultats hold-out (validation) — meilleur par modèle ===")
    order = ["ridge", "rf", "svr", "cat"]
    pretty = {"ridge": "Ridge", "rf": "RandomForest", "svr": "SVR (RBF)", "cat": "CatBoost"}
    for h in HORIZONS:
        if h not in results or not results[h]:
            continue
        print(f"\n[H={h} jours]")
        for name in order:
            if name not in results[h]:
                continue
            res = results[h][name]
            nfeat = len(res.get("features", []))
            mse = res.get("mse", float("nan"))
            r2v = res.get("r2", float("nan"))
            params = res.get("best_params", {})
            print(f"  {pretty[name]:12s} | MSE={mse:.6f} | R²={r2v:.4f} | n_feat={nfeat} | params={params}")


    print("\nListe des variables candidates :")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:3d}. {col}")


    models_order = ["ridge", "rf", "svr", "cat"]
    pretty_name = {"ridge": "Ridge", "rf": "RandomForest", "svr": "SVR (RBF)", "cat": "CatBoost"}
    n_rows, n_cols = len(HORIZONS), len(models_order)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 16))
    for i, h in enumerate(HORIZONS):
        target_col = f"logratio_RV{h}"
        if target_col not in data.columns:
            for j in range(n_cols):
                axes[i, j].axis("off")
            continue
        y = data[target_col]
        X_full = data[feature_cols]
        mask_train = data["date"] < validation_start_date
        mask_validation = data["date"] >= validation_start_date
        X_train_full = X_full.loc[mask_train]
        y_train_full = y.loc[mask_train]
        X_validation_full = X_full.loc[mask_validation]
        y_validation = y.loc[mask_validation]
        gap_rows = embargo_rows_from_horizon(h)
        if len(X_train_full) > gap_rows:
            X_train_full = X_train_full.iloc[:-gap_rows]
            y_train_full = y_train_full.iloc[:-gap_rows]
        for j, m in enumerate(models_order):
            ax = axes[i, j]
            if h not in results or m not in results[h]:
                ax.axis("off")
                continue
            model = results[h][m]["model"]
            sel = results[h][m]["features"]
            base_cols = results[h][m].get("base_cols", feature_cols)
            X_validation_aligned = X_validation_full[base_cols]
            if len(sel) == 0 or model is None:
                ax.axis("off")
                continue
            y_pred = model.predict(X_validation_aligned[sel])
            r2v = r2_vs_persistence(y_validation, y_pred)
            ax.set_title(f"{pretty_name[m]} - {h}j  R²: {r2v:.2f}")
            ax.scatter(y_validation, y_pred, s=18, alpha=0.7)
            lim = float(max(
                abs(np.nanmin(y_validation.values)),
                abs(np.nanmax(y_validation.values)),
                abs(np.nanmin(y_pred)),
                abs(np.nanmax(y_pred)),
            ))
            if lim == 0:
                lim = 1e-3
            ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.3)
            ax.axhline(0, linewidth=0.8, alpha=0.6)
            ax.axvline(0, linewidth=0.8, alpha=0.6)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            if i == n_rows - 1:
                ax.set_xlabel("logratio_RV (observé)")
            if j == 0:
                ax.set_ylabel("logratio_RV (prédit)")
    plt.tight_layout()
    plt.savefig(REGRESSION_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Regression diagnostics saved to: {REGRESSION_FIGURE_PATH.resolve()}")
    return results


def load_best_model(horizon: int, rank: int = 1) -> Optional[ModelArtifact]:


    return get_best_model(horizon, rank=rank)


def apply_best_model_on_dataset(test_df: pd.DataFrame, horizon: int, rank: int = 1) -> pd.Series:


    return predict_with_best_model(test_df, horizon, rank=rank)


if __name__ == "__main__":
    main()

