from __future__ import annotations
import sys, io, contextlib
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.dates as mdates

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from pathlib import Path
from datetime import datetime, timezone, timedelta


# Plot configuration
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "legend.frameon": False,
})

import Regression as R
import classification as CLF

# Project output paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BACKTEST_ROOT = RESULTS_DIR / "backtest"

BACKTEST_SUMMARY_CSV = RESULTS_DIR / "backtest_summary.csv"
CLASSIFICATION_METRICS_CSV = RESULTS_DIR / "classification_metrics.csv"

RUN_TAG = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
RUN_DIR = BACKTEST_ROOT / RUN_TAG
REG_DIR = RUN_DIR / "regression_models"
CLF_DIR = RUN_DIR / "classification"
FIG_DIR = RUN_DIR / "figures"
for directory in (REG_DIR, CLF_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CLF.OUTPUT_DIR = CLF_DIR
CLF.CSV_DIR = CLF_DIR
CLF.FIG_DIR = FIG_DIR
CLF.METRICS_VAL_CSV = CLF_DIR / "metrics_validation.csv"
CLF.VAL_SIGNALS_CSV = CLF_DIR / "validation_signals.csv"
CLF.TEST_SIGNALS_CSV = CLF_DIR / "test_signals.csv"
CLF.PLOT_ROC = False
try:
    CLF._KMEANS_CACHE.clear()
except Exception:
    pass


# Quiet classifier retraining
@contextlib.contextmanager
def _silence_io():
    _out, _err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = _out
        sys.stderr = _err


try:
    import yfinance as yf
except Exception:
    yf = None


TRAIN_START = R.START_DATE
TRAIN_END   = datetime(2018, 9, 1)
BACKTEST_START = datetime(2018, 9, 1)
BACKTEST_END   = datetime(2023, 9, 1)
HORIZONS = R.HORIZONS
CAP_LOG_FLOOR = 0.30
CAP_K         = 3.0


def _make_estimator(name: str):
    if name == "ridge": return make_pipeline(StandardScaler(), Ridge())
    if name == "svr":   return make_pipeline(StandardScaler(), SVR(kernel="rbf", cache_size=1000))
    if name == "rf":    return RandomForestRegressor(random_state=0, n_jobs=1)
    if name == "cat":   return CatBoostRegressor(loss_function="RMSE", random_state=0,
                                                 verbose=0, thread_count=-1, allow_writing_files=False)
    raise ValueError(f"Modèle inconnu : {name}")

# Refit selected regression models

def retrain_best_regressions(data: pd.DataFrame) -> None:
    for h in HORIZONS:
        art_prev = R.get_best_model(h, rank=1)
        if art_prev is None:
            print(f"[H={h}] artefact régression absent."); continue
        ycol = f"logratio_RV{h}"
        if ycol not in data.columns:
            print(f"[H={h}] cible {ycol} absente."); continue

        Xb = data.reindex(columns=art_prev.base_cols, copy=False).astype(float)
        feats = [f for f in art_prev.features if f in Xb.columns]
        if not feats:
            print(f"[H={h}] aucune feature alignée."); continue

        X = Xb[feats]; y = data[ycol].astype(float)

        est = _make_estimator(art_prev.model_name)
        if art_prev.best_params:
            try: est.set_params(**art_prev.best_params)
            except Exception:
                est = _make_estimator(art_prev.model_name); est.set_params(**art_prev.best_params)
        est.fit(X, y)

        art_new = R.ModelArtifact(
            horizon=h, model_name=art_prev.model_name, estimator=est,
            features=list(feats), base_cols=list(art_prev.base_cols),
            best_params=dict(art_prev.best_params), k_star=int(art_prev.k_star),
            mse=np.nan, r2=np.nan,
            train_start=pd.to_datetime(data["date"].min()),
            train_end=pd.to_datetime(data["date"].max()),
            val_start=None, val_end=None,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
            lib_versions=art_prev.lib_versions, rank=1,
        )


        old_dir = R.MODEL_STORE_DIR
        try:
            R.MODEL_STORE_DIR = REG_DIR
            R.save_artifact(art_new)
        finally:
            R.MODEL_STORE_DIR = old_dir


def retrain_classifier_global(data: pd.DataFrame,
                              valid_start: pd.Timestamp = R.VALIDATION_START,
                              valid_end: pd.Timestamp = R.END_DATE,
                              run_tag: Optional[str] = None,
                              quiet: bool = True) -> dict:
    if quiet:
        with _silence_io():
            stats = CLF.train_pooled_classifier_between(data, valid_start, valid_end, run_tag=run_tag)
    else:
        stats = CLF.train_pooled_classifier_between(data, valid_start, valid_end, run_tag=run_tag)
    return stats


def load_backtest_data() -> pd.DataFrame:
    return R.load_dataset_from_csv(start=BACKTEST_START, end=BACKTEST_END)


def discrete_signal(df: pd.DataFrame, h: int, exposure: float = 1.0) -> pd.Series:
    try:
        d = CLF.predict_with_details(df, horizon=h)
        dec = pd.to_numeric(d["decision"], errors="coerce").fillna(0.0)
    except Exception:

        dec = CLF.apply_saved_classifier(df, h).reindex(df.index, fill_value=0)
    pos = exposure * dec.astype(float)
    return pos.clip(-abs(exposure), abs(exposure)).fillna(0.0)


# Weekly straddle return

def compute_straddle_step(df: pd.DataFrame, h: int) -> tuple[pd.Series, pd.Series]:
    c, p = f"call_price_{h}", f"put_price_{h}"
    c_raw = pd.to_numeric(df[c], errors="coerce")
    p_raw = pd.to_numeric(df[p], errors="coerce")

    valid_raw = c_raw.notna() & p_raw.notna() & c_raw.shift(-1).notna() & p_raw.shift(-1).notna()


    S = (c_raw + p_raw).replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(lower=1e-8)


    with np.errstate(divide="ignore", invalid="ignore"):
        step_log_raw = np.log(S.shift(-1) / S)
    step_log_raw = step_log_raw.where(valid_raw)


    med = np.nanmedian(step_log_raw)
    mad = np.nanmedian(np.abs(step_log_raw - med))
    sigma = 1.4826 * mad if np.isfinite(mad) else 0.0
    cap_log = max(CAP_LOG_FLOOR, CAP_K * sigma)

    step_log = step_log_raw.clip(-cap_log, cap_log).fillna(0.0)
    step_simple = np.expm1(step_log)
    return S, step_simple


def backtest_straddle(df: pd.DataFrame, h: int,
                      exposure: float = 1.0, capital0: float = 1000.0) -> pd.DataFrame:

    pos = discrete_signal(df, h, exposure=exposure)


    S, step = compute_straddle_step(df, h)


    strat_ret = (pos * step).clip(-0.999, 0.999).fillna(0.0)

    capital = (1.0 + strat_ret).cumprod() * float(capital0)

    pnl_step = capital.diff()
    if not pnl_step.empty:
        pnl_step.iloc[0] = capital.iloc[0] - float(capital0)
    pnl_step = pnl_step.fillna(0.0)

    pnl_cum = pnl_step.cumsum()

    out = pd.DataFrame({
        "date": df["date"].values,
        "signal": np.sign(pos).astype(int).values,
        "position": pos.values,
        "straddle_price": S.values,
        "step_change": step.values,
        "strategy_ret": strat_ret.values,
        "capital": capital.values,
        "pnl_step": pnl_step.values,
        "pnl_cum": pnl_cum.values,
    }).set_index("date")
    out.attrs["meta"] = {"exposure": exposure, "capital0": capital0}
    return out


def _infer_steps_per_year(dates: pd.Index) -> float:
    if len(dates) < 2: return 52.0
    d = pd.to_datetime(dates)
    dt = np.median(np.diff(d.values).astype("timedelta64[D]").astype(float))
    if not np.isfinite(dt) or dt <= 0: return 52.0
    return float(365.25 / dt)

# Performance metrics and benchmark

def perf_metrics(step_series: pd.Series, dates: Optional[pd.Index] = None) -> dict:
    r = pd.to_numeric(step_series, errors="coerce").fillna(0.0)
    steps_per_year = _infer_steps_per_year(dates if dates is not None else r.index)
    mu_ann = r.mean() * steps_per_year
    vol_ann = r.std(ddof=1) * np.sqrt(max(1.0, steps_per_year))
    sharpe = np.nan if vol_ann == 0 else mu_ann / vol_ann

    eq = (1.0 + r).cumprod()
    if dates is not None and len(dates) > 1:
        d = pd.to_datetime(dates)
        years = (d[-1] - d[0]).days / 365.25
    else:
        years = len(r) / max(1.0, steps_per_year)

    running_max = eq.cummax().clip(lower=1.0)
    dd = eq / running_max - 1.0
    max_dd = float(dd.min())
    try:
        cagr = (eq.iloc[-1] ** (1.0 / max(1e-9, years))) - 1.0
    except Exception:
        cagr = np.nan

    return {
        "mu_ann": float(mu_ann),
        "vol_ann": float(vol_ann),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "cagr": float(cagr),
    }

def _yf_pick_close(df: pd.DataFrame) -> pd.Series:

    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.astype(float)
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")

    return df.select_dtypes(include=[np.number]).iloc[:, 0].astype(float)

def compute_spx_equity(df: pd.DataFrame) -> pd.Series:

    for px_col in ["SPX","spx","^GSPC","SPX_Index","PX_LAST","close","Close","Adj Close","adj_close","px","price"]:
        if px_col in df.columns:
            px = pd.to_numeric(df[px_col], errors="coerce")
            rets = px.pct_change().fillna(0.0)
            eq = (1.0 + rets).cumprod()
            return eq / (eq.iloc[0] if len(eq) else 1.0)


    if yf is None:
        raise RuntimeError("Installe yfinance (pip install yfinance) ou fournis une colonne SPX.")
    start = pd.to_datetime(df["date"].min()).to_pydatetime() - timedelta(days=7)
    end   = pd.to_datetime(df["date"].max()).to_pydatetime() + timedelta(days=7)
    data = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise RuntimeError("Téléchargement SPX vide via yfinance.")
    px = _yf_pick_close(data).rename("SPX_px")
    px.index = pd.to_datetime(px.index).tz_localize(None)

    dates = pd.to_datetime(df["date"]).dt.tz_localize(None)
    px_on_panel = px.reindex(dates).ffill()
    rets = px_on_panel.pct_change().fillna(0.0)
    eq = (1.0 + rets).cumprod()
    return eq / (eq.iloc[0] if len(eq) else 1.0)


def plot_equity_and_drawdown(eq_sig: pd.Series, eq_spx: pd.Series, h: int, subtitle: str = ""):
    ref = eq_sig.index

    def _prep(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").astype(float).reindex(ref).ffill().bfill()
        s = s.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if s.empty:
            return s
        return s.clip(lower=1e-8)

    e_sig, e_spx = _prep(eq_sig), _prep(eq_spx)


    c_spx, c_strat, c_base = "#2563eb", "#f59e0b", "#9ca3af"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.4, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.3]}
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.10)


    ax1.plot(e_spx.index, e_spx, lw=1.8, label="SPX", color=c_spx)
    ax1.plot(e_sig.index, e_sig, lw=2.2, label="Straddle (signal)", color=c_strat)


    ax1.axhline(1.0, ls=(0, (3, 3)), lw=1.1, color=c_base, alpha=0.8, zorder=0)

    ax1.set_yscale("log")

    ymin = float(min(e_sig[e_sig > 0].min(), e_spx[e_spx > 0].min())) * 0.98
    ymax = float(max(e_sig.max(), e_spx.max())) * 1.02
    ax1.set_ylim(max(ymin, 1e-8), ymax)


    grid = np.geomspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 160)
    xlog = np.log10(grid)
    keep = np.abs(xlog - np.rint(xlog)) > 1e-3
    ticks = grid[keep]
    if len(ticks) > 6:
        idx = np.linspace(0, len(ticks) - 1, 6).round().astype(int)
        ticks = ticks[idx]
    ax1.yaxis.set_major_locator(FixedLocator(ticks))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}×"))
    ax1.yaxis.set_minor_locator(FixedLocator([]))


    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", length=4, width=0.8)

    ax1.set_title(f"Équity normalisée × — H={h} j  {subtitle}".strip())
    ax1.set_ylabel("P&L")
    ax1.legend(loc="upper left")


    for s, c in ((e_spx, c_spx), (e_sig, c_strat)):
        ax1.annotate(f"{s.iloc[-1]:.2f}×",
                     xy=(s.index[-1], s.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     va="center", color=c,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


    dd_sig = e_sig / e_sig.cummax().clip(lower=1.0) - 1.0
    dd_spx = e_spx / e_spx.cummax().clip(lower=1.0) - 1.0

    ax2.plot(dd_spx.index, dd_spx, lw=1.6, label="SPX", color=c_spx)
    ax2.plot(dd_sig.index, dd_sig, lw=1.6, label="straddle", color=c_strat)

    ax2.fill_between(dd_sig.index, dd_sig, 0.0, alpha=0.10, color=c_strat)
    ax2.fill_between(dd_spx.index, dd_spx, 0.0, alpha=0.08, color=c_spx)

    ax2.axhline(0.0, color=c_base, lw=1.0, ls="--", alpha=0.8)

    ax2.set_ylim(-1.02, 0.02)
    ax2.set_ylabel("Drawdown")
    ax2.set_title(f"Underwater — H={h} j")

    ax2.yaxis.set_major_locator(FixedLocator([-0.95, -0.80, -0.60, -0.40, -0.20, 0.0]))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.legend(loc="lower left")


    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig_path = FIG_DIR / f"backtest_H{h}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return fig_path


def regression_test_metrics(df_bt: pd.DataFrame, h: int) -> Tuple[float, float, int]:
    art = R.get_best_model(h, rank=1)
    if art is None: return (np.nan, np.nan, 0)
    ycol = f"logratio_RV{h}"
    if ycol not in df_bt.columns: return (np.nan, np.nan, 0)
    Xb = df_bt.reindex(columns=art.base_cols, copy=False).astype(float)
    feats = [f for f in art.features if f in Xb.columns]
    if not feats: return (np.nan, np.nan, 0)
    X = Xb[feats].dropna()
    y_true = df_bt.loc[X.index, ycol].astype(float)
    y_pred = art.estimator.predict(X)
    mse = float(mean_squared_error(y_true, y_pred))
    r2  = float(R.r2_vs_persistence(y_true.values, y_pred))
    return mse, r2, len(X)

def classifier_test_error(df_bt: pd.DataFrame, h: int) -> Tuple[float, int, float]:
    try:
        details = CLF.predict_with_details(df_bt, horizon=h)
        yhat = details["decision"].astype(int)
    except Exception:
        return (np.nan, 0, np.nan)


    try:
        iv = CLF._iv_mean(df_bt, h).astype(float)
    except Exception:
        return (np.nan, 0, np.nan)
    rv = df_bt[f"RV{h}"].astype(float)
    ytrue = np.where((rv - iv).values >= 0.0, 1, -1)
    ypred = yhat.reindex(df_bt.index).fillna(0).values

    mask = (ypred != 0)
    n_decided = int(mask.sum())
    coverage = float(n_decided / max(1, len(ypred)))
    if n_decided == 0:
        return (np.nan, 0, coverage)
    err = float(np.mean(np.sign(ypred[mask]) != ytrue[mask]))
    return (err, n_decided, coverage)


# Out-of-sample backtest

def run_backtest(alloc: float = 0.2) -> Dict[int, pd.DataFrame]:
    df_bt = load_backtest_data()
    spx_equity = compute_spx_equity(df_bt)

    results: Dict[int, pd.DataFrame] = {}
    summary_rows = []

    for h in HORIZONS:
        if R.load_artifact(h, rank=1) is None:
            print(f"[H={h}] régression absente.")
            continue

        try:
            _ = CLF.apply_saved_classifier(df_bt, h)
        except Exception as e:
            print(f"[H={h}] classif GLOBAL indisponible : {e}")
            continue

        mse_t, r2_t, n_reg = regression_test_metrics(df_bt, h)
        err_t, n_clf, cov_t = classifier_test_error(df_bt, h)

        res = backtest_straddle(
            df_bt,
            h,
            exposure=alloc,
            capital0=1.0,
        )

        results[h] = res
        res.to_csv(RUN_DIR / f"backtest_H{h}.csv")

        m = perf_metrics(res["strategy_ret"], dates=res.index)

        spx_eq_aligned = spx_equity.reindex(res.index).ffill().bfill()
        spx_ret = spx_eq_aligned.pct_change().fillna(0.0)
        m_spx = perf_metrics(spx_ret, dates=res.index)

        summary_rows.append({
            "Horizon": h,
            "Strategy_Final_Value": float(res["capital"].iloc[-1]),
            "Strategy_CAGR": float(m["cagr"]),
            "Strategy_Volatility": float(m["vol_ann"]),
            "Strategy_Sharpe": float(m["sharpe"]),
            "Strategy_Max_Drawdown": float(m["max_dd"]),
            "SPX_Final_Value": float(spx_eq_aligned.iloc[-1]),
            "SPX_CAGR": float(m_spx["cagr"]),
            "SPX_Volatility": float(m_spx["vol_ann"]),
            "SPX_Sharpe": float(m_spx["sharpe"]),
            "Regression_MSE": float(mse_t),
            "Regression_R2": float(r2_t),
            "Classification_Error": float(err_t),
            "Classification_Coverage": float(cov_t),
            "N_Regression": int(n_reg),
            "N_Classification": int(n_clf),
        })

        def _fmt(x, nd=6):
            return "N/A" if pd.isna(x) else f"{x:.{nd}f}"

        def _fmt_pct(x, nd=2):
            return "N/A" if pd.isna(x) else f"{x:.{nd}%}"

        print(
            f"[H={h}] TEST 2018-2023 | "
            f"Régression: MSE={_fmt(mse_t)} R²={_fmt(r2_t, 4)} (n={n_reg}) "
            f"| Classif: err_décisions={_fmt_pct(err_t)} "
            f"(n={n_clf}, coverage={_fmt_pct(cov_t)})"
        )

        print(
            f"         Straddle | Capital_fin={res['capital'].iloc[-1]:.2f} "
            f"| vol={_fmt(m['vol_ann'], 4)} "
            f"sharpe={_fmt(m['sharpe'], 2)} "
            f"| CAGR={_fmt_pct(m['cagr'])}"
        )

        print(
            f"         SPX      | Capital_fin={spx_eq_aligned.iloc[-1]:.2f} "
            f"| vol={_fmt(m_spx['vol_ann'], 4)} "
            f"sharpe={_fmt(m_spx['sharpe'], 2)} "
            f"| CAGR={_fmt_pct(m_spx['cagr'])}"
        )

        equity_x = res["capital"].astype(float)
        plot_equity_and_drawdown(
            equity_x,
            spx_equity,
            h,
            subtitle=f"(alloc={alloc})",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        BACKTEST_SUMMARY_CSV,
        index=False,
        float_format="%.6f",
    )

    print(f"\n[Backtest summary] -> {BACKTEST_SUMMARY_CSV.resolve()}")

    return results

def main() -> None:
    print("=== Chargement des données d’entraînement ===")

    data = R.load_dataset_from_csv(
        start=TRAIN_START,
        end=TRAIN_END,
    )

    print(
        f"Fenêtre d’entraînement : "
        f"{data['date'].min():%Y-%m-%d} → "
        f"{data['date'].max():%Y-%m-%d} | "
        f"N={len(data)}"
    )

    missing = [
        h for h in HORIZONS
        if R.get_best_model(h, rank=1) is None
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing regression artifacts for horizons {missing}. "
            "Run Regression.py first."
        )

    _ = retrain_classifier_global(
        data,
        valid_start=R.VALIDATION_START,
        valid_end=R.END_DATE,
        quiet=True,
    )

    if CLF.METRICS_VAL_CSV.exists():
        classification_metrics = pd.read_csv(CLF.METRICS_VAL_CSV)

        classification_metrics.to_csv(
            CLASSIFICATION_METRICS_CSV,
            index=False,
            float_format="%.6f",
        )

        print(
            f"[Classification metrics] -> "
            f"{CLASSIFICATION_METRICS_CSV.resolve()}"
        )

    retrain_best_regressions(data)

    R.MODEL_REGISTRY.clear()
    R.MODEL_STORE_DIR = REG_DIR

    print("\n=== Backtest 2018-2023 ===")
    _ = run_backtest(alloc=0.2)

if __name__ == "__main__":
    main()

