from __future__ import annotations

import os
import math
import getpass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Sequence, Callable
import contextlib
import io

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from sqlalchemy import create_engine, URL, text

import pandas_market_calendars as mcal
import yfinance as yf


# Project paths and sample definition
SECID_SPX: int = 108105


HORIZONS: List[int] = [10, 30, 60]


START_DATE: datetime = datetime(2010, 9, 1)
END_DATE: datetime = datetime(2023, 9, 1)

SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "wrds_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def embargo_rows_from_horizon(h: int, rows_per_week: int = 1, bdays_per_week: int = 5) -> int:
    return max(1, math.ceil(h / bdays_per_week) * rows_per_week)


def _interp_by_ttm(ttm: np.ndarray, values: np.ndarray, target: int) -> float:
    if ttm.size == 0:
        return np.nan
    if ttm.size >= 2 and (ttm.min() <= target <= ttm.max()):
        return float(np.interp(target, ttm, values))

    idx = int(np.argmin(np.abs(ttm - target)))
    return float(values[idx])


def _calculate_ttm_business_days(dates: pd.Series, expirations: pd.Series) -> np.ndarray:
    nyse = mcal.get_calendar("XNYS")
    trading_days = nyse.valid_days(start_date=START_DATE, end_date=END_DATE).tz_localize(None)
    pos_d = trading_days.searchsorted(dates.values.astype("datetime64[ns]"))
    pos_e = trading_days.searchsorted(expirations.values.astype("datetime64[ns]"))
    return (pos_e - pos_d).astype(np.int32)


# Secure WRDS connection

def get_wrds() -> Optional[object]:
    user = os.getenv("WRDS_USER") or input("WRDS username: ").strip()
    password = os.getenv("WRDS_PASS") or getpass.getpass("WRDS password: ")
    if not user or not password:
        return None
    try:
        url = URL.create(
            drivername="postgresql+psycopg2",
            username=user,
            password=password,
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            database="wrds",
            query={"sslmode": "require"},
        )
        return create_engine(
            url,
            pool_size=2,
            max_overflow=0,
            pool_pre_ping=True,
            echo=False,
            future=True,
            connect_args={
                "sslmode": "require",
                "application_name": "spx_vol_pipeline",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
                "options": "-c statement_timeout=1800000",
            },
        )
    except Exception:
        return None


# Market and macro data

def download_spx_data(start: datetime | str = START_DATE,
                      end: datetime | str = END_DATE) -> pd.DataFrame:
    end_dt = pd.to_datetime(end)
    df = yf.download(
        "^GSPC",
        start=start,
        end=end_dt + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=True,
        progress=False,
    ).reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").rename(columns=str.title)
    return df


def _yf_download_compat(ticker: str, **kwargs):
    try:
        return yf.download(ticker, **kwargs)
    except TypeError as e:
        if "raise_errors" in str(e):
            kwargs.pop("raise_errors", None)
            return yf.download(ticker, **kwargs)
        raise


def fetch_macro_features() -> pd.DataFrame:
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)

    def fred(code: str, name: Optional[str] = None) -> pd.Series:
        s = pdr.DataReader(code, "fred", start_dt, end_dt).iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = name or code
        return s

    def _silent(fn: Callable, *args, **kwargs):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            return fn(*args, **kwargs)

    def yf_close(ticker: str, name: str) -> pd.Series:
        try:
            y = _silent(
                _yf_download_compat,
                ticker,
                start=pd.to_datetime(START_DATE),
                end=pd.to_datetime(END_DATE) + pd.Timedelta(days=1),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                raise_errors=False,
            )
        except Exception:
            return pd.Series(dtype=float, name=name)

        if y is None or y.empty:
            return pd.Series(dtype=float, name=name)

        s = y.get("Close", y.get("Adj Close"))
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = name
        return s.astype(float)


    breakeven10y = fred("T10YIE", "breakeven10y_pct")
    wti = fred("DCOILWTICO", "WTI_usd_bbl")
    anFCI = fred("ANFCI", "ANFCI")
    ig_oas_pct = fred("BAMLC0A0CM", "IG_OAS_pct")
    ig_oas_bp = (ig_oas_pct * 100).rename("IG_OAS_bp") if not ig_oas_pct.empty else pd.Series(dtype=float, name="IG_OAS_bp")
    ust10 = fred("DGS10", "UST10Y_pct")
    ust2 = fred("DGS2", "UST2Y_pct")
    slope_bp = ((ust10 - ust2) * 100).rename("UST_10Y_2Y_slope_bp")
    dxy = fred("DTWEXBGS", "DXY_broad")


    move_raw = yf_close("^MOVE", "MOVE_index")
    if move_raw.empty:
        d10 = ust10.diff() * 100
        d2 = ust2.diff() * 100
        move_bp = (np.sqrt(d10.rolling(20).std() ** 2 + d2.rolling(20).std() ** 2) * np.sqrt(252)).rename("MOVE_bp")
    else:
        move_bp = move_raw
        move_bp.name = "MOVE_bp"


    vix_pct = yf_close("^VIX", "VIX_pct")
    vix3m_pct = yf_close("^VIX3M", "VIX3M_pct")
    vvix = yf_close("^VVIX", "VVIX")
    vix9d_pct = yf_close("^VIX9D", "VIX9D_pct")
    vix6m_pct = yf_close("^VIX6M", "VIX6M_pct")

    macro = pd.concat([
        breakeven10y, dxy, wti, vix_pct, vix3m_pct, vvix, vix9d_pct, vix6m_pct,
        move_bp, slope_bp, ig_oas_bp, anFCI,
    ], axis=1).sort_index().ffill().reset_index().rename(columns={"index": "date"})
    macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
    return macro


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dates = pd.to_datetime(d["date"]).dt.normalize()
    d["date"] = dates
    start = dates.min()
    end = dates.max()


    opex_fridays = pd.date_range(start, end, freq="WOM-3FRI").normalize()
    opex_window = pd.DatetimeIndex(
        np.unique(np.concatenate([(opex_fridays + pd.Timedelta(days=i)).values for i in range(-2, 3)]))
    )
    d["is_opex_week"] = d["date"].isin(opex_window).astype(int)


    month_ends = pd.date_range(start, end, freq="BME").normalize()
    last3_bdays = pd.DatetimeIndex(
        np.unique(np.concatenate([(month_ends - pd.tseries.offsets.BDay(i)).values for i in range(0, 3)]))
    )
    d["is_month_end"] = d["date"].isin(last3_bdays).astype(int)


    q_ends = pd.date_range(start, end, freq="BQE").normalize()
    last5_q_bdays = pd.DatetimeIndex(
        np.unique(np.concatenate([(q_ends - pd.tseries.offsets.BDay(i)).values for i in range(0, 5)]))
    )
    d["is_quarter_end"] = d["date"].isin(last5_q_bdays).astype(int)
    return d


def compute_forward_realized_vol_from_prices(spx_df: pd.DataFrame,
                                             horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    px = spx_df.copy()
    px["date"] = pd.to_datetime(px["Date"]).dt.tz_localize(None).dt.normalize()
    returns = px["Close"].pct_change()
    out = px[["date"]].copy()

    for h in horizons:
        future_vol = returns.shift(-h).rolling(h).std() * np.sqrt(252)
        out[f"RV{h}"] = future_vol
        past_vol = returns.rolling(h).std() * np.sqrt(252)
        out[f"RV{h}_past"] = past_vol
        out[f"dRV{h}"] = future_vol - past_vol
        log_ratio = np.log(future_vol / past_vol.replace(0, np.nan))
        out[f"logratio_RV{h}"] = log_ratio.replace([np.inf, -np.inf], np.nan).clip(-2, 2)

    return out.dropna().reset_index(drop=True)


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    keep = [c for c in Xn.columns if Xn[c].notna().all() and Xn[c].nunique() > 1]
    return Xn[keep]


# Weekly technical features

def compute_equity_indicators(df: pd.DataFrame) -> pd.DataFrame:
    from ta.trend import ADXIndicator, AroonIndicator
    from ta.volatility import BollingerBands
    from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
    from ta.momentum import ROCIndicator
    from ta.volume import VolumeWeightedAveragePrice

    data = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    H, L, C, V = data["High"], data["Low"], data["Close"], data["Volume"]
    r = C.pct_change()


    data["skew20"] = r.rolling(20, min_periods=20).skew().shift(1)
    data["kurt20"] = r.rolling(20, min_periods=20).kurt().shift(1)
    data["var20"]  = r.rolling(20, min_periods=20).var().shift(1)


    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()
    macd_line = ema(C, 12) - ema(C, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    data["MACDh_12_26_9"] = macd_line - macd_sig


    dollar_vol = (C * V).replace(0, np.nan)
    data["AMIHUD_20"] = (r.abs() / dollar_vol).rolling(20, min_periods=10).mean()


    def rsi_ewm(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(0.0)
    data["RSI_14"] = rsi_ewm(C, window=14)


    prev_close = C.shift(1)
    tr = pd.concat([(H - L).abs(), (H - prev_close).abs(), (L - prev_close).abs()], axis=1)
    true_range = tr.max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    data["ATRn_14"] = atr / C
    data["ADX_14"]  = ADXIndicator(high=H, low=L, close=C, window=14).adx()


    ar = AroonIndicator(high=H, low=L, window=25)
    data["AROONOSC_25"] = ar.aroon_up() - ar.aroon_down()


    bb = BollingerBands(close=C, window=20, window_dev=2)
    lower_band = bb.bollinger_lband()
    upper_band = bb.bollinger_hband()
    data["BBP_20_2.0"] = ((C - lower_band) / (upper_band - lower_band)).replace([np.inf, -np.inf], np.nan)


    n, d = 10, 10
    ema_hl = (H - L).ewm(span=n, adjust=False).mean()
    chv = ((ema_hl - ema_hl.shift(d)) / ema_hl.shift(d)) * 100
    data[f"CHV_{n}_{d}"] = chv
    data[f"CHV_{n}_{d}_delta"] = chv.diff()


    rng = (H - L).replace(0, np.nan)
    data["CLV"] = ((2 * C - H - L) / rng).fillna(0.0)


    data["MFI_14"] = MFIIndicator(high=H, low=L, close=C, volume=V, window=14).money_flow_index()
    data["OBV"]     = OnBalanceVolumeIndicator(close=C, volume=V).on_balance_volume()


    data["ROC_12"] = ROCIndicator(close=C, window=12).roc()
    vwap = VolumeWeightedAveragePrice(high=H, low=L, close=C, volume=V).volume_weighted_average_price()
    data["VWAP_dev"] = ((C - vwap) / vwap).replace([np.inf, -np.inf], np.nan)


    def roll_spread(price: pd.Series, window: int = 20) -> pd.Series:
        dp = price.diff()
        mu = dp.rolling(window).mean()
        dm = dp - mu
        gamma1 = (dm * dm.shift(1)).rolling(window).mean()
        spread_abs = 2.0 * np.sqrt(np.maximum(-gamma1, 0.0))
        return spread_abs / price
    data["ROLL_spread_20"] = roll_spread(C, window=20)

    is_wed = (
        (data["Date"] >= START_DATE)
        & (data["Date"] < END_DATE)
        & (data["Date"].dt.weekday == 2)
    )
    keep_cols = [
        "Date",
        "skew20", "kurt20", "var20", f"CHV_{n}_{d}", f"CHV_{n}_{d}_delta",
        "MACDh_12_26_9", "RSI_14", "ATRn_14", "ADX_14", "AROONOSC_25",
        "BBP_20_2.0", "CLV", "MFI_14", "OBV", "ROC_12", "VWAP_dev",
        "ROLL_spread_20", "AMIHUD_20",
    ]
    data = data.loc[is_wed, keep_cols].replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None).dt.normalize()
    return data


# OptionMetrics cache and queries

def _cache_path_for_year(year: int) -> Path:
    return CACHE_DIR / f"spx_{year}.parquet"


def _sql_year(year: int) -> str:
    start = START_DATE.strftime("%Y-%m-%d")
    end = END_DATE.strftime("%Y-%m-%d")
    return (
        f"SELECT op.date, op.exdate AS expiration, op.cp_flag AS cp, "
        f"op.strike_price/1000.0 AS k, op.best_bid, op.best_offer, "
        f"op.impl_volatility AS iv, op.delta, op.gamma, op.vega, "
        f"op.volume AS volume, und.close AS s, op.secid "
        f"FROM optionm.opprcd{year} op "
        f"JOIN optionm.secprd und ON und.secid = op.secid AND und.date = op.date "
        f"WHERE op.secid = {SECID_SPX} "
        f"AND op.best_offer > 0 "
        f"AND op.impl_volatility > 0 AND op.impl_volatility < 2 "
        f"AND op.date BETWEEN DATE '{start}' AND DATE '{end}' "
        f"ORDER BY op.date, op.exdate, op.cp_flag, op.strike_price"
    )


def _read_one_year(year: int,
                   engine: Optional[object],
                   use_cache: bool = True) -> Path:
    p = _cache_path_for_year(year)
    if use_cache and p.exists():
        return p

    tmp = p.with_suffix(".parquet.tmp")
    if tmp.exists():
        tmp.unlink()

    if engine is None:
        engine = get_wrds()
    if engine is None:
        raise RuntimeError("WRDS engine is unavailable; cannot fetch option quotes.")

    query = _sql_year(year)
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    with engine.connect().execution_options(stream_results=True) as conn:
        for chunk in pd.read_sql(text(query), conn, chunksize=1_000_000):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
    if writer is not None:
        writer.close()

    _ = pq.read_metadata(tmp)
    if p.exists():
        p.unlink()
    tmp.replace(p)
    return p


def load_options_data(years: Sequence[int],
                      engine: Optional[object] = None,
                      use_cache: bool = True) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pyarrow.parquet as pq

    years = list(sorted(int(y) for y in years))
    cache_map: Dict[int, Path] = {y: _cache_path_for_year(y) for y in years}
    missing = [y for y, p in cache_map.items() if not p.exists()]


    if use_cache and not missing:
        frames: List[pd.DataFrame] = [pd.read_parquet(cache_map[y]) for y in years]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


    if engine is None:
        engine = get_wrds()
    if engine is None:
        raise RuntimeError(
            "WRDS engine is unavailable; cannot fetch option quotes. "
            f"Missing cache for years: {missing}. "
            "Soit définissez WRDS_USER/WRDS_PASS, soit placez les fichiers parquet "
            f"dans {CACHE_DIR} aux noms attendus: " +
            ", ".join(str(_cache_path_for_year(y).name) for y in missing)
        )


    from concurrent.futures import ThreadPoolExecutor, as_completed
    year_to_path: Dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_read_one_year, y, engine, use_cache=True): y
            for y in years
        }
        for fut in as_completed(futures):
            y = futures[fut]
            year_to_path[y] = fut.result()

    frames: List[pd.DataFrame] = [pd.read_parquet(year_to_path[y]) for y in sorted(year_to_path)]
    if not frames:
        raise RuntimeError("No option data retrieved (cache and WRDS both unavailable).")
    return pd.concat(frames, ignore_index=True)


# Option smile and Greeks

def compute_smile_25d_by_horizon(options: pd.DataFrame,
                                 horizons: Sequence[int] = HORIZONS,
                                 target_abs_delta: float = 0.25) -> pd.DataFrame:
    opts = options.copy()
    opts[["date", "expiration"]] = opts[["date", "expiration"]].apply(pd.to_datetime)
    opts["date"] = opts["date"].dt.tz_localize(None).dt.normalize()
    opts["expiration"] = opts["expiration"].dt.tz_localize(None).dt.normalize()
    opts = opts[(opts["iv"] > 0) & (opts["iv"] < 2) & (opts["best_offer"] > 0)].copy()
    opts["ttm_bdays"] = _calculate_ttm_business_days(opts["date"], opts["expiration"])

    def nearest_by_delta(g: pd.DataFrame, side: str) -> float:
        d = g[g["cp"] == side][["delta", "iv"]].dropna()
        if d.empty:
            return np.nan
        target = +target_abs_delta if side == "C" else -target_abs_delta
        idx = (d["delta"] - target).abs().idxmin()
        return float(d.loc[idx, "iv"])

    rows = []
    for (d0, exp), group in opts.groupby(["date", "expiration"], sort=False):
        rows.append({
            "date": d0,
            "expiration": exp,
            "ttm_bdays": int(group["ttm_bdays"].iloc[0]),
            "call25_iv": nearest_by_delta(group, "C"),
            "put25_iv": nearest_by_delta(group, "P"),
        })
    df_exp = pd.DataFrame(rows)
    if df_exp.empty:
        return pd.DataFrame(columns=["date"] + [f"call25_iv_{h}" for h in horizons] + [f"put25_iv_{h}" for h in horizons])

    df_mean = (
        df_exp.groupby(["date", "ttm_bdays"], as_index=False)
              .mean(numeric_only=True)
              .sort_values(["date", "ttm_bdays"])
    )
    out_rows = []
    for d0, g in df_mean.groupby("date", sort=False):
        t = g["ttm_bdays"].to_numpy()
        call_vals = g["call25_iv"].to_numpy()
        put_vals = g["put25_iv"].to_numpy()
        row = {"date": d0}
        for h in horizons:
            row[f"call25_iv_{h}"] = _interp_by_ttm(t, call_vals, h)
            row[f"put25_iv_{h}"] = _interp_by_ttm(t, put_vals, h)
        out_rows.append(row)
    res = pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)
    for h in horizons:
        for col in (f"call25_iv_{h}", f"put25_iv_{h}"):
            if col in res.columns:
                res[col] = res[col].ffill()
    return res


def compute_atm_greeks_by_horizon(options: pd.DataFrame,
                                  horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    opts = options.copy()
    opts[["date", "expiration"]] = opts[["date", "expiration"]].apply(pd.to_datetime)
    opts["date"] = opts["date"].dt.tz_localize(None).dt.normalize()
    opts["expiration"] = opts["expiration"].dt.tz_localize(None).dt.normalize()
    opts["mid_raw"] = (opts["best_bid"] + opts["best_offer"]) / 2.0
    opts["eff_mid"] = np.where(opts["mid_raw"] > 0, opts["mid_raw"], 0.5 * opts["best_offer"])
    opts["spread"] = (opts["best_offer"] - opts["best_bid"]).clip(lower=0)
    opts["rel_spread"] = opts["spread"] / opts["best_offer"].replace(0, np.nan)
    opts["moneyness"] = opts["k"] / opts["s"]
    opts = opts[(opts["iv"] > 0) & (opts["iv"] < 2) & (opts["best_offer"] > 0)].copy()
    opts["ttm_bdays"] = _calculate_ttm_business_days(opts["date"], opts["expiration"])
    opts["rel_spread_mid"] = opts["spread"] / opts["eff_mid"].replace(0, np.nan)
    mask_put = (opts["cp"] == "P") & (opts["rel_spread"] <= 0.85)
    mask_call = (opts["cp"] == "C") & (opts["best_offer"] >= 0.05) & (opts["rel_spread_mid"] <= 1.6)
    opts = opts.loc[mask_put | mask_call].copy()

    def nearest_atm_value(group: pd.DataFrame, col: str, side: Optional[str] = None) -> float:
        d = group if side is None else group[group["cp"] == side]
        d = d[["moneyness", col]].dropna()
        if d.empty:
            return np.nan
        idx = (d["moneyness"] - 1.0).abs().idxmin()
        return float(d.loc[idx, col])

    def agg_one_exp(g: pd.DataFrame) -> Dict[str, float]:
        g = g.sort_values("moneyness", kind="mergesort")
        out: Dict[str, float] = {"ttm_bdays": int(g["ttm_bdays"].iloc[0])}
        out["gamma_atm"] = nearest_atm_value(g, "gamma")
        out["vega_atm"] = nearest_atm_value(g, "vega")
        out["call_iv_atm"] = nearest_atm_value(g, "iv", side="C")
        out["put_iv_atm"] = nearest_atm_value(g, "iv", side="P")
        out["iv_atm"] = 0.5 * (out["call_iv_atm"] + out["put_iv_atm"])
        out["call_volume"] = g.loc[g["cp"] == "C", "volume"].sum()
        out["put_volume"] = g.loc[g["cp"] == "P", "volume"].sum()
        out["call_mid_atm"] = nearest_atm_value(g, "eff_mid", side="C")
        out["put_mid_atm"] = nearest_atm_value(g, "eff_mid", side="P")
        return out

    def reduce_by_ttm_mean(df_exp_agg: pd.DataFrame) -> pd.DataFrame:
        return (
            df_exp_agg.groupby("ttm_bdays", as_index=False)
                      .mean(numeric_only=True)
                      .sort_values("ttm_bdays")
        )

    rows: List[Dict[str, float]] = []
    for date, df_d in opts.groupby("date", sort=False):
        exp_rows = [agg_one_exp(g) for _, g in df_d.groupby("expiration", sort=False)]
        agg_d = reduce_by_ttm_mean(pd.DataFrame(exp_rows))
        row: Dict[str, float] = {"date": date}
        tb30 = 30
        row["gamma_30"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["gamma_atm"].to_numpy(), tb30)
        row["vega_30"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["vega_atm"].to_numpy(), tb30)
        row["call_iv_30"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["call_iv_atm"].to_numpy(), tb30)
        row["put_iv_30"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["put_iv_atm"].to_numpy(), tb30)
        for tb in horizons:
            row[f"call_iv_{tb}"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["call_iv_atm"].to_numpy(), tb)
            row[f"put_iv_{tb}"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["put_iv_atm"].to_numpy(), tb)
            row[f"iv_atm{tb}"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["iv_atm"].to_numpy(), tb)
            row[f"call_price_{tb}"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["call_mid_atm"].to_numpy(), tb)
            row[f"put_price_{tb}"] = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["put_mid_atm"].to_numpy(), tb)
        S_date = float(df_d["s"].mean())
        T = tb30 / 252.0
        sqrtT = math.sqrt(T) if T > 0 else 0.0
        sigma_30 = 0.5 * (row["call_iv_30"] + row["put_iv_30"])
        vega_30 = row["vega_30"]
        _eps = np.finfo(float).eps
        sigma = float(sigma_30) if (pd.notna(sigma_30) and sigma_30 > 0) else _eps
        S_safe = float(S_date) if (S_date is not None and S_date > 0) else _eps
        rtT = sqrtT if sqrtT > 0 else _eps
        d1 = 0.5 * sigma * rtT
        d2 = -d1
        row["vomma_30"] = vega_30 * (d1 * d2) / sigma
        row["vanna_30"] = (vega_30 / S_safe) * (1.0 - (d1 / (sigma * rtT)))
        from math import erf, sqrt
        def Nd(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))
        delta_T = Nd(d1)
        T2 = (tb30 + 1) / 252.0
        sqrtT2 = math.sqrt(T2) if T2 > 0 else _eps
        d1_T2 = 0.5 * sigma * (sqrtT2 if sqrtT2 > 0 else _eps)
        delta_T2 = Nd(d1_T2)
        row["charm_30"] = (delta_T - delta_T2) / (1.0 / 252.0)
        call_v_30 = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["call_volume"].to_numpy(), tb30)
        put_v_30 = _interp_by_ttm(agg_d["ttm_bdays"].to_numpy(), agg_d["put_volume"].to_numpy(), tb30)
        row["call_put_vol_ratio_30"] = float(call_v_30 / (put_v_30 if put_v_30 != 0 else np.nan))
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for h in horizons:
        for col in (f"call_iv_{h}", f"put_iv_{h}"):
            if col in res.columns:
                res[col] = res[col].ffill()
    for col in ["gamma_30", "vega_30", "call_iv_30", "put_iv_30",
                "vomma_30", "vanna_30", "charm_30",
                "call_put_vol_ratio_30"]:
        if col in res.columns:
            res[col] = res[col].ffill()
    return res


# Assemble the modeling panel

def prepare_dataset(engine: Optional[object], years: Sequence[int]) -> pd.DataFrame:


    spx_df = download_spx_data(start=START_DATE, end=END_DATE)

    eq = compute_equity_indicators(spx_df)

    rv = compute_forward_realized_vol_from_prices(spx_df, HORIZONS)


    options = load_options_data(years, engine=engine)
    greek = compute_atm_greeks_by_horizon(options, HORIZONS)
    smile25 = compute_smile_25d_by_horizon(options, HORIZONS)

    macro = fetch_macro_features()

    data = eq.rename(columns={"Date": "date"})
    data = pd.merge(data, macro, how="left", on="date")
    if not greek.empty:
        data = pd.merge(data, greek, how="left", on="date")
    if not smile25.empty:
        data = pd.merge(data, smile25, how="left", on="date")
    data = pd.merge(data, rv, how="inner", on="date")
    data = data.sort_values("date").reset_index(drop=True)

    for col in data.columns:
        if col.startswith(("RV", "dRV", "logratio_RV")):
            continue
        data[col] = data[col].ffill()

    # Derived volatility features
    data["RV_over_IV_30_past"] = data["RV30_past"] / (0.5 * (data["call_iv_30"] + data["put_iv_30"])).replace(0, np.nan)
    data["iv_asymmetry_30"] = data["put_iv_30"] - data["call_iv_30"]
    for h in HORIZONS:
        c25 = f"call25_iv_{h}"
        p25 = f"put25_iv_{h}"
        cATM = f"call_iv_{h}"
        pATM = f"put_iv_{h}"
        if {c25, p25}.issubset(data.columns):
            denom = 0.5 * (data[c25] + data[p25])
            data[f"rr25_rel_{h}"] = (data[c25] - data[p25]) / denom.replace(0, np.nan)
        if {c25, p25, cATM, pATM}.issubset(data.columns):
            atm_mean = 0.5 * (data[cATM] + data[pATM])
            wings_mean = 0.5 * (data[c25] + data[p25])
            data[f"bf25_{h}"] = wings_mean - atm_mean
            data[f"skew_slope25_{h}"] = (data[p25] - data[c25]) / (2 * 0.25)
        vix_dec = (data["VIX_pct"] / 100.0).astype(float)
        data["VRP_30"] = (vix_dec ** 2) - (data["RV30_past"] ** 2)
        def _vrp_from_atm(horizon: int):
            c, p, rvp = f"call_iv_{horizon}", f"put_iv_{horizon}", f"RV{horizon}_past"
            if {c, p, rvp}.issubset(data.columns):
                sigma = 0.5 * (data[c].astype(float) + data[p].astype(float))
                data[f"VRP_{horizon}"] = (sigma ** 2) - (data[rvp].astype(float) ** 2)
        for _h in (10, 60):
            _vrp_from_atm(_h)
        data["dVRP_5"] = data["VRP_30"] - data["VRP_30"].shift(1)
    if "VIX3M_pct" in data.columns:
        data["VIX_term_slope"] = (data["VIX3M_pct"] - data["VIX_pct"]) / data["VIX_pct"].replace(0, np.nan)
        data["dVIX_term_5"] = data["VIX_term_slope"] - data["VIX_term_slope"].shift(1)
    if "VVIX" in data.columns:
        data["VVIX_level"] = data["VVIX"].astype(float)
        data["dVVIX_5"] = data["VVIX_level"] - data["VVIX_level"].shift(1)
    if {"VIX9D_pct", "VIX_pct"}.issubset(data.columns):
        data["VIX_short_slope"] = (data["VIX_pct"] - data["VIX9D_pct"]) / data["VIX9D_pct"].replace(0, np.nan)
        data["dVIX_short_5"] = data["VIX_short_slope"] - data["VIX_short_slope"].shift(1)
    if {"VIX3M_pct", "VIX_pct"}.issubset(data.columns):
        data["VIX_mid_slope"] = (data["VIX3M_pct"] - data["VIX_pct"]) / data["VIX_pct"].replace(0, np.nan)
        data["dVIX_mid_5"] = data["VIX_mid_slope"] - data["VIX_mid_slope"].shift(1)
    if {"VIX6M_pct", "VIX3M_pct"}.issubset(data.columns):
        data["VIX_curve"] = (data["VIX6M_pct"] - data["VIX3M_pct"]) / data["VIX3M_pct"].replace(0, np.nan)
        data["dVIX_curve_5"] = data["VIX_curve"] - data["VIX_curve"].shift(1)

    px = spx_df.copy()
    px["date"] = pd.to_datetime(px["Date"]).dt.normalize()
    r_on = np.log(px["Open"] / px["Close"].shift(1))
    r_intra = np.log(px["Close"] / px["Open"])
    vol_on_20 = r_on.rolling(20, min_periods=20).std() * np.sqrt(252)
    vol_intra_20 = r_intra.rolling(20, min_periods=20).std() * np.sqrt(252)
    on_share_20 = vol_on_20 / (vol_on_20 + vol_intra_20)
    overnight = pd.DataFrame({
        "date": px["date"],
        "vol_on_20": vol_on_20,
        "vol_intra_20": vol_intra_20,
        "on_share_20": on_share_20,
    })
    overnight_w = overnight[overnight["date"].dt.weekday == 2].copy()
    overnight_w["d_on_share_5"] = overnight_w["on_share_20"] - overnight_w["on_share_20"].shift(1)
    data = pd.merge(data, overnight_w.dropna().reset_index(drop=True), on="date", how="left")


    r1d = np.log(px["Close"] / px["Close"].shift(1))
    lev = pd.DataFrame({
        "date": px["date"],
        "ret1d": r1d,
        "neg_ret": (r1d < 0).astype(int),
        "neg_abs_ret": r1d.abs() * (r1d < 0).astype(int),
    })
    data = pd.merge(data, lev.dropna().reset_index(drop=True), on="date", how="left")

    pos_sq = (r1d.where(r1d > 0, 0.0) ** 2)
    neg_sq = (r1d.where(r1d < 0, 0.0) ** 2)
    RS_minus_20 = np.sqrt((252 / 20) * neg_sq.rolling(20, min_periods=20).sum())
    RS_plus_20 = np.sqrt((252 / 20) * pos_sq.rolling(20, min_periods=20).sum())
    sv = pd.DataFrame({"date": px["date"], "RSminus_20": RS_minus_20, "RSplus_20": RS_plus_20})
    sv["RSratio_20"] = sv["RSminus_20"] / sv["RSplus_20"].replace(0, np.nan)
    data = pd.merge(data, sv.dropna().reset_index(drop=True), on="date", how="left")

    def _sector_dispersion_features(start_dt, end_dt):
        tickers = ["XLK", "XLF", "XLE", "XLY", "XLV", "XLI", "XLU", "XLB"]
        y = yf.download(tickers, start=start_dt, end=end_dt + pd.Timedelta(days=1),
                        interval="1d", auto_adjust=True, progress=False, threads=False)
        closes = y["Close"].copy() if isinstance(y.columns, pd.MultiIndex) else y.copy()
        closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
        r = np.log(closes / closes.shift(1))
        RV30_sector = r.rolling(30, min_periods=30).std() * np.sqrt(252)
        mean_var_sector = (RV30_sector.pow(2)).mean(axis=1)
        tmp = pd.DataFrame({"date": RV30_sector.index, "mean_var_sector": mean_var_sector.values})
        return tmp.reset_index(drop=True)
    tmp = _sector_dispersion_features(START_DATE, END_DATE)
    data = pd.merge(data, tmp, on="date", how="left")
    if "RV30_past" in data.columns and "mean_var_sector" in data.columns:
        data["dispersion_30"] = data["mean_var_sector"] - (data["RV30_past"] ** 2)
        data["realized_corr_30"] = (data["RV30_past"] ** 2) / data["mean_var_sector"]
        data["d_dispersion_5"] = data["dispersion_30"] - data["dispersion_30"].shift(1)
        data["d_realized_corr_5"] = data["realized_corr_30"] - data["realized_corr_30"].shift(1)

    data = add_event_flags(data)

    all_nan_cols = [c for c in data.columns if data[c].isna().all()]
    if all_nan_cols:
        data = data.drop(columns=all_nan_cols)

    # Keep complete forecast targets
    target_cols = [f"logratio_RV{h}" for h in HORIZONS]
    data = data.dropna(subset=target_cols).reset_index(drop=True)

    return data


# Export the final feature panel

def main(write_csv: bool = True) -> pd.DataFrame:
    engine = None
    years = list(range(START_DATE.year, END_DATE.year + 1))
    data = prepare_dataset(engine, years)
    if write_csv:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cols = ["date"] + [c for c in data.columns if c != "date"]
        out_path = DATA_DIR / f"spx_feature_panel_{START_DATE:%Y%m%d}_{END_DATE:%Y%m%d}.csv"
        data[cols].to_csv(out_path, index=False, float_format="%.10g")
        print(f"[EXPORT] Features écrites dans: {out_path.resolve()}  | shape={data.shape}")
    return data


if __name__ == "__main__":

    main()

