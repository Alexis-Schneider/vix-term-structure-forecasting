"""
ts_extract.py — extraction de la term structure depuis les options brutes SPX/SPXW,
fit Nelson-Siegel, et chargement des indices CBOE.
À placer à côté du notebook :  import ts_extract as tx

Pipeline : options brutes -> term structure quotidienne (Day, DaysToExpiration, ImpliedVol)
           -> Nelson-Siegel ridge -> courbe à maturité constante (iv9, iv30, iv90) -> features.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

LAMBDA   = 0.186   # Guo-Han-Zhao (2014) : lambda=0.0147 avec tau en JOURS -> 1/(365*0.0147)=0.186 en annees ; = aussi l'optimum empirique du fit (grid search)
RIDGE_NS = 1e-2
SNAP_TIME = pd.Timestamp("15:30").time()   # 15:30 (Davide) : évite le bruit des 30 dernières minutes

# ----------------------------------------------------------------------
# 1. Indexation des fichiers (rglob -> inclut monthly/nuove)
# ----------------------------------------------------------------------
def list_parquet_files(folder):
    rows = []
    for p in sorted(Path(folder).rglob("*.parquet")):
        parts = p.stem.split("_")
        try:
            if len(parts) == 2:   start, end = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
            elif len(parts) == 1: start = end = pd.Timestamp(parts[0])
            else: continue
            rows.append({"path": p, "start": start, "end": end})
        except Exception:
            continue
    return pd.DataFrame(rows)

def opt_folders(data_root):
    return {k: Path(data_root) / "bid_ask_quotes" / k for k in ["0dte_5dte", "5friday", "monthly"]}

# ----------------------------------------------------------------------
# 2. Snapshot + filtres qualité
# ----------------------------------------------------------------------
OPEN_TIME = pd.Timestamp("10:00").time()   # début de séance mais après l'illiquidité de l'ouverture (09:30)

def load_and_snap(path, at="open", snap_time=SNAP_TIME, open_time=OPEN_TIME):
    """at='open' -> premier snapshot de la journée (début de séance, préférence de Davide) ;
       at='close' -> dernier snapshot <= 16:00."""
    df = pd.read_parquet(path)
    if df.empty: return df
    dt = df["datetime"]
    if dt.dt.tz is None: dt = dt.dt.tz_localize("America/New_York")
    df = df.assign(_date=dt.dt.date, _time=dt.dt.time)
    keep = []
    for d, sub in df.groupby("_date"):
        if at == "open":
            valid = sub[sub["_time"] >= open_time]
            if valid.empty: continue
            chosen = valid["_time"].min()          # début de journée
        else:
            valid = sub[sub["_time"] <= snap_time]
            if valid.empty: continue
            chosen = valid["_time"].max()          # clôture
        keep.append(valid[valid["_time"] == chosen])
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]

def quality_filter(snap, max_rel_spread=0.25, require_size=True):
    snap = snap[(snap["bid"] > 0) & (snap["ask"] > 0) & (snap["ask"] >= snap["bid"])].copy()
    if require_size and "bid_size" in snap and "ask_size" in snap:
        snap = snap[(snap["bid_size"] > 0) & (snap["ask_size"] > 0)]
    snap["mid"] = (snap["bid"] + snap["ask"]) / 2
    snap["spread"] = snap["ask"] - snap["bid"]
    return snap[snap["spread"] / snap["mid"] < max_rel_spread]

# ----------------------------------------------------------------------
# 3. Forward sans taux + Black-Scholes + IV ATM
# ----------------------------------------------------------------------
def implied_forward(df_exp, n_near=15):
    piv = df_exp.pivot_table(index="strike", columns="right", values="mid", aggfunc="first")
    if not {"C", "P"}.issubset(piv.columns): return np.nan, np.nan, np.nan
    piv = piv.dropna(subset=["C", "P"])
    if len(piv) < 3: return np.nan, np.nan, np.nan
    Kstar = (piv["C"] - piv["P"]).abs().idxmin()
    near = piv.reindex(piv.index[np.argsort(np.abs(piv.index - Kstar))[:n_near]]).sort_index()
    b, a = np.polyfit(near.index.values.astype(float), (near["C"] - near["P"]).values, 1)
    if b >= 0: return np.nan, np.nan, float(Kstar)
    return float(-a / b), float(-b), float(Kstar)

_RATE = pd.Series({2019:.022,2020:.004,2021:.0005,2022:.018,2023:.050,2024:.052,2025:.043})
def rate_for(date):
    y = pd.Timestamp(date).year
    return float(_RATE.loc[y]) if y in _RATE.index else float(_RATE.iloc[-1])

def bs_price_forward(F, K, T, r, sigma, right="C"):
    if T <= 0 or sigma <= 0:
        return np.exp(-r*T) * (max(F-K, 0) if right == "C" else max(K-F, 0))
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T)); d2 = d1 - sigma*np.sqrt(T)
    if right == "C": return np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))
    return np.exp(-r*T) * (K*norm.cdf(-d2) - F*norm.cdf(-d1))

def bs_implied_vol(price, F, K, T, r, right="C"):
    if T <= 0 or price <= 0: return np.nan
    intr = np.exp(-r*T) * (max(F-K, 0) if right == "C" else max(K-F, 0))
    if price < intr - 1e-6: return np.nan
    try: return brentq(lambda s: bs_price_forward(F, K, T, r, s, right) - price, 1e-4, 5.0, xtol=1e-6, maxiter=100)
    except Exception: return np.nan

def atm_iv_for_expiration(sub, F, T, r, n_atm=6):
    sub = sub.copy()
    sub["otm"] = np.where(sub["strike"] >= F, sub["right"] == "C", sub["right"] == "P")
    otm = sub[sub["otm"]].copy(); otm["dist"] = (otm["strike"] - F).abs()
    otm = otm.sort_values("dist").head(n_atm)
    pts = []
    for _, o in otm.iterrows():
        iv = bs_implied_vol(o["mid"], F, o["strike"], T, r, o["right"])
        if iv is not None and not np.isnan(iv): pts.append((o["strike"], iv))
    if not pts: return np.nan, 0
    pts.sort(); Ks = np.array([k for k, _ in pts]); IVs = np.array([v for _, v in pts])
    iv = float(np.interp(F, Ks, IVs)) if Ks.min() <= F <= Ks.max() else float(IVs[np.argmin(np.abs(Ks-F))])
    return iv, len(pts)

def term_structure_day(snap, trade_date, r=None, min_dte=1, min_atm=2):
    if r is None: r = rate_for(trade_date)
    out = []
    for exp in sorted(snap["expiration"].unique()):
        sub = snap[snap["expiration"] == exp]
        Td = (pd.Timestamp(exp) - pd.Timestamp(trade_date)).days
        if Td < min_dte: continue
        F, DF, Ks = implied_forward(sub)
        if np.isnan(F): continue
        iv, n = atm_iv_for_expiration(sub, F, Td/365.0, r)
        if np.isnan(iv) or n < min_atm: continue
        out.append({"Day": pd.Timestamp(trade_date).date(), "Expiration": exp, "DaysToExpiration": Td,
                    "Forward": F, "NbATM": n, "ImpliedVol": iv*100})
    return pd.DataFrame(out)

# ----------------------------------------------------------------------
# 4. Build complet (par fenêtre annuelle, mémoire bornée)
# ----------------------------------------------------------------------
def build_window(data_root, start, end, at="open"):
    folders = opt_folders(data_root); files = []
    for src, fld in folders.items():
        fi = list_parquet_files(fld)
        if not fi.empty: fi["src"] = src; files.append(fi)
    files = pd.concat(files, ignore_index=True); start, end = pd.Timestamp(start), pd.Timestamp(end)
    files = files[files["end"] >= start].sort_values("start")
    per_date = {}
    for _, fr in files.iterrows():
        snap = load_and_snap(fr["path"], at=at)
        if snap.empty: continue
        snap = quality_filter(snap); snap["src"] = fr["src"]
        for d, sub in snap.groupby("_date"):
            if start.date() <= d <= end.date(): per_date.setdefault(d, []).append(sub)
    ts_all = []
    for d in sorted(per_date):
        snap = pd.concat(per_date[d], ignore_index=True)
        snap = snap.drop_duplicates(subset=["root", "expiration", "strike", "right"], keep="first")
        t = term_structure_day(snap, d)
        if not t.empty: ts_all.append(t)
    return pd.concat(ts_all, ignore_index=True) if ts_all else pd.DataFrame()

def build_all(data_root, years=range(2020, 2026), verbose=True, at="open"):
    parts = []
    for y in years:
        if verbose: print(f"Année {y}...")
        t = build_window(data_root, f"{y}-01-01", f"{y}-12-31", at=at)
        if len(t): parts.append(t)
        if verbose: print(f"   -> {0 if not len(t) else t['Day'].nunique()} jours, {len(t)} lignes")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# ----------------------------------------------------------------------
# 5. Nelson-Siegel ridge -> courbe à maturité constante + features
# ----------------------------------------------------------------------
def L2(t, lam=LAMBDA):
    t = np.asarray(t, float); x = t/lam; o = np.empty_like(x); s = x < 1e-6
    o[s] = 1.0; o[~s] = (1 - np.exp(-x[~s]))/x[~s]; return o
def L3(t, lam=LAMBDA): return L2(t, lam) - np.exp(-np.asarray(t, float)/lam)

def fit_ns_ridge(ivs, mat, lam=LAMBDA, alpha=RIDGE_NS):
    X = np.column_stack([np.ones_like(mat), L2(mat, lam), L3(mat, lam)])
    D = np.diag([0., 1., 1.])
    return np.linalg.solve(X.T @ X + alpha*D, X.T @ ivs)

def build_curve(ts, min_nexp=4, max_mindte=21, min_maxdte=90):
    """ts -> courbe constante quotidienne iv9/iv30/iv90 (filtre de couverture)."""
    ts = ts.copy(); ts["Day"] = pd.to_datetime(ts["Day"])
    rows = []
    for day, g in ts.groupby("Day"):
        g = g.sort_values("DaysToExpiration"); dte = g["DaysToExpiration"]
        if dte.nunique() < min_nexp or dte.min() > max_mindte or dte.max() < min_maxdte: continue
        b = fit_ns_ridge(g["ImpliedVol"].values, dte.values/365.0)
        ivc = lambda d: b[0] + b[1]*L2(np.array([d/365]))[0] + b[2]*L3(np.array([d/365]))[0]
        rows.append({"Day": day, "iv9": ivc(9), "iv30": ivc(30), "iv90": ivc(91)})
    return pd.DataFrame(rows).set_index("Day").sort_index()

def build_betas(ts, min_nexp=4, max_mindte=21, min_maxdte=90, min_dte_fit=5):
    """Estime chaque jour (beta0, beta1, beta2) en fittant Nelson-Siegel sur toutes les
       maturités disponibles, SAUF le front ultra-court (< min_dte_fit jours) : les options
       0-DTE/1-DTE portent des primes d'événement/week-end qui cassent la forme NS et
       effondrent le R². Renvoie aussi la qualité de fit (RMSE, R2, nb de maturités)."""
    ts = ts.copy(); ts["Day"] = pd.to_datetime(ts["Day"])
    rows = []
    for day, g in ts.groupby("Day"):
        g = g[g["DaysToExpiration"] >= min_dte_fit].sort_values("DaysToExpiration")  # exclut le front bruité
        dte = g["DaysToExpiration"]
        if dte.nunique() < min_nexp or dte.min() > max_mindte or dte.max() < min_maxdte:
            continue
        mat = dte.values / 365.0; ivs = g["ImpliedVol"].values
        b = fit_ns_ridge(ivs, mat)
        fit = b[0] + b[1]*L2(mat) + b[2]*L3(mat)
        rmse = float(np.sqrt(np.mean((ivs - fit)**2)))
        ss = np.sum((ivs - ivs.mean())**2); r2 = float(1 - np.sum((ivs-fit)**2)/ss) if ss > 0 else np.nan
        rows.append({"Day": day, "beta0": b[0], "beta1": b[1], "beta2": b[2],
                     "n_mat": int(dte.nunique()), "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows).set_index("Day").sort_index()

def iv_from_betas(betas_row, days):
    """Reconstruit l'IV à `days` jours depuis (beta0,beta1,beta2)."""
    t = np.array([days]) / 365.0
    return betas_row["beta0"] + betas_row["beta1"]*L2(t)[0] + betas_row["beta2"]*L3(t)[0]

def build_features_betas(betas):
    """Features basées sur les facteurs eux-mêmes (pour comparer aux modèles séries temporelles)."""
    df = pd.DataFrame(index=betas.index)
    df["beta0"] = betas["beta0"]; df["beta1"] = betas["beta1"]; df["beta2"] = betas["beta2"]
    df["beta0_chg5"] = betas["beta0"].diff(5)
    return df.dropna()

def build_features(curve):
    df = pd.DataFrame(index=curve.index)
    df["iv30"] = curve["iv30"]; df["iv30_chg5"] = curve["iv30"].diff(5)
    df["slope"] = curve["iv90"] - curve["iv30"]
    df["slope_short"] = curve["iv30"] - curve["iv9"]
    return df.dropna()

FEATS = ["iv30", "iv30_chg5", "slope", "slope_short"]

# ----------------------------------------------------------------------
# 6. Indices CBOE (benchmark long, gratuit) + features
# ----------------------------------------------------------------------
def load_cboe(fname, local_csv=None):
    import io, ssl, urllib.request
    if local_csv:
        df = pd.read_csv(local_csv)
    else:
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{fname}"
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = [c.strip().upper() for c in df.columns]; df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE")["CLOSE"].sort_index()

def build_features_cboe(vix9d, vix, vix3m, vix6m):
    df = pd.DataFrame(index=vix.index)
    df["iv30"] = vix; df["iv30_chg5"] = vix.diff(5)
    df["slope"] = vix3m - vix; df["slope_short"] = vix - vix9d; df["curve"] = vix6m - vix3m
    return df.dropna()

FEATS_CBOE = ["iv30", "iv30_chg5", "slope", "slope_short", "curve"]

def split_index(feats, test_year):
    yrs = sorted(feats.index.year.unique())
    if test_year not in yrs: test_year = yrs[-1]
    return feats.index.get_loc(feats.index[feats.index.year == test_year][0]), test_year
