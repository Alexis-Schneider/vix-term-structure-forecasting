"""
vix_lab.py — boîte à outils Phases 1 & 2 (code de niveau publication).
À placer dans le même dossier que le notebook, puis :  import vix_lab as vl

Phase 1 : skill vs horizon x tenor ; robustesse (sensibilité + bootstrap par blocs).
Phase 2 : instrument tradable réaliste (VXX) ; prévision de la variation ancrée ;
          horse-race de modèles (Random Walk / Ridge direct / VAR(1)) ;
          backtest carry-aware avec overlay de régime (Markov-switching) optionnel.

Toutes les prévisions sont en walk-forward strict (entraînement uniquement sur des
cibles déjà réalisées) -> pas de lookahead.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# ----------------------------------------------------------------------
# Ridge maison (intercept non pénalisé)
# ----------------------------------------------------------------------
def _ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    Xc = np.column_stack([np.ones(len(Xs)), Xs])
    P = np.eye(Xc.shape[1]); P[0, 0] = 0
    return (np.linalg.solve(Xc.T @ Xc + alpha * P, Xc.T @ y), mu, sd)

def _ridge_pred(model, X):
    beta, mu, sd = model; Xs = (X - mu) / sd
    return np.column_stack([np.ones(len(Xs)), Xs]) @ beta

# ----------------------------------------------------------------------
# Modèle "Ridge direct" : prévision de la VARIATION, ré-ancrée sur le niveau du jour
# ----------------------------------------------------------------------
def wf_forecast_target(features, target, horizon, split_i, feat_cols,
                       alpha=15.0, refit_every=21, predict_change=True):
    df = features.copy()
    lvl = target.reindex(df.index)
    df["y"] = (lvl.shift(-horizon) - lvl) if predict_change else lvl.shift(-horizon)
    v = df.dropna(subset=["y"] + list(feat_cols))
    X, y, idx = v[feat_cols].values, v["y"].values, v.index
    base = lvl.reindex(idx).values
    preds, m = {}, None
    for i, t in enumerate(range(split_i, len(v))):
        te = t - horizon
        if te < 30: continue
        if m is None or i % refit_every == 0:
            m = _ridge_fit(X[:te], y[:te], alpha)
        d = float(_ridge_pred(m, X[t:t+1])[0])
        preds[idx[t]] = base[t] + d if predict_change else d
    return pd.Series(preds)

# ----------------------------------------------------------------------
# Benchmarks : random walk + VAR(1) joint (Diebold-Li)
# ----------------------------------------------------------------------
def rw_forecast(target, index):
    return target.reindex(index)

def var_forecast(tenor_df, target_col, horizon, split_i, refit_every=63, maxlag=1):
    from statsmodels.tsa.api import VAR
    df = tenor_df.dropna(); cols = list(df.columns); ti = cols.index(target_col)
    idx = df.index; preds, res = {}, None
    for i, t in enumerate(range(split_i, len(df))):
        hist = df.iloc[:t+1]
        if res is None or i % refit_every == 0:
            res = VAR(hist).fit(maxlag)
        fc = res.forecast(hist.values[-res.k_ar:], steps=horizon)
        preds[idx[t]] = fc[-1, ti]
    return pd.Series(preds)

# ----------------------------------------------------------------------
# Skill (réduction de RMSE vs random walk)
# ----------------------------------------------------------------------
def skill(pred, target, horizon):
    tgt = target.shift(-horizon).reindex(pred.index); rw = target.reindex(pred.index)
    c = pred.dropna().index.intersection(tgt.dropna().index)
    rm = np.sqrt(((pred.reindex(c) - tgt.reindex(c))**2).mean())
    rr = np.sqrt(((rw.reindex(c)   - tgt.reindex(c))**2).mean())
    return 1 - rm/rr
_skill = skill  # alias

def skill_vs_horizon(features, tenors, split_i, feat_cols, alpha=15.0, horizons=range(1, 31)):
    rows = []
    for h in horizons:
        row = {"horizon": h}
        for name, s in tenors.items():
            row[name] = skill(wf_forecast_target(features, s, h, split_i, feat_cols, alpha), s, h)
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")

# ----------------------------------------------------------------------
# Instrument tradable réaliste : VXX (Stooq, gratuit) ; sinon proxy de future
# ----------------------------------------------------------------------
def load_vxx(local_csv=None):
    import io, ssl, urllib.request
    if local_csv:
        df = pd.read_csv(local_csv)
    else:
        url = "https://stooq.com/q/d/l/?s=vxx.us&i=d"
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = [c.strip().capitalize() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Close"].sort_index()

def proxy_return(short_vol, long_vol, beta_spot=0.5, roll_days=21):
    spot = short_vol.pct_change()
    roll = ((long_vol - short_vol).shift(1) / short_vol.shift(1)) / roll_days
    return (beta_spot*spot - roll).rename("fut_ret")

# ----------------------------------------------------------------------
# Overlay de régime : prob. filtrée (temps réel) d'un régime de vol haute (Markov-switching)
# ----------------------------------------------------------------------
def regime_highvol_prob(level, split_i, refit_every=126):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    s = level.dropna(); idx = s.index; out = {}; res = None; hi = None
    for i, t in enumerate(range(split_i, len(s))):
        if res is None or i % refit_every == 0:
            res = MarkovRegression(s.iloc[:t+1], k_regimes=2, switching_variance=True).fit(disp=False)
            consts = [res.params.get(f"const[{k}]", np.nan) for k in range(2)]
            hi = int(np.nanargmax(consts))
        out[idx[t]] = float(res.filtered_marginal_probabilities.iloc[-1, hi])
    return pd.Series(out)

# ----------------------------------------------------------------------
# Backtest (directionnel OU carry-aware) + sizing vol-target + stop-loss + coûts
# ----------------------------------------------------------------------
def backtest(pred, lvl_now, instr_ret, slope=None, target_vol=0.15, cost_bps=5.0,
             max_lev=2.0, edge_thresh=0.3, stop_loss=0.15, carry_w=1.0, fcst_w=1.0,
             regime_prob=None):
    df = pd.DataFrame({"pred": pred, "now": lvl_now, "ret": instr_ret}).dropna()
    if slope is not None:
        df["slope"] = slope.reindex(df.index)
    edge = df["pred"] - df["now"]
    if slope is None:                                            # signal directionnel
        raw = pd.Series(np.where(edge > edge_thresh, 1.0,
                        np.where(edge < -edge_thresh, -1.0, 0.0)), index=df.index)
    else:                                                        # carry-aware
        def z(x): m = x.expanding(120).mean(); sd = x.expanding(120).std(); return ((x-m)/sd).clip(-3, 3)
        raw = (-carry_w*z(df["slope"]) + fcst_w*z(edge)).clip(-1, 1)
        raw = raw.where(~((df["slope"] < 0) & (raw < 0)), 0.0)   # disjoncteur backwardation
    if regime_prob is not None:                                  # réduit le short si vol haute
        rp = regime_prob.reindex(df.index).fillna(0.0)
        raw = raw.where(raw > 0, raw * (1 - rp))
    rv = df["ret"].rolling(20).std()*np.sqrt(252)
    size = (target_vol/rv).clip(upper=max_lev).shift(1).fillna(0)
    df["position"] = raw*size
    turn = df["position"].diff().abs().fillna(df["position"].abs())
    df["strat_ret"] = df["position"].shift(1)*df["ret"] - turn*cost_bps/1e4
    if stop_loss is not None:
        df["strat_ret"] = df["strat_ret"].clip(lower=-abs(stop_loss))
    return df.dropna(subset=["strat_ret"])

def metrics(r, freq=252):
    r = r.dropna()
    if len(r) < 5 or r.std() == 0:
        return {"Sharpe": np.nan, "CAGR%": np.nan, "MaxDD%": np.nan, "Hit%": np.nan, "N": len(r)}
    cum = (1+r).cumprod(); peak = cum.cummax()
    return {"Sharpe": round(np.sqrt(freq)*r.mean()/r.std(), 2),
            "CAGR%": round((cum.iloc[-1]**(freq/len(r))-1)*100, 1),
            "MaxDD%": round(((cum-peak)/peak).min()*100, 1),
            "Hit%": round((r > 0).mean()*100, 1), "N": len(r)}

def metrics_by_year(bt):
    d = {y: metrics(g["strat_ret"]) for y, g in bt.groupby(bt.index.year)}
    d["ALL"] = metrics(bt["strat_ret"])
    return pd.DataFrame(d).T

# ----------------------------------------------------------------------
# Robustesse : sensibilité aux paramètres + bootstrap par blocs (significativité)
# ----------------------------------------------------------------------
def sensitivity(features, tenor, lvl_now, instr_ret, split_i, feat_cols, slope=None,
                alphas=(5, 15, 40), horizons=(10, 21, 30), edges=(0.0, 0.3, 0.6)):
    rows = []
    for a in alphas:
        for h in horizons:
            p = wf_forecast_target(features, tenor, h, split_i, feat_cols, a)
            for e in edges:
                bt = backtest(p, lvl_now.reindex(p.index), instr_ret, slope=slope, edge_thresh=e)
                rows.append({"alpha": a, "horizon": h, "edge": e, "Sharpe": metrics(bt["strat_ret"])["Sharpe"]})
    return pd.DataFrame(rows)

def block_bootstrap_sharpe(returns, block=21, n=2000, freq=252, seed=0):
    r = returns.dropna().values; L = len(r)
    if L < block*3: return {"Sharpe": np.nan}
    rng = np.random.default_rng(seed); nb = int(np.ceil(L/block)); sh = []
    for _ in range(n):
        starts = rng.integers(0, L-block, nb)
        samp = np.concatenate([r[s:s+block] for s in starts])[:L]
        sd = samp.std()
        sh.append(np.sqrt(freq)*samp.mean()/sd if sd > 0 else 0.0)
    sh = np.array(sh)
    return {"Sharpe_obs": round(np.sqrt(freq)*r.mean()/r.std(), 2),
            "CI95": (round(np.percentile(sh, 2.5), 2), round(np.percentile(sh, 97.5), 2)),
            "p(>0)": round((sh > 0).mean(), 3)}

# ======================================================================
#  PUBLICATION UPGRADES
#  (B) vrais futures VIX à maturité constante (M1/M2 + roll)
#  (C) significativité du skill par block-bootstrap (Diebold-Mariano)
#  (bonus) régime Markov-switching intégré comme feature de la Ridge
# ======================================================================

# ---- (B0) Chargeur des futures VIX du CFE -> format long -----------------
def load_cfe_vix_futures(path):
    """Lit les CSV de futures VIX du CFE (cfe.cboe.com/market-data) -> DataFrame
       long ['date','expiry','settle']. 'path' = dossier de CSV (un par contrat)
       OU un seul CSV combiné. expiry = dernier jour coté du contrat (règlement final).
       Robuste au format CFE : détecte les colonnes Trade Date / Settle / Futures."""
    import glob, os
    def _read(fp):
        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]
        dcol = next((c for c in df.columns if "date" in c.lower()), None)
        scol = (next((c for c in df.columns if c.lower() == "settle"), None)
                or next((c for c in df.columns if "settle" in c.lower()), None)
                or next((c for c in df.columns if c.lower() == "close"), None))
        idcol = next((c for c in df.columns if c.lower() in ("futures", "symbol", "contract")), None)
        if dcol is None or scol is None:
            return pd.DataFrame()
        cols = [c for c in [dcol, scol, idcol] if c]
        df = df[cols].copy()
        df.columns = ["date", "settle"] + (["cid"] if idcol else [])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["settle"] = pd.to_numeric(df["settle"], errors="coerce")
        df = df.dropna(subset=["date", "settle"]); df = df[df["settle"] > 0]
        if "cid" not in df:
            df["cid"] = os.path.basename(fp)            # 1 fichier = 1 contrat
        return df
    files = glob.glob(os.path.join(path, "*.csv")) if os.path.isdir(path) else [path]
    parts = [d for d in (_read(f) for f in files) if len(d)]
    if not parts:
        raise ValueError("Aucun CSV CFE lisible dans " + str(path))
    df = pd.concat(parts, ignore_index=True)
    df["expiry"] = df.groupby("cid")["date"].transform("max")
    return df[["date", "expiry", "settle"]].sort_values("date").reset_index(drop=True)

# ---- (B) Future VIX à maturité constante --------------------------------
def constant_maturity_return(fut_long, target_days=30):
    """Rendement journalier d'un future VIX à maturité CONSTANTE (standard académique).
       fut_long : DataFrame long avec colonnes ['date','expiry','settle'] (un contrat par ligne).
       On détient en permanence les deux contrats encadrant `target_days`, pondérés
       linéairement par le time-to-maturity, et on roule chaque jour -> le coût du roll
       (négatif en contango) est capturé naturellement.
       NB : assembler fut_long depuis les CSV CFE (un fichier par contrat) :
            date = 'Trade Date', settle = 'Settle', expiry = date d'expiration du contrat."""
    f = fut_long.copy()
    f["date"] = pd.to_datetime(f["date"]); f["expiry"] = pd.to_datetime(f["expiry"])
    settle = f.pivot_table(index="date", columns="expiry", values="settle").sort_index()
    exps = np.array(settle.columns); dates = settle.index
    ret = settle / settle.shift(1) - 1
    out = pd.Series(index=dates, dtype=float)
    for d in dates[1:]:
        ttm = (exps - d.to_datetime64()) / np.timedelta64(1, "D")
        s = settle.loc[d]; valid = (~s.isna().values) & (ttm > 0)
        near = np.where(valid & (ttm <= target_days))[0]
        far  = np.where(valid & (ttm >  target_days))[0]
        if len(near) == 0 or len(far) == 0:
            continue
        i1 = near[np.argmax(ttm[near])]; i2 = far[np.argmin(ttm[far])]
        t1, t2 = ttm[i1], ttm[i2]; w1 = (t2 - target_days) / (t2 - t1)
        r1, r2 = ret.iloc[:, i1].loc[d], ret.iloc[:, i2].loc[d]
        if np.isnan(r1) or np.isnan(r2):
            continue
        out.loc[d] = w1 * r1 + (1 - w1) * r2
    return out.dropna().rename("cm_fut_ret")

# ---- (C) Significativité du skill (block-bootstrap, Diebold-Mariano) -----
def skill_significance(pred, target, horizon, block=21, n=2000, seed=0):
    """Teste si l'amélioration de RMSE vs random walk est statistiquement significative.
       H0 : le modèle ne fait pas mieux que la marche aléatoire (différentiel de perte <= 0).
       Renvoie le skill et la p-value (block-bootstrap du différentiel de perte quadratique)."""
    tgt = target.shift(-horizon).reindex(pred.index)
    c = pred.dropna().index.intersection(tgt.dropna().index)
    em = (pred.reindex(c) - tgt.reindex(c)).values**2
    er = (target.reindex(c).values - tgt.reindex(c).values)**2
    d = er - em                                    # >0 quand le modèle est meilleur
    L = len(d); rng = np.random.default_rng(seed); nb = int(np.ceil(L/block)); mu = []
    for _ in range(n):
        st = rng.integers(0, L-block, nb)
        mu.append(np.concatenate([d[k:k+block] for k in st])[:L].mean())
    mu = np.array(mu)
    return {"skill": round(1 - np.sqrt(em.mean())/np.sqrt(er.mean()), 3),
            "mean_loss_diff": round(float(d.mean()), 4),
            "p_value": round(float((mu <= 0).mean()), 4)}

# ---- (bonus) régime Markov-switching comme feature -----------------------
def regime_prob_series(level, train_end_i=None):
    """Probabilité de régime de vol HAUTE (Markov-switching 2 états, variance commutante).
       Probabilités FILTRÉES (récursives, n'utilisent que le passé pour l'inférence d'état).
       Fit unique sur tout l'échantillon (le lookahead se limite aux paramètres, pratique
       standard pour une feature de régime ; pour la version 100% temps réel, utiliser
       regime_highvol_prob qui ré-estime en walk-forward)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    s = level.dropna()
    fit_sample = s.iloc[:train_end_i] if train_end_i else s
    res = MarkovRegression(fit_sample, k_regimes=2, switching_variance=True).fit(disp=False)
    consts = [res.params.get(f"const[{k}]", np.nan) for k in range(2)]
    hi = int(np.nanargmax(consts))
    full = MarkovRegression(s, k_regimes=2, switching_variance=True).fit(disp=False)
    fp = full.filtered_marginal_probabilities
    col = fp.iloc[:, hi] if hasattr(fp, "iloc") else fp[:, hi]
    return pd.Series(np.asarray(col), index=s.index, name="regime")


# ======================================================================
# 8. Helpers de pipeline — persistés ici pour survivre à un redémarrage de kernel.
#    Une nouvelle session ne demande plus que :
#        import vix_lab as vl
#        P = vl.load_pipeline(r'.../ts_optionsdx.parquet')
#    (aucune cellule longue à retrouver ; l'extraction reste en cache parquet).
# ======================================================================
def build_final_features(betas):
    """Reconstruit la courbe (iv9, iv30, iv90) depuis les facteurs Nelson-Siegel,
       puis les features du modèle final (iv30, momentum, pentes).
       Renvoie (features, FEATS)."""
    import ts_extract as tx
    curve = pd.DataFrame(index=betas.index)
    for days, name in [(9, "iv9"), (30, "iv30"), (90, "iv90")]:
        curve[name] = betas.apply(lambda r: tx.iv_from_betas(r, days), axis=1)
    return tx.build_features(curve), tx.FEATS


def ridge_extra_factory(features, feat_cols, alpha=15.0):
    """Fabrique la fonction (series, horizon, split_i) -> prévisions du modèle ridge
       final, utilisable comme 'extra' dans compare_models et dans l'axe 3."""
    def fn(series, horizon, split_i):
        return wf_forecast_target(features, series, horizon, split_i, feat_cols, alpha=alpha)
    return fn


def load_vx_cfe(folder, pattern="CFE_VX_*.csv", start=None, end=None, verbose=True):
    """Charge les futures VIX (VX) du CFE : un CSV par contrat.

    Format CFE : 'Trade Date, Futures, Open, High, Low, Close, Settle, ...'.
    ATTENTION : selon le millésime, c'est Settle OU Close qui vaut 0 -- on prend
    Settle en priorité et on se rabat sur Close. L'échéance est le dernier jour
    coté du contrat, calculée sur TOUT l'historique avant le filtrage de dates.
    Renvoie ['date','expiry','settle'], prêt pour constant_maturity_return()."""
    import glob, os
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"Aucun fichier {pattern} dans {folder}")
    rows, skipped = [], []
    for f in files:
        name, d = os.path.basename(f), None
        for enc in ("utf-8", "latin-1"):
            for skip in (0, 1):                     # certains millésimes ont un préambule
                try:
                    t = pd.read_csv(f, encoding=enc, skiprows=skip)
                    t.columns = [str(c).strip() for c in t.columns]
                    if "Trade Date" in t.columns:
                        d = t; break
                except Exception:
                    continue
            if d is not None:
                break
        if d is None:
            skipped.append(name); continue
        d = d.rename(columns={"Trade Date": "date", "Futures": "cid"})
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        se = pd.to_numeric(d.get("Settle"), errors="coerce")
        cl = pd.to_numeric(d.get("Close"),  errors="coerce")
        d["settle"] = se.where(se > 0, cl)
        if "cid" not in d.columns:
            d["cid"] = name
        d = d.dropna(subset=["date", "settle", "cid"])
        d = d[d["settle"] > 0]
        if len(d):
            rows.append(d[["date", "cid", "settle"]])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["date", "cid"])
    out["expiry"] = out.groupby("cid")["date"].transform("max")
    if start: out = out[out["date"] >= pd.Timestamp(start)]
    if end:   out = out[out["date"] <= pd.Timestamp(end)]
    if verbose:
        print(f"{len(files)} fichiers | {len(out)} lignes | {out['cid'].nunique()} contrats"
              + (f" | ignorés : {skipped[:3]}" if skipped else ""))
    return out[["date", "expiry", "settle"]].sort_values("date").reset_index(drop=True)


def load_pipeline(cache_path, test_year=2016, min_dte_fit=5, min_maxdte=90):
    """Recharge TOUT le pipeline depuis le cache parquet d'IV ATM — sans ré-extraction.
       Renvoie un dict : betas, iv30, si, feats, FE, ridge_fn, ts_opt.
       Ex. :  P = vl.load_pipeline(r'.../ts_optionsdx.parquet')
              betas_long, iv30, si, ridge_fn = P['betas'], P['iv30'], P['si'], P['ridge_fn']"""
    import ts_extract as tx
    ts_opt = pd.read_parquet(cache_path)
    betas = tx.build_betas(ts_opt, min_dte_fit=min_dte_fit, min_maxdte=min_maxdte)
    iv30 = betas.apply(lambda r: tx.iv_from_betas(r, 30), axis=1)
    si, _ = tx.split_index(betas, test_year)
    feats, FE = build_final_features(betas)
    ridge_fn = ridge_extra_factory(feats, FE)
    return {"betas": betas, "iv30": iv30, "si": si, "feats": feats,
            "FE": FE, "ridge_fn": ridge_fn, "ts_opt": ts_opt}
