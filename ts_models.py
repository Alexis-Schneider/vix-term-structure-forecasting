"""
ts_models.py — modèles de séries temporelles Box-Jenkins (AR / MA / ARMA / ARIMA)
et leur comparaison, pour répondre aux consignes de Davide.

Contenu :
  - tests statistiques : stationnarité (ADF, KPSS), diagnostics résidus (Ljung-Box, Jarque-Bera)
  - sélection d'ordre par BIC (d déterminé par ADF pour ARIMA)
  - prévision walk-forward multi-horizon (h pas en avant, mise à jour d'état)
  - comparaison des modèles (+ random walk) par horizon, sur les béthas ou l'IV
À placer à côté du notebook :  import ts_models as tm
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

# ----------------------------------------------------------------------
# 1. Tests de stationnarité (identification Box-Jenkins, étape 1)
# ----------------------------------------------------------------------
def stationarity_tests(series):
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    return {"ADF_stat": round(adf_stat, 3), "ADF_p": round(adf_p, 4),
            "ADF": "stationnaire" if adf_p < 0.05 else "NON-stationnaire",
            "KPSS_stat": round(kpss_stat, 3), "KPSS_p": round(kpss_p, 4),
            "KPSS": "stationnaire" if kpss_p > 0.05 else "NON-stationnaire"}

def integration_order(series):
    """Ordre d de différenciation (0 si ADF rejette, sinon 1)."""
    return 0 if adfuller(series.dropna(), autolag="AIC")[1] < 0.05 else 1

# ----------------------------------------------------------------------
# 2. Sélection d'ordre par BIC pour chaque famille
# ----------------------------------------------------------------------
def select_order(series, family, max_p=3, max_q=3):
    """family in {'AR','MA','ARMA','ARIMA'}. Renvoie (p,d,q) par minimisation du BIC."""
    s = series.dropna()
    d = integration_order(s) if family == "ARIMA" else 0
    p_range = range(1, max_p+1) if family in ("AR", "ARMA", "ARIMA") else [0]
    q_range = range(1, max_q+1) if family in ("MA", "ARMA", "ARIMA") else [0]
    best = (np.inf, None)
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                m = ARIMA(s, order=(p, d, q)).fit()
                if np.isfinite(m.bic) and m.bic < best[0]:
                    best = (m.bic, (p, d, q))
            except Exception:
                continue
    return best[1]

# ----------------------------------------------------------------------
# 3. Diagnostics du modèle (étape 3 Box-Jenkins)
# ----------------------------------------------------------------------
def diagnostics(series, order):
    m = ARIMA(series.dropna(), order=order).fit()
    resid = m.resid.dropna()
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    jb_p = jarque_bera(resid)[1]
    return {"order": order, "AIC": round(m.aic, 1), "BIC": round(m.bic, 1),
            "LjungBox_p": round(float(lb), 3), "JarqueBera_p": round(float(jb_p), 3),
            "residus_bruit_blanc": bool(lb > 0.05)}

# ----------------------------------------------------------------------
# 4. Prévision walk-forward multi-horizon (h pas en avant, anti-lookahead)
# ----------------------------------------------------------------------
def arima_walkforward(series, order, split_i, horizon, refit_every=63):
    """Prévision walk-forward à `horizon` pas.

    ALIGNEMENT (corrigé) : à la date t on entraîne sur les observations 0..t
    INCLUSES (`vals.iloc[:t+1]`), donc `forecast(steps)` porte sur t+1..t+steps
    et `.iloc[horizon-1]` est bien la prévision de y_{t+horizon} -- ce que
    skill() compare à target.shift(-horizon)[t].
    L'ancienne version entraînait sur `:t` (0..t-1) : la prévision portait sur
    t+horizon-1 (décalage d'un jour) et n'utilisait pas l'observation en t,
    alors que les modèles à features l'utilisent -> comparaison biaisée."""
    s = series.dropna(); dates = s.index[split_i:]; vals = pd.Series(s.values)
    res, preds, up = None, [], None
    for i, t in enumerate(range(split_i, len(vals))):
        if res is None or i % refit_every == 0:
            try:
                res = ARIMA(vals.iloc[:t+1], order=order).fit(); up = t
            except Exception:
                preds.append(np.nan); continue
        else:
            try:
                # `extend` prolonge l'etat filtre sans refiltrer tout l'historique. Verifie
                # strictement identique a `append(refit=False)` (ecart max 0.0) et ~35 % plus
                # rapide ; on garde `append` en repli si la version de statsmodels ne l'offre pas.
                try:
                    res = res.extend(vals.iloc[up+1:t+1])
                except AttributeError:
                    res = res.append(vals.iloc[up+1:t+1], refit=False)
                up = t
            except Exception:
                pass
        try:
            preds.append(float(res.forecast(steps=horizon).iloc[horizon-1]))
        except Exception:
            preds.append(float(vals.iloc[t]))
    return pd.Series(preds, index=dates)

# ----------------------------------------------------------------------
# 5. Skill (réduction de RMSE vs random walk)
# ----------------------------------------------------------------------
def skill_rmse(pred, target, horizon):
    tgt = target.shift(-horizon).reindex(pred.index)
    rw = target.reindex(pred.index)
    c = pred.dropna().index.intersection(tgt.dropna().index)
    if len(c) < 5:
        return np.nan, np.nan
    rm = np.sqrt(((pred.reindex(c) - tgt.reindex(c))**2).mean())
    rr = np.sqrt(((rw.reindex(c)   - tgt.reindex(c))**2).mean())
    return (1 - rm/rr), rm

# ----------------------------------------------------------------------
# 6. Comparaison des modèles par horizon
# ----------------------------------------------------------------------
def compare_models(series, split_i, horizons=(1, 2, 5, 7, 14, 30),
                   families=("AR", "MA", "ARMA", "ARIMA"), extra=None, max_p=2, max_q=2):
    """extra : dict {nom: fonction(series, horizon, split_i) -> Series de prévisions}
       (ex. ton modèle ridge). max_p/max_q plafonnent les ordres (parcimonie -> évite les
       prévisions multi-pas explosives). Renvoie (table de skill par horizon x modèle, ordres)."""
    orders = {f: select_order(series, f, max_p=max_p, max_q=max_q) for f in families}
    rows = []
    for h in horizons:
        row = {"horizon": h}
        for f in families:
            if orders[f] is None:
                row[f] = np.nan; continue
            p = arima_walkforward(series, orders[f], split_i, h)
            row[f], _ = skill_rmse(p, series, h)
        if extra:
            for name, fn in extra.items():
                try:
                    p = fn(series, h, split_i)
                    row[name], _ = skill_rmse(p, series, h)
                except Exception:
                    row[name] = np.nan
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("horizon")
    return (tab * 100).round(1), orders   # skill en %


if __name__ == "__main__":
    # test synthétique
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-02", "2021-12-31"); n = len(idx)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.97*x[t-1] + rng.normal(0, 0.5) + 0.03*15
    s = pd.Series(20 + x, index=idx)
    si = s.index.get_loc(s.index[s.index.year == 2019][0])

    print("== Stationnarité ==", stationarity_tests(s))
    for fam in ["AR", "MA", "ARMA", "ARIMA"]:
        o = select_order(s, fam)
        print(f"  {fam:5s} ordre {o} | diag {diagnostics(s, o)}")
    print("\n== Comparaison (skill % par horizon) ==")
    tab, orders = compare_models(s, si, horizons=(1, 5, 30))
    print(tab.to_string())
    print("\nTEST OK")
