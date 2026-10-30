"""
futures_signed.py — spécification principale du test économique (révision d'août 2026).

Motivation (critique de referee) : le signal du papier prédit une variation de la
volatilité implicite AU COMPTANT, alors qu'un future ne monétise que l'écart entre
la prévision et le PRIX COTÉ. La règle testée n'était donc pas celle qu'un desk
utiliserait. Ce module implémente la bonne règle et son placebo.

Règle « modèle »  : edge_t = f̂_{t+h} − F_t      (F_t = future 30j à maturité constante)
Règle « placebo » : edge_t = iv30_t − F_t        (marche aléatoire à la place du modèle)

Les deux règles sont IDENTIQUES en tout point sauf la prévision : même seuil, même
ciblage de risque, mêmes coûts, mêmes jours. Le placebo est donc le cas emboîté de
la règle modèle, exactement comme le random walk est le cas emboîté des prévisions
de l'axe 1. La différence des deux isole la contribution du modèle, et rien d'autre.

Inférence : erreurs-types HAC du Sharpe annualisé par la fonction d'influence de
Lo (2002), et test apparié modèle − placebo (les deux séries partagent l'instrument).

Usage :  python futures_signed.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import vix_lab as vl

CFE = "data/vix_futures_cfe"
COST_BPS = 33.0          # un tick de 0.05 pt sur un VIX à 15
DELTA = 0.3              # seuil en points de vol (identique au papier)
TARGET_VOL, MAX_LEV = 0.15, 2.0
TEST_START, TEST_END = "2016-01-01", "2023-12-31"


# ---------------------------------------------------------------- inférence
def nw_lrv(x, q=None):
    """Variance de long terme Newey-West (noyau de Bartlett)."""
    x = np.asarray(x, float) - np.mean(x); T = len(x)
    if q is None:
        q = int(np.floor(4 * (T / 100) ** (2 / 9)))
    s = (x @ x) / T
    for l in range(1, q + 1):
        s += 2 * (1 - l / (q + 1)) * (x[l:] @ x[:-l]) / T
    return max(s, 1e-12), q


def _if_sharpe(r):
    """Fonction d'influence du Sharpe par période (Lo 2002)."""
    r = np.asarray(r, float); mu = r.mean(); sd = r.std(ddof=1); s = mu / sd
    return (r - mu) / sd - 0.5 * s * (((r - mu) ** 2) / sd ** 2 - 1.0), s


def sharpe_se(r, freq=252):
    """Sharpe annualisé et son erreur-type HAC. Validé contre la formule
       analytique de Lo (2002, eq. 9) sous i.i.d. : voir __main__."""
    r = pd.Series(r).dropna().values
    IFs, s = _if_sharpe(r); lrv, q = nw_lrv(IFs)
    return np.sqrt(freq) * s, np.sqrt(freq) * np.sqrt(lrv / len(r)), len(r), q


def sharpe_diff(ra, rb, freq=252):
    """Test apparié de différence de Sharpe (HAC). Les deux séries étant
       exposées au même instrument, l'appariement est nettement plus puissant
       que la comparaison de deux intervalles séparés."""
    d = pd.DataFrame({"a": ra, "b": rb}).dropna()
    ia, sa = _if_sharpe(d.a.values); ib, sb = _if_sharpe(d.b.values)
    X = np.column_stack([ia, ib]) - np.column_stack([ia, ib]).mean(0)
    T = len(d); q = int(np.floor(4 * (T / 100) ** (2 / 9)))
    S = X.T @ X / T
    for l in range(1, q + 1):
        G = X[l:].T @ X[:-l] / T; S += (1 - l / (q + 1)) * (G + G.T)
    var = (S[0, 0] + S[1, 1] - 2 * S[0, 1]) / T
    diff = np.sqrt(freq) * (sa - sb)
    se = np.sqrt(freq) * np.sqrt(max(var, 1e-16))
    return diff, se, diff / se, T


# ------------------------------------------------- future à maturité constante
def cm_price(fut_long, target_days=30):
    """PRIX (et non rendement) du future VIX à maturité constante `target_days`,
       par interpolation linéaire en time-to-maturity entre les deux contrats
       encadrants. C'est le prix contre lequel la prévision doit être signée."""
    f = fut_long.copy()
    f["date"] = pd.to_datetime(f["date"]); f["expiry"] = pd.to_datetime(f["expiry"])
    settle = f.pivot_table(index="date", columns="expiry", values="settle").sort_index()
    exps = np.array(settle.columns); out = pd.Series(index=settle.index, dtype=float)
    for d in settle.index:
        ttm = (exps - d.to_datetime64()) / np.timedelta64(1, "D")
        s = settle.loc[d]; ok = (~s.isna().values) & (ttm > 0)
        near = np.where(ok & (ttm <= target_days))[0]
        far = np.where(ok & (ttm > target_days))[0]
        if len(near) == 0 or len(far) == 0:
            continue
        i1 = near[np.argmax(ttm[near])]; i2 = far[np.argmin(ttm[far])]
        t1, t2 = ttm[i1], ttm[i2]; w = (t2 - target_days) / (t2 - t1)
        out.loc[d] = w * s.iloc[i1] + (1 - w) * s.iloc[i2]
    return out.dropna().rename("cm_price")


# ---------------------------------------------------------------- backtest
def run_rule(edge, instr_ret, delta=DELTA, cost_bps=COST_BPS,
             target_vol=TARGET_VOL, max_lev=MAX_LEV):
    """Règle ternaire à seuil, ciblage de volatilité, coûts au turnover.
       La position est décalée d'un jour (`shift(1)`) : le signal d'un jour n'est
       exécuté qu'au jour suivant, ce qui rend le décalage horaire entre la cote
       SPX de 16:00 ET et le settlement CFE sans conséquence."""
    df = pd.DataFrame({"edge": edge, "ret": instr_ret}).dropna()
    raw = pd.Series(np.where(df.edge > delta, 1.0,
                    np.where(df.edge < -delta, -1.0, 0.0)), index=df.index)
    rv = df["ret"].rolling(20).std() * np.sqrt(252)
    size = (target_vol / rv).clip(upper=max_lev).shift(1).fillna(0)
    pos = raw * size
    turn = pos.diff().abs().fillna(pos.abs())
    df["r"] = pos.shift(1) * df["ret"] - turn * cost_bps / 1e4
    df["pos"] = pos
    return df.dropna(subset=["r"])


def risk(r, freq=252):
    r = pd.Series(r).dropna()
    cum = (1 + r).cumprod()
    return {"CAGR%": round((cum.iloc[-1] ** (freq / len(r)) - 1) * 100, 1),
            "MaxDD%": round((cum / cum.cummax() - 1).min() * 100, 1)}


# ------------------------------------------------ projection dans l'espace des futures
def wf_projection(F_, iv_, min_obs=250):
    """Regression F = a + b*iv30 estimee en EXPANDING WINDOW : (a_t, b_t) n'utilisent
       que l'information disponible jusqu'a t inclus.

       Motivation : le VIX est un taux de variance-swap couvrant toute la nappe de strikes,
       notre cible est une volatilite at-the-money. L'ecart de definition (le skew) vaut
       plusieurs points de vol et n'est PAS une prime de risque. Signer une prevision ATM
       contre un prix variance-swap donnerait un short permanent -- la regle ne testerait
       rien. On projette donc la prevision dans les unites du prix cote avant de la signer.

       Renvoie (a, b), deux Series alignees sur l'index commun."""
    d = pd.DataFrame({"F": F_, "iv": iv_}).dropna()
    a = pd.Series(index=d.index, dtype=float); b = pd.Series(index=d.index, dtype=float)
    x, y = d.iv.values, d.F.values
    cx, cy = np.cumsum(x), np.cumsum(y); cxx, cxy = np.cumsum(x * x), np.cumsum(x * y)
    for k in range(min_obs - 1, len(d)):
        n = k + 1; den = n * cxx[k] - cx[k] ** 2
        if den <= 0:
            continue
        bb = (n * cxy[k] - cx[k] * cy[k]) / den
        b.iloc[k] = bb; a.iloc[k] = (cy[k] - bb * cx[k]) / n
    return a, b


def roll_turnover(fut_long, target_days=30, start=TEST_START, end=TEST_END):
    """Turnover quotidien impose par le maintien d'une maturite constante : chaque jour les
       poids des deux contrats encadrants se deplacent. Renvoie (turnover moyen/jour,
       unites de notionnel par an). Ce cout est SUPPORTE A L'IDENTIQUE par la strategie,
       le placebo et le passif -- il ne change aucune comparaison, mais il peut annuler
       la rentabilite absolue."""
    f = fut_long.copy()
    f["date"] = pd.to_datetime(f["date"]); f["expiry"] = pd.to_datetime(f["expiry"])
    settle = f.pivot_table(index="date", columns="expiry", values="settle").sort_index()
    exps = np.array(settle.columns); rec = []
    for d in settle.index:
        ttm = (exps - d.to_datetime64()) / np.timedelta64(1, "D"); sv = settle.loc[d]
        ok = (~sv.isna().values) & (ttm > 0)
        near = np.where(ok & (ttm <= target_days))[0]; far = np.where(ok & (ttm > target_days))[0]
        if len(near) == 0 or len(far) == 0:
            continue
        i1 = near[np.argmax(ttm[near])]; i2 = far[np.argmin(ttm[far])]
        t1, t2 = ttm[i1], ttm[i2]
        rec.append((d, exps[i1], exps[i2], (t2 - target_days) / (t2 - t1)))
    W = pd.DataFrame(rec, columns=["d", "c1", "c2", "w1"]).set_index("d").loc[start:end]
    tn = []
    for k in range(1, len(W)):
        p_, c_ = W.iloc[k - 1], W.iloc[k]
        prev = {p_.c1: p_.w1, p_.c2: 1 - p_.w1}; cur = {c_.c1: c_.w1, c_.c2: 1 - c_.w1}
        tn.append(sum(abs(cur.get(x, 0) - prev.get(x, 0)) for x in set(prev) | set(cur)))
    tn = np.array(tn)
    return tn.mean(), tn.mean() * 252


# ---------------------------------------------------------------- pipeline
def main(cache="ts_optionsdx.parquet", horizons=(5, 14, 30), label="principal"):
    P = vl.load_pipeline(cache)
    iv30, si, ridge = P["iv30"], P["si"], P["ridge_fn"]
    vx = vl.load_vx_cfe(CFE, verbose=False)
    F = cm_price(vx, 30)
    ret = vl.constant_maturity_return(vx, 30)
    win = slice(TEST_START, TEST_END)
    F, ret, iv30w = F[win], ret[win], iv30[win]

    # --- base entre l'objet prédit (vol ATM 30j) et l'instrument traité (future 30j)
    b = pd.DataFrame({"F": F, "iv": iv30w}).dropna()
    beta, a = np.polyfit(b.iv, b.F, 1)
    r2 = np.corrcoef(b.iv, b.F)[0, 1] ** 2
    basis = (b.F - b.iv)
    print(f"\n=== Base instrument/cible ({len(b)} jours) ===")
    print(f"  F = {a:.2f} + {beta:.3f} * iv30   R2 = {r2:.3f}")
    print(f"  base F - iv30 : moyenne {basis.mean():+.2f} pt, ecart-type {basis.std():.2f}, "
          f"positive {100*(basis>0).mean():.0f}% du temps")

    rows, series = [], {}
    for h in horizons:
        f_hat = ridge(iv30, h, si).reindex(F.index)
        m = run_rule(f_hat - F, ret)                    # règle modèle
        p = run_rule(iv30w - F, ret)                    # placebo : marche aléatoire
        idx = m.index.intersection(p.index)
        m, p = m.loc[idx], p.loc[idx]
        series[h] = (m["r"], p["r"])
        sm, sem, T, q = sharpe_se(m["r"]); sp, sep, _, _ = sharpe_se(p["r"])
        d, sd, t, _ = sharpe_diff(m["r"], p["r"])
        rows.append({"h": h, "SR_modele": round(sm, 2), "SE_m": round(sem, 2),
                     "SR_placebo": round(sp, 2), "SE_p": round(sep, 2),
                     "diff": round(d, 2), "SE_diff": round(sd, 2), "t": round(t, 2),
                     "CAGR%": risk(m["r"])["CAGR%"], "MaxDD%": risk(m["r"])["MaxDD%"],
                     "expo%": round(100 * (m["pos"] != 0).mean(), 0), "N": T, "lags": q})
    tab = pd.DataFrame(rows).set_index("h")
    print(f"\n=== Regle signee contre le future — {label} ({cache}) ===")
    print(tab.to_string())
    return tab, series


if __name__ == "__main__":
    # validation de l'estimateur d'erreur-type avant tout usage
    rng = np.random.default_rng(0); r = rng.normal(0.0004, 0.01, 4000)
    sr, se, T, _ = sharpe_se(r); s = sr / np.sqrt(252)
    print("VALIDATION Lo(2002) : SE estimee %.4f | formule analytique %.4f"
          % (se, np.sqrt(252 * (1 + s ** 2 / 2) / T)))
    main()
