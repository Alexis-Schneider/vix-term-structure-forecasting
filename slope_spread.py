"""
slope_spread.py — trader le facteur qui porte le poids économique (section 5.4 du papier).

MOTIVATION
----------
La §3.4 montre que l'avantage multivarié se loge sur la COURBURE, dont la contribution
plafonne à 1,0 point de vol à trente jours, alors que le facteur qui porte le poids est la
PENTE, à 4,97 points. Or l'axe économique principal trade le point 30 jours, objet dominé
par le niveau et par la prime de variance. Conclure que la prime écrase le signal y est
presque circulaire. Le spread calendaire corrige cela :
  (a) il cible la pente directement ;
  (b) les deux jambes corrélant à 0,985, il annule l'essentiel de l'exposition sèche ;
  (c) il ne demande aucune chaîne d'options, donc sa version inconditionnelle se mesure sur
      tout l'échantillon de futures (2013-2026) et pas seulement sur les huit ans gratuits.

DEUX POINTS DE MÉTHODE, appris à nos dépens
--------------------------------------------
1. PROJECTION. L'edge du modèle est en points de volatilité d'options, le P&L du spread en
   points de futures. On projette donc la prévision de pente dans les unités du spread par
   une régression walk-forward S_t = a_t + b_t * s_t, exactement comme l'équation (4) le
   fait pour la règle sur le point 30 jours. Sans cela on compare des unités différentes.

2. PLACEBO EMBOÎTÉ. Une version antérieure comparait la règle au spread INCONDITIONNEL et
   trouvait une différence de -1,97 avec t = -3,22. C'était un artefact : le spread
   inconditionnel n'est pas le cas emboîté de la règle, et l'écart mesurait le renoncement
   au carry, non la qualité de la prévision. Avec la projection, le placebo emboîté cesse
   d'être dégénéré — la marche aléatoire projetée donne un edge égal au négatif de la pente
   résiduelle — et la différence appariée redevient nulle. C'est ce qu'on rapporte.

CONVENTIONS : identiques au reste du papier (33 bp, sigma* = 15 %, plafond 2, décalage d'un
jour, erreurs-types de Lo et test apparié via futures_signed).

Usage :  python slope_spread.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import vix_lab as vl
import futures_signed as fs

NEAR, FAR = 30, 60
COST_BPS = 33.0
TARGET_VOL, MAX_LEV = 0.15, 2.0
DELTA_MULT = 0.5          # seuil = DELTA_MULT x edge absolu median (règle de pouce assumée)


# ---------------------------------------------------------------- instrument
def spread_return(fut_long, near=NEAR, far=FAR):
    """Rendement quotidien du spread : long `far`, short `near`, en notionnel égal.
       On utilise les RENDEMENTS des positions à maturité constante et non la variation des
       prix interpolés : leur différence est exactement le roll (voir §4.1 du papier)."""
    return (vl.constant_maturity_return(fut_long, far)
            - vl.constant_maturity_return(fut_long, near)).dropna()


def spread_diagnostics(fut_long, near=NEAR, far=FAR, s_opt=None):
    S, rs_, _, _ = _projection(fut_long, s_opt, near, far)
    rn = vl.constant_maturity_return(fut_long, near)
    d = pd.DataFrame({"rn": rn, "rs": rs_, "S": S}).dropna()
    d["rf"] = d.rn + d.rs; d["pn"] = np.nan; d["pf"] = np.nan
    rs = d.rs
    ann = lambda x: (x.mean()*252*100, x.std()*np.sqrt(252)*100, np.sqrt(252)*x.mean()/x.std())
    return {"n": len(d), "slope_mean_pts": float(d.S.mean()),
            "slope_positive_pct": float(100*(d.S > 0).mean()),
            "leg_corr": float(d.rn.corr(d.rf)), "spread_vol_share": float(rs.std()/d.rn.std()),
            "near": ann(d.rn), "far": ann(d.rf), "spread": ann(rs)}


def roll_cost_spread(fut_long, near=NEAR, far=FAR, mean_position=1.0):
    """Turnover de roll du spread : les DEUX jambes roulent chaque jour, donc le coût
       structurel est le double de celui d'une position simple à maturité constante."""
    _, a_near = fs.roll_turnover(fut_long, near)
    _, a_far = fs.roll_turnover(fut_long, far)
    units = (a_near + a_far) * mean_position
    return {"units_unit_notional": a_near + a_far, "units_effective": units,
            "cost_17bp_pct": units*17/1e4*100, "cost_33bp_pct": units*33/1e4*100}


# ---------------------------------------------------------------- backtest
def run_spread(ret_, weight=None, cost_bps=COST_BPS):
    rv = ret_.rolling(20).std() * np.sqrt(252)
    size = (TARGET_VOL / rv).clip(upper=MAX_LEV).shift(1).fillna(0)
    w = pd.Series(1.0, index=ret_.index) if weight is None else weight.reindex(ret_.index).fillna(0.0)
    pos = w * size
    turn = pos.diff().abs().fillna(pos.abs())
    return (pos.shift(1)*ret_ - turn*cost_bps/1e4).dropna(), pos


def _ternary(edge, delta):
    return pd.Series(np.where(edge > delta, 1.0, np.where(edge < -delta, -1.0, 0.0)), index=edge.index)


_PROJ_CACHE = {}
CACHE_FILE = "_slope_cache.pkl"     # cache disque : ces series ne dependent ni de h ni du modele


def _projection(fut_long, s_opt, near, far):
    """Renvoie (S, rendement du spread, a_t, b_t) pour un couple de ténors.

       Les prix à maturité constante et la projection walk-forward coûtent quelques dizaines
       de secondes chacun et ne dépendent ni de l'horizon ni du modèle : on les mémoïse en
       mémoire et sur disque, pour que le notebook de réplication se relance en quelques
       secondes. Supprimer `_slope_cache.pkl` force le recalcul complet."""
    import os, pickle, hashlib
    # la cle DOIT inclure la courbe d'options : a_t et b_t dependent de s_opt, donc de la
    # construction retenue. Sans cela, changer de pipeline reutiliserait une projection fausse.
    tag = hashlib.md5(np.asarray(s_opt.dropna().values).tobytes()).hexdigest()[:8]
    key = (near, far, tag)
    if key in _PROJ_CACHE:
        return _PROJ_CACHE[key]
    if os.path.exists(CACHE_FILE):
        try:
            disk = pickle.load(open(CACHE_FILE, "rb"))
            if key in disk:
                _PROJ_CACHE[key] = disk[key]
                return _PROJ_CACHE[key]
        except Exception:
            disk = {}
    else:
        disk = {}
    S = (fs.cm_price(fut_long, far) - fs.cm_price(fut_long, near))
    rs = spread_return(fut_long, near, far)
    a_, b_ = fs.wf_projection(S, s_opt.reindex(S.index))
    _PROJ_CACHE[key] = (S, rs, a_, b_)
    disk[key] = _PROJ_CACHE[key]
    pickle.dump(disk, open(CACHE_FILE, "wb"))
    return _PROJ_CACHE[key]


def build_signals(fut_long, s_opt, forecast, near=NEAR, far=FAR):
    """Renvoie (rendement du spread, edge du modèle, edge du placebo), tous projetés.

       S_t = a_t + b_t * s_t estimée en expanding window ;
       edge modèle  = (a_t + b_t * prévision) - S_t
       edge placebo = (a_t + b_t * s_t)       - S_t = - (pente résiduelle)."""
    S, rs, a_, b_ = _projection(fut_long, s_opt, near, far)
    idx = forecast.dropna().index.intersection(rs.index).intersection(a_.dropna().index)
    aa, bb, SS = a_.reindex(idx), b_.reindex(idx), S.reindex(idx)
    return rs.reindex(idx), (aa + bb*forecast.reindex(idx)) - SS, (aa + bb*s_opt.reindex(idx)) - SS


def paired_test(rs, e_model, e_placebo, delta_mult=DELTA_MULT):
    delta = float(pd.concat([e_model, e_placebo]).abs().median() * delta_mult)
    m, pm = run_spread(rs, _ternary(e_model, delta))
    p, pp = run_spread(rs, _ternary(e_placebo, delta))
    i = m.index.intersection(p.index)
    sm, sem, n, _ = fs.sharpe_se(m.loc[i]); sp_, sep, _, _ = fs.sharpe_se(p.loc[i])
    d, sd, t, _ = fs.sharpe_diff(m.loc[i], p.loc[i])
    return {"model": round(sm, 2), "se_m": round(sem, 2), "placebo": round(sp_, 2), "se_p": round(sep, 2),
            "diff": round(d, 2), "se_d": round(sd, 2), "t": round(t, 2), "delta": round(delta, 2),
            "long%": round(100*(pm.loc[i] > 0).mean()), "short%": round(100*(pm.loc[i] < 0).mean()),
            "corr_pos": round(float(pm.loc[i].corr(pp.loc[i])), 2), "n": n}


# ---------------------------------------------------------------- pipeline
def slope_series(betas):
    import ts_extract as tx
    return (betas.apply(lambda r: tx.iv_from_betas(r, 90), axis=1)
            - betas.apply(lambda r: tx.iv_from_betas(r, 30), axis=1)).rename("slope")


def main(cache="ts_optionsdx.parquet", horizons=(5, 14, 30)):
    P = vl.load_pipeline(cache)
    betas, si, feats, FE = P["betas"], P["si"], P["feats"], P["FE"]
    vx = vl.load_vx_cfe(fs.CFE, verbose=False)
    s_opt = slope_series(betas)
    fh = {h: vl.wf_forecast_target(feats, s_opt, h, si, FE, alpha=15.0) for h in horizons}

    dg = spread_diagnostics(vx, s_opt=s_opt)
    print("=== Instrument (spread %d/%d) ===" % (FAR, NEAR))
    print("  n=%d | pente moyenne %+.2f pt, positive %.0f%% | correlation des jambes %.3f"
          % (dg["n"], dg["slope_mean_pts"], dg["slope_positive_pct"], dg["leg_corr"]))
    print("  ecart-type du spread = %.0f%% de celui de la jambe avant" % (100*dg["spread_vol_share"]))
    for k in ("near", "far", "spread"):
        m, v, s = dg[k]; print("  %-7s moyenne %+7.1f%%/an | vol %5.1f%% | Sharpe %+.2f" % (k, m, v, s))

    print("\n=== Spread inconditionnel : l'echantillon long change la reponse ===")
    u_all, _ = run_spread(_projection(vx, s_opt, NEAR, FAR)[1])
    for lab, sl in [("2013-2026 (tout)", slice(None)), ("2013-2015", slice("2013", "2015")),
                    ("2016-2019", slice("2016", "2019")), ("2020-2023", slice("2020", "2023")),
                    ("2024-2026", slice("2024", "2026")), ("2016-2023 (fenetre du papier)", slice("2016", "2023"))]:
        s_, se_, n_, _ = fs.sharpe_se(u_all[sl])
        print("  %-30s Sharpe %+.2f [%.2f]  n=%5d" % (lab, s_, se_, n_))
    rc = roll_cost_spread(vx, mean_position=0.65)
    print("  cout de roll du spread : %.1f unites/an a notionnel unitaire, %.1f effectives"
          % (rc["units_unit_notional"], rc["units_effective"]))
    print("    -> %.1f%% par an a 17 bp, %.1f%% a 33 bp, contre %+.1f%% de rendement brut"
          % (rc["cost_17bp_pct"], rc["cost_33bp_pct"], dg["spread"][0]))

    print("\n=== Regle projetee contre placebo emboite ===")
    rows = []
    for h in horizons:
        rs, em, ep = build_signals(vx, s_opt, fh[h])
        rows.append({"h": h, **paired_test(rs, em, ep)})
    tab = pd.DataFrame(rows).set_index("h")
    print(tab.to_string())

    print("\n=== Sensibilite au seuil : le signe de la difference n'est pas stable ===")
    for mult in (0.0, 0.25, 0.5, 1.0, 2.0):
        out = []
        for h in horizons:
            rs, em, ep = build_signals(vx, s_opt, fh[h])
            r = paired_test(rs, em, ep, delta_mult=mult)
            out.append("%+.2f (t=%+.2f)" % (r["diff"], r["t"]))
        print("  seuil x%.2f : %s" % (mult, " | ".join(out)))

    print("\n=== Second couple de tenors (30/90) ===")
    for h in horizons:
        rs, em, ep = build_signals(vx, s_opt, fh[h], near=30, far=90)
        r = paired_test(rs, em, ep)
        print("  h=%2d : modele %+.2f  placebo %+.2f  diff %+.2f [%.2f] t=%+.2f"
              % (h, r["model"], r["placebo"], r["diff"], r["se_d"], r["t"]))
    return tab


if __name__ == "__main__":
    main()
    print("""
CE QUI RESTE HORS DE PORTEE
---------------------------
- Le test conditionnel reste borne a 2016-2023 : il exige les chaines d'options. Seul
  l'inconditionnel beneficie de l'echantillon 2013-2026.
- Spread sur contrats listes (F1/F2) plutot que sur maturites constantes interpolees.
- Ponderation des jambes par le beta plutot qu'en notionnel egal.
""")
