"""
paper_numbers.py — recalcule TOUS les chiffres du papier sur la construction principale
(parité call-put + inversion Black-Scholes, `ts_reinvert.parquet`).

La construction alternative (IV du fournisseur, `ts_optionsdx.parquet`) est traitée en
robustesse par la partie `alt`.

Usage :  python paper_numbers.py <partie>
         parties : stat | mcs | fwd | cw | mult | boot | econ | span | alt | app | conc | decomp | coverage | eiv | extra
                   cwboot | rw | spa | mech   (revision rapporteur)
Chaque partie écrit ses résultats dans `results/numbers.json` (fusion incrémentale), pour que
le tout puisse être relancé morceau par morceau.
"""
import sys, os, json, pickle, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import numpy as np, pandas as pd
import vix_lab as vl, ts_extract as tx, futures_signed as fs

CACHE_MAIN, CACHE_ALT = "ts_reinvert.parquet", "ts_optionsdx.parquet"
H = [1, 2, 5, 7, 14, 30]
HS = [5, 14, 30]
OUT = "results/numbers.json"


def save(d):
    cur = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cur.update(d); json.dump(cur, open(OUT, "w"), indent=1, default=float)


def load_all(cache=CACHE_MAIN):
    P = vl.load_pipeline(cache)
    pre = pickle.load(open("cache/preds_reinvert.pkl", "rb"))
    arma = pickle.load(open("cache/arma_bic_reinvert.pkl", "rb"))
    for h in H:
        pre[h]["ARMA"] = arma[("iv30", h)]
    return P, pre, arma


# ------------------------------------------------------------------ benchmarks ancrés
def rmse_(pred, tgt, h):
    y = tgt.shift(-h); c = pred.dropna().index.intersection(y.dropna().index)
    return float(np.sqrt(((pred.reindex(c) - y.reindex(c)) ** 2).mean()))


def bench_mean(y, split_i):
    return y.expanding(250).mean().iloc[split_i:]


def bench_ar1(y, h, split_i):
    mu = y.expanding(250).mean()
    rho = y.rolling(500).apply(lambda v: pd.Series(v).autocorr(1), raw=False)
    out = {}
    for t in range(split_i, len(y)):
        m_, r_ = mu.iloc[t], rho.iloc[t]
        if np.isnan(m_) or np.isnan(r_):
            continue
        out[y.index[t]] = m_ + (r_ ** h) * (y.iloc[t] - m_)
    return pd.Series(out)


def har(y, h, split_i, refit=126):
    X = pd.DataFrame({"d": y, "w": y.rolling(5).mean(), "m": y.rolling(22).mean()}).dropna()
    lvl = y.reindex(X.index); d = pd.concat([X, (lvl.shift(-h) - lvl).rename("t")], axis=1).dropna()
    Xv, yv, idx = d[["d", "w", "m"]].values, d["t"].values, d.index
    out, co_ = {}, None
    for k, dt in enumerate(y.index[split_i:]):
        if dt not in X.index:
            continue
        j = idx.searchsorted(dt)
        if k % refit == 0 or co_ is None:
            e = max(j - h, 1)
            if e < 30:
                continue
            co_ = np.linalg.lstsq(np.c_[np.ones(e), Xv[:e]], yv[:e], rcond=None)[0]
        out[dt] = y.loc[dt] + float(np.r_[1, X.loc[dt].values] @ co_)
    return pd.Series(out)


CAL, TD = 365., 252.


def forward_vol(b, h, tenor=30):
    hc = h * CAL / TD
    v = b.apply(lambda r: ((tx.iv_from_betas(r, hc + tenor) / 100.) ** 2 * ((hc + tenor) / CAL)
                           - (tx.iv_from_betas(r, hc) / 100.) ** 2 * (hc / CAL)) / (tenor / CAL), axis=1)
    return (100 * np.sqrt(v[v > 0])).dropna()


# ------------------------------------------------------------------ MCS et Clark-West
def mcs(L, alpha=0.10, nboot=2000, seed=0, block=None):
    """Model confidence set, statistique T_max. Le bloc DOIT couvrir le recouvrement des
       erreurs à h pas : passer block=max(2h,5). Un bloc plus court rend les différentiels
       rééchantillonnés trop indépendants et élimine trop facilement."""
    rng = np.random.default_rng(seed); L = L.dropna(); n = len(L); models = list(L.columns)
    B = int(block) if block else max(int(n ** (1 / 3)), 5)
    nbk = int(np.ceil(n / B))
    boot = [np.concatenate([np.arange(k, min(k + B, n))
            for k in rng.integers(0, max(n - B, 1), nbk)])[:n] for _ in range(nboot)]
    dropped, pstop = [], None
    while len(models) > 1:
        X = L[models].values; dbar = X.mean(0) - X.mean(0).mean()
        db = np.array([X[ix].mean(0) - X[ix].mean(0).mean() for ix in boot])
        var = np.where(((db - dbar) ** 2).mean(0) <= 0, 1e-12, ((db - dbar) ** 2).mean(0))
        t = dbar / np.sqrt(var)
        p = float((((db - dbar) / np.sqrt(var)).max(1) > t.max()).mean())
        if p >= alpha:
            pstop = p; break
        dropped.append((models[int(np.argmax(t))], round(p, 3))); models.pop(int(np.argmax(t)))
    return models, dropped, pstop


def clark_west(pred, rw, tgt, h, lags=None):
    """Statistique de Clark-West (2007) pour modèles emboîtés, erreurs-types Newey-West."""
    y = tgt.shift(-h)
    c = pred.dropna().index.intersection(rw.dropna().index).intersection(y.dropna().index)
    p, r, a = pred.reindex(c).values, rw.reindex(c).values, y.reindex(c).values
    f = (a - r) ** 2 - ((a - p) ** 2 - (r - p) ** 2)
    T = len(f)
    if lags is None:
        lags = max(2 * h, 7)
    lrv, _ = fs.nw_lrv(f, q=int(lags))
    return float(f.mean() / np.sqrt(lrv / T)), T


def holm_bh(pvals):
    p = np.asarray(pvals, float); m = len(p); o = np.argsort(p); ps = p[o]
    # Holm
    adj = np.maximum.accumulate(ps * (m - np.arange(m)))
    holm = np.empty(m); holm[o] = np.minimum(adj, 1.0)
    # Benjamini-Hochberg
    bh_ = np.minimum.accumulate((ps * m / (np.arange(m) + 1))[::-1])[::-1]
    bh = np.empty(m); bh[o] = np.minimum(bh_, 1.0)
    return holm, bh


# ================================================================== parties
def part_stat():
    P, pre, arma = load_all()
    iv30, si, b = P["iv30"], P["si"], P["betas"]
    r = {"n_days": len(b), "r2_med": float(b.R2.median()), "n_mat_med": float(b.n_mat.median()),
         "test_start": str(iv30.index[si].date()), "sd_test": float(iv30.iloc[si:].std())}
    r["skills"] = {m: [round(100 * vl.skill(pre[h][m], iv30, h), 1) for h in H]
                   for m in ["ARMA", "HAR", "VAR", "Ridge"]}
    anc = {}
    for h in HS:
        rw = rmse_(iv30.iloc[si:], iv30, h); mn = rmse_(bench_mean(iv30, si), iv30, h)
        ar = rmse_(bench_ar1(iv30, h, si), iv30, h); rd = rmse_(pre[h]["Ridge"], iv30, h)
        anc[h] = dict(RW=round(rw, 3), Mean=round(mn, 3), AR1=round(ar, 3), Ridge=round(rd, 3),
                      s_rw=round(100 * (1 - rd / rw), 1), s_mn=round(100 * (1 - rd / mn), 1),
                      s_ar=round(100 * (1 - rd / ar), 1))
    r["anchor"] = anc
    fac = {}
    for nm, s in [("beta0", b.beta0), ("beta1", b.beta1), ("beta2", b.beta2)]:
        fac[nm] = {"ARMA": [round(100 * vl.skill(arma[(nm, h)], s, h), 1) for h in HS],
                   "Ridge": [round(100 * vl.skill(P["ridge_fn"](s, h, si), s, h), 1) for h in HS]}
    r["factors"] = fac
    # contributions : loading * sd
    lam = 68.0 / 365.0
    def load(tau_d):
        x = (tau_d / 365.0) / lam
        l1 = (1 - np.exp(-x)) / x
        return 1.0, l1, l1 - np.exp(-x)
    L30 = load(30)
    r["contrib30"] = [round(L30[k] * s.std(), 2) for k, s in enumerate([b.beta0, b.beta1, b.beta2])]
    grid = np.arange(1, 1500)
    l2 = np.array([load(t)[2] for t in grid])
    r["curv_peak"] = [round(float(l2.max()), 4), int(grid[l2.argmax()]),
                      round(float(l2.max() * b.beta2.std()), 2)]
    save({"stat": r}); print(json.dumps(r, indent=1, default=float))


def part_mcs():
    P, pre, _ = load_all()
    iv30, si, b = P["iv30"], P["si"], P["betas"]
    res = {}
    for h in HS:
        tgt = iv30.shift(-h)
        pr = {"RW": iv30.iloc[si:], **{m: pre[h][m] for m in ["ARMA", "HAR", "VAR", "Ridge"]}}
        L = pd.DataFrame({k: (v - tgt.reindex(v.index)) ** 2 for k, v in pr.items()}).dropna()
        keep, dropped, ps = mcs(L, block=max(2 * h, 5))
        keep12, dr12, ps12 = mcs(L, block=max(int(len(L) ** (1 / 3)), 5))
        res[h] = {"keep": keep, "dropped": dropped, "p": round(ps, 3) if ps else None,
                  "block": max(2 * h, 5), "n": len(L),
                  "short_block_keep": keep12, "short_block_dropped": dr12}
    save({"mcs": res}); print(json.dumps(res, indent=1, default=float))


def part_fwd():
    P, pre, _ = load_all()
    iv30, si, b = P["iv30"], P["si"], P["betas"]
    fwd_all = forward_vol(b, 1)  # positivité
    res = {"fwd_pos_days": {}}
    rows = {}
    for h in H:
        tgt = iv30.shift(-h)
        fw = forward_vol(b, h).iloc[si:]
        res["fwd_pos_days"][h] = int(len(forward_vol(b, h)))
        pr = {"RW": iv30.iloc[si:], "Ridge": pre[h]["Ridge"], "HAR": pre[h]["HAR"], "Forward": fw}
        L = pd.DataFrame({k: (v - tgt.reindex(v.index)) ** 2 for k, v in pr.items()}).dropna()
        keep, dropped, ps = mcs(L, block=max(2 * h, 5))
        rows[h] = {**{k: round(float(np.sqrt(L[k].mean())), 3) for k in pr},
                   "keep": keep, "p": round(ps, 3) if ps else None, "n": len(L)}
    res["table"] = rows
    save({"fwd": res}); print(json.dumps(res, indent=1, default=float))


def part_cw():
    P, pre, arma = load_all()
    iv30, si, b = P["iv30"], P["si"], P["betas"]
    rw = iv30.iloc[si:]
    res = {"iv30": {}, "trunc": {}}
    for h in H:
        res["iv30"][h] = {m: round(clark_west(pre[h][m], rw, iv30, h)[0], 2)
                          for m in ["ARMA", "HAR", "VAR", "Ridge"]}
    for lg in [7, 29, 45, 60]:
        t, _ = clark_west(pre[30]["Ridge"], rw, iv30, 30, lags=lg)
        from scipy.stats import norm
        res["trunc"][lg] = [round(t, 2), round(float(1 - norm.cdf(t)), 4)]
    save({"cw": res}); print(json.dumps(res, indent=1, default=float))


def part_mult():
    """Famille explicitement énumérée : 24 tests du Tableau 2 (4 modèles x 6 horizons)
       + 18 du Tableau 5 (2 modèles x 3 facteurs x 3 horizons) = 42."""
    from scipy.stats import norm
    P, pre, arma = load_all()
    iv30, si, b = P["iv30"], P["si"], P["betas"]
    names, tstats = [], []
    rw = iv30.iloc[si:]
    for h in H:
        for m in ["ARMA", "HAR", "VAR", "Ridge"]:
            t, _ = clark_west(pre[h][m], rw, iv30, h)
            names.append(f"{m}/iv30/h={h}"); tstats.append(t)
    for nm, s in [("beta0", b.beta0), ("beta1", b.beta1), ("beta2", b.beta2)]:
        rwf = s.iloc[si:]
        for h in HS:
            for m, pr in [("ARMA", arma[(nm, h)]), ("Ridge", P["ridge_fn"](s, h, si))]:
                t, _ = clark_west(pr, rwf, s, h)
                names.append(f"{m}/{nm}/h={h}"); tstats.append(t)
    p = np.array([1 - norm.cdf(t) for t in tstats])
    holm, bh = holm_bh(p)
    res = {"n_tests": len(p), "n_raw": int((p < .05).sum()),
           "n_holm": int((holm < .05).sum()), "n_bh": int((bh < .05).sum()),
           "survivors_holm": [names[i] for i in np.argsort(p)[:len(p)] if holm[i] < .05],
           "detail": {names[i]: [round(tstats[i], 2), round(float(p[i]), 5),
                                 round(float(holm[i]), 4), round(float(bh[i]), 4)]
                      for i in np.argsort(p)}}
    save({"mult": res}); print(json.dumps(res, indent=1, default=float))


def part_boot():
    P, pre, _ = load_all()
    iv30, si = P["iv30"], P["si"]
    rng = np.random.default_rng(0); res = {}
    for h in HS:
        tgt = iv30.shift(-h)
        d = pd.DataFrame({"m": pre[h]["Ridge"], "r": iv30.iloc[si:], "t": tgt}).dropna()
        em, er = (d.m - d.t).values ** 2, (d.r - d.t).values ** 2
        n = len(d); B = max(2 * h, 5); nbk = int(np.ceil(n / B)); out = []
        for _ in range(2000):
            ix = np.concatenate([np.arange(k, min(k + B, n))
                                 for k in rng.integers(0, max(n - B, 1), nbk)])[:n]
            out.append(100 * (1 - np.sqrt(em[ix].mean()) / np.sqrt(er[ix].mean())))
        res[h] = [round(100 * (1 - np.sqrt(em.mean()) / np.sqrt(er.mean())), 1),
                  round(float(np.percentile(out, 2.5)), 1), round(float(np.percentile(out, 97.5)), 1)]
    save({"boot": res}); print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ économie
def _futures():
    vx = vl.load_vx_cfe(fs.CFE, verbose=False)
    F = fs.cm_price(vx, 30)
    ret_full = vl.constant_maturity_return(vx, 30)
    return vx, F, ret_full


def part_econ():
    P, pre, _ = load_all()
    iv30, si, ridge = P["iv30"], P["si"], P["ridge_fn"]
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]; ivw = iv30["2016":"2023"]
    a_wf, b_wf = fs.wf_projection(Fall, iv30.reindex(Fall.index))
    aa, bb = a_wf.reindex(F.index), b_wf.reindex(F.index)
    res = {"proj": [round(float(aa.min()), 2), round(float(aa.max()), 2),
                    round(float(bb.min()), 3), round(float(bb.max()), 3)]}
    d = pd.DataFrame({"F": F, "iv": ivw}).dropna()
    beta, a = np.polyfit(d.iv, d.F, 1); basis = d.F - d.iv
    res["basis"] = {"a": round(float(a), 2), "b": round(float(beta), 3),
                    "r2": round(float(np.corrcoef(d.iv, d.F)[0, 1] ** 2), 3), "n": len(d),
                    "mean": round(float(basis.mean()), 2), "sd": round(float(basis.std()), 2),
                    "pos": round(100 * float((basis > 0).mean()))}
    # passif : mêmes conventions que la règle (ciblage, coûts), position -1
    rv = ret_full.rolling(20).std().mul(np.sqrt(252))
    size = (0.15 / rv).clip(upper=2.0).shift(1).fillna(0)
    pos_p = -1.0 * size
    turn_p = pos_p.diff().abs().fillna(pos_p.abs())
    passive_full = (pos_p.shift(1) * ret_full - turn_p * 33 / 1e4).dropna()

    def block_se_diff(ra, rb, nb=2000, block=21, seed=0):
        rng = np.random.default_rng(seed)
        dd = pd.DataFrame({"a": ra, "b": rb}).dropna(); n = len(dd)
        A, Bv = dd.a.values, dd.b.values; nbk = int(np.ceil(n / block)); out = []
        for _ in range(nb):
            ix = np.concatenate([np.arange(k, min(k + block, n))
                                 for k in rng.integers(0, max(n - block, 1), nbk)])[:n]
            sa = np.sqrt(252) * A[ix].mean() / A[ix].std(ddof=1)
            sb = np.sqrt(252) * Bv[ix].mean() / Bv[ix].std(ddof=1)
            out.append(sa - sb)
        return float(np.std(out, ddof=1))

    rows, cur = {}, {}
    for h in HS:
        f_hat = ridge(iv30, h, si).reindex(F.index)
        m = fs.run_rule((aa + bb * f_hat) - F, ret)
        p = fs.run_rule((aa + bb * ivw.reindex(F.index)) - F, ret)
        i = m.index.intersection(p.index); m, p = m.loc[i], p.loc[i]; cur[h] = (m, p)
        sm, sem, N, _ = fs.sharpe_se(m.r); sp, sep, _, _ = fs.sharpe_se(p.r)
        dd, sd, t, _ = fs.sharpe_diff(m.r, p.r)
        em = ((aa + bb * f_hat) - F).reindex(i); ep = ((aa + bb * ivw.reindex(F.index)) - F).reindex(i)
        opp = (np.sign(m.pos) * np.sign(p.pos)) < 0
        diffp = (np.sign(m.pos) != np.sign(p.pos))
        contrib = 252 * 100 * float((m.r[opp] - p.r[opp]).sum()) / len(m)
        pa = passive_full.reindex(i).dropna()
        spa, sepa, _, _ = fs.sharpe_se(pa)
        dpp, sdpp, tpp, _ = fs.sharpe_diff(p.r.reindex(pa.index), pa)
        rho_pp = float(p.pos.reindex(pa.index).corr(pos_p.reindex(pa.index)))
        rows[h] = {"model": round(sm, 2), "se_m": round(sem, 2), "placebo": round(sp, 2),
                   "se_p": round(sep, 2), "diff": round(dd, 2), "se_d": round(sd, 2),
                   "t": round(t, 2), "boot_se_d": round(block_se_diff(m.r, p.r), 2),
                   "CAGR": fs.risk(m.r)["CAGR%"], "MaxDD": fs.risk(m.r)["MaxDD%"],
                   "long": round(100 * float((m.pos > 0).mean())), "n": N,
                   "med_edge_gap": round(float((em - ep).abs().median()), 2),
                   "med_edge": round(float(em.abs().median()), 2),
                   "diff_pos_pct": round(100 * float(diffp.mean())),
                   "opp_pct": round(100 * float(opp.mean())),
                   "rho": round(float(m.pos.corr(p.pos)), 2),
                   "opp_contrib_pp": round(contrib, 1),
                   "passive_same_days": round(spa, 2),
                   "plac_minus_pass": [round(dpp, 2), round(sdpp, 2), round(tpp, 2), round(rho_pp, 2)],
                   "meanabs_w": round(float(m.pos.abs().mean()), 3)}
    res["projected"] = rows
    # non projeté et spot
    unp, spot = {}, {}
    for h in HS:
        f_hat = ridge(iv30, h, si).reindex(F.index)
        m = fs.run_rule(f_hat - F, ret); p = fs.run_rule(ivw.reindex(F.index) - F, ret)
        i = m.index.intersection(p.index); m, p = m.loc[i], p.loc[i]
        dd, sd, t, _ = fs.sharpe_diff(m.r, p.r)
        unp[h] = {"model": round(fs.sharpe_se(m.r)[0], 2), "placebo": round(fs.sharpe_se(p.r)[0], 2),
                  "diff": round(dd, 2), "se_d": round(sd, 2), "t": round(t, 2),
                  "long": round(100 * float((m.pos > 0).mean()))}
        bt = vl.backtest(ridge(iv30, h, si), iv30, ret, cost_bps=33.0, stop_loss=None)
        spot[h] = round(fs.sharpe_se(bt["strat_ret"])[0], 2)
    res["unprojected"] = unp; res["spot"] = spot
    sfull, sefull, _, _ = fs.sharpe_se(passive_full["2016":"2023"])
    res["passive"] = {"sharpe": round(sfull, 2), "se": round(sefull, 2),
                      **fs.risk(passive_full["2016":"2023"])}
    tj, ann = fs.roll_turnover(vx, 30)
    res["roll"] = {"per_day": round(tj, 4), "per_year": round(ann, 1)}
    inst = ret_full["2013":"2023"]
    res["instrument"] = {"vol": round(100 * float(inst.std() * np.sqrt(252)), 1),
                         "mean": round(100 * float(inst.mean() * 252), 1),
                         "worst": round(100 * float(inst.min()), 1),
                         "mean_test": round(100 * float(ret.mean() * 252), 1),
                         "price_drift_test": round(100 * float((F.iloc[-1] / F.iloc[0]) ** (252 / len(F)) - 1), 1)}
    # robustesse : proportionnel, plein horizon, décalé
    rob = {}
    for name in ["proportional", "fullhorizon", "lagged"]:
        r_ = {}
        for h in HS:
            f_hat = ridge(iv30, h, si).reindex(F.index)
            e_m = (aa + bb * f_hat) - F; e_p = (aa + bb * ivw.reindex(F.index)) - F
            if name == "lagged":
                e_m, e_p = e_m.shift(1), e_p.shift(1)
            if name == "proportional":
                def rule(e):
                    z = (e / e.rolling(250).std()).clip(-2, 2).fillna(0)
                    rv_ = ret.rolling(20).std() * np.sqrt(252)
                    sz = (0.15 / rv_).clip(upper=2.0).shift(1).fillna(0)
                    pos = (z * sz).reindex(ret.index).fillna(0)
                    tu = pos.diff().abs().fillna(pos.abs())
                    return (pos.shift(1) * ret - tu * 33 / 1e4).dropna()
                rm, rp = rule(e_m), rule(e_p)
            elif name == "fullhorizon":
                def rule(e):
                    raw = pd.Series(np.where(e > .3, 1., np.where(e < -.3, -1., 0.)), index=e.index)
                    raw = raw.rolling(h, min_periods=1).mean()
                    rv_ = ret.rolling(20).std() * np.sqrt(252)
                    sz = (0.15 / rv_).clip(upper=2.0).shift(1).fillna(0)
                    pos = (raw * sz).reindex(ret.index).fillna(0)
                    tu = pos.diff().abs().fillna(pos.abs())
                    return (pos.shift(1) * ret - tu * 33 / 1e4).dropna()
                rm, rp = rule(e_m), rule(e_p)
            else:
                rm = fs.run_rule(e_m, ret)["r"]; rp = fs.run_rule(e_p, ret)["r"]
            i = rm.index.intersection(rp.index)
            dd, sd, t, _ = fs.sharpe_diff(rm.loc[i], rp.loc[i])
            r_[h] = [round(fs.sharpe_se(rm.loc[i])[0], 2), round(fs.sharpe_se(rp.loc[i])[0], 2),
                     round(dd, 2), round(t, 2)]
        rob[name] = r_
    # sous-périodes à h=30
    m30, p30 = cur[30]; sub = {}
    for lab, sl in [("2016-2019", slice("2016", "2019")), ("2020-2023", slice("2020", "2023"))]:
        mm, pp = m30.r[sl], p30.r[sl]
        dd, sd, t, _ = fs.sharpe_diff(mm, pp)
        sub[lab] = [round(fs.sharpe_se(mm)[0], 2), round(fs.sharpe_se(pp)[0], 2), round(dd, 2), round(t, 2)]
    res["robust"] = rob; res["subperiods"] = sub
    save({"econ": res}); print(json.dumps(res, indent=1, default=float))


def part_span():
    P, pre, _ = load_all()
    iv30, si, ridge = P["iv30"], P["si"], P["ridge_fn"]
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]; ivw = iv30["2016":"2023"]
    a_wf, b_wf = fs.wf_projection(Fall, iv30.reindex(Fall.index))
    aa, bb = a_wf.reindex(F.index), b_wf.reindex(F.index)
    out, out_ret = {}, {}
    for h in H:
        f_hat = ridge(iv30, h, si).reindex(F.index)
        pred_dF = (aa + bb * f_hat) - F
        basis = F - ivw.reindex(F.index)
        y = F.shift(-h) - F
        d = pd.DataFrame({"y": y, "m": pred_dF, "b": basis}).dropna()
        X = np.c_[np.ones(len(d)), d.m, d.b]
        co = np.linalg.lstsq(X, d.y.values, rcond=None)[0]
        e = d.y.values - X @ co
        XtXi = np.linalg.inv(X.T @ X); q = max(2 * h, 7); n = len(d)
        S = (X * e[:, None]).T @ (X * e[:, None])
        for l in range(1, q + 1):
            G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
            S += (1 - l / (q + 1)) * (G + G.T)
        V = XtXi @ S @ XtXi
        r2 = 1 - (e ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
        out[h] = [round(float(co[1]), 3), round(float(co[1] / np.sqrt(V[1, 1])), 2),
                  round(float(co[2]), 3), round(float(co[2] / np.sqrt(V[2, 2])), 2),
                  round(float(r2), 3), n, round(float(np.corrcoef(d.m, d.b)[0, 1]), 2)]
        # même régression sur le RENDEMENT cumulé effectivement monétisé
        yr = (1 + ret).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        d2 = pd.DataFrame({"y": yr, "m": pred_dF, "b": basis}).dropna()
        X2 = np.c_[np.ones(len(d2)), d2.m, d2.b]
        co2 = np.linalg.lstsq(X2, d2.y.values, rcond=None)[0]
        e2 = d2.y.values - X2 @ co2
        XtXi2 = np.linalg.inv(X2.T @ X2)
        S2 = (X2 * e2[:, None]).T @ (X2 * e2[:, None])
        for l in range(1, q + 1):
            G = (X2[l:] * e2[l:, None]).T @ (X2[:-l] * e2[:-l, None])
            S2 += (1 - l / (q + 1)) * (G + G.T)
        V2 = XtXi2 @ S2 @ XtXi2
        out_ret[h] = [round(float(co2[1] / np.sqrt(V2[1, 1])), 2),
                      round(float(1 - (e2 ** 2).sum() / ((d2.y - d2.y.mean()) ** 2).sum()), 3)]
    res = {"price": out, "return": out_ret}
    # collinéarité en espace spot
    coll = {}
    for h in HS:
        f_hat = ridge(iv30, h, si)
        fw = forward_vol(P["betas"], h)
        carry = (fw - iv30).reindex(f_hat.index)
        dm = (f_hat - iv30.reindex(f_hat.index))
        d = pd.DataFrame({"m": dm, "c": carry}).dropna()
        coll[h] = round(float(np.corrcoef(d.m, d.c)[0, 1]), 3)
    res["spot_collinearity"] = coll
    save({"span": res}); print(json.dumps(res, indent=1, default=float))


def part_alt():
    """Construction alternative : IV du fournisseur, plus proche strike (robustesse)."""
    Pa = vl.load_pipeline(CACHE_ALT)
    Pb = vl.load_pipeline(CACHE_MAIN)
    iv_a, iv_b = Pa["iv30"], Pb["iv30"]
    c = iv_a.index.intersection(iv_b.index)
    res = {"corr": round(float(iv_a[c].corr(iv_b[c])), 4),
           "mad": round(float((iv_a[c] - iv_b[c]).abs().mean()), 3),
           "n_common": len(c), "n_alt": len(Pa["betas"]),
           "r2_alt": round(float(Pa["betas"].R2.median()), 4)}
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]
    a_wf, b_wf = fs.wf_projection(Fall, iv_a.reindex(Fall.index))
    aa, bb = a_wf.reindex(F.index), b_wf.reindex(F.index)
    iva_w = iv_a["2016":"2023"].reindex(F.index)
    rows, spot = {}, {}
    for h in HS:
        f_hat = Pa["ridge_fn"](iv_a, h, Pa["si"]).reindex(F.index)
        m = fs.run_rule((aa + bb * f_hat) - F, ret); p = fs.run_rule((aa + bb * iva_w) - F, ret)
        i = m.index.intersection(p.index); m, p = m.loc[i], p.loc[i]
        dd, sd, t, _ = fs.sharpe_diff(m.r, p.r)
        rows[h] = [round(fs.sharpe_se(m.r)[0], 2), round(fs.sharpe_se(p.r)[0], 2),
                   round(dd, 2), round(t, 2)]
        bt = vl.backtest(Pa["ridge_fn"](iv_a, h, Pa["si"]), iv_a, ret, cost_bps=33.0, stop_loss=None)
        spot[h] = round(fs.sharpe_se(bt["strat_ret"])[0], 2)
    res["projected"] = rows; res["spot"] = spot
    save({"alt": res}); print(json.dumps(res, indent=1, default=float))


def part_app():
    """Annexes : cible brute, régime, pénalité du fit, non-linéaire."""
    P, pre, _ = load_all()
    iv30, si, b, ridge = P["iv30"], P["si"], P["betas"], P["ridge_fn"]
    ts = P["ts_opt"]
    # cible brute : interpolation linéaire en maturité entre les deux expiries encadrant 30j
    raw = {}
    for day, g in ts.groupby("Day"):
        g = g[g.DaysToExpiration >= 5].sort_values("DaysToExpiration")
        lo = g[g.DaysToExpiration <= 30]; hi = g[g.DaysToExpiration > 30]
        if len(lo) == 0 or len(hi) == 0:
            continue
        l, hh = lo.iloc[-1], hi.iloc[0]
        w = (hh.DaysToExpiration - 30) / (hh.DaysToExpiration - l.DaysToExpiration)
        raw[day] = w * l.ImpliedVol + (1 - w) * hh.ImpliedVol
    raw = pd.Series(raw).sort_index()
    c = raw.index.intersection(iv30.index)
    res = {"raw": {"corr": round(float(raw[c].corr(iv30[c])), 4),
                   "mad": round(float((raw[c] - iv30[c]).abs().mean()), 3),
                   "ac1_raw": round(float(raw.autocorr(1)), 4), "ac1_fit": round(float(iv30.autocorr(1)), 4),
                   "ac21_raw": round(float(raw.autocorr(21)), 4), "ac21_fit": round(float(iv30.autocorr(21)), 4)}}
    sk = {}
    for h in HS:
        sk[h] = [round(100 * vl.skill(pre[h]["Ridge"], iv30, h), 1),
                 round(100 * vl.skill(pre[h]["Ridge"].reindex(raw.index).dropna(), raw, h), 1)]
    res["raw"]["skill_fit_vs_raw"] = sk
    # régime
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]; ivw = iv30["2016":"2023"]
    a_wf, b_wf = fs.wf_projection(Fall, iv30.reindex(Fall.index))
    aa, bb = a_wf.reindex(F.index), b_wf.reindex(F.index)
    q3 = iv30.expanding(250).quantile(0.75).reindex(F.index)
    reg = {}
    for h in HS:
        f_hat = ridge(iv30, h, si).reindex(F.index)
        m = fs.run_rule((aa + bb * f_hat) - F, ret); p = fs.run_rule((aa + bb * ivw.reindex(F.index)) - F, ret)
        i = m.index.intersection(p.index)
        hi = (ivw.reindex(i) > q3.reindex(i))
        reg[h] = {"calm": [round(fs.sharpe_se(m.r.loc[i][~hi])[0], 2), round(fs.sharpe_se(p.r.loc[i][~hi])[0], 2)],
                  "vol": [round(fs.sharpe_se(m.r.loc[i][hi])[0], 2), round(fs.sharpe_se(p.r.loc[i][hi])[0], 2)]}
    res["regime"] = reg
    save({"app": res}); print(json.dumps(res, indent=1, default=float))


def part_conc():
    """Concentration du skill : le gain vient-il d'une poignee de jours ?
       Reponse : oui pour tout le monde (les differences de carres sont a queues lourdes),
       et MOINS pour beta2/Ridge que pour les sept autres cellules du Tableau 5."""
    import pickle
    P, pre, arma = load_all()
    b, si, iv30 = P["betas"], P["si"], P["iv30"]
    def diag(pred, s, h, k=50):
        y = s.shift(-h)
        c = pred.dropna().index.intersection(y.dropna().index)
        em = pred.reindex(c) - y.reindex(c); er = s.reindex(c) - y.reindex(c)
        d = (er ** 2 - em ** 2).sort_values(ascending=False)
        m2, r2 = em.drop(d.index[:k]), er.drop(d.index[:k])
        return {"n": len(c),
                "rmse": round(100 * (1 - np.sqrt((em ** 2).mean()) / np.sqrt((er ** 2).mean())), 1),
                "mae": round(100 * (1 - em.abs().mean() / er.abs().mean()), 1),
                "days_won": round(100 * float((em.abs() < er.abs()).mean()), 1),
                "top50_share": round(100 * float(d.iloc[:k].sum() / d.sum())),
                "rmse_ex_top50": round(100 * (1 - np.sqrt((m2 ** 2).mean()) / np.sqrt((r2 ** 2).mean())), 1)}
    res = {}
    for nm, s in [("iv30", iv30), ("beta0", b.beta0), ("beta1", b.beta1), ("beta2", b.beta2)]:
        for h in HS:
            res[f"ARMA/{nm}/h={h}"] = diag(arma[(nm, h)] if nm != "iv30" else arma[("iv30", h)], s, h)
            res[f"Ridge/{nm}/h={h}"] = diag(P["ridge_fn"](s, h, si) if nm != "iv30" else pre[h]["Ridge"], s, h)
    save({"conc": res}); print(json.dumps(res, indent=1, default=float))


def part_decomp():
    """Decomposition E[w r] = E[w]E[r] + Cov(w,r) : ce qu'une regle gagne par son
       EXPOSITION MOYENNE a l'instrument, et ce qu'elle gagne en faisant VARIER cette
       exposition. Sert a montrer que le contraste spot/future mesure surtout la premiere."""
    P, pre, _ = load_all()
    iv30, si, ridge = P["iv30"], P["si"], P["ridge_fn"]
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]
    ivw = iv30["2016":"2023"].reindex(F.index)
    a_, b_ = fs.wf_projection(Fall, iv30.reindex(Fall.index))
    aa, bb = a_.reindex(F.index), b_.reindex(F.index)

    def split(pos, r):
        """Deux decompositions du MEME total, sur les MEMES positions :
             (a) exposition moyenne + timing        : E[w]E[r] + Cov(w, r)
             (b) signe a l'echelle moyenne + interaction, avec w = signe(w) * |w|
           Les deux doivent redonner le total ; c'est verifie par assertion."""
        d = pd.DataFrame({"w": pos.shift(1), "r": r}).dropna()
        tot = float(d.w.mul(d.r).mean() * 252 * 100)
        expo = float(d.w.mean() * d.r.mean() * 252 * 100)
        wd = d.w - d.w.mean(); rt = wd * d.r
        s, z = np.sign(d.w), d.w.abs(); zbar = z.mean()
        sign_term = float(zbar * (s * d.r).mean() * 252 * 100)
        inter = float((s * (z - zbar) * d.r).mean() * 252 * 100)
        assert abs((expo + (tot - expo)) - tot) < 1e-6
        assert abs((sign_term + inter) - tot) < 1e-6, "les deux decompositions divergent"
        return {"total": round(tot, 1), "exposure": round(expo, 1),
                "timing": round(tot - expo, 1), "mean_w": round(float(d.w.mean()), 3),
                "sign_at_mean_scale": round(sign_term, 1), "interaction": round(inter, 1),
                "timing_sharpe": round(float(np.sqrt(252) * rt.mean() / rt.std()), 2)}

    res = {}
    for h in HS:
        fh = ridge(iv30, h, si).reindex(F.index)
        for lab, edge in [("projected", (aa + bb * fh) - F),
                          ("placebo", (aa + bb * ivw) - F),
                          ("spot", fh - ivw)]:
            r_ = fs.run_rule(edge, ret)
            res[f"{lab}/h={h}"] = split(r_.pos, ret.reindex(r_.index))
    save({"decomp": res}); print(json.dumps(res, indent=1, default=float))


def _proj_setup():
    P, pre, _ = load_all()
    iv30, si, ridge = P["iv30"], P["si"], P["ridge_fn"]
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]
    ivw = iv30["2016":"2023"].reindex(F.index)
    a_, b_ = fs.wf_projection(Fall, iv30.reindex(Fall.index))
    return P, iv30, si, ridge, Fall, ret_full, F, ret, ivw, a_.reindex(F.index), b_.reindex(F.index)


def part_coverage():
    """La borne appariee est-elle bien calibree ? On reechantillonne par blocs les
       rendements REELS des deux bras, demoyennes pour que la vraie difference soit nulle,
       et on mesure la couverture de l'intervalle HAC a 95 %."""
    from scipy import stats
    P, iv30, si, ridge, Fall, ret_full, F, ret, ivw, aa, bb = _proj_setup()
    fh = ridge(iv30, 30, si).reindex(F.index)
    m = fs.run_rule((aa + bb * fh) - F, ret); p = fs.run_rule((aa + bb * ivw) - F, ret)
    i = m.index.intersection(p.index)
    A = (m.r.loc[i] - m.r.loc[i].mean()).values
    B = (p.r.loc[i] - p.r.loc[i].mean()).values
    n = len(A); rng = np.random.default_rng(0)
    res = {"n": n, "skew": [round(float(stats.skew(A)), 1), round(float(stats.skew(B)), 1)],
           "kurtosis": [round(float(stats.kurtosis(A, fisher=False))), round(float(stats.kurtosis(B, fisher=False)))],
           "corr": round(float(np.corrcoef(A, B)[0, 1]), 2), "blocks": {}}
    for blk in (5, 21, 63):
        nbk = int(np.ceil(n / blk)); cov = 0; ses = []; ds = []
        for _ in range(3000):
            ix = np.concatenate([np.arange(k, k + blk) for k in rng.integers(0, n - blk, nbk)])[:n]
            d, se, _, _ = fs.sharpe_diff(pd.Series(A[ix]), pd.Series(B[ix]))
            ds.append(d); ses.append(se)
            if abs(d) <= 1.96 * se:
                cov += 1
        res["blocks"][blk] = {"coverage": round(100 * cov / 3000, 1),
                              "se_hac": round(float(np.mean(ses)), 3),
                              "se_empirical": round(float(np.std(ds, ddof=1)), 3)}
    save({"coverage": res}); print(json.dumps(res, indent=1, default=float))


def part_eiv():
    """Erreurs sur variables : ajustement NS separe sur les echeances de rang pair et impair.
       Donne le ratio de fiabilite de chaque facteur et le skill que le seul bruit de mesure
       produirait contre la marche aleatoire."""
    P, pre, _ = load_all()
    b, si = P["betas"], P["si"]
    ts = P["ts_opt"]; ts["Day"] = pd.to_datetime(ts["Day"])
    rows = []
    for day, g in ts.groupby("Day"):
        g = g[g.DaysToExpiration >= 5].sort_values("DaysToExpiration")
        o, e = g.iloc[0::2], g.iloc[1::2]
        if len(o) < 4 or len(e) < 4:
            continue
        bo = tx.fit_ns_ridge(o.ImpliedVol.values, o.DaysToExpiration.values / 365.0)
        be = tx.fit_ns_ridge(e.ImpliedVol.values, e.DaysToExpiration.values / 365.0)
        rows.append((day,) + tuple(bo) + tuple(be))
    S = pd.DataFrame(rows, columns=["Day", "o0", "o1", "o2", "e0", "e1", "e2"]).set_index("Day")
    res = {"n_days": len(S)}
    for k, (co, ce, ser) in {"beta0": ("o0", "e0", b.beta0), "beta1": ("o1", "e1", b.beta1),
                             "beta2": ("o2", "e2", b.beta2)}.items():
        x, y = S[co], S[ce]
        var_full = ((x - y).var() / 2) / 2     # erreur d'un demi-ajustement, puis 2x plus d'echeances
        d = {"reliability": round(float(x.corr(y)), 3)}
        for h in HS:
            Vh = x.diff(h).cov(y.diff(h))
            d[f"mechanical_h{h}"] = round(100 * float(1 - np.sqrt((Vh + var_full) / (Vh + 2 * var_full))), 1)
            d[f"observed_h{h}"] = round(100 * vl.skill(P["ridge_fn"](ser, h, si), ser, h), 1)
        res[k] = d
    save({"eiv": res}); print(json.dumps(res, indent=1, default=float))


def part_extra():
    """Sensibilite au seuil sur la regle directionnelle, queues et evenements, volatilites
       des deux bras, et bêta de la projection en niveau contre en variations."""
    from scipy import stats
    P, iv30, si, ridge, Fall, ret_full, F, ret, ivw, aa, bb = _proj_setup()
    res = {"delta": {}, "beta_level_vs_change": {}}
    d = pd.DataFrame({"F": F, "iv": ivw}).dropna()
    res["beta_level_vs_change"]["level"] = round(float(np.polyfit(d.iv, d.F, 1)[0]), 3)
    for k in (1, 5, 14, 30):
        dd = pd.DataFrame({"dF": d.F.diff(k), "div": d.iv.diff(k)}).dropna()
        res["beta_level_vs_change"][f"h{k}"] = round(float(np.polyfit(dd["div"], dd["dF"], 1)[0]), 3)
    for h in HS:
        fh = ridge(iv30, h, si).reindex(F.index)
        em = (aa + bb * fh) - F; ep = (aa + bb * ivw) - F
        row = {}
        for dl in (0.0, 0.15, 0.3, 0.6, 1.2):
            m = fs.run_rule(em, ret, delta=dl); p = fs.run_rule(ep, ret, delta=dl)
            i = m.index.intersection(p.index)
            dif, _, t, _ = fs.sharpe_diff(m.r.loc[i], p.r.loc[i])
            row[str(dl)] = [round(dif, 2), round(t, 2)]
        res["delta"][h] = row
    fh = ridge(iv30, 30, si).reindex(F.index)
    m = fs.run_rule((aa + bb * fh) - F, ret); p = fs.run_rule((aa + bb * ivw) - F, ret)
    i = m.index.intersection(p.index); m, p = m.loc[i], p.loc[i]
    res["vol"] = {"model": round(100 * float(m.r.std() * np.sqrt(252)), 1),
                  "placebo": round(100 * float(p.r.std() * np.sqrt(252)), 1)}
    res["tails"] = {"skew": round(float(stats.skew(m.r)), 1),
                    "kurtosis": round(float(stats.kurtosis(m.r, fisher=False)))}
    ev = {}
    for lab, msk in [("full", np.ones(len(m), bool)),
                     ("ex_feb2018", ~((m.index >= "2018-02-01") & (m.index <= "2018-02-28"))),
                     ("ex_mar2020", ~((m.index >= "2020-03-01") & (m.index <= "2020-03-31"))),
                     ("ex_both", ~(((m.index >= "2018-02-01") & (m.index <= "2018-02-28")) |
                                   ((m.index >= "2020-03-01") & (m.index <= "2020-03-31"))))]:
        dif, _, t, _ = fs.sharpe_diff(m.r[msk], p.r[msk])
        ev[lab] = [round(dif, 2), round(t, 2)]
    res["events"] = ev
    save({"extra": res}); print(json.dumps(res, indent=1, default=float))


# ==================================================================
# ============ REVISION SUITE AU RAPPORT DE S. FORMENTIN ===========
# ==================================================================

def cw_series(pred, rw, tgt, h):
    """Serie de perte ajustee de Clark-West, dont la moyenne donne la statistique.
       Sous H0 (le modele emboite est vrai) son esperance est nulle."""
    y = tgt.shift(-h)
    c = pred.dropna().index.intersection(rw.dropna().index).intersection(y.dropna().index)
    p, r, a = pred.reindex(c).values, rw.reindex(c).values, y.reindex(c).values
    return pd.Series((a - r) ** 2 - ((a - p) ** 2 - (r - p) ** 2), index=c)


def _blocks(n, B, rng, nboot):
    nbk = int(np.ceil(n / B))
    return [np.concatenate([np.arange(k, min(k + B, n))
            for k in rng.integers(0, max(n - B, 1), nbk)])[:n] for _ in range(nboot)]


def _nw_se(x, q):
    lrv, _ = fs.nw_lrv(np.asarray(x, float), q=int(q))
    return float(np.sqrt(lrv / len(x)))


# ------------------------------------------------------------------ POINT 1
def part_cwboot(nboot=2000):
    """Point 1 du rapporteur : a horizon long la statistique de Clark-West est referee a une
       normale dont la validite n'est pas etablie sous schema recursif (West 1996 ;
       Clark et McCracken 2005). On remplace la reference asymptotique par deux references
       empiriques :
         (a) bootstrap par blocs mobiles de la perte ajustee RECENTREE (donc engendree sous H0),
             studentisee par l'erreur-type Newey-West de l'echantillon d'origine. On NE studentise
             PAS a l'interieur de chaque replication : l'echantillon rebloque a des ruptures de
             dependance aux jointures, l'estimateur Newey-West y est biaise vers le bas de ~18 %
             et le rapport obtenu a une loi degeneree (ecart-type 3,7 mais quantile a 95 % de 1,26).
             Avec l'erreur-type d'origine l'ecart-type bootstrap vaut 1,00 a 1,03, ce qui valide
             au passage la longueur de bloc ; ce qui est teste est alors la FORME de la loi nulle,
             c'est-a-dire exactement l'hypothese de normalite mise en cause.
         (b) sous-echantillons espaces de h jours, qui n'ont aucun recouvrement, donc pour
             lesquels la theorie a un pas s'applique sans correction."""
    from scipy.stats import norm
    P, pre, _ = load_all()
    iv30, si = P["iv30"], P["si"]
    rw = iv30.iloc[si:]
    rng = np.random.default_rng(7)
    res = {}
    for h in HS:
        for m in ["ARMA", "HAR", "VAR", "Ridge"]:
            f = cw_series(pre[h][m], rw, iv30, h).values
            n, q = len(f), max(2 * h, 7)
            se0 = _nw_se(f, q)
            t_obs = float(f.mean() / se0)
            # (a) bootstrap par blocs sous H0, studentise par l'erreur-type d'origine
            B = max(2 * h, 7)
            fc = f - f.mean()
            tb = np.array([fc[ix].mean() / se0 for ix in _blocks(n, B, rng, nboot)])
            p_boot = float((tb >= t_obs).mean())
            assert 0.85 < tb.std() < 1.20, f"bloc inadapte : sd bootstrap = {tb.std():.2f}"
            # (b) sous-echantillons espaces de h
            ts = []
            for off in range(h):
                s = f[off::h]
                if len(s) < 20:
                    continue
                ts.append(float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))))
            ts = np.array(ts)
            res[f"{m}/h={h}"] = {
                "t": round(t_obs, 2),
                "p_normal": round(float(1 - norm.cdf(t_obs)), 4),
                "p_boot": round(p_boot, 4),
                "q95_boot": round(float(np.percentile(tb, 95)), 2),
                "sd_boot": round(float(tb.std()), 2),
                "skew_boot": round(float(pd.Series(tb).skew()), 2),
                "sub_n": int(len(ts)), "sub_obs": int(len(f[0::h])),
                "sub_t_med": round(float(np.median(ts)), 2),
                "sub_t_range": [round(float(ts.min()), 2), round(float(ts.max()), 2)],
                "sub_frac_sig": round(float((ts > 1.645).mean()), 2)}
    save({"cwboot": res}); print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ POINT 2a
def part_rw(nboot=2000, alpha=0.05):
    """Point 2 du rapporteur : Holm et Benjamini-Yekutieli ignorent la dependance entre les
       42 tests, qui est forte (horizons emboites, erreurs a h pas qui se recouvrent).
       Procedure stepdown de Romano et Wolf (2005) : la distribution du maximum des
       statistiques studentisees est bootstrappee CONJOINTEMENT par blocs mobiles, ce qui
       exploite cette dependance au lieu de payer une penalite pour elle."""
    from scipy.stats import norm
    P, pre, arma = load_all()
    iv30, si, b = P["iv30"], P["si"], P["betas"]
    rw = iv30.iloc[si:]
    cols, hs = {}, {}
    for h in H:
        for m in ["ARMA", "HAR", "VAR", "Ridge"]:
            k = f"{m}/iv30/h={h}"; cols[k] = cw_series(pre[h][m], rw, iv30, h); hs[k] = h
    for nm, s in [("beta0", b.beta0), ("beta1", b.beta1), ("beta2", b.beta2)]:
        rwf = s.iloc[si:]
        for h in HS:
            for m, pr in [("ARMA", arma[(nm, h)]), ("Ridge", P["ridge_fn"](s, h, si))]:
                k = f"{m}/{nm}/h={h}"; cols[k] = cw_series(pr, rwf, s, h); hs[k] = h
    D = pd.DataFrame(cols).dropna()
    names = list(D.columns); X = D.values; n, mm = X.shape
    # Studentisation PROPRE A CHAQUE TEST (max(2h,7), comme partout ailleurs dans le papier) ;
    # seul le BLOC du bootstrap est commun, et il doit couvrir le plus long recouvrement.
    q = max(2 * max(hs.values()), 7)
    se = np.array([_nw_se(X[:, j], max(2 * hs[names[j]], 7)) for j in range(mm)])
    tstat = X.mean(0) / se
    # bootstrap conjoint : memes blocs pour toutes les colonnes -> dependance preservee
    rng = np.random.default_rng(11)
    Tb = np.empty((nboot, mm))
    for j, ix in enumerate(_blocks(n, q, rng, nboot)):
        Tb[j] = (X[ix].mean(0) - X.mean(0)) / se
    order = np.argsort(-tstat)
    padj = np.ones(mm); prev = 0.0
    remaining = list(order)
    while remaining:
        mx = Tb[:, remaining].max(1)
        j = remaining[0]                                   # deja trie par t decroissant
        p = float((mx >= tstat[j]).mean())
        prev = max(prev, p); padj[j] = prev
        remaining.pop(0)
    p_ind = np.array([1 - norm.cdf(t) for t in tstat])
    holm, bh = holm_bh(p_ind)
    surv = [names[i] for i in order if padj[i] < alpha]
    res = {"n_tests": mm, "n_obs_common": n, "block": q,
           "n_raw": int((p_ind < alpha).sum()),
           "n_holm": int((holm < alpha).sum()),
           "n_rw": len(surv),
           "survivors_rw": surv,
           "n_rw_curvature": sum("beta2" in s for s in surv),
           "detail": {names[i]: [round(float(tstat[i]), 2), round(float(p_ind[i]), 5),
                                 round(float(holm[i]), 4), round(float(padj[i]), 4)]
                      for i in order}}
    save({"rw": res}); print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ POINT 2b
def part_spa(nboot=2000):
    """Point 2, volet economique : test SPA de Hansen (2005) applique directement a la
       metrique economique. L'espace de recherche est la grille (horizon, seuil delta, cible
       de volatilite) que nous avons effectivement exploree ; le benchmark est le placebo
       apparie. H0 : aucune configuration ne bat le placebo. Le recentrage de Hansen retire
       les configurations trop mauvaises pour etre plausiblement a la frontiere."""
    P, pre, _ = load_all()
    iv30, si, ridge = P["iv30"], P["si"], P["ridge_fn"]
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]; ivw = iv30["2016":"2023"]
    a_wf, b_wf = fs.wf_projection(Fall, iv30.reindex(Fall.index))
    aa, bb = a_wf.reindex(F.index), b_wf.reindex(F.index)
    grid, labels = [], []
    for h in HS:
        f_hat = ridge(iv30, h, si).reindex(F.index)
        e_m = (aa + bb * f_hat) - F
        e_p = (aa + bb * ivw.reindex(F.index)) - F
        for dl in [0.2, 0.3, 0.5, 0.75, 1.0]:
            for tv in [0.10, 0.15, 0.20]:
                m = fs.run_rule(e_m, ret, delta=dl, target_vol=tv)
                p = fs.run_rule(e_p, ret, delta=dl, target_vol=tv)
                i = m.index.intersection(p.index)
                grid.append((m.r.loc[i] - p.r.loc[i]).rename(f"h{h}_d{dl}_v{tv}"))
                labels.append({"h": h, "delta": dl, "target": tv,
                               "sharpe_m": round(fs.sharpe_se(m.r.loc[i])[0], 2),
                               "sharpe_p": round(fs.sharpe_se(p.r.loc[i])[0], 2)})
    D = pd.concat(grid, axis=1).dropna()
    X = D.values * 252 * 100                               # points de rendement annualises
    n, k = X.shape
    dbar = X.mean(0)
    om = np.array([np.sqrt(fs.nw_lrv(X[:, j], q=21)[0]) for j in range(k)])
    t = np.sqrt(n) * dbar / om
    T_spa = max(0.0, float(t.max()))
    thr = -np.sqrt(om ** 2 / n * 2 * np.log(np.log(n)))    # seuil de recentrage de Hansen
    g = np.where(dbar >= thr, dbar, 0.0)
    rng = np.random.default_rng(23)
    Zb = np.empty(nboot)
    for j, ix in enumerate(_blocks(n, 21, rng, nboot)):
        Zb[j] = max(0.0, float((np.sqrt(n) * (X[ix].mean(0) - g) / om).max()))
    p_spa = float((Zb >= T_spa).mean())
    # Reality Check de White (recentrage complet, plus conservateur)
    Zl = np.empty(nboot)
    for j, ix in enumerate(_blocks(n, 21, rng, nboot)):
        Zl[j] = max(0.0, float((np.sqrt(n) * (X[ix].mean(0) - dbar) / om).max()))
    p_rc = float((Zl >= T_spa).mean())
    best = int(np.argmax(t))
    res = {"n_configs": k, "n_obs": n, "T_spa": round(T_spa, 2),
           "p_spa": round(p_spa, 3), "p_reality_check": round(p_rc, 3),
           "best": {**labels[best], "excess_pp": round(float(dbar[best]), 2),
                    "t": round(float(t[best]), 2)},
           "n_positive": int((dbar > 0).sum()),
           "mean_excess_pp": round(float(dbar.mean()), 2),
           "range_excess_pp": [round(float(dbar.min()), 2), round(float(dbar.max()), 2)]}
    save({"spa": res}); print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ POINT 3
def part_mech(nboot=2000):
    """Point 3 du rapporteur : dans la regression en espace negocie, F_t figure dans les deux
       regresseurs ET dans la variable expliquee (y = F_{t+h} - F_t). Si F est moyenne-reversif,
       un coefficient positif sur (f_hat - F) apparait MECANIQUEMENT, sans aucun contenu
       previsionnel. Trois tests :
         (a) regression NON CONTRAINTE : y sur [f_hat, F, IV] separement. La specification du
             papier impose un coefficient commun ; la liberer isole ce que la prevision apporte
             au-dela des niveaux. C'est le test decisif.
         (b) PLACEBOS sans information : la moyenne glissante de F a la place de f_hat, puis
             la regression sans prevision du tout.
         (c) PERMUTATION par decalage circulaire de f_hat : detruit l'alignement temporel de la
             prevision tout en conservant exactement sa loi marginale, son autocorrelation et
             la structure en -F_t. Donne la loi de t_model sous une nulle ou le design est
             intact mais la prevision n'informe pas."""
    P, pre, _ = load_all()
    iv30, si, ridge = P["iv30"], P["si"], P["ridge_fn"]
    vx, Fall, ret_full = _futures()
    F = Fall["2016":"2023"]; ret = ret_full["2016":"2023"]; ivw = iv30["2016":"2023"]
    a_wf, b_wf = fs.wf_projection(Fall, iv30.reindex(Fall.index))
    aa, bb = a_wf.reindex(F.index), b_wf.reindex(F.index)

    def hac(X, y, q):
        co = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ co; XtXi = np.linalg.inv(X.T @ X)
        S = (X * e[:, None]).T @ (X * e[:, None])
        for l in range(1, q + 1):
            G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
            S += (1 - l / (q + 1)) * (G + G.T)
        V = XtXi @ S @ XtXi
        r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return co, np.sqrt(np.diag(V)), float(r2)

    # persistance de F : source de la composante mecanique
    lf = F.dropna()
    rho1 = float(pd.Series(lf.values).autocorr(1))
    hl = float(np.log(0.5) / np.log(abs(rho1))) if 0 < abs(rho1) < 1 else np.nan
    res = {"F_ar1": round(rho1, 4), "F_halflife_days": round(hl, 1),
           "F_ar1_20d": round(float(pd.Series(lf.values).autocorr(20)), 3)}

    unres, plac, perm, retv = {}, {}, {}, {}
    rng = np.random.default_rng(31)
    for h in H:
        q = max(2 * h, 7)
        f_hat = ridge(iv30, h, si).reindex(F.index)
        fproj = (aa + bb * f_hat)
        y_ = F.shift(-h) - F
        base = pd.DataFrame({"y": y_, "fp": fproj, "F": F,
                             "iv": ivw.reindex(F.index)}).dropna()
        # (a) non contrainte
        Xu = np.c_[np.ones(len(base)), base.fp, base.F, base.iv]
        co, se, r2u = hac(Xu, base.y.values, q)
        # contrainte (reproduit le tableau du papier)
        Xr = np.c_[np.ones(len(base)), base.fp - base.F, base.F - base.iv]
        cor, ser, r2r = hac(Xr, base.y.values, q)
        unres[h] = {"b_fhat": round(float(co[1]), 3), "t_fhat": round(float(co[1] / se[1]), 2),
                    "b_F": round(float(co[2]), 3), "t_F": round(float(co[2] / se[2]), 2),
                    "b_iv": round(float(co[3]), 3), "t_iv": round(float(co[3] / se[3]), 2),
                    "r2_unrestricted": round(r2u, 3),
                    "b_model_restricted": round(float(cor[1]), 3),
                    "t_model_restricted": round(float(cor[1] / ser[1]), 2),
                    "r2_restricted": round(r2r, 3), "n": len(base)}
        # (b) placebos
        mF = F.expanding(250).mean()
        b2 = pd.DataFrame({"y": y_, "x": mF - F, "b": F - ivw.reindex(F.index)}).dropna()
        X2 = np.c_[np.ones(len(b2)), b2.x, b2.b]
        c2, s2, r22 = hac(X2, b2.y.values, q)
        X3 = np.c_[np.ones(len(base)), base.F, base.iv]
        _, _, r23 = hac(X3, base.y.values, q)
        plac[h] = {"t_runmean": round(float(c2[1] / s2[1]), 2), "b_runmean": round(float(c2[1]), 3),
                   "r2_runmean": round(r22, 3), "r2_levels_only": round(r23, 3)}
        # (c) permutation par decalage circulaire
        fv = fproj.reindex(base.index).values; Fv = base.F.values
        Bv = (base.F - base.iv).values; yv = base.y.values; nn = len(base)
        t_obs = float(cor[1] / ser[1])
        cnt = 0; tt = np.empty(nboot)
        for j in range(nboot):
            s_ = int(rng.integers(h + 1, nn - h - 1))
            xs = np.roll(fv, s_) - Fv
            Xs = np.c_[np.ones(nn), xs, Bv]
            cs, ss, _ = hac(Xs, yv, q)
            tt[j] = cs[1] / ss[1]
            cnt += tt[j] >= t_obs
        perm[h] = {"t_obs": round(t_obs, 2), "p_perm": round(cnt / nboot, 4),
                   "q95_perm": round(float(np.percentile(tt, 95)), 2),
                   "mean_perm": round(float(tt.mean()), 2)}
        # meme diagnostic sur le RENDEMENT : y ne contient plus -F_t
        yr = (1 + ret).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        br = pd.DataFrame({"y": yr, "fp": fproj, "F": F,
                           "iv": ivw.reindex(F.index)}).dropna()
        Xru = np.c_[np.ones(len(br)), br.fp, br.F, br.iv]
        cr, sr, r2ru = hac(Xru, br.y.values, q)
        Xrr = np.c_[np.ones(len(br)), br.fp - br.F, br.F - br.iv]
        crr, srr, r2rr = hac(Xrr, br.y.values, q)
        retv[h] = {"b_fhat": round(float(cr[1]), 4), "t_fhat": round(float(cr[1] / sr[1]), 2),
                   "t_model_restricted": round(float(crr[1] / srr[1]), 2),
                   "r2_restricted": round(r2rr, 3), "n": len(br)}
    res.update({"unrestricted": unres, "placebo": plac,
                "permutation": perm, "return_space": retv})
    save({"mech": res}); print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ POINT 5
def _vx_curve():
    """Courbe des futures VIX en maturites constantes, a partir des seuls settlements CFE.
       Aucune option, aucune volatilite spot : tout est deja dans les unites de l'objet negocie."""
    vx = vl.load_vx_cfe(fs.CFE, verbose=False)
    F = pd.DataFrame({d: fs.cm_price(vx, d) for d in [30, 60, 90, 120]}).dropna()
    F.columns = ["F30", "F60", "F90", "F120"]
    F["slope"] = F.F60 - F.F30
    F["curv"] = (F.F120 - F.F90) - (F.F60 - F.F30)
    ret = vl.constant_maturity_return(vx, 30).reindex(F.index).dropna()
    return F, ret


def _ridge_wf(X, y, split_i, alpha=15.0, embargo=0, minobs=250):
    """Ridge walk-forward a fenetre croissante, forme fermee, refit a chaque date.
       `embargo` retire les dernieres observations d'entrainement pour qu'aucune cible
       d'entrainement ne recouvre la date prevue."""
    Xv, yv, idx = X.values, y.values, X.index
    out = {}
    for t in range(split_i, len(idx)):
        e = t - embargo
        if e < minobs:
            continue
        A, b = Xv[:e], yv[:e]
        ok = ~np.isnan(b) & ~np.isnan(A).any(1)
        if ok.sum() < minobs:
            continue
        A, b = A[ok], b[ok]
        m, sd = A.mean(0), A.std(0); sd = np.where(sd == 0, 1, sd)
        As = (A - m) / sd
        w = np.linalg.solve(As.T @ As + alpha * np.eye(As.shape[1]), As.T @ (b - b.mean()))
        out[idx[t]] = float(b.mean() + ((Xv[t] - m) / sd) @ w)
    return pd.Series(out)


def part_vxdirect(nboot=2000):
    """Point 5 : modeliser DIRECTEMENT la courbe des futures VIX au lieu de prevoir la
       volatilite spot puis de la projeter. L'objet prevu devient l'objet negocie, ce qui
       supprime d'un coup les deux ecarts documentes dans le papier (le coin d'unites
       variance-swap contre at-the-money, et l'ecart entre prix a maturite constante et
       rendement d'une position roulee). L'echantillon va de 2013 a 2026.

       DEUX exercices, a ne pas confondre :
         (A) PRIX : prevoir F30 a h jours, skill contre marche aleatoire. Analogue exact de
             la section 3 du papier, mais en unites negociees.
         (B) RENDEMENT : prevoir le rendement cumule a h jours de la position a maturite
             constante, c'est-a-dire exactement ce que le portefeuille encaisse. C'est le
             seul exercice ou l'objet prevu et l'objet negocie coincident vraiment.

       Le placebo economique n'est PLUS la marche aleatoire : en espace futures elle donne un
       edge identiquement nul, donc aucune position. C'est le CARRY (short quand la courbe est
       en contango), soit le trade de Simon et Campasano (2014), benchmark plus exigeant. La
       comparaison est donc NON EMBOITEE et doit etre presentee comme telle : l'appariement
       achete toujours de la precision, mais la logique de nullite emboitee ne s'applique pas.
       Les deux bras recoivent un seuil calibre au MEME quantile de leur propre |edge|, estime
       sur la seule fenetre d'entrainement, pour qu'aucun ne trade plus de jours que l'autre."""
    from scipy.stats import norm
    F, ret = _vx_curve()
    split = int(F.index.searchsorted(pd.Timestamp("2018-01-01")))
    res = {"n_days": len(F), "start": str(F.index.min().date()), "end": str(F.index.max().date()),
           "test_start": str(F.index[split].date()), "n_test": len(F) - split,
           "F30_ar1": round(float(F.F30.autocorr(1)), 4),
           "slope_mean": round(float(F.slope.mean()), 2),
           "contango_pct": round(100 * float((F.slope > 0).mean()))}
    rng = np.random.default_rng(101)
    feat = pd.DataFrame({"lvl": F.F30, "d5": F.F30.diff(5), "slope": F.slope, "curv": F.curv,
                         "rv20": ret.rolling(20).std() * np.sqrt(252)}).dropna()

    # ---------------- (A) PRIX : skill contre la marche aleatoire
    A = {}
    for h in H:
        y = F.F30; rw = y.iloc[split:]
        d = feat.copy(); d["dy"] = y.shift(-h) - y          # on prevoit le CHANGEMENT
        d = d.dropna(subset=[c for c in d.columns if c != "dy"])
        si2 = int(d.index.searchsorted(F.index[split]))
        dhat = _ridge_wf(d.drop(columns="dy"), d.dy, si2, embargo=h)
        preds = {"Ridge": (y.reindex(dhat.index) + dhat).dropna(),
                 "HAR": har(y, h, split), "AR1anch": bench_ar1(y, h, split)}
        row = {}
        for m, pr in preds.items():
            pr = pr.dropna()
            sk = 100 * (1 - rmse_(pr, y, h) / rmse_(rw.reindex(pr.index), y, h))
            t, n = clark_west(pr, rw, y, h)
            f = cw_series(pr, rw, y, h).values
            q = max(2 * h, 7); se0 = _nw_se(f, q); fc = f - f.mean()
            tb = np.array([fc[ix].mean() / se0 for ix in _blocks(len(f), q, rng, nboot)])
            row[m] = {"skill": round(float(sk), 1), "t_cw": round(t, 2),
                      "p_norm": round(float(1 - norm.cdf(t)), 4),
                      "p_boot": round(float((tb >= t).mean()), 4), "n": n}
        A[h] = row
    res["price"] = A

    # ---------------- (B) RENDEMENT de la position, puis test economique contre le carry
    B, E = {}, {}
    for h in HS:
        cumr = (1 + ret).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        d = feat.copy(); d["y"] = cumr
        d = d.dropna(subset=[c for c in d.columns if c != "y"])
        si2 = int(d.index.searchsorted(F.index[split]))
        pr = _ridge_wf(d.drop(columns="y"), d.y, si2, embargo=h).dropna()
        c = pr.index.intersection(d.dropna().index)
        yy, pp = d.y.reindex(c).values, pr.reindex(c).values
        mu = d.y.expanding(250).mean().shift(1).reindex(c).values
        ok = ~np.isnan(mu)
        B[h] = {"r2_oos_vs_mean": round(float(1 - ((yy[ok] - pp[ok]) ** 2).sum()
                                               / ((yy[ok] - mu[ok]) ** 2).sum()), 4),
                "corr": round(float(np.corrcoef(yy[ok], pp[ok])[0, 1]), 3), "n": int(ok.sum())}
        edge_m = pd.Series(pp * 100, index=c)          # rendement prevu, en %
        edge_c = (-F.slope).reindex(c)                 # short si contango, en points de VIX
        rr = ret.reindex(c)
        tr = c < F.index[split]                        # calibration du seuil hors test
        qq = 0.25
        dm = float(np.nanquantile(edge_m[tr].abs(), qq)) if tr.sum() > 50 else float(np.nanquantile(edge_m.abs(), qq))
        dc = float(np.nanquantile(edge_c[tr].abs(), qq)) if tr.sum() > 50 else float(np.nanquantile(edge_c.abs(), qq))
        m_ = fs.run_rule(edge_m, rr, delta=dm); p_ = fs.run_rule(edge_c, rr, delta=dc)
        i = m_.index.intersection(p_.index)
        i = i[i >= F.index[split]]                     # evaluation hors echantillon seulement
        m_, p_ = m_.loc[i], p_.loc[i]
        sm, sem, N, _ = fs.sharpe_se(m_.r); sp, sep, _, _ = fs.sharpe_se(p_.r)
        dd, sd, t, _ = fs.sharpe_diff(m_.r, p_.r)
        E[h] = {"delta_m": round(dm, 3), "delta_c": round(dc, 3),
                "sharpe_model": round(sm, 2), "se_m": round(sem, 2),
                "sharpe_carry": round(sp, 2), "se_c": round(sep, 2),
                "diff": round(dd, 2), "se_d": round(sd, 2), "t": round(t, 2),
                "upper95": round(dd + 1.96 * sd, 2),
                "rho_pos": round(float(m_.pos.corr(p_.pos)), 2),
                "long_m": round(100 * float((m_.pos > 0).mean())),
                "long_c": round(100 * float((p_.pos > 0).mean())), "n": N}
    res["return_forecast"] = B; res["economic"] = E
    save({"vxdirect": res}); print(json.dumps(res, indent=1, default=float))


def part_split():
    """La fenetre d'entrainement de six ans est-elle la contrainte ? On deplace la date de
       separation. Tot : moins d'entrainement, plus de test (donc plus de puissance).
       Tard : l'inverse. Si le resultat bouge peu, la longueur d'entrainement ne limite pas."""
    P, pre, _ = load_all()
    iv30, ridge = P["iv30"], P["ridge_fn"]
    res = {}
    for yr in [2014, 2015, 2016, 2017, 2018]:
        si = int(iv30.index.searchsorted(pd.Timestamp(f"{yr}-01-01")))
        row = {"train_years": round(si / 252, 1), "test_years": round((len(iv30) - si) / 252, 1)}
        for h in HS:
            pr = ridge(iv30, h, si); rw = iv30.iloc[si:]
            row[f"h{h}"] = [round(float(100 * (1 - rmse_(pr, iv30, h) / rmse_(rw, iv30, h))), 1),
                            round(clark_west(pr, rw, iv30, h)[0], 2)]
        res[yr] = row
    save({"split": res}); print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "stat"
    {"stat": part_stat, "mcs": part_mcs, "fwd": part_fwd, "cw": part_cw,
     "mult": part_mult, "boot": part_boot, "econ": part_econ, "span": part_span,
     "alt": part_alt, "app": part_app, "conc": part_conc, "decomp": part_decomp, "coverage": part_coverage,
     "eiv": part_eiv, "extra": part_extra,
     "cwboot": part_cwboot, "rw": part_rw, "spa": part_spa, "mech": part_mech,
     "vxdirect": part_vxdirect, "split": part_split}[part]()
