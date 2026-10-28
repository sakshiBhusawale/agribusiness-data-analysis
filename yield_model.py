"""Predictive modelling framework demo: ONE pipeline function, re-used for any crop (or any panel target).
Steps: time-aware split -> leak-free feature engineering -> baselines -> tuned models (walk-forward CV) -> validation-based selection
-> refit -> single test evaluation -> reliability checks (leakage demo, subgroup, intervals, importance)."""
import json, itertools, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
SEED = 42
df_all = pd.read_csv("data/synthetic_yield_data.csv")
TRAIN, VAL, TEST = (2008, 2017), (2018, 2020), (2021, 2024)   # development = 2008-2020, TEST = 2021-2024 (held out once)
FOLDS = list(range(2013, 2021))                      # 8 walk-forward folds: validate on each year 2013-2020, train on all earlier years

# ---------------- leak-free feature engineering (fit on TRAIN rows only) ----------------
class FE:
    def __init__(self, heat_thr, dry_thr): self.heat_thr, self.dry_thr = heat_thr, dry_thr
    def fit(self, d):
        self.rain_mean = d.groupby("district").rain_season_mm.mean(); self.yield_mean = d.groupby("district").yield_t_ha.mean(); return self
    def transform(self, d):
        o = d.copy()
        o["year_index"] = o.year - 2008
        o["district_mean_yield"] = o.district.map(self.yield_mean).fillna(self.yield_mean.mean())
        o["rain_anomaly"] = o.rain_season_mm - o.district.map(self.rain_mean).fillna(self.rain_mean.mean())
        o["dry_excess"] = (o.dry_spell_days - self.dry_thr).clip(lower=0)
        o["heat_excess"] = (o.tmax_critical - self.heat_thr).clip(lower=0)
        return o
SETS = {
 "Pre-season": ["soil_index", "irrigation_pct", "fert_kg_ha", "price_lag", "area_ha", "year_index", "prev_yield", "roll3_yield", "district_mean_yield"],
 "In-season": ["soil_index", "irrigation_pct", "fert_kg_ha", "price_lag", "area_ha", "year_index", "prev_yield", "roll3_yield", "district_mean_yield",
               "rain_season_mm", "rain_anomaly", "dry_spell_days", "dry_excess", "tmax_critical", "heat_excess"]}
SETS["In-season + NDVI"] = SETS["In-season"] + ["ndvi_peak"]
GRIDS = {"Ridge": [dict(alpha=a) for a in [0.01, 0.1, 1, 10, 100]],
         "Random forest": [dict(max_depth=m, min_samples_leaf=l) for m, l in itertools.product([4, 8, None], [2, 5])],
         "Gradient boosting": [dict(n_estimators=n, learning_rate=lr, max_depth=m) for n, lr, m in itertools.product([150, 300], [0.05, 0.1], [2, 3])]}
def build(name, p):
    if name == "Ridge": return make_pipeline(StandardScaler(), Ridge(**p))
    if name == "Random forest": return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1, **p)
    return GradientBoostingRegressor(random_state=SEED, **p)
def metrics(y, p):
    e = p - y; ss = ((y - y.mean())**2).sum()
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e**2).mean())), MAPE=float((np.abs(e)/y).mean()*100), R2=float(1 - (e**2).sum()/ss), bias=float(e.mean()))
def rows_in(d, lo, hi): return d[(d.year >= lo) & (d.year <= hi)]

def run_pipeline(crop, feats, cfg, keep_preds=False):
    d = df_all[df_all.crop == crop].dropna(subset=["prev_yield", "roll3_yield"]).reset_index(drop=True)
    dev, te = rows_in(d, TRAIN[0], VAL[1]), rows_in(d, *TEST)             # development period 2008-2020 (walk-forward CV inside); TEST held out once
    out = {"n": dict(dev=len(dev), test=len(te), folds=len(FOLDS)), "models": {}}
    def fold_data(fy):
        a, b = d[(d.year >= TRAIN[0]) & (d.year < fy)], d[d.year == fy]; f = FE(**cfg).fit(a); return a, b, f.transform(a), f.transform(b)
    # baselines scored on the same walk-forward folds
    for nm in ["Naive: last year", "District mean"]:
        e = []
        for fy in FOLDS:
            a, b, Ta, Tb = fold_data(fy); p = b.prev_yield.values if nm.startswith("Naive") else Tb.district_mean_yield.values
            e.append(np.sqrt(((p - b.yield_t_ha.values)**2).mean()))
        out["models"][nm] = dict(cv_rmse=float(np.mean(e)))
    best, oof = {}, {}
    for name, grid in GRIDS.items():
        res_by_p = []
        for p in grid:
            errs, resid = [], []
            for fy in FOLDS:
                a, b, Ta, Tb = fold_data(fy); m = build(name, p).fit(Ta[feats], a.yield_t_ha); r = m.predict(Tb[feats]) - b.yield_t_ha.values
                errs.append(np.sqrt((r**2).mean())); resid.extend(np.abs(r))
            res_by_p.append((float(np.mean(errs)), resid, p))
        sc, resid, p = min(res_by_p, key=lambda t: t[0]); best[name] = p; oof[name] = resid
        out["models"][name] = dict(params=p, cv_rmse=sc)
    ml = ["Ridge", "Random forest", "Gradient boosting"]
    sel = min(ml, key=lambda n: out["models"][n]["cv_rmse"]); out["selected"] = sel
    r_ = np.array(oof[sel]); q = float(np.quantile(r_, min(1, 0.9*(1 + 1/len(r_))))); out["interval_halfwidth"] = q; out["n_resid"] = len(r_)
    fe2 = FE(**cfg).fit(dev); Tdev, Tte = fe2.transform(dev), fe2.transform(te); y_te = te.yield_t_ha.values
    out["models"]["Naive: last year"]["test"] = metrics(y_te, te.prev_yield.values); out["models"]["District mean"]["test"] = metrics(y_te, Tte.district_mean_yield.values)
    fitted = {}
    for name in ml:
        m = build(name, best[name]).fit(Tdev[feats], dev.yield_t_ha); fitted[name] = m; out["models"][name]["test"] = metrics(y_te, m.predict(Tte[feats]))
    pred = fitted[sel].predict(Tte[feats]); out["cover90"] = float(((y_te >= pred - q) & (y_te <= pred + q)).mean())
    dr = te.drought_year.values
    out["subgroup"] = {"normal": metrics(y_te[~dr], pred[~dr]), "drought": metrics(y_te[dr], pred[dr]) if dr.any() else None,
                       "cover_drought": float(((y_te[dr] >= pred[dr] - q) & (y_te[dr] <= pred[dr] + q)).mean()) if dr.any() else None}
    out["skill_vs_naive"] = float(1 - out["models"][sel]["test"]["RMSE"]/out["models"]["Naive: last year"]["test"]["RMSE"])
    if keep_preds:
        pi = permutation_importance(fitted[sel], Tte[feats], y_te, n_repeats=30, random_state=SEED, scoring="neg_root_mean_squared_error")
        out["importance"] = dict(sorted({f: float(v) for f, v in zip(feats, pi.importances_mean)}.items(), key=lambda kv: -kv[1]))
        out["pred"] = dict(district=te.district.tolist(), year=te.year.tolist(), actual=y_te.tolist(), pred=pred.tolist(), drought=dr.tolist())
    return out

R = {}
cfgs = {"Soybean": dict(heat_thr=33.0, dry_thr=10), "Wheat": dict(heat_thr=31.5, dry_thr=10)}
for crop in ["Soybean", "Wheat"]:
    R[crop] = {}
    for sname, feats in SETS.items(): R[crop][sname] = run_pipeline(crop, feats, cfgs[crop], keep_preds=(sname == "In-season"))
# ---------- leakage demonstration (soybean, in-season set): random K-fold vs time-ordered split ----------
d = df_all[df_all.crop == "Soybean"].dropna(subset=["prev_yield", "roll3_yield"]).reset_index(drop=True); feats = SETS["In-season"]; cfg = cfgs["Soybean"]
p = R["Soybean"]["In-season"]["models"]["Random forest"]["params"]; r2 = []; rm = []
for a, b in KFold(5, shuffle=True, random_state=SEED).split(d):
    A, B = d.iloc[a], d.iloc[b]; f = FE(**cfg).fit(A); m = build("Random forest", p).fit(f.transform(A)[feats], A.yield_t_ha); pr = m.predict(f.transform(B)[feats]); mm = metrics(B.yield_t_ha.values, pr); r2.append(mm["R2"]); rm.append(mm["RMSE"])
R["leak"] = dict(random_R2=float(np.mean(r2)), random_RMSE=float(np.mean(rm)), time_R2=R["Soybean"]["In-season"]["models"]["Random forest"]["test"]["R2"], time_RMSE=R["Soybean"]["In-season"]["models"]["Random forest"]["test"]["RMSE"])
# ---------- robustness: noisy weather inputs at forecast time (soybean, in-season, selected model) ----------
dd = df_all[df_all.crop == "Soybean"].dropna(subset=["prev_yield", "roll3_yield"]).reset_index(drop=True); dev_, te_ = rows_in(dd, TRAIN[0], VAL[1]), rows_in(dd, *TEST)
fe_ = FE(**cfg).fit(dev_); sel_ = R["Soybean"]["In-season"]["selected"]; mdl = build(sel_, R["Soybean"]["In-season"]["models"][sel_]["params"]).fit(fe_.transform(dev_)[SETS["In-season"]], dev_.yield_t_ha)
rg = np.random.default_rng(SEED); R["robust"] = {}
for frac in [0.0, 0.05, 0.10, 0.20]:
    rm = []
    for _ in range(30):
        t2 = te_.copy()
        for col in ["rain_season_mm", "dry_spell_days", "tmax_critical"]: t2[col] = t2[col] + rg.normal(0, frac*dd[col].std(), len(t2))
        rm.append(metrics(t2.yield_t_ha.values, mdl.predict(fe_.transform(t2)[SETS["In-season"]]))["RMSE"])
    R["robust"][str(frac)] = float(np.mean(rm))
print("robust:", R["robust"])
json.dump(R, open("data/yield_results.json", "w"), indent=1, default=float)
for crop in ["Soybean", "Wheat"]:
    for s in SETS:
        o = R[crop][s]; print(crop, s, "selected:", o["selected"], "| cv:", {k: round(v["cv_rmse"], 3) for k, v in o["models"].items()}, "| test RMSE:", {k: round(v["test"]["RMSE"], 3) for k, v in o["models"].items()}, "| skill vs naive: %.2f cover90: %.2f" % (o["skill_vs_naive"], o["cover90"]))
print(R["leak"]); print(R["Soybean"]["In-season"]["subgroup"]); print(list(R["Soybean"]["In-season"]["importance"].items())[:8])
