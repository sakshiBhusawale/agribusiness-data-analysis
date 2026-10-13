"""Evidence: how wrong are conclusions if we skip cleaning? Compare naive vs cleaned vs ground truth."""
import json, numpy as np, pandas as pd
raw = pd.read_csv("data/raw_mandi_prices.csv", dtype=str, keep_default_na=False)
clean = pd.read_csv("data/clean_mandi_prices.csv", parse_dates=["date"]); truth = pd.read_csv("data/ground_truth_clean.csv", parse_dates=["date"])
out = {}
# --- naive: just coerce to numbers, drop blanks, lower-case nothing
naive = raw.copy()
for c in ["modal_price","arrivals_tonnes"]: naive[c] = pd.to_numeric(naive[c], errors="coerce")
naive["commodity_raw"] = naive["commodity"]
rows = []
for crop in ["Soybean","Onion","Wheat"]:
    nv = naive[naive.commodity == crop]               # exact-string match only (what a naive filter does)
    rows.append(dict(commodity=crop, naive_rows=len(nv), naive_mean=nv.modal_price.mean(), naive_max=nv.modal_price.max(),
                     clean_rows=int((clean.commodity==crop).sum()), clean_mean=clean[clean.commodity==crop].modal_price.mean(),
                     clean_max=clean[clean.commodity==crop].modal_price.max(),
                     true_rows=int((truth.commodity==crop).sum()), true_mean=truth[truth.commodity==crop].modal_price.mean(), true_max=truth[truth.commodity==crop].modal_price.max()))
out["summary"] = rows
# --- imputation method comparison on the exact cells that were imputed
m = clean.merge(truth[["record_id","modal_price","arrivals_tonnes"]], on="record_id", suffixes=("","_true"))
ip = m[m.modal_price_was_imputed].copy(); ia = m[m.arrivals_was_imputed].copy()
obs = clean[~clean.modal_price_was_imputed & ~clean.flag_modal_out_of_range]
def mape(a, b): return float((np.abs(a-b)/b).mean()*100)
gm_price = ip.commodity.map(obs.groupby("commodity").modal_price.mean())
gmk_price = ip.set_index(["commodity","market"]).index.map(obs.groupby(["commodity","market"]).modal_price.median())
out["price_methods"] = {"Overall mean (all crops pooled)": mape(np.full(len(ip), obs.modal_price.mean()), ip.modal_price_true),
                        "Crop mean": mape(gm_price, ip.modal_price_true),
                        "Crop + market median": mape(np.array(gmk_price), ip.modal_price_true),
                        "Time interpolation (used)": mape(ip.modal_price, ip.modal_price_true), "n": len(ip)}
oa = clean[~clean.arrivals_was_imputed]
ga = ia.commodity.map(oa.groupby("commodity").arrivals_tonnes.mean())
out["arrival_methods"] = {"Overall mean (all crops pooled)": mape(np.full(len(ia), oa.arrivals_tonnes.mean()), ia.arrivals_tonnes_true),
                          "Crop mean": mape(ga, ia.arrivals_tonnes_true),
                          "Crop + market median (used)": mape(ia.arrivals_tonnes, ia.arrivals_tonnes_true), "n": len(ia)}
# --- relationship: arrivals vs price (correlation) per crop
cor = {}
for crop in ["Soybean","Onion","Wheat"]:
    nv = naive[(naive.commodity==crop)].dropna(subset=["modal_price","arrivals_tonnes"])
    c_ = clean[clean.commodity==crop]; t_ = truth[truth.commodity==crop]
    cor[crop] = dict(naive=float(nv.modal_price.corr(nv.arrivals_tonnes)), clean=float(c_.modal_price.corr(c_.arrivals_tonnes)), true=float(t_.modal_price.corr(t_.arrivals_tonnes)))
out["corr"] = cor
json.dump(out, open("data/impact.json","w"), indent=1)
pd.set_option("display.width",200); print(pd.DataFrame(rows).round(0).to_string()); print(json.dumps({k:out[k] for k in ["price_methods","arrival_methods","corr"]}, indent=1))
