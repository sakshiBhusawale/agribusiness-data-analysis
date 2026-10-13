import json, pandas as pd, numpy as np
raw = pd.read_csv("data/raw_mandi_prices.csv", dtype=str, keep_default_na=False)
clean = pd.read_csv("data/clean_mandi_prices.csv"); clean["record_id"] = clean.record_id.astype(int); raw["record_id"] = raw.record_id.astype(int)
ex = {}
# A: spelling + date variant
a = raw[raw.commodity.isin(["SOYBEAN"]) & raw.date.str.contains("/")].iloc[0]; c = clean[clean.record_id == a.record_id].iloc[0]
ex["A"] = dict(rid=int(a.record_id), raw_date=a.date, raw_commodity=a.commodity, clean_date=c.date, clean_commodity=c.commodity)
# B: Rs/Kg
b = raw[raw.price_unit.str.lower() == "rs/kg"].iloc[0]; c = clean[clean.record_id == b.record_id].iloc[0]
ex["B"] = dict(rid=int(b.record_id), market=c.market, commodity=c.commodity, raw_modal=float(b.modal_price), raw_unit=b.price_unit, clean_modal=float(c.modal_price))
# C: typo
t = clean[(clean.commodity=="Onion")&(clean.market=="Pune")&(clean.date=="2025-10-21")].iloc[0]
rawrow = raw[raw.record_id == t.record_id].iloc[0]
med = clean[(clean.commodity=="Onion")&(clean.market=="Pune")].modal_price.median()
ex["C"] = dict(rid=int(t.record_id), raw_modal=float(rawrow.modal_price), median=float(med), clean_modal=float(t.modal_price), imputed=bool(t.modal_price_was_imputed))
# D: missing arrivals
d = raw[raw.arrivals_tonnes.isin(["N/A"])].iloc[0]; c = clean[clean.record_id == d.record_id].iloc[0]
ex["D"] = dict(rid=int(d.record_id), market=c.market, commodity=c.commodity, raw="N/A", clean=float(c.arrivals_tonnes))
# E: normalisation example
e = clean[(clean.commodity=="Soybean")].iloc[5]
ex["E"] = dict(modal=float(e.modal_price), minmax=float(e.modal_price_minmax), z=float(e.modal_price_zscore), arr=float(e.arrivals_tonnes), logarr=float(e.log_arrivals), spread=float(e.price_spread), min=float(clean[clean.commodity=="Soybean"].modal_price.min()), max=float(clean[clean.commodity=="Soybean"].modal_price.max()))
ex["clean_head"] = clean.head(5).round(2).astype(str).to_dict("records")
json.dump(ex, open("data/examples.json","w"), indent=1); print(json.dumps({k:ex[k] for k in "ABCDE"}, indent=1))
