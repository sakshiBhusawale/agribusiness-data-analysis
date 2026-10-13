"""Step 0: build a HYPOTHETICAL (synthetic) mandi-price dataset with realistic quality problems injected.
Seeded, so anyone running this gets the identical file."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

crops   = {"Soybean": 4400, "Onion": 1800, "Wheat": 2500}          # Rs/quintal base
markets = {"Pune": ("Pune", 1.03), "Lasalgaon": ("Nashik", 0.95),
           "Latur": ("Latur", 0.98), "Nagpur": ("Nagpur", 1.01)}
dates = pd.date_range("2025-10-01", "2025-11-30", freq="D")

rows = []
for c, base in crops.items():
    for m, (dist, off) in markets.items():
        t = np.arange(len(dates))
        season = 1 + 0.06*np.sin(t/9) + 0.0006*t
        modal = base*off*season*(1 + rng.normal(0, 0.015, len(dates)))
        arr_base = {"Soybean": 180, "Onion": 420, "Wheat": 260}[c]
        arrivals = arr_base*(1 - 3*(modal/(base*off) - 1))*(1 + rng.normal(0, 0.12, len(dates)))
        for i, d in enumerate(dates):
            if rng.random() < 0.18:       # market closed / no report that day
                continue
            mo = modal[i]
            rows.append(dict(date=d, state="Maharashtra", district=dist, market=m, commodity=c,
                             variety={"Soybean":"Yellow","Onion":"Red","Wheat":"Lokwan"}[c],
                             min_price=mo*rng.uniform(0.90, 0.97), max_price=mo*rng.uniform(1.03, 1.10),
                             modal_price=mo, arrivals_tonnes=max(arrivals[i], 5), price_unit="Rs/Quintal"))
truth = pd.DataFrame(rows).reset_index(drop=True)
truth.insert(0, "record_id", range(1, len(truth)+1))
truth.round(2).to_csv("data/ground_truth_clean.csv", index=False)     # the 'answer key' (never used in cleaning)

raw = truth.copy()
n = len(raw)
def pick(frac): return rng.choice(n, int(n*frac), replace=False)

# --- 1. inconsistent date formats
fmt = rng.choice(["iso", "dmy_slash", "dmy_mon"], n, p=[.55, .30, .15])
raw["date"] = [d.strftime({"iso":"%Y-%m-%d","dmy_slash":"%d/%m/%Y","dmy_mon":"%d-%b-%Y"}[f]) for d, f in zip(raw["date"], fmt)]
# --- 2. spelling / case / whitespace variants
cv = {"Soybean":["Soybean","Soyabean","SOYBEAN","soybean ","Soya bean"], "Onion":["Onion","onion","Onion ","ONION"], "Wheat":["Wheat","wheat","Wheat ","WHEAT","Wheat(Lokwan)"]}
raw["commodity"] = [rng.choice(cv[c]) if rng.random() < .30 else c for c in raw["commodity"]]
mv = {"Pune":["Pune","pune","Pune APMC"], "Lasalgaon":["Lasalgaon","Lasalgoan","Lasalgaon(Niphad)"], "Latur":["Latur","Latur ","LATUR"], "Nagpur":["Nagpur","Nagpur(Kalamna)","nagpur"]}
raw["market"] = [rng.choice(mv[m]) if rng.random() < .25 else m for m in raw["market"]]
# --- 3. unit inconsistency (a few feeds report Rs/kg)
idx = pick(0.06)
for col in ["min_price","max_price","modal_price"]: raw.loc[idx, col] = raw.loc[idx, col]/100
raw.loc[idx, "price_unit"] = rng.choice(["Rs/Kg","Rs/kg"], len(idx))
raw.loc[raw.index.difference(idx), "price_unit"] = rng.choice(["Rs/Quintal","Rs./Quintal"], n-len(idx), p=[.9,.1])
# --- 4. typo outliers in modal price (extra/missing zero) + genuine-looking spikes
typo = pick(0.012); raw.loc[typo, "modal_price"] = raw.loc[typo, "modal_price"]*rng.choice([10, 0.1], len(typo))
spike = pick(0.01);  raw.loc[spike, "modal_price"] = raw.loc[spike, "modal_price"]*1.35   # plausible but extreme
# --- 5. logical errors
sw = pick(0.015); raw.loc[sw, ["min_price","max_price"]] = raw.loc[sw, ["max_price","min_price"]].values
# --- 6. negative arrivals
neg = pick(0.008); raw.loc[neg, "arrivals_tonnes"] = -raw.loc[neg, "arrivals_tonnes"]
# --- 7. missing values with mixed placeholders
raw = raw.astype({"arrivals_tonnes":"object","modal_price":"object","min_price":"object","max_price":"object"})
for col, frac in [("arrivals_tonnes", .14), ("modal_price", .04), ("min_price", .02), ("max_price", .02)]:
    ix = pick(frac); raw.loc[ix, col] = rng.choice(["NA","-","","N/A", np.nan], len(ix))
# --- 8. exact duplicate rows (re-downloaded batch)
dup = raw.sample(int(n*.035), random_state=1); raw = pd.concat([raw, dup]).sample(frac=1, random_state=7).reset_index(drop=True)
raw.to_csv("data/raw_mandi_prices.csv", index=False)
print("raw shape:", raw.shape, "| truth shape:", truth.shape)
