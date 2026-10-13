"""Reproducible cleaning pipeline for the hypothetical mandi price dataset.
Every step: (1) does ONE job, (2) logs what it changed, (3) never overwrites the raw file."""
import hashlib, json, difflib
import numpy as np, pandas as pd

SEED = 42
RAW, CLEAN = "data/raw_mandi_prices.csv", "data/clean_mandi_prices.csv"
log = {"steps": []}
def record(step, **kw): log["steps"].append({"step": step, **kw})

# ---------- Step 1: load raw as text + fingerprint ----------
raw_bytes = open(RAW, "rb").read()
log["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()[:16]
df = pd.read_csv(RAW, dtype=str, keep_default_na=False)       # read EVERYTHING as text first
log["raw_rows"] = len(df)

# ---------- Step 2: profile BEFORE cleaning ----------
PLACEHOLDERS = {"", "NA", "N/A", "-", "na", "n/a", "null", "None", "nan", "NaN"}
num_cols = ["min_price", "max_price", "modal_price", "arrivals_tonnes"]
def missing_report(d):
    return {c: int(d[c].astype(str).str.strip().isin(PLACEHOLDERS).sum()) for c in num_cols}
log["missing_before"] = missing_report(df)
log["duplicates_before"] = int(df.duplicated().sum())
log["variants_before"] = {"commodity": int(df.commodity.nunique()), "market": int(df.market.nunique()),
                          "unit": int(df.price_unit.nunique()), "date_formats": 3}

# ---------- Step 3: placeholders -> real NaN, text -> numbers ----------
n_ph = 0
for c in num_cols:
    s = df[c].str.strip()
    n_ph += int(s.isin(PLACEHOLDERS).sum())
    df[c] = pd.to_numeric(s.mask(s.isin(PLACEHOLDERS)), errors="coerce")
record("Placeholder to NaN", changed=n_ph)

# ---------- Step 4: remove exact duplicates ----------
before = len(df); df = df.drop_duplicates().reset_index(drop=True)
record("Remove exact duplicates", changed=before-len(df), rows_after=len(df))

# ---------- Step 5: standardise text (case, spaces, spelling) ----------
COMMODITIES = ["Soybean", "Onion", "Wheat"]; MARKETS = ["Pune", "Lasalgaon", "Latur", "Nagpur"]
ALIASES = {"soya bean": "Soybean", "soyabean": "Soybean", "wheat(lokwan)": "Wheat",
           "lasalgaon(niphad)": "Lasalgaon", "nagpur(kalamna)": "Nagpur", "pune apmc": "Pune", "lasalgoan": "Lasalgaon"}
def canon(x, vocab):
    k = " ".join(str(x).strip().lower().split())
    if k in ALIASES: return ALIASES[k]
    hit = difflib.get_close_matches(k, [v.lower() for v in vocab], n=1, cutoff=0.8)   # fuzzy fallback
    return next((v for v in vocab if v.lower() == hit[0]), np.nan) if hit else np.nan
for col, vocab in [("commodity", COMMODITIES), ("market", MARKETS)]:
    old = df[col].copy(); df[col] = df[col].map(lambda x: canon(x, vocab))
    record(f"Standardise {col}", changed=int((old != df[col]).sum()), unmatched=int(df[col].isna().sum()),
           distinct_before=int(old.nunique()), distinct_after=int(df[col].nunique()))

# ---------- Step 6: parse dates (3 known formats, explicit - never guess) ----------
def parse_date(s):
    for f in ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"):
        try: return pd.to_datetime(s, format=f)
        except ValueError: pass
    return pd.NaT
df["date"] = df["date"].map(parse_date)
record("Parse dates", unparsed=int(df.date.isna().sum()), min=str(df.date.min().date()), max=str(df.date.max().date()))

# ---------- Step 7: harmonise units to Rs/Quintal ----------
is_kg = df.price_unit.str.lower().str.contains("kg")
for c in ["min_price", "max_price", "modal_price"]: df.loc[is_kg, c] = df.loc[is_kg, c]*100
df["price_unit"] = "Rs/Quintal"
record("Convert Rs/Kg to Rs/Quintal", changed=int(is_kg.sum()))

# ---------- Step 8: logical-consistency rules ----------
neg = df.arrivals_tonnes < 0
df.loc[neg, "arrivals_tonnes"] = np.nan
swap = df.min_price > df.max_price
df.loc[swap, ["min_price", "max_price"]] = df.loc[swap, ["max_price", "min_price"]].values
record("Negative arrivals to NaN", changed=int(neg.sum()))
record("Swap min/max when min > max", changed=int(swap.sum()))

# ---------- Step 9: outlier detection (robust, per crop+market series) ----------
df = df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)
g = df.groupby(["commodity", "market"])["modal_price"]
med = g.transform("median"); mad = g.transform(lambda s: (s - s.median()).abs().median())
df["robust_z"] = 0.6745*(df.modal_price - med)/mad
df["ratio_to_median"] = df.modal_price/med
df["flag_outlier"] = df.robust_z.abs() > 3.5
typo = df.flag_outlier & ((df.ratio_to_median > 5) | (df.ratio_to_median < 0.2))      # 10x / 0.1x = data-entry error
df["flag_price_spike_kept"] = df.flag_outlier & ~typo                                  # plausible but extreme -> KEEP, flag
log["outlier_examples"] = df.loc[typo, ["date","market","commodity","modal_price","ratio_to_median"]].head(4).round(1).astype(str).to_dict("records")
spike_examples = df.loc[df.flag_price_spike_kept, ["date","market","commodity","modal_price","ratio_to_median"]].head(3).round(2).astype(str).to_dict("records")
log["spike_examples"] = spike_examples
df.loc[typo, "modal_price"] = np.nan
record("Detect outliers (robust z > 3.5)", flagged=int(df.flag_outlier.sum()), typos_set_to_NaN=int(typo.sum()), plausible_spikes_kept=int(df.flag_price_spike_kept.sum()))

# ---------- Step 10: min/max/modal consistency (modal must lie inside [min, max]) ----------
bad = (df.modal_price < df.min_price) | (df.modal_price > df.max_price)
record("Modal outside [min,max] (flagged)", flagged=int(bad.sum()))
df["flag_modal_out_of_range"] = bad

# ---------- Step 11: imputation with audit flags ----------
df["modal_price_was_imputed"] = df.modal_price.isna()
df["arrivals_was_imputed"] = df.arrivals_tonnes.isna()
df["min_price_was_imputed"] = df.min_price.isna()
df["max_price_was_imputed"] = df.max_price.isna()
snap_before_imp = df[["modal_price", "arrivals_tonnes"]].isna().sum().to_dict()
# prices: time-based interpolation inside each crop+market series (max gap 5 days), then group median as fallback
df = df.set_index("date")
def interp(s): return s.interpolate(method="time", limit=5, limit_direction="both")
df["modal_price"] = df.groupby(["commodity", "market"])["modal_price"].transform(interp)
# min/max imputed from modal using that crop+market's typical ratio
for c in ["min_price", "max_price"]:
    ratio = (df[c]/df["modal_price"]).groupby([df.commodity, df.market]).transform("median")
    df[c] = df[c].fillna(df["modal_price"]*ratio)
# arrivals: group median (skewed, so median not mean)
df["arrivals_tonnes"] = df["arrivals_tonnes"].fillna(df.groupby(["commodity", "market"])["arrivals_tonnes"].transform("median"))
df = df.reset_index()
record("Impute modal_price (time interpolation)", imputed=int(df.modal_price_was_imputed.sum()))
record("Impute min/max (ratio to modal)", imputed=int(df.min_price_was_imputed.sum()+df.max_price_was_imputed.sum()))
record("Impute arrivals (group median)", imputed=int(df.arrivals_was_imputed.sum()))

# ---------- Step 12: derived + normalised features ----------
df["price_spread"] = df.max_price - df.min_price
df["log_arrivals"] = np.log1p(df.arrivals_tonnes)
df["modal_price_minmax"] = df.groupby("commodity")["modal_price"].transform(lambda s: (s - s.min())/(s.max() - s.min()))
df["modal_price_zscore"] = df.groupby("commodity")["modal_price"].transform(lambda s: (s - s.mean())/s.std())

# ---------- Step 13: validation gate ----------
checks = {
 "no_duplicate_keys": not df.duplicated(["date","market","commodity"]).any(),
 "no_missing_in_core_cols": bool(df[["date","market","commodity","modal_price","arrivals_tonnes"]].notna().all().all()),
 "min<=max_everywhere": bool((df.min_price <= df.max_price).all()),
 "all_prices_positive": bool((df[["min_price","max_price","modal_price"]] > 0).all().all()),
 "only_3_commodities_4_markets": df.commodity.nunique() == 3 and df.market.nunique() == 4,
 "single_unit": df.price_unit.nunique() == 1}
log["validation"] = checks
log["missing_remaining"] = int(df[["modal_price","arrivals_tonnes","min_price","max_price"]].isna().sum().sum())
log["clean_rows"] = len(df)
df.drop(columns=["robust_z", "ratio_to_median"]).to_csv(CLEAN, index=False)

# ---------- Evaluation vs ground truth (only possible because the data are synthetic) ----------
truth = pd.read_csv("data/ground_truth_clean.csv", parse_dates=["date"])
raw_ids = pd.read_csv(RAW, dtype=str, keep_default_na=False)
df["record_id"] = df["record_id"].astype(int)
m = df.merge(truth[["record_id","modal_price","arrivals_tonnes"]], on="record_id", suffixes=("", "_true"))
ip = m[m.modal_price_was_imputed]; ia = m[m.arrivals_was_imputed]
log["eval"] = {
 "modal_price_imputed_n": len(ip), "modal_price_MAE": float((ip.modal_price - ip.modal_price_true).abs().mean()),
 "modal_price_MAPE_pct": float(((ip.modal_price - ip.modal_price_true).abs()/ip.modal_price_true).mean()*100),
 "arrivals_imputed_n": len(ia), "arrivals_MAE": float((ia.arrivals_tonnes - ia.arrivals_tonnes_true).abs().mean()),
 "arrivals_MAPE_pct": float(((ia.arrivals_tonnes - ia.arrivals_tonnes_true).abs()/ia.arrivals_tonnes_true).mean()*100),
 "clean_rows": len(df), "truth_rows": len(truth)}
json.dump(log, open("data/cleaning_log.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in log.items() if k != "steps"}, indent=1, default=str))
for s in log["steps"]: print(s)
