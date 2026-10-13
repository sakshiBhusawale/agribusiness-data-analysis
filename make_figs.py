import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
G, DG, AMB, RED, GREY = "#2E7D32", "#1B5E20", "#F9A825", "#C62828", "#607D8B"
log = json.load(open("data/cleaning_log.json")); imp = json.load(open("data/impact.json"))
st = {s["step"]: s for s in log["steps"]}

# ---------- Fig 1: workflow diagram ----------
fig, ax = plt.subplots(figsize=(10, 6.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
boxes = [("1  Ingest & preserve", "Read raw file as text\nSave SHA-256 fingerprint", GREY),
         ("2  Profile", "Row counts, missing %,\nduplicates, value ranges", GREY),
         ("3  Structural fixes", "Placeholders → NaN, data types,\nparse dates, drop duplicates", G),
         ("4  Standardise", "Spelling, case, units\n(Rs/Kg → Rs/Quintal)", G),
         ("5  Rule validation", "min ≤ modal ≤ max,\nno negative arrivals", G),
         ("6  Detect outliers", "Robust z-score per crop+market;\ntypos vs genuine spikes", AMB),
         ("7  Impute", "Time interpolation (prices),\ngroup median (arrivals) + flags", AMB),
         ("8  Transform", "Spread, log(arrivals),\nmin-max / z-score scaling", G),
         ("9  Validate & export", "Assertion checks, cleaning log,\nversioned clean dataset", DG)]
pos = []
for i, (t, d, c) in enumerate(boxes):
    r, k = divmod(i, 3); x = 0.35 + (k if r % 2 == 0 else 2-k)*3.2; y = 4.85 - r*2.15
    ax.add_patch(FancyBboxPatch((x, y), 2.9, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=c, lw=2.2))
    ax.add_patch(FancyBboxPatch((x, y+1.05), 2.9, 0.45, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec=c))
    ax.text(x+1.45, y+1.27, t, ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)
    ax.text(x+1.45, y+0.48, d, ha="center", va="center", fontsize=9, color="#263238"); pos.append((x, y))
def arrow(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=2, color="#455A64"))
for i in range(8):
    (x1, y1), (x2, y2) = pos[i], pos[i+1]
    if y1 == y2: arrow((x1+2.9, y1+0.75) if x2 > x1 else (x1, y1+0.75), (x2, y2+0.75) if x2 > x1 else (x2+2.9, y2+0.75))
    else: arrow((x1+1.45, y1), (x2+1.45, y2+1.5))
ax.text(5, 6.45, "Preprocessing workflow: raw file in, validated dataset out", ha="center", fontsize=12.5, fontweight="bold", color=DG)
ax.text(5, 0.05, "Grey = understand   Green = fix   Amber = judgement call (always flagged and logged)", ha="center", fontsize=9, color="#455A64")
plt.savefig("figs/fig_workflow.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------- Fig 2: issues found ----------
raw = pd.read_csv("data/raw_mandi_prices.csv", dtype=str, keep_default_na=False)
non_iso = int((~raw.date.str.match(r"^\d{4}-\d{2}-\d{2}$")).sum())
issues = {"Missing values (placeholders)": st["Placeholder to NaN"]["changed"], "Inconsistent commodity spelling": st["Standardise commodity"]["changed"],
          "Inconsistent market spelling": st["Standardise market"]["changed"], "Non-ISO date format": non_iso,
          "Exact duplicate rows": st["Remove exact duplicates"]["changed"], "Wrong unit (Rs/Kg)": st["Convert Rs/Kg to Rs/Quintal"]["changed"],
          "min > max price": st["Swap min/max when min > max"]["changed"], "Typo outliers (10× / 0.1×)": st["Detect outliers (robust z > 3.5)"]["typos_set_to_NaN"],
          "Plausible price spikes": st["Detect outliers (robust z > 3.5)"]["plausible_spikes_kept"], "Negative arrivals": st["Negative arrivals to NaN"]["changed"]}
issues = dict(sorted(issues.items(), key=lambda kv: kv[1]))
fig, ax = plt.subplots(figsize=(8.2, 4.4)); bars = ax.barh(list(issues), list(issues.values()), color=G)
for b, v in zip(bars, issues.values()): ax.text(v+2, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=9)
ax.set_xlabel("Number of affected records (raw file = 630 rows)"); ax.set_title("Data quality problems found in the raw file", fontweight="bold", color=DG, loc="left")
plt.savefig("figs/fig_issues.png", dpi=200, bbox_inches="tight"); plt.close(); json.dump(issues, open("data/issues.json", "w"))

# ---------- Fig 3: before / after, one market-crop series ----------
clean = pd.read_csv("data/clean_mandi_prices.csv", parse_dates=["date"]); truth = pd.read_csv("data/ground_truth_clean.csv", parse_dates=["date"])
c = clean[(clean.commodity == "Onion") & (clean.market == "Pune")].sort_values("date")
r = raw.copy(); r["modal_price"] = pd.to_numeric(r.modal_price, errors="coerce")
r["commodity"] = r.commodity.str.strip().str.lower(); r["market"] = r.market.str.strip().str.lower()
r = r[(r.commodity == "onion") & (r.market == "pune")].copy()
def _pd(x):
    for f in ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"):
        try: return pd.to_datetime(x, format=f)
        except ValueError: pass
r["date"] = r.date.map(_pd)
r["unit_kg"] = r.price_unit.str.lower().str.contains("kg")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
a1.scatter(r.date, r.modal_price, s=14, color=GREY, label="reported value"); a1.set_yscale("log"); a1.set_title("BEFORE: raw values (log scale)", fontweight="bold", color=RED, loc="left")
a1.set_ylabel("Modal price (Rs/Quintal)")
out = r[(r.modal_price > 5000) | (r.modal_price < 800)]
out = out[out.modal_price.notna()]
for _, row in out.iterrows(): a1.annotate(("wrong unit (Rs/Kg)" if row.unit_kg else "10x typo"), (row.date, row.modal_price), xytext=(-62, 5), textcoords="offset points", fontsize=8, color=RED)
a1.scatter(out.date, out.modal_price, s=40, color=RED, zorder=3)
a2.plot(c.date, c.modal_price, color=G, lw=1.6, label="Cleaned series")
im = c[c.modal_price_was_imputed]; a2.scatter(im.date, im.modal_price, color=AMB, s=45, zorder=3, edgecolor="k", label="Imputed value")
sp = c[c.flag_price_spike_kept]; a2.scatter(sp.date, sp.modal_price, color=RED, marker="^", s=55, zorder=3, label="Spike kept & flagged")
a2.set_title("AFTER: cleaned series (linear scale)", fontweight="bold", color=G, loc="left"); a2.legend(frameon=False, fontsize=8, loc="upper right")
for a in (a1, a2): a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("figs/fig_before_after.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------- Fig 4: imputation comparison ----------
pm = {k: v for k, v in imp["price_methods"].items() if k != "n"}
fig, ax = plt.subplots(figsize=(7.4, 3.2)); cols = [GREY, GREY, GREY, G]
b = ax.barh(list(pm)[::-1], list(pm.values())[::-1], color=cols[::-1])
for bar, v in zip(b, list(pm.values())[::-1]): ax.text(v+0.5, bar.get_y()+bar.get_height()/2, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Mean absolute % error vs true value (lower is better)"); ax.set_title(f"Imputing modal price: which method recovers the truth best? (n={pm and imp['price_methods']['n']})", fontsize=10, fontweight="bold", color=DG, loc="left")
plt.savefig("figs/fig_imputation.png", dpi=200, bbox_inches="tight"); plt.close(); print("ok", issues)
