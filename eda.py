"""Week 3: exploratory analysis of the CLEANED hypothetical mandi dataset (synthetic) -> numbers + figures for the EDA report outline."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.spines.top":False,"axes.spines.right":False})
G, DG, AMB, RED, GREY, BLUE = "#2E7D32", "#1B5E20", "#F9A825", "#C62828", "#607D8B", "#1565C0"
CROP_COL = {"Soybean": G, "Onion": AMB, "Wheat": BLUE}
d = pd.read_csv("data/clean_mandi_prices.csv", parse_dates=["date"]); R = {}
crops = ["Soybean", "Onion", "Wheat"]

# ---- 1. descriptive statistics
rows = []
for c in crops:
    s = d[d.commodity == c].modal_price
    rows.append(dict(crop=c, n=len(s), mean=s.mean(), median=s.median(), sd=s.std(), cv=s.std()/s.mean()*100, min=s.min(), max=s.max(), skew=stats.skew(s)))
R["desc"] = rows
arr = []
for c in crops:
    s = d[d.commodity == c].arrivals_tonnes
    arr.append(dict(crop=c, mean=s.mean(), median=s.median(), skew=stats.skew(s), skew_log=stats.skew(np.log1p(s))))
R["arr"] = arr

# ---- 2. regional comparison (price index = price / crop mean * 100)
d["price_index"] = d.modal_price / d.groupby("commodity").modal_price.transform("mean") * 100
mk = d.groupby("market").price_index.agg(["mean", "std"]).sort_values("mean", ascending=False)
R["market"] = {m: dict(mean=float(r["mean"]), sd=float(r["std"])) for m, r in mk.iterrows()}
R["market_gap"] = float(mk["mean"].max() - mk["mean"].min())
# ANOVA across markets (price index)
groups = [g.price_index.values for _, g in d.groupby("market")]
f, p = stats.f_oneway(*groups); R["anova"] = dict(F=float(f), p=float(p))

# ---- 3. trend over the 61 days (weekly)
d["week"] = d.date.dt.to_period("W").dt.start_time
wk = d.groupby(["commodity", "week"]).modal_price.mean().unstack(0)
R["trend"] = {c: dict(first=float(wk[c].dropna().iloc[:2].mean()), last=float(wk[c].dropna().iloc[-2:].mean())) for c in crops}
for c in crops: R["trend"][c]["pct"] = (R["trend"][c]["last"]/R["trend"][c]["first"]-1)*100

# ---- 4. correlations
cor = {}
for c in crops:
    s = d[d.commodity == c]; pr = stats.pearsonr(s.arrivals_tonnes, s.modal_price); sp = stats.spearmanr(s.arrivals_tonnes, s.modal_price)
    cor[c] = dict(pearson=float(pr[0]), p=float(pr[1]), spearman=float(sp[0]), n=len(s))
R["cor"] = cor
pooled = stats.pearsonr(d.arrivals_tonnes, d.modal_price); R["pooled"] = dict(r=float(pooled[0]))
z = d.copy()
for col in ["modal_price", "min_price", "max_price", "arrivals_tonnes", "price_spread"]:
    z[col] = z.groupby("commodity")[col].transform(lambda s: (s - s.mean())/s.std())
cm = z[["modal_price", "min_price", "max_price", "arrivals_tonnes", "price_spread"]].corr(); R["cm"] = cm.round(2).to_dict()

# ---- 5. anomalies (robust z within crop+market, same rule as Week 2)
g = d.groupby(["commodity", "market"]).modal_price
d["ratio"] = d.modal_price / g.transform("median")
mad = g.transform(lambda s: (s - s.median()).abs().median()); d["rz"] = 0.6745*(d.modal_price - g.transform("median"))/mad
fl = d[d.rz.abs() > 3.5]
R["anom"] = dict(n=len(fl), pct=len(fl)/len(d)*100, by_crop=fl.commodity.value_counts().to_dict(),
                 rows=fl.sort_values("ratio", ascending=False).head(4)[["date", "market", "commodity", "modal_price", "ratio"]].assign(date=lambda x: x.date.dt.strftime("%d %b")).round(2).to_dict("records"))
# effect of the spikes on the mean and sd
ex = d[d.rz.abs() <= 3.5]
R["anom"]["effect"] = {c: dict(sd_all=float(d[d.commodity==c].modal_price.std()), sd_ex=float(ex[ex.commodity==c].modal_price.std()), mean_all=float(d[d.commodity==c].modal_price.mean()), mean_ex=float(ex[ex.commodity==c].modal_price.mean())) for c in crops}

# ---- 6. day-of-week check
d["dow"] = d.date.dt.day_name(); R["dow"] = d.groupby("dow").price_index.mean().round(1).to_dict()
obs = d[~d.arrivals_was_imputed & ~d.modal_price_was_imputed]
R["sens"] = {c: dict(n=int((obs.commodity == c).sum()), r=float(stats.pearsonr(obs[obs.commodity == c].arrivals_tonnes, obs[obs.commodity == c].modal_price)[0])) for c in crops}
R["spike_min"] = float(fl.ratio.min()); R["spike_max"] = float(fl.ratio.max()); R["n"] = len(d)
R["imp"] = dict(modal=int(d.modal_price_was_imputed.sum()), arr=int(d.arrivals_was_imputed.sum()))
json.dump(R, open("data/eda.json", "w"), indent=1, default=float)

# =============== FIGURES ===============
# F1 distributions
fig, ax = plt.subplots(2, 3, figsize=(10, 5.4))
for i, c in enumerate(crops):
    s = d[d.commodity == c]
    ax[0, i].hist(s.modal_price, bins=22, color=CROP_COL[c], edgecolor="white"); ax[0, i].axvline(s.modal_price.median(), color="k", ls="--", lw=1)
    ax[0, i].set_title(f"{c}: modal price", fontweight="bold", loc="left"); ax[0, i].set_xlabel("Rs per quintal")
    ax[1, i].hist(s.arrivals_tonnes, bins=22, color=CROP_COL[c], alpha=.75, edgecolor="white"); ax[1, i].set_title(f"{c}: arrivals", fontweight="bold", loc="left"); ax[1, i].set_xlabel("tonnes per day")
ax[0, 0].set_ylabel("Number of days"); ax[1, 0].set_ylabel("Number of days")
plt.tight_layout(); plt.savefig("figs/fig_eda_hist.png", dpi=200); plt.close()
# F2 box by market
fig, ax = plt.subplots(figsize=(7.2, 3.6)); order = list(mk.index)
bp = ax.boxplot([d[d.market == m].price_index for m in order], tick_labels=order, patch_artist=True, medianprops=dict(color="k"), flierprops=dict(marker="o", markerfacecolor=RED, markeredgecolor=RED, markersize=4))
for b in bp["boxes"]: b.set_facecolor("#A5D6A7")
ax.axhline(100, color=GREY, ls=":"); ax.set_ylabel("Price index (crop average = 100)"); ax.set_title("Regional comparison: price level by market", fontweight="bold", color=DG, loc="left")
plt.savefig("figs/fig_eda_box.png", dpi=200, bbox_inches="tight"); plt.close()
# F3 weekly trend
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for c in crops:
    s = wk[c].dropna(); ax.plot(s.index, s / s.iloc[0] * 100, marker="o", ms=3.5, lw=1.8, color=CROP_COL[c], label=c)
ax.axhline(100, color=GREY, ls=":"); ax.set_ylabel("Weekly mean price (first week = 100)"); ax.legend(frameon=False); ax.tick_params(axis="x", rotation=30)
ax.set_title("Trend: weekly average modal price, 1 Oct to 30 Nov 2025", fontweight="bold", color=DG, loc="left")
plt.savefig("figs/fig_eda_trend.png", dpi=200, bbox_inches="tight"); plt.close()
# F4 scatter per crop
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for i, c in enumerate(crops):
    s = d[d.commodity == c]; ax[i].scatter(s.arrivals_tonnes, s.modal_price, s=11, color=CROP_COL[c], alpha=.7)
    m_, b_ = np.polyfit(s.arrivals_tonnes, s.modal_price, 1); xs = np.linspace(s.arrivals_tonnes.min(), s.arrivals_tonnes.max(), 50); ax[i].plot(xs, m_*xs + b_, color="k", lw=1.4)
    ax[i].set_title(f"{c}  (r = {cor[c]['pearson']:.2f})", fontweight="bold", loc="left"); ax[i].set_xlabel("Arrivals (tonnes)")
ax[0].set_ylabel("Modal price (Rs/quintal)"); plt.tight_layout(); plt.savefig("figs/fig_eda_scatter.png", dpi=200); plt.close()
# F5 pooled vs within
fig, ax = plt.subplots(figsize=(6.8, 3.6))
for c in crops:
    s = d[d.commodity == c]; ax.scatter(s.arrivals_tonnes, s.modal_price, s=9, color=CROP_COL[c], alpha=.6, label=c)
m_, b_ = np.polyfit(d.arrivals_tonnes, d.modal_price, 1); xs = np.linspace(d.arrivals_tonnes.min(), d.arrivals_tonnes.max(), 50); ax.plot(xs, m_*xs + b_, color=RED, lw=2, ls="--", label=f"Pooled line (r = {pooled[0]:.2f})")
ax.set_xlabel("Arrivals (tonnes)"); ax.set_ylabel("Modal price (Rs/quintal)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Pooling crops hides the real relationship", fontweight="bold", color=DG, loc="left")
plt.savefig("figs/fig_eda_pooled.png", dpi=200, bbox_inches="tight"); plt.close()
# F6 heatmap
fig, ax = plt.subplots(figsize=(5.2, 4.3)); labs = ["Modal", "Min", "Max", "Arrivals", "Spread"]
im = ax.imshow(cm.values, cmap="RdYlGn", vmin=-1, vmax=1); ax.set_xticks(range(5)); ax.set_xticklabels(labs); ax.set_yticks(range(5)); ax.set_yticklabels(labs)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
for s_ in ax.spines.values(): s_.set_visible(False)
plt.colorbar(im, fraction=.046); ax.set_title("Correlation matrix (scaled within crop)", fontweight="bold", color=DG, loc="left")
plt.savefig("figs/fig_eda_heat.png", dpi=200, bbox_inches="tight"); plt.close()
# F7 anomaly view
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ok = d[d.rz.abs() <= 3.5]; ax.scatter(ok.date, ok.ratio, s=7, color=GREY, alpha=.5, label="Normal days")
ax.scatter(fl.date, fl.ratio, s=40, color=RED, zorder=3, label=f"Flagged (|robust z| > 3.5), n = {len(fl)}")
ax.axhline(1, color="k", lw=.8); ax.set_ylabel("Price / series median"); ax.set_ylim(0.88, 1.6); ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.tick_params(axis="x", rotation=30)
ax.set_title("Anomaly screen: each dot is one market-day", fontweight="bold", color=DG, loc="left")
plt.savefig("figs/fig_eda_anom.png", dpi=200, bbox_inches="tight"); plt.close()
# F8 EDA workflow
fig, ax = plt.subplots(figsize=(10, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.9); ax.axis("off")
steps = [("1 Question", "Business question\nand scope"), ("2 Data check", "Types, gaps, units,\ncleaning summary"), ("3 Univariate", "Distributions,\nsummary statistics"), ("4 Bivariate", "Price vs arrivals,\ncorrelations"),
         ("5 Time", "Trend, seasonality,\nlags"), ("6 Regional", "Market and district\ncomparison"), ("7 Anomalies", "Detect, classify,\ntreat or flag"), ("8 Insight", "Findings and\nrecommendations")]
cols = [GREY, GREY, G, G, G, G, AMB, DG]
for i, ((t, dsc), c) in enumerate(zip(steps, cols)):
    r, k = divmod(i, 4); x = 0.2 + k*2.45; y = 2.05 - r*1.95
    ax.add_patch(FancyBboxPatch((x, y), 2.1, 1.5, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=c, lw=2)); ax.add_patch(FancyBboxPatch((x, y+1.05), 2.1, 0.45, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec=c))
    ax.text(x+1.05, y+1.27, t, ha="center", va="center", color="white", fontweight="bold", fontsize=10); ax.text(x+1.05, y+0.5, dsc, ha="center", va="center", fontsize=9, color="#263238")
    if k < 3: ax.add_patch(FancyArrowPatch((x+2.1, y+.75), (x+2.45, y+.75), arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#455A64"))
ax.plot([8.6, 8.6, 1.25], [2.05, 1.83, 1.83], color="#455A64", lw=1.8, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((1.25, 1.83), (1.25, 1.6), arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#455A64"))
plt.savefig("figs/fig_eda_flow.png", dpi=200, bbox_inches="tight"); plt.close()
print(json.dumps({k: R[k] for k in ["desc","arr","market","market_gap","anova","trend","cor","pooled","anom","dow"]}, indent=1, default=float)); print(cm.round(2))
