import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
G, DG, AMB, RED, GREY, BLUE = "#2E7D32", "#1B5E20", "#F9A825", "#C62828", "#607D8B", "#1565C0"
R = json.load(open("data/yield_results.json"))

# ---- F1 workflow (3x3 snake)
fig, ax = plt.subplots(figsize=(10, 6.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
boxes = [("1  Define problem", "Target, horizon, decision,\nforecast date", GREY), ("2  Prepare data", "Collect, clean, align\nby district and year", GREY),
         ("3  Engineer features", "Lags, anomalies, stress\nindicators (train-only fit)", G),
         ("4  Split by time", "Develop 2008-2020, test\n2021-2024 held out", G), ("5  Baselines", "Last year, district mean:\nthe bar to beat", G), ("6  Train and tune", "Ridge, random forest,\nboosting; walk-forward CV", G),
         ("7  Select and calibrate", "Best CV model; interval\nwidth from CV residuals", AMB), ("8  Test once", "MAE, RMSE, MAPE, R2, bias;\ndrought-year check", AMB), ("9  Deploy and monitor", "Refresh yearly; track error\nand drift; retrain", DG)]
pos = []
for i, (t, dsc, c) in enumerate(boxes):
    r, k = divmod(i, 3); x = 0.35 + (k if r % 2 == 0 else 2 - k)*3.2; y = 4.85 - r*2.15
    ax.add_patch(FancyBboxPatch((x, y), 2.9, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=c, lw=2.2)); ax.add_patch(FancyBboxPatch((x, y+1.05), 2.9, 0.45, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec=c))
    ax.text(x+1.45, y+1.27, t, ha="center", va="center", color="white", fontweight="bold", fontsize=10.5); ax.text(x+1.45, y+0.48, dsc, ha="center", va="center", fontsize=9, color="#263238"); pos.append((x, y))
def arrow(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=2, color="#455A64"))
for i in range(8):
    (x1, y1), (x2, y2) = pos[i], pos[i+1]
    if y1 == y2: arrow((x1+2.9, y1+0.75) if x2 > x1 else (x1, y1+0.75), (x2, y2+0.75) if x2 > x1 else (x2+2.9, y2+0.75))
    else: arrow((x1+1.45, y1), (x2+1.45, y2+1.5))
ax.text(5, 6.45, "Predictive modelling framework: from question to monitored forecast", ha="center", fontsize=12.5, fontweight="bold", color=DG)
ax.text(5, 0.05, "Grey = frame the problem   Green = build   Amber = judge and verify   Dark green = use", ha="center", fontsize=9, color="#455A64")
plt.savefig("figs/fig_w4_workflow.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- F2 split timeline
fig, ax = plt.subplots(figsize=(8.4, 3.9))
folds = list(range(2013, 2021))
for i, fy in enumerate(folds):
    y = len(folds) - i
    ax.barh(y, fy - 2008, left=2008, height=0.6, color="#90CAF9"); ax.barh(y, 1, left=fy, height=0.6, color=AMB)
    ax.text(2007.7, y, f"Fold {i+1}", ha="right", va="center", fontsize=8)
ax.barh(0, 4, left=2021, height=0.6, color=RED); ax.barh(0, 13, left=2008, height=0.6, color="#B0BEC5"); ax.text(2007.7, 0, "Final", ha="right", va="center", fontsize=8)
ax.text(2014.5, 0, "refit on 2008-2020", ha="center", va="center", fontsize=8, color="k"); ax.text(2023, 0, "TEST", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_xlim(2007.9, 2025); ax.set_xticks(range(2008, 2025, 2)); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.legend(handles=[Rectangle((0, 0), 1, 1, color="#90CAF9"), Rectangle((0, 0, ), 1, 1, color=AMB), Rectangle((0, 0), 1, 1, color=RED)], labels=["Train (all earlier years)", "Validate (next year)", "Test (held out once)"], frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
ax.set_title("Time-aware splitting: expanding-window walk-forward validation", fontweight="bold", color=DG, loc="left", fontsize=10.5)
plt.savefig("figs/fig_w4_split.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- F3 model comparison (test RMSE), two crops
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=False); names = ["Naive: last year", "District mean", "Ridge", "Random forest", "Gradient boosting"]; cols = [GREY, GREY, G, G, G]
for ax, crop in zip(axs, ["Soybean", "Wheat"]):
    m = R[crop]["In-season"]["models"]; vals = [m[n]["test"]["RMSE"] for n in names]; b = ax.bar(range(5), vals, color=cols)
    b[names.index(R[crop]["In-season"]["selected"])].set_edgecolor("k"); b[names.index(R[crop]["In-season"]["selected"])].set_linewidth(2)
    for i, v in enumerate(vals): ax.text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["Naive", "District\nmean", "Ridge", "Random\nforest", "Gradient\nboosting"], fontsize=8); ax.set_title(f"{crop} (t/ha)", fontweight="bold", loc="left")
axs[0].set_ylabel("Test RMSE (lower is better)"); plt.tight_layout(); plt.savefig("figs/fig_w4_models.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- F4 lead time (feature sets)
fig, ax = plt.subplots(figsize=(7.6, 3.6)); sets = ["Pre-season", "In-season", "In-season + NDVI"]; w = 0.36
for j, (crop, col) in enumerate([("Soybean", G), ("Wheat", BLUE)]):
    v = [R[crop][s]["models"][R[crop][s]["selected"]]["test"]["RMSE"] for s in sets]; b = ax.bar(np.arange(3) + (j - .5)*w, v, w, color=col, label=f"{crop} (selected model)")
    for i, x in enumerate(v): ax.text(i + (j - .5)*w, x + 0.004, f"{x:.2f}", ha="center", fontsize=8)
    ax.axhline(R[crop]["Pre-season"]["models"]["Naive: last year"]["test"]["RMSE"], color=col, ls=":", lw=1.3)
ax.set_xticks(range(3)); ax.set_xticklabels(["Pre-season\n(before sowing)", "In-season\n(weather to harvest)", "In-season + NDVI\n(satellite greenness)"]); ax.set_ylabel("Test RMSE (t/ha)"); ax.set_ylim(0, 0.37); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
ax.set_title("Later forecasts are more accurate (dotted = naive baseline)", fontweight="bold", color=DG, loc="left", fontsize=10.5)
plt.savefig("figs/fig_w4_leadtime.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- F5 predicted vs actual
P = R["Soybean"]["In-season"]["pred"]; a = np.array(P["actual"]); p = np.array(P["pred"]); dr = np.array(P["drought"]); mt = R["Soybean"]["In-season"]["models"][R["Soybean"]["In-season"]["selected"]]["test"]
fig, ax = plt.subplots(figsize=(4.8, 4.4)); ax.scatter(a[~dr], p[~dr], s=18, color=G, label="Normal years", alpha=.8); ax.scatter(a[dr], p[dr], s=22, color=RED, marker="^", label="Drought year (2022)")
lo, hi = min(a.min(), p.min()) - .05, max(a.max(), p.max()) + .05; ax.plot([lo, hi], [lo, hi], color="k", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Actual yield (t/ha)"); ax.set_ylabel("Predicted yield (t/ha)"); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.text(hi - .02, lo + .04, f"R2 = {mt['R2']:.2f}\nMAE = {mt['MAE']:.2f} t/ha", ha="right", fontsize=9)
ax.set_title("Soybean test years 2021-2024", fontweight="bold", color=DG, loc="left", fontsize=10.5); plt.savefig("figs/fig_w4_pred.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- F6 leakage demo
L = R["leak"]; fig, ax = plt.subplots(figsize=(5.4, 3.3)); b = ax.bar(["Random 5-fold\n(wrong for forecasting)", "Time-ordered split\n(correct)"], [L["random_R2"], L["time_R2"]], color=[RED, G], width=.55)
for bar, v in zip(b, [L["random_R2"], L["time_R2"]]): ax.text(bar.get_x() + bar.get_width()/2, v + .012, f"R2 = {v:.2f}", ha="center", fontsize=9)
ax.set_ylim(0, 1); ax.set_ylabel("R2 on unseen data"); ax.set_title("Random splitting makes the model look better than it is", fontweight="bold", color=DG, loc="left", fontsize=9.5)
plt.savefig("figs/fig_w4_leak.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- F7 importance
imp = R["Soybean"]["In-season"]["importance"]; top = list(imp.items())[:8][::-1]; fig, ax = plt.subplots(figsize=(6.2, 3.4)); ax.barh([k for k, _ in top], [v for _, v in top], color=G)
ax.set_xlabel("Increase in RMSE when the feature is shuffled (t/ha)"); ax.set_title("Permutation importance (soybean, test years)", fontweight="bold", color=DG, loc="left", fontsize=10.5)
plt.savefig("figs/fig_w4_imp.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- F8 interval for one district
df = pd.DataFrame(P); q = R["Soybean"]["In-season"]["interval_halfwidth"]; fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True)
for ax, dname in zip(axs, ["Latur", "Pune"]):
    s = df[df.district == dname].sort_values("year"); ax.fill_between(s.year, s.pred - q, s.pred + q, color="#A5D6A7", alpha=.7, label="90% interval"); ax.plot(s.year, s.pred, color=G, marker="o", label="Forecast"); ax.plot(s.year, s.actual, color="k", marker="s", ls="--", label="Actual")
    ax.set_title(dname, fontweight="bold", loc="left"); ax.set_xticks(s.year); ax.set_xlabel("Year")
axs[0].set_ylabel("Soybean yield (t/ha)"); axs[0].legend(frameon=False, fontsize=8, loc="lower left"); plt.tight_layout(); plt.savefig("figs/fig_w4_interval.png", dpi=200, bbox_inches="tight"); plt.close()
print("figs ok")
