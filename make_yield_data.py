"""Hypothetical (synthetic) district-year crop yield dataset for a Week 4 modelling framework demo.
15 Maharashtra districts x 20 years x 2 crops. Drivers and drought years are built in by design; seed fixed."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
districts = ["Pune","Nashik","Latur","Nagpur","Amravati","Akola","Jalna","Beed","Solapur","Satara","Kolhapur","Yavatmal","Wardha","Parbhani","Nanded"]
years = list(range(2005, 2025))
year_rain = {y: rng.lognormal(0, 0.10) for y in years}
for y, f in {2009: .72, 2014: .70, 2015: .66, 2018: .74, 2022: .70, 2020: 1.30}.items(): year_rain[y] = f      # drought and very wet years
year_heat = {y: rng.normal(0, 1.1) + 0.03*(y-2005) for y in years}                                          # shared warming + anomaly
drought_years = [2009, 2014, 2015, 2018, 2022]

dist = pd.DataFrame({"district": districts, "rain_base": rng.uniform(620, 1100, 15), "soil_index": rng.uniform(0.40, 0.90, 15),
                     "base_soy": rng.uniform(0.80, 1.20, 15), "base_wheat": rng.uniform(2.1, 2.9, 15),
                     "irr_soy": rng.uniform(4, 30, 15), "irr_wheat": rng.uniform(40, 85, 15), "heat_base": rng.uniform(-0.8, 1.0, 15)})
rows = []
for crop in ["Soybean", "Wheat"]:
    for _, d in dist.iterrows():
        for y in years:
            t = y - 2005
            rain = d.rain_base*year_rain[y]*rng.uniform(0.92, 1.08) if crop == "Soybean" else rng.uniform(25, 120)*(0.6+0.4*year_rain[y])
            dry = float(np.clip(24 - 0.012*(d.rain_base*year_rain[y]-600) + rng.normal(0, 3), 3, 42))
            tmax = (33.0 if crop == "Soybean" else 31.5) + d.heat_base + year_heat[y] + rng.normal(0, 0.5)
            irr = float(np.clip((d.irr_soy if crop == "Soybean" else d.irr_wheat) + 0.4*t + rng.normal(0, 1.5), 1, 98))
            fert = float(np.clip((55 if crop == "Soybean" else 95) + 2.2*t + rng.normal(0, 8), 20, 200))
            area = float(rng.uniform(40, 220)*1000)
            price_lag = (3200 if crop == "Soybean" else 1500)*(1 + 0.055*t) * rng.uniform(0.93, 1.07)
            if crop == "Soybean":
                y_t = (d.base_soy + 0.011*t + 0.9*(d.soil_index-0.65) + 0.0011*(min(rain, 950)-750) - 0.0016*max(rain-1250, 0)
                       - 0.013*max(dry-10, 0) - 0.06*max(tmax-33, 0) + 0.0035*(fert-70) + 0.010*(irr-15))
                y_t = y_t + rng.normal(0, 0.07)
            else:
                y_t = (d.base_wheat + 0.018*t + 1.0*(d.soil_index-0.65) + 0.0035*(rain-70) + 0.011*(irr-60)
                       - 0.16*max(tmax-31.5, 0) + 0.0045*(fert-120)) + rng.normal(0, 0.12)
            y_t = float(max(y_t, 0.3))
            ndvi = float(np.clip(0.30 + 0.20*(y_t/(1.5 if crop == "Soybean" else 3.2)) + rng.normal(0, 0.035), 0.2, 0.9))
            rows.append(dict(crop=crop, district=d.district, year=y, yield_t_ha=y_t, rain_season_mm=rain, dry_spell_days=dry, tmax_critical=tmax,
                             irrigation_pct=irr, fert_kg_ha=fert, soil_index=d.soil_index, area_ha=area, price_lag=price_lag, ndvi_peak=ndvi))
df = pd.DataFrame(rows).sort_values(["crop", "district", "year"]).reset_index(drop=True)
g = df.groupby(["crop", "district"]).yield_t_ha
df["prev_yield"] = g.shift(1); df["roll3_yield"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
df["drought_year"] = df.year.isin(drought_years)
df.round(3).to_csv("data/synthetic_yield_data.csv", index=False)
print(df.shape); print(df.groupby("crop").yield_t_ha.describe().round(2))
print(df.groupby(["crop", "drought_year"]).yield_t_ha.mean().round(2))
