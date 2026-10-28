# Week 4: Predictive modeling framework for crop yield (hypothetical data)

The data are SYNTHETIC. Relationships were written into make_yield_data.py (seed 7), so results illustrate the method, not real farms.

    pip install -r requirements.txt
    python make_yield_data.py   # builds data/synthetic_yield_data.csv (15 districts x 20 years x 2 crops)
    python yield_model.py       # full framework: splits, features, baselines, tuning, test, reliability -> data/yield_results.json
    python make_w4_figs.py      # charts -> figs/

Framework rules implemented in yield_model.py:
- Development years 2008-2020 (8 walk-forward folds); test years 2021-2024 scored once.
- Feature statistics (district means, anomalies) are fitted on training years only.
- Baselines: last year's yield, district mean. Models: Ridge, random forest, gradient boosting.
- Model chosen by walk-forward CV only; 90% interval width from out-of-fold errors (split-conformal style).
- Reliability: leakage demo (random vs time split), drought-year subgroup, interval coverage, noisy-weather test, permutation importance.
- The same run_pipeline() function is reused for soybean and wheat; only the heat threshold differs.
