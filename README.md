# Week 3: EDA on the cleaned hypothetical mandi dataset

The data are SYNTHETIC (generated and cleaned in Week 2). eda.py reads data/clean_mandi_prices.csv and writes
every number (data/eda.json) and chart (figs/) used in the Week 3 document.

    pip install -r requirements.txt
    python eda.py

Order of analysis in eda.py: descriptive statistics, regional comparison (price index + ANOVA), weekly trend,
correlations (Pearson, Spearman, pooled vs within-crop), anomaly screen (robust z-score), weekday check, charts.
