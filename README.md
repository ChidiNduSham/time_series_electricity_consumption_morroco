# Electricity Consumption Analysis: Week 5 Data Science Project

Statistical and time-series analysis of 52,416 ten-minute electricity readings from three zones in Morocco (1 January – 30 December 2017), completed for the AnalystLab Africa Experience Lab Internship Programme (Track 2: Data Science).

## Questions answered

- How does consumption change through the day, week and year?
- Are temperature, humidity and wind speed associated with consumption?
- How do the three zones differ in level, variability and pattern?

## Key findings

- **Evening peak:** all three zones peak at 20:00, 34–46% above their own means.
- **Summer lift:** monthly means peak in July–August.
- **Weather:** temperature is the strongest association (Spearman ρ = 0.72 in Zone 1). These are associations, not causes.
- **Zones:** Zone 1 is higher than Zones 2 and 3 on all 364 days. Zone 3 is the most variable (CV 0.37).
- **Zone 3 step-down:** its mean roughly halves after August and does not recover. The cause is unknown and needs verification.

## Methods

Data validation, hourly and daily mean aggregation, descriptive statistics, STL decomposition (period = 7), Spearman correlations with Holm correction, Friedman test with Wilcoxon follow-ups and Kendall's W.

## Repository contents

| File | Purpose |
|---|---|
| `Week5_Data_Science_Analysis.ipynb` | Executed analysis notebook |
| `report_figures.py` | Regenerates report charts and `results.json` |
| `Week5_Data_Science_Report.docx` | Full report |
| `data/powerconsumption.csv` | Source data (add before running) |
| `data/powerconsumption_datadictionary.csv` | Data dictionary (optional; see notes) |

## How to run

```bash
pip install pandas numpy scipy statsmodels matplotlib plotly jupyter
mkdir -p data figs assets
# place powerconsumption.csv in data/
jupyter notebook Week5_Data_Science_Analysis.ipynb
python report_figures.py
```

Tested with Python 3.12, pandas 3.0, NumPy 2.5, SciPy 1.18, statsmodels 0.15, Matplotlib 3.11 and Plotly 7.1.

`report_figures.py` uses the Poppins font and looks for it at a Linux path; edit the font paths at the top of the script on other systems.

## Notes and limitations

- **One code change:** the notebook now loads the data dictionary only if the file exists, and continues with a message if it does not. Nothing else was altered.
- **Units:** the interval basis of the recorded values is not established, so the analysis uses means and never sums readings into total energy.
- **Coverage:** the data cover 364 days of one year, so annual seasonality cannot be established.
- **Diffuse flows:** `GeneralDiffuseFlows` and `DiffuseFlows` are not interpreted because their meaning is not established.
- **Dependence:** daily means are strongly autocorrelated (lag-1 of 0.87–0.99), so p-values are optimistic.
