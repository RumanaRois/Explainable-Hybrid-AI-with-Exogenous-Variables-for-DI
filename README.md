# Explainable Hybrid AI for Three-hourly Discomfort Index Forecasting in Dhaka

Code for the manuscript **"Explainable Hybrid AI with Exogenous Variables for Three-hourly Discomfort Index Forecasting in a Dense Tropical City"** by Rawfun Nahin Kowbi, Mohammad Mahboob Hussain Khan and Rumana Rois (Department of Statistics and Data Science, Jahangirnagar University; Bangladesh Meteorological Department).

## Overview

Dhaka is one of the most heat-exposed cities in the world. This repository forecasts Thom's Discomfort Index (DI) three hours ahead from 34,576 three-hourly observations (January 2014 – October 2025). It uses a two-stage hybrid framework and SHAP to explain what drives each forecast.

**Discomfort Index (Thom, 1959)**

DI = T − 0.55 (1 − 0.01 RH) (T − 14.5), where T is air temperature (°C) and RH is relative humidity (%).

## Framework

| Stage | Inputs | Models |
|---|---|---|
| **1. Primary learner** | Exogenous variables (rainfall, cloud cover, pressure, wind), DI lags (3, 9, 18 h), calendar terms | DTX, RFX, XGBX, SVRX, FPX (Prophet), ANNX, LSTMX, GRUX, SARIMAX |
| **2. Residual learner** | Only the error term from stage 1 | ETS, SARIMA, TBATS (ML–TS hybrids, e.g. **GRUX-SARIMA**), or an ML/DL learner on lagged SARIMAX residuals (TS–ML hybrids, e.g. **SARIMAX-GRU**) |

The suffix **X** marks the stage that receives the exogenous variables. Every model is run on the raw DI series and on an STL-adjusted series, giving 9 single models + 32 hybrids = 41 per track and **82 configurations** in total. These are compared against persistence and seasonal-naive forecasts.

## Evaluation

- Chronological split: fitting (2014–Jul 2021), calibration (Jul 2021–Jun 2023) and an independent test period (Jun 2023–Oct 2025, 6,916 forecasts)
- MSE, RMSE, MAE, MAPE, MASE and R²
- Diebold–Mariano tests with a Newey–West variance
- Skill on the six Thom discomfort classes, including detection of DI ≥ 29 °C
- Rolling conformal 90% and 95% prediction intervals
- An operational check (scenario B) in which the exogenous inputs are lagged by 3 h
- Explanations: exact TreeSHAP for XGBX and expected-gradients SHAP for GRUX

## Main results

| | Value |
|---|---|
| Best model | **GRUX-SARIMA** (RMSE 0.590 °C, R² 0.975) |
| MSE reduction vs persistence | 78% |
| Detection of severe discomfort (DI ≥ 29 °C) | 70% (false alarm ratio 0.22) |
| 95% interval coverage | 94.8% |
| STL seasonal adjustment | Significantly worse in all 41 pairings |
| Trend in annual mean DI (2014–2024) | +0.69 °C per decade (p = 0.010) |
| SHAP attribution | Past DI 64%, calendar 25%, exogenous weather 11% |

## Repository contents

```
DI_XAI_analysis.py   all analysis code in one script
requirements.txt     Python dependencies
CITATION.cff         citation metadata
LICENSE              MIT licence
```

## Data

The BMD three-hourly observations are **not included** in this repository. They are available from the Bangladesh Meteorological Department on reasonable request. The script expects an Excel file named `DI_data.xlsx` with these columns:

`DateTime, Season, Temperature, Humidity, DI, Rainfall, Atmosphere, Cloud, Wind`

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/RumanaRois/DI-XAI-Dhaka.git
cd DI-XAI-Dhaka
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Place `DI_data.xlsx` in the repository folder, then run the steps in this order:

```bash
python DI_XAI_analysis.py eda           # trend tests (Mann–Kendall, Sen's slope) and observed-DI figures
python DI_XAI_analysis.py framework     # workflow diagram
python DI_XAI_analysis.py all-models    # all 82 configurations, benchmarks and scenario B
python DI_XAI_analysis.py analyse       # metrics, Diebold–Mariano tests, category skill, conformal intervals
python DI_XAI_analysis.py xai           # TreeSHAP for XGBX
python DI_XAI_analysis.py xai-gru       # expected-gradients SHAP for GRUX
python DI_XAI_analysis.py values        # every number reported in the manuscript -> out/values.json
```

To fit one model group only, run `python DI_XAI_analysis.py model <raw|stl> <A|B> <group>`, for example:

```bash
python DI_XAI_analysis.py model raw A GRU
```

The groups are `naive SARIMAX DT RF XGB SVR ANN LSTM GRU FP SX-DT SX-RF SX-XGB SX-SVR SX-ANN SX-LSTM SX-GRU SX-FP`. Run `SARIMAX` before the `SX-*` groups of the same track.

You can change the input and output locations with `--data`, `--out` and `--fig`.

## Outputs

- `out/`: test-period predictions, metric tables (CSV), Diebold–Mariano tests, category skill, SHAP values and `values.json`
- `fig/`: 300 dpi figures (observed DI, workflow, forecast skill, SHAP summary, dependence and waterfall plots)

**Run time:** `all-models` takes several hours on a 2-core CPU. Most of that time goes to the SVR and TBATS models. Random seeds are fixed at 42.

## Citation

If you use this code, please cite the article (details will be added on publication) and this repository. Use the **"Cite this repository"** button on GitHub or the metadata in `CITATION.cff`.

## Licence

Released under the MIT licence. See `LICENSE`.

## Contact

- Rumana Rois: rois@juniv.edu
- Rawfun Nahin Kowbi: rawfunkouvi7@gmail.com
