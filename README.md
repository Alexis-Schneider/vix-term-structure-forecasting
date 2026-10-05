# 📉 The Premium, Not the Forecast: Forecasting S&P 500 Volatility Term Structure & Trading the VIX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Executive Summary:** This repository contains the code and the full research report for my Master's thesis / Internship project conducted at Politecnico di Milano (L2C Lab). The project investigates whether the statistical predictability of the S&P 500 implied volatility curve can be economically exploited through VIX futures. 
> 
> **Spoiler:** It cannot. By developing a rigorous "Nested Placebo" evaluation framework, this research proves that the strategy's returns come entirely from passively harvesting the volatility risk premium, not from the model's forecasting ability.

📄 **[Read the full 30-page Research Report here (PDF)](Report/Alexis_Schneider_Internship_Report.pdf)**
📊 **[View the Presentation Slides here (PDF)](Report/Alexis_Schneider_Presentation.pdf)**

---

## 💡 Key Contributions

1. **End-to-End Pipeline:** Reconstructed the S&P 500 implied volatility surface from raw options chain data (2010–2023), fitting a daily dynamic **Nelson-Siegel** model.
2. **Methodological Rigor ("Nested Placebo"):** Designed an economic evaluation protocol that isolates the exact contribution of the predictive model. The forecasting model is compared to a "Placebo" (a random walk) that passes through the exact same economic pipeline (same threshold, risk targeting, transaction costs, and roll drag).
3. **Market Efficiency Finding:** Demonstrated that while the term structure exhibits statistical predictability (especially on the curvature factor), this edge is economically inert. The market's own forward quote (the carry) spans the forecast.

## 🛠️ Data and Methodology

*   **Data:** Raw end-of-day S&P 500 options chains (OptionsDX) and CBOE VIX futures settlements (2010–2023). *(Note: Raw data files are not included in this repo due to size and licensing, but the processing logic is fully available).*
*   **Curve Modeling:** Dynamic Nelson-Siegel decomposition (Level, Slope, Curvature) estimated via ridge-regularized least squares.
*   **Forecasting Models:** Walk-forward estimation of ARMA, HAR (Heterogeneous Autoregressive), VAR(2), and Ridge Regression.
*   **Trading Strategy:** Constant-maturity risk-targeted position on CBOE VIX futures.
*   **Statistical Tests:** Clark-West test for nested models, Model Confidence Set (MCS), and Moving Block Bootstrap for Sharpe ratio differentials.

## 📊 Core Findings

*   **Statistical Predictability:** Forecasting models beat the random walk by 13% to 14% at a 30-day horizon (Out-of-Sample). However, most of this edge is concentrated on the *curvature* factor, which is heavily noisy and economically too small to move traded prices.
*   **The Reality of Roll Drag:** Holding a constant 30-day maturity VIX futures position costs heavily when the curve is in contango, resulting in a structural drag of -47.4% on the test window.
*   **Economic Evaluation:** The predictive strategy yields a Sharpe ratio (0.29 - 0.35) indistinguishable from zero. More importantly, the **Placebo strategy (random walk) performs slightly better** (Sharpe 0.44 - 0.46). 
*   **Conclusion:** The model contributes nothing measurable. The strategy earns returns not by timing the market, but simply by holding a structurally short volatility book (harvesting the variance risk premium). 

## 📂 Repository Structure

```text
├── Report/
│   ├── Alexis_Schneider_Internship_Report.pdf   # Full academic report
│   └── Alexis_Schneider_Presentation.pdf        # Defense slides
├── src/
│   ├── ts_extract.py    # Option chain parsing, Black-Scholes inversion, and Nelson-Siegel curve fitting
│   ├── ts_models.py     # Walk-forward forecasting models (ARMA, HAR, VAR, Ridge) & MCS evaluation
│   └── vix_lab.py       # Trading simulator, Nested Placebo framework, and Sharpe ratio inference
├── README.md
└── LICENSE              # MIT License
