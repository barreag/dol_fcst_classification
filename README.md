# USD/BRL Exchange Rate Forecasting with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## DISCLAIMER

**THIS PROJECT IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

This machine learning model was developed as a personal data science portfolio project to demonstrate technical skills in:

- Time series forecasting
- Feature engineering
- Ensemble machine learning
- Model optimization and interpretability
- Trading system simulation

**THIS IS NOT FINANCIAL ADVICE.** The predictions, strategies, and results presented in this repository should NOT be used for actual trading or investment decisions. Past performance does not guarantee future results. Forex trading involves substantial risk of loss.

**The author assumes no liability for any financial losses incurred from the use of this code or methodology.**

---

## Project Overview

This project implements a comprehensive machine learning pipeline to predict USD/BRL (Brazilian Real) exchange rate movements. The system uses a **stacking ensemble** approach optimized through Bayesian hyperparameter tuning, with **SHAP-based feature selection** to identify the most predictive technical indicators.

### Key Features

- **Automated Data Acquisition**: Downloads historical USD/BRL data from Yahoo Finance
- **Advanced Feature Engineering**: Creates 17+ technical indicators including moving averages, standard deviations, RSI, and momentum features
- **Model Selection**: Evaluates 40+ classification algorithms using LazyPredict
- **Hyperparameter Optimization**: Uses Optuna with 100 trials for Bayesian optimization
- **Model Interpretability**: SHAP analysis for feature importance and selection
- **Trading System Simulation**: Backtests trading strategies with transaction costs and probability filtering

### Performance Highlights

- **Best Model**: Stacking Ensemble (QuadraticDiscriminantAnalysis + LinearDiscriminantAnalysis + LinearSVC → LogisticRegression)
- **Test Accuracy**: 54.69% (with SHAP-selected features)
- **Features Selected**: 13 out of 17 (76% dimensionality reduction)
- **Optimization**: Improved from 52.86% to 54.69% through feature selection (+1.83%)
- **Trading Strategy**: Filtering uncertainty zone (probabilities 0.5013-0.5148) improves returns

---

## Repository Structure

```
dol_fcst/
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb          # Download USD/BRL data from Yahoo Finance
│   ├── 02_feature_creation.ipynb          # Engineer technical indicators
│   ├── 03_lazyclassifier_evaluation.ipynb # Evaluate 40+ ML models
│   ├── 04_stacking_ensemble.ipynb         # Build stacking ensemble
│   ├── 05_stacking_optuna.ipynb           # Optimize hyperparameters
│   ├── 06_shap_feature_selection.ipynb    # SHAP analysis & feature selection
│   └── 07_trading_system_analysis.ipynb   # Trading system simulation
│
├── data/
│   ├── raw/
│   │   └── BRL_X_raw.csv                  # Raw exchange rate data
│   ├── processed/
│   │   ├── BRL_X_features.csv             # Engineered features dataset
│   │   └── metrics/
│   │       ├── lazyclassifier_results.csv
│   │       ├── top_3_model_names.txt
│   │       ├── best_params_stacking.json
│   │       ├── shap_feature_importance.csv
│   │       ├── shap_selected_features.txt
│   │       └── shap_results.json
│
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── LICENSE                                 # MIT License

```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/dol_fcst.git
cd dol_fcst
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

5. **Run notebooks sequentially** (01 → 07)

---

## Methodology

### 1. Data Acquisition (Notebook 01)

- Downloads historical USD/BRL exchange rates from Yahoo Finance
- Date range: 2010-01-02 to present
- Data includes: Open, High, Low, Close, Volume, Adjusted Close

### 2. Feature Engineering (Notebook 02)

Creates **17 technical indicators**:

- **Moving Averages**: 6-day, 12-day (a, m)
- **Standard Deviations**: 6-day, 12-day (v, M)
- **Moving Max/Min**: 6-day and 12-day ranges (k, w, f, T)
- **Standardized Metrics**: mm_std6, mm_std12, std6, std12
- **RSI**: 6-day and 12-day Relative Strength Index (RSL_6, RSL_12)
- **Momentum**: Price change (g), volume change (tau)
- **Categorical**: Price vs. MA (cat)
- **Target**: Binary classification (1 = price up, 0 = price down)

### 3. Model Selection (Notebook 03)

- Uses **LazyPredict** to evaluate 40+ classifiers
- **Top 3 Models**:
  1. QuadraticDiscriminantAnalysis (55.32%)
  2. LinearDiscriminantAnalysis (55.32%)
  3. LinearSVC (54.53%)

### 4. Stacking Ensemble (Notebook 04)

- Combines top 3 models as base learners
- Tests 4 meta-learners: LogisticRegression, RandomForest, GradientBoosting, XGBoost
- **Best Meta-Learner**: LogisticRegression (53.90%)

### 5. Hyperparameter Optimization (Notebook 05)

- **Optuna** Bayesian optimization with 100 trials
- Optimizes all base learners and meta-learner simultaneously
- **Result**: 53.84% accuracy (optimized parameters saved)

### 6. SHAP Feature Selection (Notebook 06)

- Uses **SHAP KernelExplainer** for model-agnostic feature importance
- **Recursive feature addition**: Tests 1 to 17 features
- **Optimal**: 13 features → **54.69% accuracy** (+1.83% improvement)
- **Top Features**: a, v, cat, g, m, k, mm_std6, std12, M, mm_std12, std6, w, f

### 7. Trading System Analysis (Notebook 07)

- Simulates trading with transaction costs (0.025%)
- **Probability Binning**: Analyzes performance by 10 probability bins
- **Key Finding**: Bins 4-6 (0.5013-0.5148) are uncertainty zones with poor performance
- **Optimized Strategy**: Exclude bins 4-6 → Improved returns
- Visualizes cumulative results and probability distributions

---

## Results Summary

| Metric                        | Value                                                                |
| ----------------------------- | -------------------------------------------------------------------- |
| **Dataset Size**        | 4,103 samples (80% train, 20% test)                                  |
| **Date Range**          | 2010-01-04 to 2025-10-25 (15+ years)                                 |
| **Features (Original)** | 17 technical indicators                                              |
| **Features (Selected)** | 13 (SHAP-based selection)                                            |
| **Base Learners**       | QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis, LinearSVC |
| **Meta-Learner**        | LogisticRegression (C=0.622, solver='lbfgs')                         |
| **Test Accuracy**       | 54.69%                                                               |
| **Optimization Trials** | 100 (Optuna Bayesian)                                                |
| **SHAP Samples**        | 100                                                                  |

### Trading Performance by Probability Bin (Test Set)

| Bin | Probability Range | Cumulative Return | Avg Return | Trade? |
| --- | ----------------- | ----------------- | ---------- | ------ |
| 1   | 0.4453 - 0.4890   | +28.98            | +0.3491    | YES    |
| 2   | 0.4890 - 0.4965   | +7.98             | +0.0973    | YES    |
| 3   | 0.4965 - 0.5013   | +5.98             | +0.0729    | YES    |
| 4   | 0.5013 - 0.5065   | -4.02             | -0.0490    | NO     |
| 5   | 0.5065 - 0.5110   | +3.98             | +0.0485    | NO     |
| 6   | 0.5110 - 0.5148   | -24.02            | -0.2929    | NO     |
| 7   | 0.5148 - 0.5194   | +17.98            | +0.2193    | YES    |
| 8   | 0.5194 - 0.5239   | -8.02             | -0.0978    | WEAK   |
| 9   | 0.5239 - 0.5307   | +11.98            | +0.1461    | YES    |
| 10  | 0.5307 - 0.5895   | +7.98             | +0.0973    | YES    |

**Key Insight**: The model performs poorly in the uncertainty zone (bins 4-6) near 0.5 probability. Filtering these trades significantly improves overall performance.

---

## Technical Stack

- **Language**: Python 3.8+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, lightgbm, catboost
- **AutoML**: lazypredict
- **Optimization**: optuna
- **Interpretability**: shap
- **Data Source**: yfinance

---

## Usage Notes

### Running Individual Notebooks

Each notebook is designed to:

1. Load results from previous notebooks (when applicable)
2. Perform its specific analysis or training step
3. Save outputs for subsequent notebooks
4. Display comprehensive comments and explanations

### Key Configuration Parameters

```python
# Data Split
TEST_SIZE = 0.2
RANDOM_STATE = 42
SHUFFLE = False  # Preserves temporal order for time series

# Optimization
N_TRIALS = 100  # Optuna trials
CV_FOLDS = 5    # Cross-validation folds

# SHAP Analysis
N_SHAP_SAMPLES = 100  # Background samples for KernelExplainer

# Trading System
TRANSACTION_COST = 0.00025  # 0.025% per trade
PROBABILITY_BINS = 10       # For performance analysis
```

---

## Future Improvements

- [ ] Add more sophisticated features (e.g., MACD, Bollinger Bands, Ichimoku)
- [ ] Implement deep learning models (LSTM, GRU, Transformers)
- [ ] Add macroeconomic indicators (interest rates, inflation, GDP)
- [ ] Multi-timeframe analysis (daily, weekly, monthly)
- [ ] Walk-forward validation for more robust backtesting
- [ ] Real-time prediction pipeline
- [ ] Risk management metrics (Sharpe ratio, max drawdown, VaR)
- [ ] Multi-currency prediction (EUR/USD, GBP/USD, etc.)

---

## References

- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"
- **Stacking**: Wolpert, D. H. (1992). "Stacked Generalization"
- **Optuna**: Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework"
- **LazyPredict**: https://github.com/shankarpandala/lazypredict

---

## Author

**Augusto G. Barreiros**

- LinkedIn: https://www.linkedin.com/in/augusto-barreiros/
- Email: barreiros.augusto@gmail.com

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Yahoo Finance for providing free financial data through yfinance
- The open-source data science community for excellent libraries
- scikit-learn, Optuna, and SHAP teams for their outstanding tools

---

## Final Reminder

**THIS IS AN EDUCATIONAL PROJECT. DO NOT USE FOR ACTUAL TRADING.**

If you use this code or methodology in your research or projects, please provide appropriate attribution.

---

**Star this repository if you find it helpful for learning about machine learning in finance!**
