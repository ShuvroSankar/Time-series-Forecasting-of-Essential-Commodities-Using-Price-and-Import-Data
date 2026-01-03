# Time-Series Forecasting of Essential Commodities (Rice, Onion, Oil) Using Past Price and Import/Export Data

## Project Overview

This project develops and evaluates multivariate LSTM, univariate LSTM, and ARIMA models to forecast wholesale prices of three essential food commodities in Bangladesh: **rice**, **onion**, and **edible oil**. The study integrates historical price data with production and import/export volumes to assess whether external trade data improve forecasting accuracy, and identifies commodity-specific factors that determine model selection.

### Key Research Questions

1. **RQ1**: Do multivariate LSTM models (price + production + imports) outperform univariate (price-only) LSTM and ARIMA baselines?
2. **RQ2**: Does the value of trade data differ across commodity types (staple vs. volatile vs. import-dependent)?
3. **RQ3**: Are forecasts more accurate during normal periods or crisis periods, and which models are most robust?

### Main Findings

- **Onion (volatile perishable)**: Univariate LSTM is best (RMSE 15,897 BDT/ton, 80% directional accuracy). Multivariate LSTM performs 97.6% worse because annual trade data are too coarse to capture crisis-driven price shocks.
- **Rice (regulated staple)**: Multivariate LSTM is best (RMSE 5,189 BDT/ton, 15.2% improvement over univariate). Trade data align with policy-driven price evolution.
- **Oil (import-dependent)**: Univariate LSTM beats ARIMA by 33%, but multivariate analysis is infeasible due to sparse import data (6 observations over 39 years).
- **Counterfactual Impact**: India's 2019–2022 onion export ban created a 116% price premium, costing 54.7 billion BDT (~$456 million USD) in consumer welfare loss.

---

## Repository Structure

```
├── README.md                          # This file
├── Integration.ipynb                  # Data cleaning, merging, and preprocessing
├── LSTM.ipynb                         # Model training, evaluation, and analysis
├── Data/
│   ├── production_data.csv            # FAOSTAT production data (1972–2024)
│   ├── producer_prices_data.csv       # FAOSTAT producer prices (1991–2024)
│   ├── import_export_data.csv         # FAOSTAT trade data
│   ├── agristat_data_bd.xlsx          # Bangladesh national agriculture statistics
│   ├── rice_national_annual_panel.csv # Cleaned rice dataset (output from Integration.ipynb)
│   ├── onion_national_annual_panel.csv# Cleaned onion dataset (output from Integration.ipynb)
│   └── oil_national_annual_panel.csv  # Cleaned oil dataset (output from Integration.ipynb)
├── figures/
│   ├── performance_metrics_comparison.png    # Bar plots: RMSE, MAE, MAPE, DA
│   ├── onion_test_predictions.png           # Actual vs predicted for onion
│   ├── rice_test_predictions.png            # Actual vs predicted for rice
│   └── counterfactual_analysis.png          # India export ban welfare loss
├── paper/
│   ├── main_paper.tex                 # Full LNCS-format manuscript
│   └── references.bib                 # Bibliography
└── requirements.txt                   # Python dependencies
```

---

## How to Run This Project

### 1. Environment Setup

**Install Python 3.8+** and required packages:

```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `pandas`, `numpy` – data manipulation
- `tensorflow`, `keras` – LSTM model building
- `scikit-learn` – evaluation metrics, scaling
- `statsmodels` – ARIMA modeling
- `matplotlib`, `seaborn` – visualization
- `scipy` – statistical tests

### 2. Data Preparation (Integration.ipynb)

Run the **Integration.ipynb** notebook to:
- Download or load FAOSTAT production, price, and trade data
- Download Bangladesh national agriculture statistics
- Standardize commodity naming across sources
- Handle missing values via interpolation and forward-fill
- Construct hybrid import series (FAOSTAT + national data)
- Perform min-max scaling (0,1 range) for all features
- Output cleaned datasets: `rice_national_annual_panel.csv`, `onion_national_annual_panel.csv`, `oil_national_annual_panel.csv`

**Expected Output**:
- Rice: 32 rows (1991–2022) with columns [year, production, imports, price]
- Onion: 34 rows (1991–2024) with columns [year, production, imports, price]
- Oil: 20 rows (2005–2024) with columns [year, price] (import data too sparse)

**Run time**: ~5–10 minutes

### 3. Model Training and Evaluation (LSTM.ipynb)

Run the **LSTM.ipynb** notebook to:

#### **Phase 1: Univariate and Multivariate LSTM Training** (for each commodity)

- Create lagged sequences (window = 3 timesteps)
- Split data chronologically: Train (70%, 1991–2012), Val (10%, 2013–2015), Test (20%, 2016–2022)
- Train Univariate LSTM: price history only
- Train Multivariate LSTM: price + production + imports
- Train ARIMA baseline: automated order selection via AIC
- Evaluate on test set using RMSE, MAE, MAPE, and directional accuracy (DA)

#### **Phase 2: Crisis Detection and Evaluation**

- Define crisis periods using commodity-specific thresholds:
  - **Onion**: price > 30,000 BDT/ton OR year-over-year change > 50%
  - **Rice**: price > 35,000 BDT/ton OR year-over-year change > 30%
- Evaluate models separately on normal vs. crisis test periods
- Compare performance degradation between regimes

#### **Phase 3: Counterfactual Analysis** (Onion only)

- Fit a 2nd-order polynomial to pre-crisis onion prices (1991–2018)
- Extrapolate to 2019–2022 to estimate "no ban" prices
- Calculate welfare loss: (actual price − counterfactual price) × 500,000 tons/year × 120 BDT/USD
- Result: 54.7 billion BDT total loss (2019–2022)

**Expected Output**:
- Summary tables for each commodity (RMSE, MAE, MAPE, DA)
- Bar and line plots saved as PNG files
- Crisis period classification and performance comparison
- Welfare loss estimate for onion

**Run time**: ~15–30 minutes (depending on hardware and epoch count)

---

## Detailed Notebook Descriptions

### Integration.ipynb

**Purpose**: Merge multiple data sources (FAOSTAT, Bangladesh national statistics) into unified, model-ready datasets.

**Key Sections**:
1. **Commodity Mapping**: Define rice, onion, and oil variants across sources
2. **FAOSTAT Extraction**: Query production, prices, and trade data for Bangladesh (1972–2024)
3. **National Data Integration**: Load Bangladesh agriculture statistics and trade data
4. **Missing Data Handling**: 
   - Forward-fill for production (if sparse)
   - Interpolate imports using anchor points from both FAOSTAT and national sources
   - Output "hybrid" import series with best available coverage
5. **Scaling**: Min-Max normalize [0,1] using training-set statistics
6. **Output**: Three cleaned CSVs ready for LSTM training

**Input Files** (required):
- `production_data.csv` (from FAOSTAT)
- `producer_prices_data.csv` (from FAOSTAT)
- `import_export_data.csv` (from FAOSTAT)
- `agristat_data_bd.xlsx` (Bangladesh national statistics)

**Output Files** (generated):
- `rice_national_annual_panel.csv`
- `onion_national_annual_panel.csv`
- `oil_national_annual_panel.csv`

### LSTM.ipynb

**Purpose**: Train and evaluate forecasting models for each commodity.

**Key Sections**:

#### **For Rice and Onion** (identical structure):

1. **Data Loading & Splitting**:
   - Load cleaned CSV (e.g., `onion_national_annual_panel.csv`)
   - Create target variable (next-period price via `.shift(-1)`)
   - Split chronologically: 70% train (1991–2012), 10% val (2013–2015), 20% test (2016–2022)

2. **Feature Preparation**:
   - Univariate features: `[price_t-3, price_t-2, price_t-1]`
   - Multivariate features: `[price_t-3, prod_t-3, import_t-3, ..., price_t-1, prod_t-1, import_t-1]`
   - Scale all using training set min/max values

3. **LSTM Architecture**:
   - Layer 1: LSTM(32 units, return_sequences=True) → Dropout(0.2)
   - Layer 2: LSTM(16 units) → Dropout(0.2)
   - Layer 3: Dense(8, activation='relu') → Dense(1)
   - Loss: MSE; Optimizer: Adam(lr=0.001)
   - Training: 200 epochs, batch_size=4, early stopping on validation loss

4. **ARIMA Baseline**:
   - Auto-fit ARIMA(p,d,q) using AIC
   - Forecast on test set

5. **Evaluation & Crisis Analysis**:
   - Compute RMSE, MAE, MAPE, Directional Accuracy on full test set
   - Identify crisis years using thresholds (price level + YoY change)
   - Evaluate models separately for normal vs. crisis periods
   - Compute performance degradation

6. **Counterfactual (Onion only)**:
   - Fit polynomial trend to 1991–2018 prices
   - Extrapolate to 2019–2022
   - Calculate welfare loss

#### **For Oil** (simplified):

1. **Data Loading & Splitting**: As above (2005–2024)
2. **Univariate LSTM**: Train and evaluate price-only model
3. **ARIMA Baseline**: Compare
4. **Note**: Multivariate analysis skipped due to sparse import data (only 6 points)

**Outputs**:
- Performance tables (RMSE, MAE, MAPE, DA)
- PNG plots: bar charts (metrics comparison), line charts (actual vs. predicted)
- Crisis period classification
- Correlation analysis (crisis periods)
- Ensemble model recommendation (for onion)
- Welfare loss quantification (onion counterfactual)

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Window size** | 3 | Lagged timesteps per sample |
| **LSTM units (Layer 1)** | 32 | Captures patterns over ~3 months |
| **LSTM units (Layer 2)** | 16 | Reduces feature dimension |
| **Dropout rate** | 0.2 | Prevents overfitting |
| **Dense units** | 8 | Feature compression |
| **Loss function** | MSE | Mean Squared Error |
| **Optimizer** | Adam | Learning rate = 0.001 |
| **Batch size** | 4 | Small batch for annual data |
| **Epochs** | 200 | Early stopping on validation loss |
| **ARIMA order** | (1,1,1) | Selected via AIC; differencing=1 for stationarity |
| **Crisis threshold (Onion)** | Price > 30k BDT/ton OR ΔYoY > 50% | Reflects historical volatility |
| **Crisis threshold (Rice)** | Price > 35k BDT/ton OR ΔYoY > 30% | Reflects policy thresholds |

---

## Interpretation of Results

### RMSE, MAE, MAPE

- **RMSE** (Root Mean Squared Error): Penalizes large errors; measured in original units (BDT/ton or BDT/liter)
- **MAE** (Mean Absolute Error): Average absolute deviation; more interpretable than RMSE
- **MAPE** (Mean Absolute Percentage Error): Scale-free; allows comparison across commodities with different price ranges

**Example**: Onion univariate LSTM achieves RMSE = 15,897 BDT/ton means typical prediction is off by ~15,900 BDT/ton; MAPE = 25% means average percentage error is 25%.

### Directional Accuracy (DA)

Percentage of test samples where the model correctly predicts whether price will increase or decrease.

**Example**: Onion univariate LSTM DA = 80% means it correctly predicts price direction 8 out of 10 times. Random guessing = 50%.

**Policy Relevance**: A model with 80% DA can guide policymakers on whether to enable imports (price rising) or release buffer stocks (price falling).

### R² (Coefficient of Determination)

Proportion of variance in target explained by model. Negative R² indicates the model performs worse than a horizontal "naive" baseline (always predicting the mean).

Note: For annual commodity data with small test sets, negative R² is common and does not invalidate the model if RMSE/MAPE are low and DA is high.

---

## Limitations and Caveats

1. **Annual Data Frequency**: Monthly or weekly data would better capture price shocks. Annual trade data lag behind rapid crisis dynamics (e.g., export ban effects).

2. **Small Test Sets**: Only 5–7 test years per commodity; directional accuracy estimates have wide confidence intervals. 10+ year test periods recommended for stronger inference.

3. **Missing Data**: 
   - Rice imports: 80% missing (27 of 34 years)
   - Oil imports: 83% missing (6 of 39 years)
   - Multivariate results depend on interpolation quality; sensitivity analyses recommended.

4. **No External Variables**: Model excludes weather, policy announcements, news sentiment, and cross-border prices (e.g., Indian onion prices). These are known drivers and could improve multivariate models if integrated.

5. **Single-Country Analysis**: Results based on Bangladesh data; generalization to other regions (India, Pakistan, Nepal) requires validation.

6. **Model Stability**: Training on 1991–2015 and testing on 2016–2024 covers a major structural break (India–Bangladesh trade tensions). Unclear if model parameters remain stable over longer horizons.

---

## Future Extensions

1. **High-Frequency Data**: Acquire monthly/weekly import and price data; retrain multivariate models to test whether fine-grained trade information improves onion crisis forecasting.

2. **Hybrid Models**: Develop ensemble architectures that dynamically blend univariate and multivariate components based on detected volatility spikes.

3. **News & Policy Integration**: Scrape policy announcements and news articles; extract sentiment embeddings to capture rapid shock signals.

4. **Regional Models**: Include India, the major supplier of onion and competitor in rice; test whether cross-border price information improves Bangladesh forecasts.

5. **Causal Inference**: Apply Granger causality tests and instrumental variables to validate whether production/imports causally drive prices or merely correlate with omitted factors.

6. **Operational Deployment**: Implement models in real-time systems and measure their ability to support policy decisions (tariffs, import quotas, buffer stock releases).

---

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{sen_sultana_2026,
  author = {Sen, Shuvro Sankar and Sultana, Ambia},
  title = {Time-Series Forecasting of Essential Commodities Using Price and Trade Data: 
           Evidence from Bangladesh},
  booktitle = {Proceedings of the Advanced Innovation Institute Conference},
  year = {2026},
  address = {London, UK}
}
```

---

## Contact & Support

- **Authors**: Shuvro Sankar Sen, Ambia Sultana
- **Institution**: American International University – Bangladesh (AIUB)
- **Email**: 25-93776-2@student.aiub.edu, 24-93389-2@student.aiub.edu

For questions about data sources, methodology, or results, please refer to the main manuscript or create a GitHub issue.

---

## Acknowledgments

- **FAOSTAT** (Food and Agriculture Organization) for production, price, and trade data
- **Bangladesh Bank** and **Ministry of Agriculture** for national statistics
- **TensorFlow/Keras** and **scikit-learn** communities for open-source tools

---

## License

This project is released under the MIT License. See LICENSE file for details.

---

## Appendix: Quick Start (5 Minutes)

1. **Clone or download** this repository
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Open `Integration.ipynb`**: Run all cells to preprocess data (~5 min)
4. **Open `LSTM.ipynb`**: Run all cells to train models and generate results (~15 min)
5. **Check outputs**:
   - Cleaned CSVs in `/data/`
   - Plots in `/figures/`
   - Performance tables printed in notebook
6. **Read paper**: Open `paper/main_paper.tex` for detailed methodology and discussion
