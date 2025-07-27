# Bond Price Prediction Using Machine Learning

This project predicts the **price of Germany's 10-Year Government Bond** using historical data and machine learning. The model is built using Python and trained with a **Random Forest Regressor**.

![Bond Prediction Chart](Bond%20Price%20Prediction%20Chart.png)

## About This Project

Bond prices are influenced by many economic indicators, including interest rates and market momentum. This project uses financial time series data and technical indicators to predict future bond prices. It can be a helpful reference for students, researchers, and analysts interested in financial forecasting with machine learning.

This project was developed as part of a personal machine learning portfolio to apply regression modeling techniques to real-world financial data.

## Files Included

- `Germany 10-Year Bond Yield Historical Data.csv` — raw bond data used for feature engineering.
- `Training_data_processing.py` — prepares training features from raw data.
- `Test_data_processing.py` — prepares test features from raw data.
- `Germany 10-Year Bond Training Data.csv` — processed training data.
- `Germany 10-Year Bond Test Data.csv` — processed test data.
- `bond_price_prediction.py` — main model training and evaluation script.
- `Bond Price Prediction Chart.png` — visual comparison between predicted and actual prices.

## Features Used

- Price Lag 1 & 2
- Price Return (1-day)
- Momentum (3-day)
- Moving Averages (5-day & 10-day)
- Volatility (5-day)
- Daily Change (%)
- Interest Rate

## Model Used

- **RandomForestRegressor** from scikit-learn  
- Parameters:
  - `n_estimators = 300`
  - `random_state = 42`
  - `oob_score = True`

## Evaluation Metrics

- **R² Score** (Goodness of Fit)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **OOB Score** (Out-of-Bag Score)

### Example Output:

R²: 0.8904 | MAE: 0.0163 | MSE: 0.0005 | OOB Score: 0.9204

## Result

The model produces a line chart showing both **Predicted Prices** and **Actual Prices**, making it easy to visualize the accuracy.

## How to Run

1. Make sure all required `.csv` files are in the same directory.
2. Run `Training_data_processing.py` to generate training data.
3. Run `Test_data_processing.py` to generate test data.
4. Run `bond_price_prediction.py` to train the model and view the results.

## Requirements

Install dependencies using:
```bash
pip install pandas scikit-learn matplotlib seaborn
```
