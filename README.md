# 📈 AI Stock Price Prediction

This project explores how machine learning can be used to predict short-term stock price direction using historical market data and engineered technical features.

The current prototype focuses on a single A-share stock and predicts whether the next trading day's closing price will increase.

## 📊 Dataset

**Data source:** AKShare

**Current stock:** `000001` — Ping An Bank

**Market:** China A-share

**Frequency:** Daily

**Period:** 2010–2026

The historical data includes:

- Open price
- Close price
- High price
- Low price
- Trading volume
- Trading amount
- Price change
- Turnover rate

Forward-adjusted (`qfq`) prices are used to reduce distortions caused by dividends and corporate actions.

## 🎯 Project Goals

1. Build a machine-learning pipeline for stock direction prediction.
2. Engineer useful features from historical market data.
3. Compare multiple classification models.
4. Evaluate predictive performance using time-based validation.
5. Reduce model overfitting and improve generalization.
6. Extend the prototype to multiple stocks, ETFs, funds, and U.S. equities.

## 🧠 Methods

### Feature Engineering

The current model uses:

- 1-day return
- 3-day return
- 5-day return
- Distance from 5-day moving average
- Distance from 20-day moving average
- 5-day volatility
- 20-day volatility
- Daily trading-volume change

### Prediction Target

Binary classification:

- `1` — next trading day closes higher
- `0` — next trading day does not close higher

### Models

- Dummy Classifier
- Logistic Regression
- Random Forest
- XGBoost
- Regularized XGBoost

The dataset is split chronologically into training, validation, and test sets to reduce time-series data leakage.

## 📈 Current Results

### Validation Performance

| Model               | Train Accuracy | Validation Accuracy | F1 Score   | ROC-AUC    |
| ------------------- | -------------- | ------------------- | ---------- | ---------- |
| Logistic Regression | 52.92%         | 54.96%              | 0.2637     | 0.5177     |
| Random Forest       | 68.28%         | **56.97%**          | 0.3600     | **0.5456** |
| XGBoost             | 80.39%         | 55.80%              | **0.4913** | 0.5440     |
| Tuned XGBoost       | 60.45%         | **56.97%**          | 0.4074     | 0.5374     |

The majority-class Dummy Classifier achieved a validation accuracy of **57.31%**.

## 🔍 Key Findings

Random Forest currently provides the strongest overall validation performance, achieving the highest ROC-AUC score of **0.5456**.

The original XGBoost model achieved high training accuracy but showed substantial overfitting.

After additional regularization, the XGBoost training-validation gap was significantly reduced, although its ROC-AUC score did not improve.

The current results suggest that basic price, momentum, volatility, and volume features contain only weak predictive signals for next-day stock direction.

## ⚠️ Limitations

- The current prototype uses only one A-share stock.
- Only basic price- and volume-based features are currently included.
- Predicting next-day stock direction is highly noisy and difficult.
- Current models have not yet been evaluated on the final test set.
- Transaction costs, slippage, suspensions, and trading constraints are not yet included.
- Backtesting has not yet been implemented.
- Historical predictive performance does not imply future profitability.

This project is intended for machine-learning research and education, not financial advice.

## 🚀 Future Work

- evaluate Extra Trees and LightGBM
- apply GA-based feature selection
- perform systematic hyperparameter tuning
- evaluate the final model on the unseen test set
- investigate prediction-confidence thresholds
- implement strategy backtesting
- include transaction costs and risk metrics
- add SHAP-based model explainability
- extend prediction to multiple A-share stocks
- build an AI stock screener and ranking system
- support U.S. stocks, ETFs, and funds
- build a web dashboard for predictions and backtesting

## 🛠 Technologies

Python, Pandas, Num
