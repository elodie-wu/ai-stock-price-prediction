# 📈 AI Stock Prediction & RL Trading

<p align="center">
  <img src="data/assets/geniustrader.png" width="200">
</p>

This project explores two approaches to A-share market modelling:

- Supervised learning for next-day price direction prediction
- Reinforcement learning for Buy / Hold / Sell trading decisions

**Current stock:** `000001` — Ping An Bank  
**Data source:** AKShare

## 🧠 Supervised Learning _(Work in Progress)_

Features include:

- Short-term returns
- Moving-average distance
- Volatility
- Volume change

Models:

- Logistic Regression
- Random Forest
- XGBoost
- Tuned XGBoost

Best validation ROC-AUC so far:

**Random Forest: 0.5456**

The results suggest that basic technical features contain only weak predictive signals for next-day stock direction.

## 🤖 Reinforcement Learning

A simple Q-learning trading agent was implemented.

State:

- Market trend
- Volatility
- Current position

Actions:

- Hold
- Buy
- Sell

The first reward design caused the agent to learn a degenerate strategy: staying entirely in cash.

After changing the reward to relative market performance, the agent began making actual trading decisions.

## 📊 Backtest Results

| Strategy        | Final Value |  Return |
| --------------- | ----------: | ------: |
| RL Strategy     |   10,905.07 |   9.05% |
| Buy and Hold    |   11,846.62 |  18.47% |
| Random Strategy |    6,901.40 | -30.99% |

The RL strategy outperformed random trading but did not beat Buy-and-Hold.

## ⚠️ Limitations

- One A-share stock only
- Simplified features and trading rules
- All-in / all-out position model
- Simplified transaction costs
- End-of-day features and closing prices are used in the current backtest

## 🚀 Future Work

- Improve feature engineering
- Test more stocks
- Use more realistic execution assumptions
- Explore DQN and PPO
- Compare RL with supervised-learning trading signals

## 🛠 Technologies

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Q-learning, AKShare

> For learning and research only. Not financial advice.
