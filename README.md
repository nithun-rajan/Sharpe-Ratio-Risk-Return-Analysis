# 📊 Sharpe Ratio Analysis: MSFT & Amazon vs. S&P 500

This project analyzes the **Sharpe Ratio** of Microsoft (MSFT) and Amazon (AMZN) stocks relative to the S&P 500 Index. The Sharpe Ratio measures risk-adjusted returns and is a widely used metric for comparing investment performance.

> **"The Sharpe ratio measures the excess return per unit of risk."**  
— William F. Sharpe (1966)

---

## 🚀 Project Overview

- 📈 **Objective:** Compare the risk-adjusted returns of MSFT and Amazon against the S&P 500.
- 🗓️ **Period Analyzed:** January 1, 2022 – June 25, 2024.
- 🧮 **Metric:** Annualized Sharpe Ratio (daily returns scaled by √252).
- 📊 **Data Sources:** Yahoo Finance (`yfinance`)

---

## ✅ Key Features

- Downloads historical price data for:
  - MSFT (Microsoft)
  - AMZN (Amazon)
  - ^GSPC (S&P 500 Index)
- Calculates:
  - Daily returns & excess returns over S&P 500.
  - Standard deviations of excess returns.
  - Annualized Sharpe Ratios.
- Visualizations:
  - Stock prices over time.
  - Daily returns & excess returns.
  - Bar plots of mean returns and Sharpe Ratios.

---

## 🛠️ Installation

```bash
pip install pandas numpy matplotlib yfinance pandas_datareader
