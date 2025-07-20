# Market Analyzer

A modular AI/ML system for real-time market analysis and buy/sell signal prediction. Currently supports stocks (via yfinance), but is designed to be easily extended to crypto, forex, and other markets.

## Features
- Free, open-source libraries only
- Modular data pipeline (easy to add new markets)
- Technical indicator feature engineering
- Baseline ML model for signal prediction
- Simple backtesting

## Free Libraries Used
- Python 3.x
- pandas, numpy
- scikit-learn
- yfinance (stock data)
- pandas-ta (technical indicators)

## Getting Started
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run the prototype: `python main.py`

## Extending to New Markets
- Add a new fetcher in `data/fetcher.py`
- Add/modify model logic in `models/`
- Add new signal logic in `signals/`

---

For questions or contributions, open an issue or PR. 