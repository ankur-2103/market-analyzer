import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Added for scatter plots

from data.fetcher import StockDataFetcher
from data.preprocess import preprocess_data
from models.stock_model import StockModel
from signals.signal_generator import generate_signals
from backtest.backtester import Backtester

if __name__ == "__main__":
    all_symbols = ["RELIANCE.BO"]
    start_date = "2020-01-01"
    end_date = "2025-07-18"

    fetcher = StockDataFetcher()
    backtester = Backtester()
    summary_results = []

    for symbol in all_symbols:
        print(f"\nAnalyzing {symbol}...")
        try:
            df = fetcher.fetch(symbol, start_date, end_date)
            
            if df is not None:
                # Flatten MultiIndex columns and rename
                df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]
                df.columns = [col.split('_')[0] if '_' in col else col for col in df.columns]

                rename_map = {}
                for col in df.columns:
                    col_str = str(col)
                    if col_str.startswith('Close'):
                        rename_map[col] = 'Close'
                    elif col_str.startswith('Open'):
                        rename_map[col] = 'Open'
                    elif col_str.startswith('High'):
                        rename_map[col] = 'High'
                    elif col_str.startswith('Low'):
                        rename_map[col] = 'Low'
                    elif col_str.startswith('Volume'):
                        rename_map[col] = 'Volume'
                    elif col_str == 'Date':
                        rename_map[col] = 'Date'
                df = df.rename(columns=rename_map)

            if df is not None and not df.empty:
                X, y, feature_names = preprocess_data(df)
                if X.shape[0] > 0 and y.shape[0] > 0:
                    model = StockModel()
                    model.train(X, y)
                    predictions = model.predict(X)

                    y = y.astype(float)
                    predictions = predictions.astype(float)

                    # 🔧 FIX: Slice close_prices to match y/predictions length
                    close_prices = df['Close'].iloc[-len(y):].values.astype(float)

                    # 🔧 FIX: Also slice dates to match
                    dates = pd.to_datetime(df['Date']).iloc[-len(y):]

                    # Plot actual vs predicted signals
                    plt.figure(figsize=(14, 6))
                    # plt.step(dates, y, label='Actual', color='blue', where='mid', linewidth=1, alpha=0.7)
                    # plt.step(dates, predictions, label='Prediction', color='orange', where='mid', linewidth=1, alpha=0.5)
                    plt.plot(dates, close_prices, label='Actual Price', color='blue')
                    plt.plot(dates, predictions, label='Predicted Price', color='orange', alpha=0.7)
                    plt.title(f"{symbol} - Actual vs Prediction (Signals)")
                    plt.xlabel("Date")
                    plt.ylabel("Signal")
                    plt.yticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])
                    plt.xticks(rotation=45)
                    plt.grid(True, linestyle='--', alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{symbol}_price_and_signals.svg", format="svg")       # Infinite zoom quality
                    plt.savefig(f"plots/{symbol}_actual_vs_prediction.png", dpi=600)
                    plt.close()

                    # # Plot price and signals
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(close_prices, label='Close Price', color='black', alpha=0.7)

                    # # Actual signals
                    # buy_idx = (y == 1)
                    # sell_idx = (y == -1)
                    # hold_idx = (y == 0)

                    # ax.scatter(np.where(buy_idx), close_prices[buy_idx], marker='^', color='green', label='Actual Buy', alpha=0.7)
                    # ax.scatter(np.where(sell_idx), close_prices[sell_idx], marker='v', color='red', label='Actual Sell', alpha=0.7)

                    # # Predicted signals
                    # buy_pred = (predictions == 1)
                    # sell_pred = (predictions == -1)
                    # hold_pred = (predictions == 0)

                    # ax.scatter(np.where(buy_pred), close_prices[buy_pred], marker='^', facecolors='none', edgecolors='green', label='Predicted Buy', alpha=0.7)
                    # ax.scatter(np.where(sell_pred), close_prices[sell_pred], marker='v', facecolors='none', edgecolors='red', label='Predicted Sell', alpha=0.7)

                    # ax.set_title(f'{symbol} - Price and Buy/Sell Signals')
                    # ax.set_xlabel('Time')
                    # ax.set_ylabel('Price')
                    # ax.legend(loc='upper left')

                    signals = generate_signals(predictions)
                    results = backtester.run(df, signals)
                    summary_results.append({"symbol": symbol, "results": results})
                    print(f"Backtest Results for {symbol}: {results}")
                else:
                    print(f"Not enough data after preprocessing for {symbol}, skipping.")
                    summary_results.append({"symbol": symbol, "results": None, "error": "Not enough data after preprocessing"})
            else:
                print(f"No data for {symbol}, skipping.")
                summary_results.append({"symbol": symbol, "results": None, "error": "No data returned"})
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            summary_results.append({"symbol": symbol, "results": None, "error": str(e)})

    print("\n===== Summary of Backtest Results =====")
    for entry in summary_results:
        symbol = entry["symbol"]
        results = entry["results"]
        if results is not None:
            print(f"{symbol}: {results}")
        else:
            print(f"{symbol}: Error - {entry.get('error', 'Unknown error')}")
