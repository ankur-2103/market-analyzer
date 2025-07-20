import pdb
import numpy as np


class Backtester:
    def run(self, df, signals):
        """
        Simulate trading based on signals and calculate total return.
        df: DataFrame with stock data.
        signals: List of 'BUY', 'SELL', 'HOLD'.
        Returns a dictionary with total return and number of trades.
        """
        df.columns = [col.replace("(", "").replace(")", "").replace("'", "").replace(",", "").strip() for col in df.columns]

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




        returns = []  # List to store profit/loss for each trade
        for i in range(len(signals)-1):
            if signals[i] == "BUY":
                # If 'BUY', profit is next day's close minus today's close
                returns.append(df['Close'].iloc[i+1] - df['Close'].iloc[i])
            elif signals[i] == "SELL":
                # If 'SELL', profit is today's close minus next day's close
                returns.append(df['Close'].iloc[i] - df['Close'].iloc[i+1])
            else:
                # 'HOLD' means no trade, no profit/loss
                returns.append(0)
        total_return = np.sum(returns)  # Sum of all trade results
        num_trades = sum([s != "HOLD" for s in signals])  # Count of trades made
        return {"total_return": total_return, "num_trades": num_trades} 