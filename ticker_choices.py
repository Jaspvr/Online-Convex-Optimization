import pandas as pd

CSV_PATH = "data/sp500_companies_nov04.csv" 

df = pd.read_csv(CSV_PATH)
tickers = df.iloc[:, 1].astype(str).str.strip()

# Drop blanks/NaNs
tickers = [t for t in tickers if t]

# For Stooq
TICKERS = [f"{t}.US" for t in tickers]

print(len(tickers))
print(tickers[:20]) 

TICKERS_ETFS = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM"]
TICKERS_SP = tickers
