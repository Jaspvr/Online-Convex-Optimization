import pandas as pd

CSV_PATH = "data/sp500_companies_nov04.csv" 

df = pd.read_csv(CSV_PATH)
tickers = df.iloc[:, 1].astype(str).str.strip()

tickers = [t for t in tickers if t] # Drop blanks/NaNs
TICKERS = [f"{t}.US" for t in tickers]

print(TICKERS)