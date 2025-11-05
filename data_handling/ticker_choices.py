import pandas as pd

# Wikipedia page with the official S&P 500 constituents
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Read all HTML tables on the page
tables = pd.read_html(url)

# First table is the "S&P 500 component stocks"
sp500 = tables[0]

# The 'Symbol' column has the tickers (includes things like BRK.B, GOOGL, etc.)
tickers = sp500["Symbol"].tolist()

print(len(tickers))
print(tickers[:20]) 

TICKERS_ETFS = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM"]

TICKERS_SP = tickers
