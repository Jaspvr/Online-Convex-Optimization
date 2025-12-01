from eg_penny import *
import matplotlib.ticker as mtick

TICKERS = TICKERS_PENNY30
START = "2020-11-01"
END = "2025-10-31"

cache_file = "data/penny20_2020-11-01_2025-10-31.csv"
prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                min_days=500, cache_path=cache_file)

relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
dates = prices.index[1:]

portfolio = OnlinePortfolioEGPenny(relativePrices)

lambdas = np.linspace(0.0, 1.0, 21)
lambdas = lambdas[1:len(lambdas)-1]


maxW = -1000
bestLambda = -1
ws = []
lam = []
for l in lambdas:
    _, wealth, _ = portfolio.eg()
    ws.append(wealth[-1])
    lam.append(l)

ws = [(w - 1.0) * 100.0 for w in ws]

lamRev = lam[::-1]
wRev = ws[::-1]



plt.figure()
plt.plot(lamRev, wRev, marker='o')
plt.title("Portfolio Percent Gain vs Lambda")
plt.xlabel("$p$")
plt.ylabel("Total Percent Gain")


ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))


plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/lam.pdf")  # vector graphic
plt.show()
