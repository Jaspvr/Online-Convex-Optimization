from online_gradient_descent import *
from exponentiated_gradient import OnlinePortfolio as OnlinePortfolioEG
from online_newton_step import OnlinePortfolio as OnlinePortfolioONS


TICKERS = TICKERS_SP20
START = "2021-01-01"
END = "2025-11-01"

cache_file = "data/sp20new_2015-11-01_2025-11-01.csv"
prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                min_days=500, cache_path=cache_file)

relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
dates = prices.index[1:]

portfolioEG = OnlinePortfolioEG(relativePrices)
_, wealthEG, loss = portfolioEG.eg()
portfolioONS = OnlinePortfolioONS(relativePrices)
_, wealthONS, loss = portfolioONS.ons(1)
portfolioOGD = OnlinePortfolioOGD(relativePrices)
_, wealthOGD, loss = portfolioOGD.odg()

# For comparison
wealthUniformCRP = uniformCRP(relativePrices)
_, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
wealthBestStock = bestInHindsight(relativePrices)

wealthUniformCRP = uniformCRP(relativePrices)
_, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
wealthBestStock = bestInHindsight(relativePrices)

# Regret vs Optimal CRP (cumulative)
regretOGD = np.log(wealthOptimalCRP) - np.log(wealthOGD)
regretONS = np.log(wealthOptimalCRP) - np.log(wealthONS)
regretEG  = np.log(wealthOptimalCRP) - np.log(wealthEG)

plt.figure()
plt.plot(dates, regretOGD, label="OGD regret vs Optimal CRP")
plt.plot(dates, regretONS, label="ONS regret vs Optimal CRP")
plt.plot(dates, regretEG,  label="EG regret vs Optimal CRP")
plt.title("Regret vs Optimal CRP")
plt.xlabel("Date")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.tight_layout()
plt.savefig("Plots/ogd_ons_eg_regret_vs_optcrp.pdf")
plt.show()


# # Plot the log wealth growth over time. Use log wealth since it matches with the loss
# plt.figure()
# plt.plot(dates, np.log(wealthOGD), label="OGD")
# plt.plot(dates, np.log(wealthONS), label="ONS")
# plt.plot(dates, np.log(wealthEG), label="EG")
# # plt.plot(dates, np.log(wealthUniformCRP),
# #             label=f"Uniform CRP")
# # plt.plot(dates, np.log(wealthOptimalCRP),
# #             label=f"Optimal CRP")
# plt.title("OGD vs ONS vs EG")
# plt.xlabel("Date")
# plt.ylabel("Portfolio Log Wealth")
# plt.legend()
# plt.tight_layout()
# plt.savefig("Plots/ogd_ons_eg.pdf")  #
# plt.show()