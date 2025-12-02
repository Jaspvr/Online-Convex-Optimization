import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import *
from best_stock import bestInHindsight
from data.tickers import *
from optimal_crp import optimalCrpWeightsCvx
from uniform_crp import uniformCRP
from follow_the_leader import OnlinePortfolio
from ftl_volatile import OnlinePortfolioFTLV

def main():
    TICKERS = TICKERS_PENNY30
    START = "2020-11-01"
    END = "2025-10-31"

    cache_file = "data/penny20_2020-11-01_2025-10-31.csv"
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)

    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:]


    portfolioFTL = OnlinePortfolio(relativePrices)
    portfolioFTLV = OnlinePortfolioFTLV(relativePrices)

    _, wealthFTL, _ = portfolioFTL.followLeader()
    _, wealthFTLV1, _ = portfolioFTLV.followLeader1Day()
    _, wealthFTLV3, _ = portfolioFTLV.followLeader3Day()
    _, wealthFTLVp, _ = portfolioFTLV.followLeaderPositives()
    _, wealthFTLVp2, _ = portfolioFTLV.followLeader2Positives()

    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    wealthBestStock = bestInHindsight(relativePrices)

    # print("Final wealth follow leader: ", wealth[-1])

    print("wftl: ", wealthFTL[-1])
    print("wftlV1: ", wealthFTLV1[-1])
    print("wftlV3: ", wealthFTLV3[-1])
    print("wftlVp: ", wealthFTLVp[-1])
    print("Optimal CRP: ", wealthOptimalCRP[-1])
    print("Uniform CRP: ", wealthUniformCRP[-1])




    plt.figure()
    plt.plot(dates, np.log(wealthFTL), label="Regular FTL")
    plt.plot(dates, np.log(wealthFTLV3), label="FTL 3-day Wndow")
    plt.plot(dates, np.log(wealthFTLV1), label="FTL 1-day Window")
    plt.plot(dates, np.log(wealthFTLVp), label="FTL Positives")
    # plt.plot(dates, np.log(wealthFTLVp2), label="FTL 2-Day Positives")
    plt.plot(dates, np.log(wealthUniformCRP), label="Uniform CRP")
    plt.plot(dates, np.log(wealthOptimalCRP), label="Optimal CRP")
    plt.title("Modified FTL Algorithms")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Log Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/follow_leader_mod.pdf")  #
    plt.show()

    
if __name__ == "__main__":
    main()