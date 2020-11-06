# Crypto-Pairs-Trading
This is a code exploring a statistical arbitrage strategy to trade crypto pairs.

First, the code (cointegration_analsis.py) analyzes the performance of 8 cryptocurrencies from 1-1-2020 to 5-31-2020 with respect to their intial values: 
![Alt text](https://github.com/coderaashir/Crypto-Pairs-Trading/blob/main/Results/Screenshot%202020-11-06%20at%203.34.15%20PM.png)

Then, the code tests for cointegration of the cryptocurrencies against each other. 

Cointegration implies that the ratio of two stationary time series will vary around a mean. This is the basis for trading pairs. For example, if BTC and ETH are highly cointegrated, and BTC diverts up from the mean while ETH diverts down from the mean, we can go long on ETH and short on BTC because they will revert back. 

The code finds the p values of each pair, and plots a heat map that allows us to pick highly cointegrated pairs for trading: 

![Alt text](https://github.com/coderaashir/Crypto-Pairs-Trading/blob/main/Results/Screenshot%202020-11-06%20at%203.34.58%20PM.png)

We can see that the best trades are BTC-ETH, BTC-LTC, and BTC-XRP 

Now it implements a strategy to trade. 

It calculate the z-scores (how many standard deviations has the asset moved from the mean) and places trades accordingly. 

# Results

Sharpe Ratio Table (> 1 is ideal)  

|              | BTC-ETH  | BTC-LTC   | BTC-XRP  |
|--------------|----------|-----------|----------|
| Training Set | 4.744124 | 3.182864  | 3.630110 |
| Testing Set  | 1.158460 | -1.808723 | 2.060752 |

Cumulative PNL: 

![Alt text](https://github.com/coderaashir/Crypto-Pairs-Trading/blob/main/Results/Screenshot%202020-11-06%20at%203.36.03%20PM.png)


