import quandl
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import requests
import statsmodels.tsa.stattools as ts 
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import seaborn

def rotate(l, n): 
	return l[n:] + l[:n]

def get_bitfinex_asset(asset, ts_ms_start, ts_ms_end):
	url = 'https://api.bitfinex.com/v2/candles/trade:1D:t' + asset + '/hist'
	params = { 'start': ts_ms_start, 'end': ts_ms_end, 'sort': 1}
	r = requests.get(url, params = params)
	data = r.json()
	return pd.DataFrame(data)[2]

start_date = 1577836800000 # 1 January 2020
end_date = 1590883200000   # 31 May 2020
assets = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XMRUSD', 'NEOUSD', 'XRPUSD', 'ZECUSD']

crypto_prices = pd.DataFrame()

for a in assets:
	print('Downloading ' + a)
	crypto_prices[a] = get_bitfinex_asset(asset = a, ts_ms_start = start_date, ts_ms_end = end_date)

print(crypto_prices.head())

norm_prices = crypto_prices.divide(crypto_prices.iloc[0])

plt.figure(figsize = (15, 10))
plt.plot(norm_prices)
plt.xlabel('days')
plt.title('Performance of cryptocurrencies')
plt.legend(assets)
plt.show()

def find_cointegrated_pairs(crypto_prices):
	n = crypto_prices.shape[1]
	score_matrix = pd.DataFrame()
	pvalue_matrix = pd.DataFrame()
	pairs = []
	for a1 in assets:
		for a2 in assets:
			if(a1 == a2): 
				continue
			S1 = crypto_prices[a1]
			S2 = crypto_prices[a2]
			test_result = ts.coint(S1, S2)
			score = test_result[0]
			pvalue = test_result[1]
			score_matrix.at[a1, a2] = score
			pvalue_matrix.at[a1, a2] = pvalue
			if pvalue < 0.02:
				pairs.append((a1, a2))
	
	return score_matrix, pvalue_matrix, pairs

scores, pvalues, pairs = find_cointegrated_pairs(crypto_prices)
m = [0,0.2,0.4,0.6,0.8,1]

print(pvalues.head())

seaborn.heatmap(pvalues, xticklabels=assets, 
				yticklabels=rotate(assets, -1), 
				mask = (pvalues >= 0.98))
plt.show()
print (pairs)