import quandl
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests

def get_bitfinex_asset(asset, ts_ms_start, ts_ms_end):
    url = 'https://api.bitfinex.com/v2/candles/trade:1D:t' + asset + '/hist'
    params = { 'start': ts_ms_start, 'end': ts_ms_end, 'sort': 1}
    r = requests.get(url, params = params)
    data = r.json()
    return pd.DataFrame(data)[2]

start_date = 1420102917000 # 1 January 2015
end_date = 1591002117000   # 1 June 2020
assets = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XMRUSD', 'NEOUSD', 'XRPUSD', 'ZECUSD']

data = pd.DataFrame()

for a in assets:
    print('Downloading ' + a)
    data[a] = get_bitfinex_asset(asset = a, ts_ms_start = start_date, ts_ms_end = end_date)

print(len(data))

close = pd.DataFrame({'BTC': data.BTCUSD,
                      'ETH': data.ETHUSD,
                      'LTC': data.LTCUSD,
                      'XRP': data.XRPUSD})

print(len(close))

id = ['BTC', 'ETH', 'LTC', 'XRP']

# Define training set
training_start = 0
training_end = 60
training_set = close[training_start:training_end]

# Define testing set
testing_start = 61
testing_end = 119
testing_set = close[testing_start:testing_end]


round_trip = 0.0010


entry_threshold = 1
exit_threshold = 0.5

# Set crypto 1 to BTC
crypto_1 = id[0]

# Initialize output
output = {id[1]: {},
          id[2]: {}, 
          id[3]: {}}

for i in range(1, len(id)):
    
    # Set crypto 2 to ETH, LTC and XRP
    crypto_2 = id[i]
    
    # Calculate the hedge ratio using the training set
    model = sm.OLS(training_set[crypto_1], training_set[crypto_2])
    result = model.fit()
    hedge_ratio = result.params[crypto_2]

    # Calculate the spread
    spread = close[crypto_1] - hedge_ratio * close[crypto_2]
    # Mean of the spread on the training set
    spread_mean = spread[training_start:training_end].mean()
    # Standard deviation of the spread calculated on the training set
    spread_std = spread[training_start:training_end].std()
    # Z-score of the spread
    z_score = (spread - spread_mean) / spread_std
    
    # Implement pair trading strategy
    # Create masks for long, short and exit positions
    longs = (z_score <= -entry_threshold)
    shorts = (z_score >= entry_threshold)
    exits = (np.abs(z_score) <= exit_threshold)
    # Initialize the positions
    positions = pd.DataFrame({crypto_1: np.nan * pd.Series(range(len(z_score))),
                              crypto_2: np.nan * pd.Series(range(len(z_score)))},
                             index=z_score.index)
    # Update the positions
    [positions[crypto_1][longs], positions[crypto_2][longs]] = [1, -1]
    [positions[crypto_1][shorts], positions[crypto_2][shorts]] = [-1, 1]
    [positions[crypto_1][exits], positions[crypto_2][exits]] = [0, 0]
    # Carry forward the positions except when there is an exit
    positions.fillna(method='ffill', inplace=True)
    # Lag the positions to the next day because we base calculations on close
    positions = positions.shift(periods=1)
    
    # Calculate the performance
    # Initialize the returns
    returns = pd.DataFrame({crypto_1: close[crypto_1],
                            crypto_2: close[crypto_2]})
    # Update the returns
    returns = returns.pct_change()
    # Calculate the pnl
    pnl = returns * positions

    # Calculate transaction costs
    # Create a mask to indicate changes in position
    mask = (~np.isnan(positions.BTC) & (positions.BTC - positions.BTC.shift(periods=1)).astype(bool))
    # mask = (~np.isnan(positions.BTC) & (positions.BTC != positions.BTC.shift(periods=1)))
    # Create a transaction costs Series
    tc = pd.Series(np.zeros(len(mask)), index=mask.index)
    tc[mask] = - round_trip
    
    # Update pnl DataFrame
    pnl['TC'] = tc
    # Calculate net pnl
    pnl_net = pnl.sum(axis='columns')
    
    # Calculate the Sharpe ratio under the training set
    sharpe_training = np.sqrt(252) * pnl_net[training_start:training_end].mean() / pnl_net[training_start:training_end].std()
    # Calculate the Sharpe ratio under the testing set
    sharpe_testing = np.sqrt(252) * pnl_net[testing_start:testing_end].mean() / pnl_net[testing_start:testing_end].std()
            
    # Generate the output
    # Gather data
    data = {'spread': z_score,
            'positions': positions,
            'pnl': pnl_net,
            'sharpe training': sharpe_training,
            'sharpe testing': sharpe_testing,
           }
    # Update the output
    output.update({crypto_2: data})

positions_eth = output['ETH']['positions'][:-100].dropna()
positions_ltc = output['LTC']['positions'][:-100].dropna()
positions_xrp = output['XRP']['positions'][:-100].dropna()

'''
sharpe = pd.DataFrame({'BTC & ETH': [output['ETH']['sharpe training'], output['ETH']['sharpe testing']],
                       'BTC & LTC': [output['LTC']['sharpe training'], output['LTC']['sharpe testing']],
                       'BTC & XRP': [output['XRP']['sharpe training'], output['XRP']['sharpe testing']]},
                      index=pd.MultiIndex.from_product([['Sharpe Ratio'], ['Training Set', 'Testing Set']]))

'''

sharpe = pd.DataFrame({'BTC & ETH': [output['ETH']['sharpe training'], output['ETH']['sharpe testing']],
                       'BTC & LTC': [output['LTC']['sharpe training'], output['LTC']['sharpe testing']],
                       'BTC & XRP': [output['XRP']['sharpe training'], output['XRP']['sharpe testing']]},
                      index=['Training Set', 'Testing Set'])

print(sharpe)

plt.figure(figsize=[20, 5])

plt.subplot(1, 2, 1)
plt.plot(output['ETH']['pnl'].cumsum()[training_start:training_end])
plt.plot(output['LTC']['pnl'].cumsum()[training_start:training_end])
plt.plot(output['XRP']['pnl'].cumsum()[training_start:training_end])
plt.title('Cumulative PnL Under The Training Set')
plt.legend(['BTC & ETH', 'BTC & LTC', 'BTC & XRP'])
plt.ylim(-1, 10)

plt.subplot(1, 2, 2)
plt.plot(output['ETH']['pnl'].cumsum()[testing_start:testing_end])
plt.plot(output['LTC']['pnl'].cumsum()[testing_start:testing_end])
plt.plot(output['XRP']['pnl'].cumsum()[testing_start:testing_end])
plt.title('Cumulative PnL Under The Testing Set')
plt.legend(['BTC & ETH', 'BTC & LTC', 'BTC & XRP'])
plt.ylim(-1, 10)

plt.show() 
