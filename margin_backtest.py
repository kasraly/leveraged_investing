# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt


# %%
def process_yahoo_csv(file_name):
    df = pd.read_csv(file_name, 
                     parse_dates=True,
                     index_col=0)

    price_ratio = df['Adj Close']/df['Close']
    for column in ["Open", "High", "Low", "Close"]:
        df[column] = df[column]*price_ratio

    return df[["Open", "High", "Low", "Close", "Volume"]]


# %%
vtsmx = process_yahoo_csv("VTSMX.csv")
vfisx = process_yahoo_csv("VFISX.csv")
loan = vfisx.copy()
load_pct = 2.0
loan['Close'] = np.power(1 + load_pct/100.0, (loan.index - loan.index[0]).days/365.0)


# %%
def assetAllocBacktest(assets_data_raw, start_date, end_date, start_balance=1.0, monthly_contrib=1.0, commission=10, bidAskSpread=0.001):

    asset_data = []
    start = start_date
    end = end_date
    for asset in assets_data_raw:
        if asset.index[0] > start:
            start = asset.index[0]
        if asset.index[-1] < end:
            end = asset.index[-1]
    for asset in assets_data_raw:
        asset['52WeekHigh'] = asset['Close'].rolling(365,min_periods=1).max()
        asset_data.append(asset[(asset.index >= start) & (asset.index <= end)])

    fund = asset_data[0]
    cash = asset_data[1]
    idx = asset_data[0].index[0]
    fund_drop = max(0.0, (1 - fund.loc[idx, 'Close']/fund.loc[idx, '52WeekHigh']))
    margin_ratio = calc_margin_ratio(fund_drop)
    fund.loc[idx, 'Volume'] = (start_balance * margin_ratio / fund.loc[idx, 'Close'])
    cash.loc[idx, 'Volume'] = (start_balance * (margin_ratio-1) / cash.loc[idx, 'Close'])
    last_contrib = start

    
    total_high = 0.0
    max_drop = 0.0
    total_fees = 0.0
    rebal_count = 0
    total_deposit = start_balance
    for i in range(1, len(asset_data[0])):
        idx_prev = fund.index[i-1]
        idx = fund.index[i]
        fund.loc[idx, 'Volume'] = fund.loc[idx_prev, 'Volume']
        cash.loc[idx, 'Volume'] = cash.loc[idx_prev, 'Volume']
        if (idx - last_contrib).days > 30:
            # monthly contribution
            last_contrib = idx
            cash.loc[idx, 'Volume'] -= monthly_contrib / cash.loc[idx, 'Close']
            total_deposit += monthly_contrib
        # print(idx, fund.loc[idx, 'Volume']*fund.loc[idx, 'Close'], cash.loc[idx, 'Volume']*cash.loc[idx, 'Close'])

        base_asset_drop = max(0.0, (1 - fund.loc[idx, 'Close']/fund.loc[idx, '52WeekHigh']))
        margin_ratio = calc_margin_ratio(base_asset_drop)

        total_asset = fund.loc[idx, 'Volume'] * fund.loc[idx, 'Close'] - cash.loc[idx, 'Volume'] * cash.loc[idx, 'Close']
        if total_asset > total_high:
            total_high = total_asset
        drop = (total_high - total_asset) / total_high
        if drop > max_drop:
            max_drop = drop
        
        buy_value = (total_asset * (margin_ratio-1)) - (cash.loc[idx, 'Volume'] * cash.loc[idx, 'Close'])
        # print(buy_value)
        if buy_value >= 5000:# or buy_value <= -10000:
            fees = abs(buy_value) * bidAskSpread + 10
            total_fees += fees * 2
            fund.loc[idx, 'Volume'] += (buy_value - fees) / fund.loc[idx, 'Close']
            cash.loc[idx, 'Volume'] += (buy_value + fees) / cash.loc[idx, 'Close']
        # print(idx, fund.loc[idx, 'Volume']*fund.loc[idx, 'Close'], cash.loc[idx, 'Volume']*cash.loc[idx, 'Close'])
        
        if fund.loc[idx, 'Volume'] * fund.loc[idx, 'Close'] * 0.7 - (cash.loc[idx, 'Volume'] * cash.loc[idx, 'Close']) < 0.0:
            buy_value = fund.loc[idx, 'Volume'] * fund.loc[idx, 'Close'] - total_asset / 0.34
            fees = abs(buy_value) * bidAskSpread + 10
            total_fees += fees * 2
            rebal_count += 1
            fund.loc[idx, 'Volume'] = (total_asset / 0.34 - fees) / fund.loc[idx, 'Close']
            cash.loc[idx, 'Volume'] = (total_asset / 0.34 * 0.66 + fees) / cash.loc[idx, 'Close']
        # print(idx, fund.loc[idx, 'Volume']*fund.loc[idx, 'Close'], cash.loc[idx, 'Volume']*cash.loc[idx, 'Close'])

    val = fund['Volume'] * fund['Close'] - cash['Volume'] * cash['Close']
    print(total_deposit)
    return val, max_drop, rebal_count, total_fees

# %%
def analyze(val):
    years = (val.index[-1] - val.index[0]).days / 365.0
    cagr = ((val[-1]/val[0])**(1/years) - 1)*100
    monthly_val = val.groupby(pd.Grouper(freq="M")).last()
    shifted_val = monthly_val.shift(periods=12)
    rolling_performance = ((monthly_val - shifted_val) / shifted_val).dropna()
    sharpe = np.average(rolling_performance) / np.std(rolling_performance)

    return cagr, sharpe

# %%
def calc_margin_ratio(drop):
    # return 1.0
    drop50pct = 0.4/(1-drop)    
    return max(1.0, min(3.0, (1.0) / ((1 - 0.7*drop50pct)*1.0)))

# %%
start = datetime.datetime(2013, 2, 1)
end = datetime.datetime(2019, 2, 1)
start = datetime.datetime(2017, 7, 1)
# start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 12, 14)
start = datetime.datetime(1990, 1, 1)
end = datetime.datetime(2021, 1, 1)
portfolio, max_drop, rebal_count, fees = assetAllocBacktest([vtsmx, loan],
    start_date=start, end_date=end, start_balance=100000.0, monthly_contrib=5000)
cagr, sharpe = analyze(portfolio)

print(f"Max Drawdown: {max_drop*100:.0f}%\nfinal: {portfolio[-1]:.2f}\nRebalances: {rebal_count}\nFees: {fees:.0f}")


# %%
