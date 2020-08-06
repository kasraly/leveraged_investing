# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import datetime
import backtrader as bt
import numpy as np
import math
import matplotlib.pyplot as plt


# %%
def sim_leverage(proxy, leverage=1, expense_ratio = 0.0, initial_value=1.0, start_date=None):
    """
    Simulates a leverage ETF given its proxy, leverage, and expense ratio.
    
    Daily percent change is calculated by taking the daily percent change of
    the proxy, subtracting the daily expense ratio, then multiplying by the leverage.
    """
    val = proxy['Close']
    pct_change = (val - val.shift(1)) / val.shift(1)
    if start_date is not None:
        pct_change = pct_change[pct_change.index > start_date]
    pct_change = pct_change * leverage
    sim = ((1 + pct_change - expense_ratio / 252).cumprod() * initial_value).to_frame("Close")
    sim.iloc[0] = initial_value
    for column in ["Open", "High", "Low"]:
        sim[column] = sim["Close"]
    sim["Volume"] = 0
    return sim


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
vix = process_yahoo_csv("VIX.csv")
vfinx = process_yahoo_csv("VFINX.csv")
vustx = process_yahoo_csv("VUSTX.csv")
tlt = process_yahoo_csv("TLT.csv")
ief = process_yahoo_csv("IEF.csv")
nasdaq100 = process_yahoo_csv("NASDAQ100.csv")
sp500 = process_yahoo_csv("SP500.csv")
qqq = process_yahoo_csv("QQQ.csv")
vtsmx = process_yahoo_csv("VTSMX.csv")
visvx = process_yahoo_csv("VISVX.csv")
vfitx = process_yahoo_csv("VFITX.csv")
tqqq = process_yahoo_csv("TQQQ.csv")
tyd = process_yahoo_csv("TYD.csv")
ty30 = process_yahoo_csv("TY30.csv").dropna()

spy = process_yahoo_csv("spy.csv")
tmf = process_yahoo_csv("TMF.csv")
upro = process_yahoo_csv("UPRO.csv")
edv = process_yahoo_csv("EDV.csv")
edc = process_yahoo_csv("EDC.csv")
eem = process_yahoo_csv("EEM.csv")
vwo = process_yahoo_csv("VWO.csv")
cash = vfinx.copy()
for column in ["Open", "High", "Low", "Close"]:
    cash[column] = 1.0

upro_sim = sim_leverage(vfinx, leverage=3.0, expense_ratio=0.015)
tmf_sim = sim_leverage(vustx, leverage=3.0, expense_ratio=0.015)
tqqq_sim = sim_leverage(nasdaq100, leverage=3.0, expense_ratio=0.015)
tyd_sim = sim_leverage(vfitx, leverage=3.0, expense_ratio=0.015)

# %%
sp500 = process_yahoo_csv("SP500.csv")
sso = process_yahoo_csv("SSO.csv")
upro = process_yahoo_csv("UPRO.csv")
vfinx = process_yahoo_csv("VFINX.csv")

upro_sim = sim_leverage(sp500, leverage=3.0, expense_ratio=0.0, initial_value=upro.iloc[500]['Close'], start_date=upro.index[500])
sso_sim = sim_leverage(sp500, leverage=2.0, expense_ratio=0.0, initial_value=sso.iloc[0]['Close'], start_date=sso.index[0])
vfinx_sim = sim_leverage(sp500, leverage=1.0, expense_ratio=0.0, initial_value=vfinx.iloc[0]['Close'], start_date=vfinx.index[0])

get_ipython().run_line_magic('matplotlib', 'widget')
# plt.plot((upro_sim['Close']- upro['Close'])/upro['Close'])
plt.plot(pd.concat((vfinx['Close'], vfinx_sim['Close']),axis=1))
plt.legend(['real', 'sim'])
# plt.title('1.5% expense')
plt.yscale('log')

# # %%
# # resample to month
# upro_sim = upro_sim.groupby(pd.Grouper(freq="M")).last()
# tmf_sim = tmf_sim.groupby(pd.Grouper(freq="M")).last()
# tqqq_sim = tqqq_sim.groupby(pd.Grouper(freq="M")).last()
# vfinx = vfinx.groupby(pd.Grouper(freq="M")).last()
# vustx = vustx.groupby(pd.Grouper(freq="M")).last()
# nasdaq = nasdaq.groupby(pd.Grouper(freq="M")).last()


 # %%
sp500 = process_yahoo_csv("SP500.csv")
sso = process_yahoo_csv("SSO.csv")
upro = process_yahoo_csv("UPRO.csv")
start = datetime.datetime(2018, 1, 1)

sp500 = sp500[sp500.index >= start]
val = sp500['Close']
sp500_pct_change = (val - val.shift(1)) / val.shift(1)
sso = sso[sso.index >= start]
val = sso['Close']
sso_pct_change = (val - val.shift(1)) / val.shift(1)
upro = upro[upro.index >= start]
val = upro['Close']
upro_pct_change = (val - val.shift(1)) / val.shift(1)

get_ipython().run_line_magic('matplotlib', 'widget')
cost = 0.15 / 252
ratio = (sso_pct_change + cost)/sp500_pct_change
ratio = ratio[np.logical_and(ratio > -5,  ratio < 5)] 
print('median', ratio.median())
print('mean', ratio.mean())
cost = 0.35 / 252
ratio = (upro_pct_change + cost)/sp500_pct_change
ratio = ratio[np.logical_and(ratio > -5,  ratio < 5)] 
print('median', ratio.median())
print('mean', ratio.mean())
# plt.plot(ratio)



# %%
def assetAllocBacktest(assets_data_all, start_date, end_date, asset_alloc=None, rebal_days=1, rebal_abs=0.02, rebal_rel=0.1, start_balance=1.0, commission=10, bidAskSpread=0.001):

    rebal_count = 0    
    asset_alloc = rebal_3x(0.0)
    if asset_alloc is None:
        asset_alloc = [1.0 / len(assets_data_all)]*len(assets_data_all)
    asset_alloc = np.array(asset_alloc)
    asset_alloc_base = asset_alloc
    asset_data = []
    for asset in assets_data_all:
        asset['52WeekHigh'] = asset['Close'].rolling(365,min_periods=1).max()
        asset_data.append(asset[(asset.index >= start_date) & (asset.index <= end_date)])

    for k in range(len(asset_data)):
        idx = asset_data[0].index[0]
        asset_data[k].loc[idx, 'Volume'] = (start_balance * asset_alloc[k] / asset_data[k].loc[idx, 'Close'])
    
    rebal_required = False
    last_rebal_time = assets_data_all[-1].index[0] - datetime.timedelta(days=rebal_days)
    total_high = 0.0
    max_drop = 0.0
    total_fees = 0.0
    for i in range(1, len(asset_data[0])):
        idx_prev = asset_data[0].index[i-1]
        idx = asset_data[0].index[i]
        total_asset = 0
        assets_value = []
        for asset in asset_data:
            asset_value = asset.loc[idx_prev, 'Volume'] * asset.loc[idx, 'Close']
            assets_value.append(asset_value)
            total_asset += asset_value

        if total_asset > total_high:
            total_high = total_asset
        drop = (total_high - total_asset) / total_high
        sp500 = asset_data[1].loc[idx]
        sp500Drop = max(0.0, (1 - sp500['Close']/sp500['52WeekHigh']))
        asset_alloc = rebal_3x(sp500Drop)
        if drop > max_drop:
            max_drop = drop

        if (idx - last_rebal_time).days >= rebal_days:
            # checking if rebalance is needed
            alloc_err = np.abs(np.array([val / total_asset for val in assets_value]) - asset_alloc)
            rebal_required = np.any(np.logical_and(alloc_err > rebal_abs, 
                                                   alloc_err > (asset_alloc * rebal_rel)))
            # if rebal_required:
            #     v = assets_value
            #     v = [v[0]/(v[0]+v[1]), v[1]/(v[0]+v[1]), v[0]/(v[0]+v[2]), v[2]/(v[0]+v[2])]
            #     a = asset_alloc
            #     a = [a[0]/(a[0]+a[1]), a[1]/(a[0]+a[1]), a[0]/(a[0]+a[2]), a[2]/(a[0]+a[2])]
            #     print(["%.3f"%item for item in [val / total_asset for val in assets_value]], ["%.3f"%item for item in asset_alloc], "%.3f"%sp500Drop, ["%.3f"%(a-v) for a, v in zip(a,v)], ["%.3f"%item for item in a])

        if rebal_required:
            last_rebal_time = idx
            rebal_count += 1
            rebal_required = False
            fees = total_asset * alloc_err.sum() * 0.0005 + 10 * sum(alloc_err > rebal_abs)
            total_asset -= fees
            total_fees += fees
            for asset, alloc in zip(asset_data, asset_alloc):
                asset.loc[idx, 'Volume'] = max(0, (total_asset * alloc / asset.loc[idx, 'Close']))
        else:
            for asset in asset_data:
                asset.loc[idx, 'Volume'] = asset.loc[idx_prev, 'Volume']

    val = 0
    for asset in asset_data:
        val += asset['Volume'] * asset['Close']
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
def rebal_3x(drop):
    # return np.array([0.0, 1.0, 0.0])
    # return np.array([0.4, 0.3, 0.3])
    # return np.array([0.5, 0.0, 0.5])
    x3_x0 = min(0.8, (1.0)/(1.0+2/3 + 0.5*drop - 2*drop))
    # x3_x0 = min(0.8, (1.0)/(1.0+3.2/4 + 0.0*drop - 2*drop))
    x3_x1 = min(0.7, (1.0)/((1.0+4.0/4.0) * (1 - drop)))
    # x3_x0 = min(0.85, (1.0-drop)/(1.0+3.0/4.0 - 3*drop))
    # x3_x1 = 0.57

    x3 = 1/(1+(1/x3_x1-1)+(1/x3_x0-1))
    x1 = (1/x3_x1-1)*x3
    x0 = (1/x3_x0-1)*x3
    return np.array([x3, x1, x0/2, x0/2])

# %%
start = datetime.datetime(2013, 2, 1)
end = datetime.datetime(2019, 2, 1)
start = datetime.datetime(2012, 7, 1)
# start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 7, 14)

portfolio, max_drop, rebal_count, fees = assetAllocBacktest([upro, spy, edv, tmf],
    start_date=start, end_date=end, rebal_days=0, rebal_abs=0.05, rebal_rel=0.0001,
    start_balance=150000.0)
# result, max_drop = assetAllocBacktest([vfinx])
cagr, sharpe = analyze(portfolio)

print(f"Max Drawdown: {max_drop*100:.0f}%\nCAGR: {cagr:.2f}%\nSharpe: {sharpe:.3f}%\nRebalances: {rebal_count}\nFees: {fees:.0f}")

# %%

# %%
bt_result = []
for start_year in range(1992, 2020):
    for end_year in range(start_year+5, 2021, 5):

        start = datetime.datetime(start_year-1, 12, 31)
        end = datetime.datetime(end_year, 1, 31)

        for pct_equity in [20, 30, 40, 50, 60]:
            ratio_equity = pct_equity/100.0
            asset_alloc=[(1-ratio_equity)*0.5, (1-ratio_equity)*0.5, ratio_equity*0.7, ratio_equity*0.3]
            portfolio, max_drop = assetAllocBacktest([tmf_sim, tyd_sim, upro_sim, tqqq_sim], 
                asset_alloc=asset_alloc, start_date=start, end_date=end, rebal_days=1, rebal_abs=0.02, rebal_rel=0.2,
                start_balance=30000.0)
            cagr, sharpe = analyze(portfolio)
            bt_result.append({'start':start, 'end': end, 'cagr': cagr, 
                              'dd':max_drop, 'sharpe':sharpe, 'pct_equity':pct_equity})
            print(f"Start {start_year}, End {end_year}, %eq %{pct_equity}, Max Drawdown: {max_drop*100:.0f}, CAGR: {cagr:.2f}, Sharpe: {sharpe:.3f}")

bt_result = pd.DataFrame(bt_result)
bt_result.to_csv('bench_result_daily.csv')


# %%
bt_result = pd.read_csv('bt_result_monthly.csv', parse_dates=True, index_col=0)
bt_result['start'] = pd.to_datetime(bt_result['start'],format='%Y-%m-%d')
bt_result['end'] = pd.to_datetime(bt_result['end'],format='%Y-%m-%d')


# %%
bt_result.tail()


# %%
bt_result['horizon'] = (bt_result['end'].dt.year - bt_result['start'].dt.year)
bt_horizon = bt_result[(bt_result['horizon'] % 5 ==0) & (bt_result['horizon'] <= 20)]
horizon5 = bt_result[(bt_result['horizon'] == 5)]


# %%
get_ipython().run_line_magic('matplotlib', 'widget')
bt_horizon.sort_values('dd').drop_duplicates(['start', 'horizon'],keep='first')['pct_equity'].hist(by=bt_horizon['horizon'], bins=range(-5,106,10))


# %%
get_ipython().run_line_magic('matplotlib', 'widget')
bt_horizon.sort_values('cagr').drop_duplicates(['start', 'horizon'],keep='last')['pct_equity'].hist(by=bt_horizon['horizon'], bins=range(-5,106,10))


# %%
get_ipython().run_line_magic('matplotlib', 'widget')
bt_horizon.sort_values('sharpe').drop_duplicates(['start', 'horizon'],keep='last')['pct_equity'].hist(by=bt_horizon['horizon'], bins=range(-5,106,10))


# %%
bt_horizon[(bt_horizon['pct_equity']==20)]['dd'].hist(by=bt_horizon['horizon'], bins=range(8,53,4))


# %%
bt_horizon[(bt_horizon['pct_equity']==40)]['cagr'].hist(by=bt_horizon['horizon'], bins=range(-2,43,4))


# %%
bt_horizon[(bt_horizon['pct_equity']==100)]['cagr'].hist(by=bt_horizon['horizon'])#, bins=range(-2,43,4))


# %%
bt_horizon[(bt_horizon['pct_equity']==100)]['dd'].hist(by=bt_horizon['horizon'])#, bins=range(8,53,4))


# %%


