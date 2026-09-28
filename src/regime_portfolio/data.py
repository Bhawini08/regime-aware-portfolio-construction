from __future__ import annotations
import numpy as np
import pandas as pd

ASSETS=["SPY","EFA","EEM","IEF","TLT","LQD","HYG","GLD","DBC","VNQ"]

def synthetic_dataset(n_months=240, seed=11):
    rng=np.random.default_rng(seed)
    idx=pd.date_range("2006-01-31", periods=n_months, freq="ME")

    # Persistent latent macro states.
    growth=np.zeros(n_months); inflation=np.zeros(n_months)
    rates=np.zeros(n_months); vol=np.zeros(n_months)
    for t in range(1,n_months):
        growth[t]=0.80*growth[t-1]+rng.normal(0,.55)
        inflation[t]=0.86*inflation[t-1]+rng.normal(0,.45)
        rates[t]=0.82*rates[t-1]+0.20*inflation[t]+rng.normal(0,.35)
        vol[t]=0.72*vol[t-1]-0.20*growth[t]+rng.normal(0,.50)

    features=pd.DataFrame({
        "growth":growth,
        "inflation":inflation,
        "rates":rates,
        "volatility":vol,
    },index=idx)

    # Asset loadings on macro surprises plus idiosyncratic noise.
    B=np.array([
        [.018,-.004,-.004,-.010], # SPY
        [.016,-.004,-.003,-.009],
        [.020,-.006,-.005,-.012],
        [-.004,-.010,-.015,.002], # IEF
        [-.006,-.015,-.025,.004], # TLT
        [.005,-.007,-.010,-.004],
        [.012,-.006,-.007,-.010],
        [.002,.014,-.006,.002],  # GLD
        [.006,.020,.006,.004],   # DBC
        [.013,-.006,-.012,-.010],
    ])
    base=np.array([.006,.005,.006,.0025,.003,.003,.0045,.004,.004,.005])
    X=np.column_stack([
        np.tanh(growth),np.tanh(inflation),np.tanh(rates),np.tanh(vol)
    ])
    eps=rng.normal(0, np.array([.035,.038,.045,.012,.025,.016,.025,.035,.045,.040]), size=(n_months,len(ASSETS)))
    returns=base + X@B.T + eps
    return pd.DataFrame(returns,index=idx,columns=ASSETS),features

def live_dataset(start="2007-01-01", end=None):
    import yfinance as yf
    from pandas_datareader import data as web

    raw=yf.download(ASSETS,start=start,end=end,auto_adjust=True,progress=False,threads=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no asset data")
    px=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]]
    px.columns=[str(c).upper() for c in px.columns]
    monthly_px=px.resample("ME").last()
    returns=monthly_px.pct_change(fill_method=None).dropna(how="any")

    fred=web.DataReader(["INDPRO","CPIAUCSL","DGS10","VIXCLS"],"fred",start,end)
    monthly=pd.DataFrame(index=returns.index)
    monthly["growth_raw"]=fred["INDPRO"].resample("ME").last().pct_change(12)
    monthly["inflation_raw"]=fred["CPIAUCSL"].resample("ME").last().pct_change(12)
    monthly["rates_raw"]=fred["DGS10"].resample("ME").mean()/100.0
    monthly["volatility_raw"]=fred["VIXCLS"].resample("ME").mean()/100.0

    # CPI and industrial production are shifted one month to approximate publication lag.
    monthly["growth"]=monthly["growth_raw"].shift(1)
    monthly["inflation"]=monthly["inflation_raw"].shift(1)
    # Market-observable rate and volatility measures are known by month end.
    monthly["rates"]=monthly["rates_raw"]
    monthly["volatility"]=monthly["volatility_raw"]
    features=monthly[["growth","inflation","rates","volatility"]]
    aligned=returns.join(features,how="inner").dropna()
    return aligned[ASSETS],aligned[["growth","inflation","rates","volatility"]]
