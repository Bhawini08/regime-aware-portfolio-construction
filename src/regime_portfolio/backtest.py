from __future__ import annotations
import numpy as np
import pandas as pd
from .portfolio import covariance,risk_parity_weights,constrained_weights,conditional_moments

def performance_stats(r:pd.Series, periods=12):
    r=r.dropna()
    wealth=(1+r).cumprod(); dd=wealth/wealth.cummax()-1
    ann=(wealth.iloc[-1]**(periods/len(r))-1) if len(r) else np.nan
    vol=r.std()*np.sqrt(periods)
    return {
        "annual_return":float(ann),
        "annual_vol":float(vol),
        "sharpe":float((r.mean()*periods)/(vol+1e-12)),
        "max_drawdown":float(dd.min()),
        "hit_rate":float((r>0).mean()),
    }

def walk_forward(returns,states,min_history=60,cost_bps=5.0,max_weight=.30):
    dates=returns.index
    strategies=["adaptive_regime","expanding_min_variance","risk_parity","equal_weight"]
    realized={s:[] for s in strategies}; out_dates=[]; weight_rows=[]; trade_rows=[]
    prev={s:pd.Series(1/returns.shape[1],index=returns.columns) for s in strategies}
    for i,date in enumerate(dates):
        if i<min_history or states.loc[date,"regime_code"].startswith("n"):
            continue
        hist=returns.iloc[:i]
        next_r=returns.loc[date]

        mu,cov,n_reg=conditional_moments(returns,states,date,min_history=min_history)
        if mu is None: continue
        w_ad=constrained_weights(mu,cov,prev["adaptive_regime"],max_weight=max_weight)

        full_cov=covariance(hist)
        zero_mu=pd.Series(0.0,index=returns.columns)
        # minimum variance is mean-variance with zero expected return and stronger variance penalty
        w_mv=constrained_weights(zero_mu,full_cov,prev["expanding_min_variance"],max_weight=max_weight,risk_aversion=20.0,turnover_penalty=.001)
        w_rp=risk_parity_weights(full_cov,max_weight=max_weight)
        w_eq=pd.Series(1/returns.shape[1],index=returns.columns)

        ws={"adaptive_regime":w_ad,"expanding_min_variance":w_mv,"risk_parity":w_rp,"equal_weight":w_eq}
        out_dates.append(date)
        for name,w in ws.items():
            turnover=float((w-prev[name]).abs().sum())
            cost=turnover*cost_bps/10000
            ret=float(w@next_r-cost)
            realized[name].append(ret)
            weight_rows.append({"date":date,"strategy":name,"regime":states.loc[date,"regime_code"],"regime_obs":n_reg,**w.to_dict()})
            trade_rows.append({"date":date,"strategy":name,"turnover":turnover,"cost":cost})
            prev[name]=w
    result=pd.DataFrame(realized,index=out_dates)
    weights=pd.DataFrame(weight_rows)
    trades=pd.DataFrame(trade_rows)
    summary=pd.DataFrame([{"strategy":c,**performance_stats(result[c]),"avg_turnover":float(trades.loc[trades.strategy==c,"turnover"].mean())} for c in result.columns])
    return result,weights,trades,summary
