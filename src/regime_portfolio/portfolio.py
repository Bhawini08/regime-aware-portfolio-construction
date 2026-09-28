from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def covariance(x:pd.DataFrame)->pd.DataFrame:
    # diagonal shrinkage for stability
    s=x.cov()
    target=pd.DataFrame(np.diag(np.diag(s)),index=s.index,columns=s.columns)
    return 0.75*s+0.25*target

def risk_parity_weights(cov:pd.DataFrame,max_weight=.30)->pd.Series:
    C=cov.to_numpy(float); n=len(C)
    def obj(w):
        var=w@C@w
        rc=w*(C@w)/(var+1e-12)
        return float(((rc-rc.mean())**2).sum())
    res=minimize(obj,np.repeat(1/n,n),method="SLSQP",
                 bounds=[(0,max_weight)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                 options={"maxiter":2000,"ftol":1e-12})
    if not res.success: raise RuntimeError(res.message)
    return pd.Series(res.x,index=cov.index)

def constrained_weights(mu:pd.Series,cov:pd.DataFrame,prev=None,max_weight=.30,risk_aversion=6.0,turnover_penalty=.002):
    assets=list(mu.index); m=mu.to_numpy(float); C=cov.loc[assets,assets].to_numpy(float); n=len(assets)
    p=np.repeat(1/n,n) if prev is None else prev.reindex(assets).fillna(0).to_numpy(float)
    def obj(w):
        utility=-(w@m-risk_aversion*(w@C@w))
        turnover=turnover_penalty*np.sum(np.sqrt((w-p)**2+1e-10))
        return float(utility+turnover)
    res=minimize(obj,p,method="SLSQP",bounds=[(0,max_weight)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                 options={"maxiter":2500,"ftol":1e-11})
    if not res.success:
        x0=np.repeat(1/n,n)
        res=minimize(obj,x0,method="SLSQP",bounds=[(0,max_weight)]*n,
                     constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                     options={"maxiter":2500,"ftol":1e-11})
    if not res.success: raise RuntimeError(res.message)
    return pd.Series(res.x,index=assets)

def conditional_moments(returns,states,date,min_history=60,min_regime_obs=12):
    hist=returns.loc[returns.index<date]
    if len(hist)<min_history: return None,None,None
    full_mu=hist.mean(); full_cov=covariance(hist)
    code=states.loc[date,"regime_code"]
    prior_codes=states.loc[hist.index,"regime_code"]
    reg=hist.loc[prior_codes==code]
    n=len(reg)
    if n<3: return full_mu,full_cov,n
    reg_mu=reg.mean(); reg_cov=covariance(reg)
    alpha=min(0.75,n/(n+min_regime_obs))
    mu=alpha*reg_mu+(1-alpha)*full_mu
    cov=alpha*reg_cov+(1-alpha)*full_cov
    return mu,cov,n
