import numpy as np
from regime_portfolio.data import synthetic_dataset
from regime_portfolio.regimes import classify_regimes
from regime_portfolio.backtest import walk_forward
from regime_portfolio.portfolio import conditional_moments

def test_regime_labels_are_lagged_and_nonempty():
    r,f=synthetic_dataset(180,seed=2)
    s=classify_regimes(f)
    assert s["regime_code"].notna().sum()>100

def test_conditional_moments_shapes():
    r,f=synthetic_dataset(180,seed=3); s=classify_regimes(f)
    d=r.index[100]
    mu,cov,n=conditional_moments(r,s,d,min_history=60)
    assert len(mu)==r.shape[1]
    assert cov.shape==(r.shape[1],r.shape[1])
    assert n>=0

def test_walk_forward_weights_and_outputs():
    r,f=synthetic_dataset(180,seed=4); s=classify_regimes(f)
    perf,w,t,summary=walk_forward(r,s,min_history=60)
    assert len(perf)>50
    sums=w.drop(columns=["date","strategy","regime","regime_obs"]).sum(axis=1)
    assert np.allclose(sums,1,atol=1e-5)
    assert np.isfinite(summary["sharpe"]).all()
