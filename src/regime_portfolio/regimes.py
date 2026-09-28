from __future__ import annotations
import pandas as pd

FEATURES=["growth","inflation","rates","volatility"]

def expanding_standardize(features:pd.DataFrame,min_periods=36)->pd.DataFrame:
    out=pd.DataFrame(index=features.index)
    for c in FEATURES:
        mean=features[c].expanding(min_periods=min_periods).mean().shift(1)
        std=features[c].expanding(min_periods=min_periods).std().shift(1)
        out[c]=(features[c].shift(1)-mean)/std.replace(0,float("nan"))
    return out

def classify_regimes(features:pd.DataFrame,min_periods=36)->pd.DataFrame:
    z=expanding_standardize(features,min_periods)
    states=pd.DataFrame(index=features.index)
    states["growth_state"]=(z["growth"]>=0).map({True:"high",False:"low"})
    states["inflation_state"]=(z["inflation"]>=0).map({True:"high",False:"low"})
    states["rate_state"]=(z["rates"]>=0).map({True:"high",False:"low"})
    states["vol_state"]=(z["volatility"]>=0).map({True:"high",False:"low"})
    states.loc[z.isna().any(axis=1),:]=None
    states["macro_regime"]=states["growth_state"].fillna("na")+" growth / "+states["inflation_state"].fillna("na")+" inflation"
    states["regime_code"]=(
        states["growth_state"].fillna("na").str[0]+
        states["inflation_state"].fillna("na").str[0]+
        states["rate_state"].fillna("na").str[0]+
        states["vol_state"].fillna("na").str[0]
    )
    return states
