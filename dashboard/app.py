import sys
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from regime_portfolio.data import synthetic_dataset,live_dataset
from regime_portfolio.regimes import classify_regimes
from regime_portfolio.backtest import walk_forward

st.set_page_config(page_title="Regime-Aware Portfolio Construction",layout="wide")
st.title("Building Portfolios for Regimes, Not Averages")
mode=st.sidebar.selectbox("Data",["synthetic","live"])
r,f=synthetic_dataset() if mode=="synthetic" else live_dataset()
s=classify_regimes(f)
perf,w,trades,summary=walk_forward(r,s)

st.dataframe(summary,use_container_width=True)
st.subheader("Cumulative performance")
st.line_chart((1+perf).cumprod())
c1,c2=st.columns(2)
with c1:
    st.subheader("Regime occupancy")
    st.bar_chart(s["regime_code"].value_counts())
with c2:
    st.subheader("Adaptive turnover")
    st.line_chart(trades.loc[trades.strategy=="adaptive_regime"].set_index("date")["turnover"])
st.subheader("Latest adaptive weights")
latest=w.loc[w.strategy=="adaptive_regime"].iloc[-1]
st.dataframe(latest.drop(labels=["date","strategy","regime","regime_obs"]).rename("weight"))
