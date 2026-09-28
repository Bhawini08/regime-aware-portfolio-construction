import argparse,json
from pathlib import Path
from regime_portfolio.data import synthetic_dataset,live_dataset
from regime_portfolio.regimes import classify_regimes
from regime_portfolio.backtest import walk_forward

p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["synthetic","live"],default="synthetic"); args=p.parse_args()
returns,features=synthetic_dataset() if args.mode=="synthetic" else live_dataset()
states=classify_regimes(features)
perf,weights,trades,summary=walk_forward(returns,states)

out=Path("results"); out.mkdir(exist_ok=True)
features.to_csv(out/f"{args.mode}_features.csv")
states.to_csv(out/f"{args.mode}_regimes.csv")
perf.to_csv(out/f"{args.mode}_strategy_returns.csv")
weights.to_csv(out/f"{args.mode}_weights.csv",index=False)
trades.to_csv(out/f"{args.mode}_trades.csv",index=False)
summary.to_csv(out/f"{args.mode}_summary.csv",index=False)
occupancy=states["regime_code"].value_counts(dropna=False).rename_axis("regime").reset_index(name="months")
occupancy.to_csv(out/f"{args.mode}_regime_occupancy.csv",index=False)
print(summary.to_json(orient="records",indent=2))
