"""Aggregate raw CSVs into seed summaries and paired OD comparisons."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

BASELINES = ["NegGrad+", "Tombstone-neg", "FT-Add (blind)", "FT-Snapshot"]

def load_tag(folder: Path, tag: str):
    subdir = {"main": "main", "abl": "ablations"}.get(tag)
    files = sorted(
        path for path in folder.rglob("*.csv")
        if path.parent.name == subdir or path.name.startswith(f"{tag}__")
    )
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    group_cols = ["dataset", "model", "scenario", "seed", "step"]
    ref_cols = ["test_mrr", "added_mrr", "removed_mrr", "ret_nb_mrr", "ret_rand_mrr"]
    derived = []
    for _, group in df.groupby(group_cols):
        group = group.set_index("method").copy()
        group["OD"] = float("nan")
        group["FE"] = float("nan")
        group["speedup"] = float("nan")
        if "Retrain (oracle)" in group.index:
            oracle = group.loc["Retrain (oracle)"]
            group["OD"] = group[ref_cols].astype(float).sub(oracle[ref_cols].astype(float)).abs().mean(axis=1, skipna=True)
            group["speedup"] = oracle["time"] / group["time"].replace(0, float("nan"))
            if "Stale" in group.index:
                stale_mrr = group.loc["Stale", "removed_mrr"]
                oracle_mrr = oracle["removed_mrr"]
                if pd.notna(stale_mrr) and abs(stale_mrr - oracle_mrr) > 1e-6:
                    group["FE"] = (stale_mrr - group["removed_mrr"]) / (stale_mrr - oracle_mrr)
        derived.append(group.reset_index())
    return pd.concat(derived, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("data/results_snapshot"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tag in ("main", "scen", "abl", "sens"):
        df = load_tag(args.input_dir, tag)
        if df is None:
            continue
        df.to_csv(args.output_dir / f"{tag}_derived.csv", index=False)
        final = df[df.step == df.groupby(["dataset", "model", "scenario", "seed"])["step"].transform("max")]
        metrics = [c for c in ["test_mrr", "added_mrr", "removed_mrr", "removed_cum_mrr",
                               "ret_nb_mrr", "ret_rand_mrr", "sub_acc", "FE", "OD", "speedup"] if c in final]
        summary = final.groupby(["dataset", "model", "scenario", "method"])[metrics].agg(["mean", "std"])
        summary.to_csv(args.output_dir / f"{tag}_summary.csv")
        if tag == "main":
            paired = final.pivot_table(index=["dataset", "model", "scenario", "seed"], columns="method", values="OD")
            rows=[]
            for baseline in BASELINES:
                if baseline not in paired or "REVISE" not in paired:
                    continue
                diff=(paired[baseline]-paired["REVISE"]).dropna()
                try: p=float(wilcoxon(diff, alternative="greater").pvalue)
                except ValueError: p=float("nan")
                rows.append({"baseline":baseline,"n":len(diff),"win_rate":float((diff>0).mean()),
                             "median_diff":float(diff.median()),"p":p})
            rows.sort(key=lambda row: row["p"] if pd.notna(row["p"]) else 1.0)
            k=len(rows)
            running_max = 0.0
            for i,row in enumerate(rows):
                if pd.notna(row["p"]):
                    running_max = max(running_max, min(1.0, row["p"] * (k-i)))
                    row["p_holm"] = running_max
                else:
                    row["p_holm"] = float("nan")
            pd.DataFrame(rows).to_csv(args.output_dir/"main_pairwise_od.csv",index=False)
            rank=final.groupby(["dataset","model","scenario","method"]).OD.mean().unstack()
            rank=rank.drop(columns=["Retrain (oracle)","Stale"],errors="ignore").rank(axis=1)
            rank.mean().sort_values().rename("average_rank").to_csv(args.output_dir/"main_average_od_rank.csv")
    print(f"Wrote summaries to {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
