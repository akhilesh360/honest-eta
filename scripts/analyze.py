"""Point ETA vs range ETA on real NYC green-taxi trips (Jan 2024), a proxy for delivery legs."""
import json, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW, DOCS = ROOT / "data" / "raw", ROOT / "docs"
df = pd.read_parquet(RAW / "green_tripdata_2024-01.parquet")
df["minutes"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
df = df[(df.minutes.between(1, 90)) & (df.trip_distance.between(0.1, 20))
        & (df.lpep_pickup_datetime.dt.year == 2024) & (df.lpep_pickup_datetime.dt.month == 1)].copy()
df["hour"] = df.lpep_pickup_datetime.dt.hour
df["weekday"] = df.lpep_pickup_datetime.dt.weekday
df["zone"] = df.PULocationID.astype(int)
df = df.sort_values("lpep_pickup_datetime")
df[["lpep_pickup_datetime", "hour", "weekday", "trip_distance", "zone", "minutes"]].sample(
    n=min(20000, len(df)), random_state=0).to_csv(ROOT / "data" / "sample_trips.csv", index=False)

X_cols = ["hour", "weekday", "trip_distance", "zone"]
cut = df.lpep_pickup_datetime < "2024-01-22"          # time split: train 3 weeks, test last ~10 days
tr, te = df[cut], df[~cut]
def fit(q):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                      categorical_features=[3], random_state=0)
    return m.fit(tr[X_cols], tr.minutes).predict(te[X_cols])
y = te.minutes.values
preds = {q: fit(q) for q in [0.1, 0.2, 0.5, 0.8, 0.9]}
mean_m = HistGradientBoostingRegressor(max_iter=300, categorical_features=[3], random_state=0)
point = mean_m.fit(tr[X_cols], tr.minutes).predict(te[X_cols])

def promise(lo, hi):
    late = np.clip(y - hi, 0, None)
    return {"on_time_pct": round(100 * np.mean(y <= hi), 1),
            "inside_range_pct": round(100 * np.mean((y >= lo) & (y <= hi)), 1),
            "late_pct": round(100 * np.mean(y > hi), 1),
            "avg_min_late_when_late": round(float(late[late > 0].mean()), 1),
            "avg_width_min": round(float(np.mean(hi - lo)), 1)}
res = {"rows_train": int(len(tr)), "rows_test": int(len(te)),
       "point_mean (one number)": promise(point, point),
       "point_median P50 (one number)": promise(preds[.5], preds[.5]),
       "range P20-P80": promise(preds[.2], preds[.8]),
       "range P10-P90": promise(preds[.1], preds[.9])}

# width vs coverage curve: symmetric quantile pairs
curve = []
for a in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    lo, hi = fit(a), fit(1 - a)
    p = promise(lo, hi); curve.append({"nominal": round(100 * (1 - 2 * a)), **p})
res["width_vs_coverage"] = curve
(DOCS / "results.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))

# demo lookup: empirical P10/P20/P50/P80/P90 by distance bin x hour band (all January data)
bins = [0, 1, 2, 3, 4, 6, 8, 12, 20]
df["dbin"] = pd.cut(df.trip_distance, bins, labels=False)
df["band"] = pd.cut(df.hour, [-1, 5, 9, 15, 19, 23], labels=["night", "am_rush", "midday", "pm_rush", "evening"])
tab = df.groupby(["dbin", "band"], observed=True).minutes.quantile([.1, .2, .5, .8, .9]).unstack().round(1)
cnt = df.groupby(["dbin", "band"], observed=True).size()
lookup = {"bins": bins, "cells": [{"dbin": int(d), "band": b, "n": int(cnt[(d, b)]),
          "p10": r[.1], "p20": r[.2], "p50": r[.5], "p80": r[.8], "p90": r[.9]} for (d, b), r in tab.iterrows()]}
(DOCS / "eta_lookup.json").write_text(json.dumps(lookup))
(DOCS / "eta_lookup.js").write_text("window.ETA_LOOKUP=" + json.dumps(lookup) + ";")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot([c["avg_width_min"] for c in curve], [c["inside_range_pct"] for c in curve], "o-", color="#2563eb")
for c in curve: ax.annotate(f"P{50-c['nominal']//2}-P{50+c['nominal']//2}", (c["avg_width_min"], c["inside_range_pct"]),
                            textcoords="offset points", xytext=(6, -12), fontsize=9, color="#555")
ax.set_xlabel("Average range width (minutes)"); ax.set_ylabel("Trips arriving inside range (%)")
ax.set_title("Wider promise, more trips kept — with diminishing returns"); fig.tight_layout()
fig.savefig(DOCS / "width_vs_coverage.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
names = ["One number\n(mean)", "One number\n(median)", "Range\nP20–P80", "Range\nP10–P90"]
keys = ["point_mean (one number)", "point_median P50 (one number)", "range P20-P80", "range P10-P90"]
vals = [res[k]["late_pct"] for k in keys]
bars = ax.bar(names, vals, color=["#9ca3af", "#9ca3af", "#2563eb", "#1e40af"])
ax.bar_label(bars, fmt="%.0f%%"); ax.set_ylabel("Orders later than promised (%)")
ax.set_title("Share of trips that arrive after the promised time"); fig.tight_layout()
fig.savefig(DOCS / "late_share.png", dpi=150)
