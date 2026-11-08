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
ax.set_title("Wider promise, more trips kept, with diminishing returns"); fig.tight_layout()
fig.savefig(DOCS / "width_vs_coverage.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
names = ["One number\n(mean)", "One number\n(median)", "Range\nP20–P80", "Range\nP10–P90"]
keys = ["point_mean (one number)", "point_median P50 (one number)", "range P20-P80", "range P10-P90"]
vals = [res[k]["late_pct"] for k in keys]
bars = ax.bar(names, vals, color=["#9ca3af", "#9ca3af", "#2563eb", "#1e40af"])
ax.bar_label(bars, fmt="%.0f%%"); ax.set_ylabel("Orders later than promised (%)")
ax.set_title("Share of trips that arrive after the promised time"); fig.tight_layout()
fig.savefig(DOCS / "late_share.png", dpi=150)

# ---------------------------------------------------------------------------
# Split conformal calibration (conformalized quantile regression, CQR).
# Train quantile models on Jan 1-17, calibrate on Jan 18-21, test on Jan 22-31.
# Conformity score E = max(lo - y, y - hi); widen both bounds by the
# ceil((n+1)(1-a))/n quantile of E, which gives >= 1-a coverage under exchangeability.
fit_cut = df.lpep_pickup_datetime < "2024-01-18"
ptr, cal = df[fit_cut & cut], df[~fit_cut & cut]
def fit2(q):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                      categorical_features=[3], random_state=0).fit(ptr[X_cols], ptr.minutes)
    return m.predict(cal[X_cols]), m.predict(te[X_cols])
yc = cal.minutes.values
def qhat_for(a, lo_c, hi_c):
    s = np.maximum(lo_c - yc, yc - hi_c); n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher"))
def cov(lo, hi, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    return {"coverage_pct": round(100 * np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])), 1),
            "late_pct": round(100 * np.mean(y[m] > hi[m]), 1),
            "avg_width_min": round(float(np.mean(hi[m] - lo[m])), 1), "n": int(m.sum())}
lo_c, lo_t = fit2(0.1); hi_c, hi_t = fit2(0.9)
qh = qhat_for(0.2, lo_c, hi_c)
lo_k, hi_k = np.maximum(lo_t - qh, 0), hi_t + qh
band_of = lambda h: pd.cut(h, [-1, 5, 9, 15, 19, 23], labels=["night", "am_rush", "midday", "pm_rush", "evening"])
te_band = np.asarray(band_of(te.hour)).astype(str)
dlabels = ["0-1 mi", "1-2 mi", "2-4 mi", "4-8 mi", "8-20 mi"]
te_dist = np.asarray(pd.cut(te.trip_distance, [0, 1, 2, 4, 8, 20], labels=dlabels)).astype(str)
segs = [("overall", "all", None)] + [("hour", b, te_band == b) for b in ["night", "am_rush", "midday", "pm_rush", "evening"]] \
       + [("distance", d, te_dist == d) for d in dlabels]
calib = {"target_coverage_pct": 80, "rows_fit": int(len(ptr)), "rows_calibration": int(len(cal)), "rows_test": int(len(te)),
         "qhat_min": round(qh, 2), "segments": []}
for kind, name, m in segs:
    calib["segments"].append({"kind": kind, "segment": name, "before": cov(lo_t, hi_t, m), "after": cov(lo_k, hi_k, m)})
# P20-P80 conformal shift too (for the demo toggle)
lo2c, lo2t = fit2(0.2); hi2c, hi2t = fit2(0.8)
qh60 = qhat_for(0.4, lo2c, hi2c)
calib["p20_p80"] = {"target_coverage_pct": 60, "qhat_min": round(qh60, 2),
                    "before": cov(lo2t, hi2t), "after": cov(np.maximum(lo2t - qh60, 0), hi2t + qh60)}

# Trust vs conversion frontier. ASSUMPTION (not measured): each extra minute of the
# upper bound vs the one-number median quote costs 0.5% of orders, relative.
ELASTICITY = -0.5
front = []
for t in [50, 60, 70, 75, 80, 85, 90, 95]:
    a = 1 - t / 100
    lc, lt_ = fit2(a / 2); hc, ht_ = fit2(1 - a / 2); q = qhat_for(a, lc, hc)
    hi_f = ht_ + q; extra = float(np.mean(hi_f - preds[.5]))
    front.append({"target_pct": t, "on_time_pct": round(100 * np.mean(y <= hi_f), 1),
                  "coverage_pct": round(100 * np.mean((y >= np.maximum(lt_ - q, 0)) & (y <= hi_f)), 1),
                  "extra_upper_min_vs_median": round(extra, 2),
                  "assumed_conversion_change_pct": round(ELASTICITY * extra, 2)})
calib["frontier"] = {"assumed_elasticity_pct_per_min": ELASTICITY,
                     "note": "ASSUMPTION: elasticity is illustrative, not measured", "points": front}
res["conformal"] = calib
(DOCS / "results.json").write_text(json.dumps(res, indent=2)); print(json.dumps(calib, indent=2))
(DOCS / "calib.js").write_text("window.CALIB=" + json.dumps({"q10": round(qh, 2), "q20": round(qh60, 2),
    "p10": {"before": calib["segments"][0]["before"], "after": calib["segments"][0]["after"]},
    "p20": {"before": calib["p20_p80"]["before"], "after": calib["p20_p80"]["after"]}, "frontier": front}) + ";")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 1.3]})
S = calib["segments"]; xs = np.arange(len(S)); w = 0.38
axs[0].bar(xs - w/2, [s["before"]["coverage_pct"] for s in S], w, color="#9ca3af", label="Raw P10-P90")
axs[0].bar(xs + w/2, [s["after"]["coverage_pct"] for s in S], w, color="#1e40af", label="Conformal (CQR)")
axs[0].axhline(80, color="#b91c1c", ls="--", lw=1); axs[0].text(len(S) - .5, 80.6, "80% target", color="#b91c1c", ha="right", fontsize=9)
axs[0].set_xticks(xs, [s["segment"].replace("_", " ") for s in S], rotation=35, ha="right", fontsize=9)
axs[0].set_ylim(60, 95); axs[0].set_ylabel("Test-week trips inside range (%)"); axs[0].legend(loc="lower left", fontsize=9, frameon=False)
axs[0].set_title("Coverage by segment, before vs after calibration")
o = S[0]; axs[1].bar(["Raw", "Conformal"], [o["before"]["avg_width_min"], o["after"]["avg_width_min"]], color=["#9ca3af", "#1e40af"])
axs[1].bar_label(axs[1].containers[0], fmt="%.1f min"); axs[1].set_title("Avg width"); axs[1].set_ylabel("Minutes")
fig.tight_layout(); fig.savefig(DOCS / "calibration.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot([-p["assumed_conversion_change_pct"] for p in front], [p["on_time_pct"] for p in front], "o-", color="#1e40af")
for p in front: ax.annotate(f"{p['target_pct']}%", (-p["assumed_conversion_change_pct"], p["on_time_pct"]),
                            textcoords="offset points", xytext=(6, -12), fontsize=9, color="#555")
ax.set_xlabel("Assumed conversion loss (%), at ASSUMED -0.5% per extra minute"); ax.set_ylabel("Promise kept (%)")
ax.set_title("Trust vs conversion frontier (conformal ranges; elasticity assumed)", fontsize=11); fig.tight_layout()
fig.savefig(DOCS / "frontier.png", dpi=150)
