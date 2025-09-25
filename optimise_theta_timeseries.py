#!/usr/bin/env python3
# optimise_theta_timeseries.py
# Usage: run from ~/Documents where tick_data.csv resides.

import os, time
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
import csv
from datetime import timedelta, datetime

# -------------------------
# CONFIGURATION (tune these)
# -------------------------
DOCS = os.path.expanduser("~/Documents")
TICK_PATH = os.path.join(DOCS, "tick_data.csv")
OUT_PATH = os.path.join(DOCS, "theta_y_timeseries.csv")
LOG_PATH = os.path.join(DOCS, "theta_y_log.csv")

TRAIN_WINDOW_MINUTES = 240       # training window length (e.g., 240 = 4 hours)
UPDATE_INTERVAL_MINUTES = 60     # how often to produce a new theta,Y (e.g., 60 = hourly)
N_GP_CALLS = 30                  # gp_minimize calls per window (decrease to speed up)
N_INITIAL_POINTS = max(5, int(N_GP_CALLS * 0.2))
N_CHUNKS = 4                     # parallel chunks per objective evaluation
DOWNSAMPLE = 5                   # sample every DOWNSAMPLE-th tick inside each window for faster simulation
MIN_TRADES_IN_WINDOW = 3         # skip windows if too few trades
THETA_BOUNDS = (5e-5, 5e-3)      # boundaries for theta
Y_BOUNDS = (0.1, 1.0)            # boundaries for Y
Y_DEFAULT_FOR_LABEL = 0.5        # use Y=0.5 to label DC events (per assignment)
N_REGIMES = 2                    # GMM regimes
RANDOM_STATE = 42

# -------------------------
# Utilities & simulators
# -------------------------
def load_tick_data(path):
    if not os.path.exists(path):
        raise SystemExit(f"Tick data not found at {path}. Run TickDumper in cTrader first.")
    df = pd.read_csv(path, parse_dates=["utc_iso"])
    df = df.sort_values("utc_iso").reset_index(drop=True)
    if not {"bid","ask"}.issubset(df.columns):
        raise SystemExit("tick_data.csv must contain 'bid' and 'ask' columns.")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df

def simulate_dc_on_series(mid, bid, ask, theta, Y):
    """
    Simulate DC strategy on provided arrays (numpy arrays).
    Returns (pnl, trades).
    Uses mid for detection; uses ask for long entry (we assume ask provided), bid for long exit.
    This is a simplified P&L measure (price difference sum).
    """
    if len(mid) < 2:
        return 0.0, 0
    ref = mid[0]
    direction = 0
    position = None
    entry_price = 0.0
    max_extrema = 0.0
    pnl = 0.0
    trades = 0
    for i in range(1, len(mid)):
        m = mid[i]; b = bid[i]; a = ask[i]
        if position is None:
            if direction != 1 and m >= ref * (1.0 + theta):
                direction = 1
                position = 'long'
                entry_price = a
                max_extrema = entry_price
            elif direction != -1 and m <= ref * (1.0 - theta):
                direction = -1
                position = 'short'
                entry_price = b
                max_extrema = entry_price
        else:
            if position == 'long':
                if b > max_extrema:
                    max_extrema = b
                maxOSV = max(0.0, (max_extrema - entry_price) / entry_price)
                theta_p = theta * Y * np.exp(-min(maxOSV, 10.0))
                if b <= max_extrema * (1.0 - theta_p):
                    exit_price = b
                    pnl += (exit_price - entry_price)
                    trades += 1
                    position = None
                    ref = b
                    direction = -1
            else:
                if a < max_extrema:
                    max_extrema = a
                maxOSV = max(0.0, (entry_price - max_extrema) / entry_price)
                theta_p = theta * Y * np.exp(-min(maxOSV, 10.0))
                if a >= max_extrema * (1.0 + theta_p):
                    exit_price = a
                    pnl += (entry_price - exit_price)
                    trades += 1
                    position = None
                    ref = a
                    direction = 1
    return pnl, trades

def detect_dc_events_and_features(mid, bid, ask, theta, Y_label=Y_DEFAULT_FOR_LABEL):
    """
    Detect DC events using theta and produce event-level features + label (profitable or not),
    using Y_label to determine exit (for labeling).
    Returns list of dicts: {entry_idx, entry_time, entry_price, exit_price, profit, maxOSV, duration_ticks, pre_vol}
    """
    events = []
    if len(mid) < 2:
        return events

    ref = mid[0]
    direction = 0
    position = None
    entry_price = 0.0
    entry_idx = 0
    max_extrema = 0.0
    for i in range(1, len(mid)):
        m = mid[i]; b = bid[i]; a = ask[i]
        if position is None:
            if direction != 1 and m >= ref * (1.0 + theta):
                direction = 1
                position = 'long'
                entry_price = a
                entry_idx = i
                max_extrema = entry_price
            elif direction != -1 and m <= ref * (1.0 - theta):
                direction = -1
                position = 'short'
                entry_price = b
                entry_idx = i
                max_extrema = entry_price
        else:
            if position == 'long':
                if b > max_extrema:
                    max_extrema = b
                maxOSV = max(0.0, (max_extrema - entry_price) / entry_price)
                theta_p = theta * Y_label * np.exp(-min(maxOSV, 10.0))
                if b <= max_extrema * (1.0 - theta_p):
                    exit_price = b
                    profit = exit_price - entry_price
                    duration = i - entry_idx
                    # pre_vol: std of log returns in 50 ticks before entry (if available)
                    pre_start = max(0, entry_idx - 50)
                    lr = np.diff(np.log(mid[pre_start:entry_idx+1])) if entry_idx - pre_start >= 1 else np.array([0.0])
                    pre_vol = float(np.std(lr)) if lr.size>0 else 0.0
                    events.append({
                        "entry_idx": entry_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "profit": profit,
                        "maxOSV": maxOSV,
                        "duration": duration,
                        "pre_vol": pre_vol
                    })
                    position = None
                    ref = b
                    direction = -1
            else:
                if a < max_extrema:
                    max_extrema = a
                maxOSV = max(0.0, (entry_price - max_extrema) / entry_price)
                theta_p = theta * Y_label * np.exp(-min(maxOSV, 10.0))
                if a >= max_extrema * (1.0 + theta_p):
                    exit_price = a
                    profit = entry_price - exit_price
                    duration = i - entry_idx
                    pre_start = max(0, entry_idx - 50)
                    lr = np.diff(np.log(mid[pre_start:entry_idx+1])) if entry_idx - pre_start >= 1 else np.array([0.0])
                    pre_vol = float(np.std(lr)) if lr.size>0 else 0.0
                    events.append({
                        "entry_idx": entry_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "profit": profit,
                        "maxOSV": maxOSV,
                        "duration": duration,
                        "pre_vol": pre_vol
                    })
                    position = None
                    ref = a
                    direction = 1
    return events

# -------------------------
# Objective wrapper for speed (caching + chunked / downsampled eval)
# -------------------------
def make_objective_for_window(mid_arr, bid_arr, ask_arr, n_chunks=N_CHUNKS, downsample=DOWNSAMPLE):
    cache = {}
    # Pre-sample arrays for downsample speed (we still split into chunks)
    # But to be safe keep full arrays and let simulation chunk; downsample inside chunk
    n = len(mid_arr)
    def objective(x):
        theta = float(x[0]); Y = float(x[1])
        key = (round(theta, 12), round(Y, 6))
        if key in cache:
            return cache[key]
        # split indices into chunks (contiguous)
        idxs = np.array_split(np.arange(n), n_chunks)
        # For each chunk, build downsampled arrays
        def eval_chunk(idxs_chunk):
            if len(idxs_chunk) < 2:
                return 0.0
            idxs_ds = idxs_chunk[::max(1, downsample)]
            mid_chunk = mid_arr[idxs_ds]
            bid_chunk = bid_arr[idxs_ds]
            ask_chunk = ask_arr[idxs_ds]
            pnl, trades = simulate_dc_on_series(mid_chunk, bid_chunk, ask_chunk, theta, Y)
            # We return pnl (sum of price diffs) and trades
            return pnl
        # Parallel evaluation across cores
        results = Parallel(n_jobs=-1)(delayed(eval_chunk)(c) for c in idxs)
        mean_pnl = np.mean(results) if len(results)>0 else 0.0
        value = -mean_pnl  # gp_minimize minimizes
        cache[key] = value
        return value
    return objective

# -------------------------
# Top-level rolling optimization
# -------------------------
def rolling_optimise(df):
    start = df["utc_iso"].iloc[0]
    end = df["utc_iso"].iloc[-1]
    train_delta = pd.Timedelta(minutes=TRAIN_WINDOW_MINUTES)
    step_delta = pd.Timedelta(minutes=UPDATE_INTERVAL_MINUTES)
    t_current = start + train_delta
    results = []
    event_records = []  # for meta-label training across windows
    window_counter = 0

    print(f"Rolling optimisation from {t_current} to {end} step {step_delta} ...")
    while t_current <= end:
        window_counter += 1
        window_end = t_current
        window_start = window_end - train_delta
        win_df = df[(df["utc_iso"] > window_start) & (df["utc_iso"] <= window_end)].reset_index(drop=True)
        print(f"[Window {window_counter}] {window_start} -> {window_end}  ticks={len(win_df)}", end=" ... ")
        if len(win_df) < 50:
            print("SKIP (too few ticks)")
            results.append((window_end, np.nan, np.nan, np.nan))
            t_current += step_delta
            continue
        mid = win_df["mid"].values
        bid = win_df["bid"].values
        ask = win_df["ask"].values

        # objective with caching and parallel chunking
        objective = make_objective_for_window(mid, bid, ask, n_chunks=N_CHUNKS, downsample=DOWNSAMPLE)

        try:
            res = gp_minimize(objective,
                              [Real(THETA_BOUNDS[0], THETA_BOUNDS[1], name='theta'),
                               Real(Y_BOUNDS[0], Y_BOUNDS[1], name='Y')],
                              n_calls=N_GP_CALLS,
                              n_initial_points=N_INITIAL_POINTS,
                              random_state=RANDOM_STATE)
            best_theta, best_Y = float(res.x[0]), float(res.x[1])
            pnl_best = -res.fun
            print(f"done. best_theta={best_theta:.6e} best_Y={best_Y:.4f} pnl_approx={pnl_best:.6f}")
        except Exception as e:
            print("optimizer failed:", e)
            best_theta, best_Y = np.nan, np.nan

        # Detect events with best_theta and label them (using Y_DEFAULT_FOR_LABEL)
        events = []
        if np.isfinite(best_theta):
            events = detect_dc_events_and_features(mid, bid, ask, best_theta, Y_label=Y_DEFAULT_FOR_LABEL)
            # attach window-end timestamp and window id to each event
            for ev in events:
                ev["window_end"] = window_end
                event_records.append(ev)

        results.append((window_end, best_theta, best_Y, len(events)))
        t_current += step_delta

    # Build timeseries dataframe
    ts_rows = []
    for row in results:
        ts_rows.append({
            "timestamp": row[0],
            "theta": row[1],
            "Y": row[2],
            "n_events": row[3],
            "vol": np.nan
        })
    ts_df = pd.DataFrame(ts_rows)

    # Compute vol per window (std of log returns in the window) by re-splitting
    # (this is approximate; we recompute vol by sampling tick_data around each timestamp)
    vols = []
    for i, r in ts_df.iterrows():
        window_end = r["timestamp"]
        window_start = window_end - pd.Timedelta(minutes=TRAIN_WINDOW_MINUTES)
        win_df = df[(df["utc_iso"] > window_start) & (df["utc_iso"] <= window_end)]
        if len(win_df) < 3:
            vols.append(np.nan)
            continue
        lr = np.diff(np.log((win_df["mid"].values)))
        vols.append(float(np.std(lr)))
    ts_df["vol"] = vols

    # Regime detection on vol using GMM
    print("Running regime detection (GMM) on window vol ...")
    valid_vol = ts_df["vol"].fillna(0.0).values.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=N_REGIMES, random_state=0)
        regimes = gmm.fit_predict(valid_vol)
        ts_df["regime"] = regimes
    except Exception as e:
        print("GMM failed:", e)
        ts_df["regime"] = 0

    # Meta-label classifier training on event_records
    avg_probs = []
    if len(event_records) >= 10:
        print("Training meta-label classifier (RandomForest) on event features ...")
        ev_df = pd.DataFrame(event_records)
        # label: profit > 0
        ev_df["label"] = (ev_df["profit"] > 0).astype(int)
        # features: maxOSV, duration, pre_vol, entry_return (we approximate entry return)
        ev_df["entry_return"] = ev_df["entry_price"].pct_change().fillna(0.0)
        feats = ["maxOSV", "duration", "pre_vol", "entry_return"]
        X = ev_df[feats].fillna(0.0).values
        y = ev_df["label"].values
        try:
            clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
            clf.fit(X, y)
            # predict probabilities for each event
            probs = clf.predict_proba(X)[:, 1]
            ev_df["prob"] = probs
            # compute average probability per window
            avg_prob_by_window = ev_df.groupby("window_end")["prob"].mean().to_dict()
            # map into ts_df
            avg_probs = [float(avg_prob_by_window.get(r["timestamp"], 0.0)) for _, r in ts_df.iterrows()]
            ts_df["avg_signal_prob"] = avg_probs
            print("Meta-label training done; avg_signal_prob added to timeseries.")
        except Exception as e:
            print("Meta-label training failed:", e)
            ts_df["avg_signal_prob"] = 0.0
    else:
        print("Not enough events to train meta-label classifier. avg_signal_prob set to 0.")
        ts_df["avg_signal_prob"] = 0.0

    # Final: write out timeseries CSV (overwrite)
    print("Writing timeseries file to:", OUT_PATH)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "theta", "Y", "regime", "avg_signal_prob"])
        for _, row in ts_df.iterrows():
            iso = row["timestamp"].to_pydatetime().strftime("%Y-%m-%dT%H:%M:%SZ")
            theta = "" if pd.isna(row["theta"]) else f"{float(row['theta']):.8e}"
            Y = "" if pd.isna(row["Y"]) else f"{float(row['Y']):.6f}"
            regime = int(row["regime"]) if not pd.isna(row["regime"]) else 0
            avgp = float(row.get("avg_signal_prob", 0.0) or 0.0)
            w.writerow([iso, theta, Y, regime, f"{avgp:.6f}"])

    # append to log with timestamped run
    print("Appending to log:", LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if os.path.getsize(LOG_PATH) == 0:
            w.writerow(["utc_run_time", "timestamp", "theta", "Y", "regime", "avg_signal_prob"])
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        for _, row in ts_df.iterrows():
            iso = row["timestamp"].to_pydatetime().strftime("%Y-%m-%dT%H:%M:%SZ")
            theta = "" if pd.isna(row["theta"]) else f"{float(row['theta']):.8e}"
            Y = "" if pd.isna(row["Y"]) else f"{float(row['Y']):.6f}"
            regime = int(row["regime"]) if not pd.isna(row["regime"]) else 0
            avgp = float(row.get("avg_signal_prob", 0.0) or 0.0)
            w.writerow([now, iso, theta, Y, regime, f"{avgp:.6f}"])

    print("Done. Timeseries file produced:", OUT_PATH)
    return ts_df

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    print("Loading tick data...")
    df = load_tick_data(TICK_PATH)
    ts_df = rolling_optimise(df)
    print("Finished. Summary of output:")
    print(ts_df.head())