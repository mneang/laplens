import os
import pandas as pd


# ----------------------------
# 1) Load raw CSVs
# ----------------------------
def load_race1_data(base_path: str) -> dict:
    """
    Load the 4 key Race 1 CSVs. Uses engine='python' for safety.
    """
    files = {
        "telemetry": "R1_cota_telemetry_data.csv",
        "lap_time": "COTA_lap_time_R1.csv",
        "lap_start": "COTA_lap_start_time_R1.csv",
        "lap_end": "COTA_lap_end_time_R1.csv",
    }
    dfs = {}
    for key, file_name in files.items():
        path = os.path.join(base_path, file_name)
        try:
            df = pd.read_csv(path, engine="python")
            dfs[key] = df
            print(f"✅ Loaded {key}: {file_name}  rows={len(df)}")
        except Exception as e:
            print(f"❌ Failed to load {key}: {e}")
    return dfs


# ----------------------------
# 2) Build lap windows (start/end) per vehicle/outing/lap
# ----------------------------
def build_lap_windows(lap_start_df: pd.DataFrame, lap_end_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lap windows using start/end timestamps.
    We treat CSV columns as:
      - lap_start_df['timestamp'] -> start markers
      - lap_end_df['timestamp']   -> end markers
      - Group by ['vehicle_id','outing','lap'] to get min start, max end.
    """
    ls = lap_start_df.copy()
    le = lap_end_df.copy()

    for c in ["timestamp", "vehicle_id", "outing", "lap"]:
        if c not in ls.columns:
            raise KeyError(f"lap_start missing column: {c}")
        if c not in le.columns:
            raise KeyError(f"lap_end missing column: {c}")

    ls["timestamp"] = pd.to_datetime(ls["timestamp"], errors="coerce")
    le["timestamp"] = pd.to_datetime(le["timestamp"], errors="coerce")

    start_agg = (
        ls.groupby(["vehicle_id", "outing", "lap"], as_index=False)["timestamp"]
        .min()
        .rename(columns={"timestamp": "start_time"})
    )
    end_agg = (
        le.groupby(["vehicle_id", "outing", "lap"], as_index=False)["timestamp"]
        .max()
        .rename(columns={"timestamp": "end_time"})
    )

    lap_windows = pd.merge(
        start_agg, end_agg, on=["vehicle_id", "outing", "lap"], how="inner"
    )
    lap_windows = lap_windows[lap_windows["end_time"] >= lap_windows["start_time"]].copy()
    lap_windows = lap_windows.sort_values(["vehicle_id", "outing", "start_time"]).reset_index(drop=True)
    return lap_windows


# ----------------------------
# 3) Pivot telemetry long->wide
# ----------------------------
def pivot_telemetry_long_to_wide(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long telemetry into wide format by timestamp & vehicle/outing.
    Expects columns: ['timestamp','vehicle_id','outing','telemetry_name','telemetry_value', ...]
    Keeps other id/meta columns if present.
    """
    t = telemetry_df.copy()
    required = ["timestamp", "vehicle_id", "outing", "telemetry_name", "telemetry_value"]
    for c in required:
        if c not in t.columns:
            raise KeyError(f"telemetry missing column: {c}")

    t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")

    idx_cols = ["timestamp", "vehicle_id", "outing"]
    value_col = "telemetry_value"
    col_col = "telemetry_name"

    t_wide = (
        t.pivot_table(
            index=idx_cols,
            columns=col_col,
            values=value_col,
            aggfunc="last",
        )
        .reset_index()
    )

    t_wide.columns = [c if isinstance(c, str) else str(c) for c in t_wide.columns]
    t_wide = t_wide.sort_values(["vehicle_id", "outing", "timestamp"]).reset_index(drop=True)
    return t_wide


# ----------------------------
# 4) Assign lap to telemetry via asof (start_time <= ts <= end_time)
# ----------------------------
def assign_laps_to_telemetry(telemetry_wide: pd.DataFrame, lap_windows: pd.DataFrame) -> pd.DataFrame:
    """
    For each vehicle_id/outing, asof-merge telemetry rows to the most recent lap start,
    then drop rows whose timestamp > end_time to ensure it's inside that window.
    """
    tw = telemetry_wide.copy()
    lw = lap_windows.copy()

    out_frames = []
    for (veh, out) in lw[["vehicle_id", "outing"]].drop_duplicates().itertuples(index=False):
        lw_g = lw[(lw["vehicle_id"] == veh) & (lw["outing"] == out)].sort_values("start_time")
        tw_g = tw[(tw["vehicle_id"] == veh) & (tw["outing"] == out)].sort_values("timestamp")

        if lw_g.empty or tw_g.empty:
            out_frames.append(tw_g.assign(lap=pd.NA, start_time=pd.NaT, end_time=pd.NaT))
            continue

        lw_starts = lw_g[["lap", "start_time"]].rename(columns={"start_time": "start_time_key"}).copy()
        lw_starts = lw_starts.sort_values("start_time_key")

        merged = pd.merge_asof(
            left=tw_g.sort_values("timestamp"),
            right=lw_starts.sort_values("start_time_key"),
            left_on="timestamp",
            right_on="start_time_key",
            direction="backward",
        )

        merged = pd.merge(
            merged,
            lw_g[["lap", "start_time", "end_time"]],
            on="lap",
            how="left",
        )

        inside = merged[
            (merged["start_time"].notna())
            & (merged["timestamp"] >= merged["start_time"])
            & (merged["timestamp"] <= merged["end_time"])
        ].copy()

        out_frames.append(inside)

    assigned = pd.concat(out_frames, ignore_index=True) if out_frames else tw.copy()
    return assigned


# ----------------------------
# 5) Build per-lap aggregates for first insights
# ----------------------------
def build_lap_aggregates(telemetry_assigned: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple lap-level aggregates: max speed, avg throttle/brake, count samples, etc.
    Note: column names depend on telemetry feed; we guard with get().
    """
    df = telemetry_assigned.copy()

    speed_col = None
    for cand in ["Speed", "speed", "vehicle_speed", "Vehicle_Speed"]:
        if cand in df.columns:
            speed_col = cand
            break

    throttle_col = None
    for cand in ["ath", "Throttle", "throttle", "Throttle_Blade", "Accelerator", "aps"]:
        if cand in df.columns:
            throttle_col = cand
            break

    brake_f_col = None
    for cand in ["pbrake_f", "Brake_Front", "brake_front"]:
        if cand in df.columns:
            brake_f_col = cand
            break

    brake_r_col = None
    for cand in ["pbrake_r", "Brake_Rear", "brake_rear"]:
        if cand in df.columns:
            brake_r_col = cand
            break

    group_cols = ["vehicle_id", "outing", "lap"]
    metrics = {
        "samples": ("timestamp", "count"),
    }
    if speed_col:
        metrics["max_speed"] = (speed_col, "max")
        metrics["avg_speed"] = (speed_col, "mean")
    if throttle_col:
        metrics["avg_throttle"] = (throttle_col, "mean")
    if brake_f_col:
        metrics["avg_brake_f"] = (brake_f_col, "mean")
    if brake_r_col:
        metrics["avg_brake_r"] = (brake_r_col, "mean")

    agg = (
        df.groupby(group_cols)
        .agg(**metrics)
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return agg


# ----------------------------
# 6) High-level pipeline helper
# ----------------------------
def build_pipeline_outputs(base_path: str) -> dict:
    """
    Convenience function: load -> lap windows -> align timestamps ->
    pivot telem -> assign laps -> aggregates.
    Returns a dict of outputs for the notebook/app.
    """
    dfs = load_race1_data(base_path)

    # 1) Lap windows from start/end CSVs
    lap_windows = build_lap_windows(dfs["lap_start"], dfs["lap_end"])

    # 2) Telemetry long -> wide
    telem_wide_raw = pivot_telemetry_long_to_wide(dfs["telemetry"])

    # 3) Align telemetry timestamps to lap windows
    telem_wide_aligned = align_timestamps(telem_wide_raw, lap_windows)

    # 4) Assign laps using time windows
    telem_with_laps = assign_laps_to_telemetry(telem_wide_aligned, lap_windows)

    # 5) Lap-level aggregates
    lap_agg = build_lap_aggregates(telem_with_laps)

    return {
        "lap_windows": lap_windows,
        "telemetry_wide": telem_wide_aligned,
        "telemetry_with_laps": telem_with_laps,
        "lap_aggregates": lap_agg,
        "lap_time_raw": dfs["lap_time"],
    }


# ----------------------------
# 7) Adjust telemetry time alignment (if needed)
# ----------------------------
def align_timestamps(telemetry_df: pd.DataFrame, lap_windows_df: pd.DataFrame):
    """
    Auto-align telemetry and lap window timestamps if their clocks differ.
    Uses the median difference between first telemetry and first lap start.
    """
    telem_start = telemetry_df["timestamp"].min()
    lap_start = lap_windows_df["start_time"].min()

    if pd.isna(telem_start) or pd.isna(lap_start):
        print("⚠️ Cannot align timestamps: missing values.")
        return telemetry_df

    offset = lap_start - telem_start
    print(f"🕒 Applying time offset: {offset}")
    telemetry_df["timestamp"] = telemetry_df["timestamp"] + offset
    return telemetry_df
