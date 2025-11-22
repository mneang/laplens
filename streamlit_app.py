import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure we can import src.preprocess when running from repo root
sys.path.append(".")

from src import preprocess


# ---------------------------------
# 0. Basic config
# ---------------------------------
BASE_PATH = "data/COTA_Race1"

st.set_page_config(
    page_title="LapLens – COTA Race 1",
    page_icon="🏎️",
    layout="wide",
)


# ---------------------------------
# 1. Cached data loading
# ---------------------------------
@st.cache_data(show_spinner=True)
def load_outputs(base_path: str):
    """Load and preprocess race data once, then cache."""
    outputs = preprocess.build_pipeline_outputs(base_path)
    return outputs


def build_driver_summary(outputs: dict, vehicle_id: str, outing: float = 0.0) -> pd.DataFrame:
    """
    Build lap-level summary + LapLens performance score for a given vehicle and outing.
    Mirrors the logic from your notebook, but kept local to the app so we don't break anything.
    """
    # 1) Align and assign laps
    aligned = preprocess.align_timestamps(outputs["telemetry_wide"], outputs["lap_windows"])
    telem_with_laps = preprocess.assign_laps_to_telemetry(aligned, outputs["lap_windows"])

    # 2) Aggregate to lap-level
    lap_agg = preprocess.build_lap_aggregates(telem_with_laps)

    # 3) Quality filter and select driver/outing
    lap_agg = lap_agg[
        (lap_agg["lap"] < 1000) &
        (lap_agg["vehicle_id"] == vehicle_id) &
        (lap_agg["outing"] == outing)
    ].copy()

    # Optional quality gate: drop laps with very few samples (e.g., in/out laps)
    lap_agg = lap_agg[lap_agg["samples"] >= 1500].copy()

    # 4) Merge in official lap times
    lap_times = outputs["lap_time_raw"].copy()
    lap_times_clean = lap_times[["vehicle_id", "outing", "lap", "value"]].rename(
        columns={"value": "official_lap_time_raw"}
    )

    summary = pd.merge(
        lap_agg,
        lap_times_clean,
        on=["vehicle_id", "outing", "lap"],
        how="left",
    )

    # Convert official lap time to seconds if values look like milliseconds
    if summary["official_lap_time_raw"].max() > 1000:
        summary["official_lap_time_s"] = summary["official_lap_time_raw"] / 1000.0
    else:
        summary["official_lap_time_s"] = summary["official_lap_time_raw"]

    # 5) Compute LapLens performance score (same idea as notebook)
    data = summary.copy()
    metrics = ["avg_speed", "avg_throttle", "avg_brake_f"]

    for m in metrics:
        if m in data.columns:
            m_min = data[m].min()
            m_max = data[m].max()
            if (m_max - m_min) > 1e-6:
                data[f"{m}_norm"] = (data[m] - m_min) / (m_max - m_min)
            else:
                data[f"{m}_norm"] = 0.5  # fallback if constant

    WEIGHT_SPEED = 0.55
    WEIGHT_THROTTLE = 0.30
    WEIGHT_BRAKE = 0.15

    data["performance_score"] = (
        data.get("avg_speed_norm", 0) * WEIGHT_SPEED +
        data.get("avg_throttle_norm", 0) * WEIGHT_THROTTLE +
        data.get("avg_brake_f_norm", 0) * WEIGHT_BRAKE
    ) * 100

    # Sort by lap for display
    data = data.sort_values("lap").reset_index(drop=True)
    return data


def compute_consistency_index(df: pd.DataFrame) -> float:
    """
    Simple Driver Consistency Index (0–100).
    Lower variance in performance_score => higher DCI.
    """
    if "performance_score" not in df.columns or len(df) < 2:
        return 50.0

    std = df["performance_score"].std()
    # Map std to [0, 100]; you can tune the scale factor
    # Assume std in [0, 20] is realistic range.
    scale = 20.0
    dci = 100 * max(0.0, 1.0 - (std / scale))
    return round(dci, 1)


def generate_coaching_note(row: pd.Series, best_row: pd.Series) -> str:
    """
    Very simple rule-based coaching note for a lap.
    Uses performance_score, throttle, brake, and lap time.
    """
    notes = []

    score = row.get("performance_score", np.nan)
    best_score = best_row.get("performance_score", np.nan)
    throttle = row.get("avg_throttle", np.nan)
    brake_f = row.get("avg_brake_f", np.nan)
    lap_time = row.get("official_lap_time_s", np.nan)

    # Relative performance vs best
    if not np.isnan(score) and not np.isnan(best_score):
        delta = best_score - score
        if delta < 5:
            notes.append("Very close to your best – strong lap overall.")
        elif delta < 15:
            notes.append("Solid lap, but there's still room to push a bit more.")
        else:
            notes.append("Significant gap to your best – opportunity to gain time here.")

    # Throttle/brake style
    if not np.isnan(throttle):
        if throttle < 60:
            notes.append("Throttle usage is conservative; focus on earlier and longer throttle application.")
        elif throttle > 90:
            notes.append("High throttle usage – ensure you’re not over-driving on corner exits.")
        else:
            notes.append("Throttle use is balanced for this lap.")

    if not np.isnan(brake_f):
        if brake_f > 6:
            notes.append("Heavy braking – check if earlier, smoother braking could stabilize entry.")
        elif brake_f < 3:
            notes.append("Very light braking – confirm you’re still making full use of braking zones.")
        else:
            notes.append("Braking looks efficient and controlled.")

    # Lap time hint
    if not np.isnan(lap_time):
        notes.append(f"Official lap time: {lap_time:.3f} s.")

    return " ".join(notes)


# ---------------------------------
# 2. App layout
# ---------------------------------
def main():
    st.title("🏎️ LapLens – COTA Race 1 Post-Event Analysis")
    st.markdown(
        """
        LapLens turns raw TRD telemetry + official lap times into a **driver-focused race debrief**.

        **For each lap**, we:
        - Rebuild the stint from ECU data (aligned with lap timing).
        - Compute lap-level metrics: speed, throttle, brake usage, sample quality.
        - Derive a **LapLens performance score (0–100)**.
        - Highlight best vs worst laps and generate short coaching notes.

        Use the controls on the left to explore different cars and outings.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")
    st.sidebar.write("LapLens – COTA Race 1 settings")

    outputs = load_outputs(BASE_PATH)
    lap_agg_all = outputs["lap_aggregates"]

    vehicles = sorted(lap_agg_all["vehicle_id"].unique())
    vehicle_id = st.sidebar.selectbox("Select car (vehicle_id)", vehicles, index=0)

    outing_options = sorted(lap_agg_all[lap_agg_all["vehicle_id"] == vehicle_id]["outing"].unique())
    outing = st.sidebar.selectbox("Select outing", outing_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Track: Circuit of the Americas • Event: COTA Race 1")

    # Build driver summary
    driver_df = build_driver_summary(outputs, vehicle_id, outing)

    if driver_df.empty:
        st.warning("No valid laps found for this configuration (after quality filters).")
        return

    # Metrics / KPIs
    best_row = driver_df.loc[driver_df["performance_score"].idxmax()]
    worst_row = driver_df.loc[driver_df["performance_score"].idxmin()]
    dci = compute_consistency_index(driver_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Lap", f"Lap {int(best_row['lap'])}", f"{best_row['performance_score']:.1f} / 100")
    col2.metric("Fastest Lap Time", f"{best_row['official_lap_time_s']:.3f} s")
    col3.metric("Worst Lap", f"Lap {int(worst_row['lap'])}", f"{worst_row['performance_score']:.1f} / 100")
    col4.metric("Driver Consistency Index", f"{dci} / 100")

    st.markdown("### 1. Lap Table with LapLens Score & Coaching Notes")

    # Build per-lap coaching notes
    coaching_notes = []
    for _, row in driver_df.iterrows():
        note = generate_coaching_note(row, best_row)
        coaching_notes.append(note)

    table_df = driver_df[[
        "lap", "official_lap_time_s", "avg_speed", "avg_throttle", "avg_brake_f", "performance_score"
    ]].copy()
    table_df.rename(columns={
        "lap": "Lap",
        "official_lap_time_s": "Lap Time (s)",
        "avg_speed": "Avg Speed (km/h)",
        "avg_throttle": "Avg Throttle (%)",
        "avg_brake_f": "Avg Front Brake (bar)",
        "performance_score": "LapLens Score (0–100)",
    }, inplace=True)
    table_df["Coaching Note"] = coaching_notes

    st.dataframe(
        table_df.style.format({
            "Lap Time (s)": "{:.3f}",
            "Avg Speed (km/h)": "{:.2f}",
            "Avg Throttle (%)": "{:.2f}",
            "Avg Front Brake (bar)": "{:.2f}",
            "LapLens Score (0–100)": "{:.1f}",
        }),
        use_container_width=True,
    )

    # -------------------------
    # 3. Visuals
    # -------------------------
    st.markdown("### 2. Performance Trends per Lap")
    left, right = st.columns(2)

    # Speed trend
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=driver_df, x="lap", y="avg_speed", marker="o", ax=ax)
        ax.set_title("Average Speed per Lap")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Avg Speed (km/h)")
        ax.grid(True)
        st.pyplot(fig)

    # Inputs trend
    with right:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=driver_df, x="lap", y="avg_throttle", marker="o", label="Throttle (%)", ax=ax2)
        sns.lineplot(data=driver_df, x="lap", y="avg_brake_f", marker="o", label="Front Brake (bar)", ax=ax2)
        ax2.set_title("Driver Inputs per Lap")
        ax2.set_xlabel("Lap")
        ax2.set_ylabel("Intensity")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    # -------------------------
    # 4. LapLens vs Lap Time
    # -------------------------
    st.markdown("### 3. LapLens Score vs Official Lap Time")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=driver_df,
        x="performance_score",
        y="official_lap_time_s",
        ax=ax3,
    )
    ax3.set_title("LapLens Score vs Official Lap Time")
    ax3.set_xlabel("LapLens Score (0–100)")
    ax3.set_ylabel("Official Lap Time (s)")
    ax3.grid(True)

    # Optional simple linear fit
    if len(driver_df) >= 2:
        x = driver_df["performance_score"].values
        y = driver_df["official_lap_time_s"].values
        coeffs = np.polyfit(x, y, deg=1)
        a, b = coeffs
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = a * x_line + b
        ax3.plot(x_line, y_line, linestyle="--", label=f"Fit: time ≈ {a:.3f}·score + {b:.1f}")
        ax3.legend()

    st.pyplot(fig3)

    st.markdown(
        """
        **Interpretation:**
        - Points further left and higher ⇒ conservative / slower laps.
        - Points further right and lower ⇒ strong laps with good LapLens scores and fast times.
        """
    )


if __name__ == "__main__":
    main()