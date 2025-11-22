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

sns.set_style("whitegrid")


# ---------------------------------
# 1. Cached data loading
# ---------------------------------
@st.cache_data(show_spinner=True)
def load_outputs(base_path: str) -> dict:
    """
    Load and preprocess race data once, then cache the outputs.
    This calls your preprocess.build_pipeline_outputs().
    """
    return preprocess.build_pipeline_outputs(base_path)


def build_driver_summary(outputs: dict, vehicle_id: str, outing: float = 0.0) -> pd.DataFrame:
    """
    Build lap-level summary + LapLens performance score for a given vehicle and outing.

    IMPORTANT: we use the already-processed lap_aggregates + lap_time_raw from preprocess.
    We do NOT realign or re-aggregate inside the app – that keeps things light and stable.
    """
    lap_agg_all = outputs["lap_aggregates"]

    # 1) Filter to this car + outing and keep valid laps
    lap_agg = lap_agg_all[
        (lap_agg_all["vehicle_id"] == vehicle_id)
        & (lap_agg_all["outing"] == outing)
        & (lap_agg_all["lap"] < 1000)
    ].copy()

    if lap_agg.empty:
        return pd.DataFrame()

    # 2) Bring in official lap times
    lap_times = outputs["lap_time_raw"].copy()
    lap_times_clean = lap_times[
        ["vehicle_id", "outing", "lap", "value"]
    ].rename(columns={"value": "official_lap_time_raw"})

    summary = pd.merge(
        lap_agg,
        lap_times_clean,
        on=["vehicle_id", "outing", "lap"],
        how="left",
    )

    # Convert to seconds if the values look like ms
    if summary["official_lap_time_raw"].max() > 1000:
        summary["official_lap_time_s"] = summary["official_lap_time_raw"] / 1000.0
    else:
        summary["official_lap_time_s"] = summary["official_lap_time_raw"]

    # 3) Quality gate on samples, with a fallback so the table is never empty
    filtered = summary[summary["samples"] >= 1500].copy()
    if filtered.empty:
        # If our strict filter kills everything, fall back to unfiltered summary
        filtered = summary.copy()

    # 4) LapLens performance score (same spirit as notebook)
    data = filtered.copy()
    metrics = ["avg_speed", "avg_throttle", "avg_brake_f"]

    for m in metrics:
        if m in data.columns:
            m_min = data[m].min()
            m_max = data[m].max()
            if (m_max - m_min) > 1e-6:
                data[f"{m}_norm"] = (data[m] - m_min) / (m_max - m_min)
            else:
                # If metric is constant, treat as neutral
                data[f"{m}_norm"] = 0.5

    WEIGHT_SPEED = 0.55
    WEIGHT_THROTTLE = 0.30
    WEIGHT_BRAKE = 0.15

    data["performance_score"] = (
        data.get("avg_speed_norm", 0) * WEIGHT_SPEED
        + data.get("avg_throttle_norm", 0) * WEIGHT_THROTTLE
        + data.get("avg_brake_f_norm", 0) * WEIGHT_BRAKE
    ) * 100

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
    # Assume realistic std range ~ [0, 20]
    scale = 20.0
    dci = 100 * max(0.0, 1.0 - (std / scale))
    return round(dci, 1)


def generate_coaching_note(row: pd.Series, best_row: pd.Series) -> str:
    """
    Rule-based coaching note per lap.
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
            notes.append("Solid lap, but there’s still room to push a bit more.")
        else:
            notes.append("Significant gap to your best – opportunity to gain time on this lap.")

    # Throttle profile
    if not np.isnan(throttle):
        if throttle < 60:
            notes.append("Throttle usage is conservative; focus on earlier and longer throttle application.")
        elif throttle > 90:
            notes.append("Very high throttle usage – check for over-driving on corner exits.")
        else:
            notes.append("Throttle use is balanced for this lap.")

    # Brake profile
    if not np.isnan(brake_f):
        if brake_f > 6:
            notes.append("Heavy braking – experiment with earlier, smoother braking for stability.")
        elif brake_f < 3:
            notes.append("Light braking – confirm you’re still fully exploiting the braking zones.")
        else:
            notes.append("Braking looks efficient and controlled.")

    # Lap time
    if not np.isnan(lap_time):
        notes.append(f"Official lap time: {lap_time:.3f} s.")

    return " ".join(notes)


# ---------------------------------
# 2. App layout
# ---------------------------------
def main():
    st.title("🏁 LapLens – COTA Race 1 Post-Event Driver Analysis")

    st.markdown(
        """
        LapLens turns raw **TRD telemetry** and **official lap times** into a driver-focused race debrief.

        For each lap we:
        - Rebuild the stint from ECU telemetry aligned with official lap timing.
        - Compute lap-level metrics: speed, throttle usage, front brake pressure, and sample quality.
        - Derive a **LapLens performance score (0–100)**.
        - Highlight best vs worst laps and auto-generate short coaching notes.

        Use the controls in the sidebar to select a car and outing.
        """
    )

    # ------------------- Sidebar -------------------
    st.sidebar.header("Configuration")
    st.sidebar.caption("LapLens – COTA Race 1 settings")

    # Reload button: clear cache & rerun
    if st.sidebar.button("🔁 Reload data (clear cache)"):
        load_outputs.clear()
        st.sidebar.success("Cache cleared – reloading data…")
        st.rerun()

    # Load data once (from cache)
    outputs = load_outputs(BASE_PATH)
    lap_agg_all = outputs["lap_aggregates"]

    if lap_agg_all is None or len(lap_agg_all) == 0:
        st.error(
            "No lap aggregates were produced. "
            "Verify that the COTA Race 1 CSVs are present under `data/COTA_Race1`."
        )
        return

    # Vehicle selection
    vehicles = sorted(lap_agg_all["vehicle_id"].unique())
    default_idx = vehicles.index("GR86-006-7") if "GR86-006-7" in vehicles else 0
    vehicle_id = st.sidebar.selectbox("Select car (vehicle_id)", vehicles, index=default_idx)

    # Outing selection (usually 0.0 for this dataset)
    outing_options = sorted(
        lap_agg_all[lap_agg_all["vehicle_id"] == vehicle_id]["outing"].unique()
    )
    outing = st.sidebar.selectbox("Select outing", outing_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Track: Circuit of the Americas • Event: COTA Race 1")

    # ------------------- Driver summary -------------------
    driver_df = build_driver_summary(outputs, vehicle_id, outing)

    if driver_df.empty:
        st.warning(
            "No valid laps found for this configuration after quality filtering. "
            "Try another car, or reduce filters in the code if needed."
        )
        return

    best_row = driver_df.loc[driver_df["performance_score"].idxmax()]
    worst_row = driver_df.loc[driver_df["performance_score"].idxmin()]
    dci = compute_consistency_index(driver_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Lap", f"Lap {int(best_row['lap'])}", f"{best_row['performance_score']:.1f} / 100")
    col2.metric("Fastest Lap Time", f"{best_row['official_lap_time_s']:.3f} s")
    col3.metric("Worst Lap", f"Lap {int(worst_row['lap'])}", f"{worst_row['performance_score']:.1f} / 100")
    col4.metric("Driver Consistency Index", f"{dci} / 100")

    # ------------------- Table with coaching notes -------------------
    st.markdown("### 1. Lap Table with LapLens Score & Coaching Notes")

    coaching_notes = [
        generate_coaching_note(row, best_row) for _, row in driver_df.iterrows()
    ]

    table_df = driver_df[
        ["lap", "official_lap_time_s", "avg_speed", "avg_throttle", "avg_brake_f", "performance_score"]
    ].copy()

    table_df.rename(
        columns={
            "lap": "Lap",
            "official_lap_time_s": "Lap Time (s)",
            "avg_speed": "Avg Speed (km/h)",
            "avg_throttle": "Avg Throttle (%)",
            "avg_brake_f": "Avg Front Brake (bar)",
            "performance_score": "LapLens Score (0–100)",
        },
        inplace=True,
    )
    table_df["Coaching Note"] = coaching_notes

    st.dataframe(
        table_df.style.format(
            {
                "Lap Time (s)": "{:.3f}",
                "Avg Speed (km/h)": "{:.2f}",
                "Avg Throttle (%)": "{:.2f}",
                "Avg Front Brake (bar)": "{:.2f}",
                "LapLens Score (0–100)": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

    # ------------------- Performance trends -------------------
    st.markdown("### 2. Performance Trends per Lap")
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=driver_df, x="lap", y="avg_speed", marker="o", ax=ax)
        ax.set_title("Average Speed per Lap")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Avg Speed (km/h)")
        ax.grid(True)
        st.pyplot(fig)

    with right:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(
            data=driver_df,
            x="lap",
            y="avg_throttle",
            marker="o",
            label="Throttle (%)",
            ax=ax2,
        )
        sns.lineplot(
            data=driver_df,
            x="lap",
            y="avg_brake_f",
            marker="o",
            label="Front Brake (bar)",
            ax=ax2,
        )
        ax2.set_title("Driver Inputs per Lap")
        ax2.set_xlabel("Lap")
        ax2.set_ylabel("Intensity")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    # ------------------- LapLens score vs time -------------------
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

        - Points farther left and higher ⇒ conservative / slower laps.
        - Points farther right and lower ⇒ strong laps with high LapLens score and fast time.
        """
    )


if __name__ == "__main__":
    main()