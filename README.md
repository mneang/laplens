# 🏎️ LapLens – COTA Race 1 Post-Event Driver Analysis

> **Hack the Track – Category:** Post-Event Analysis  
> LapLens turns raw TRD telemetry + official lap times into a **driver-focused race debrief** that a race engineer or driver can use within minutes of the checkered flag.

---

## 1. Concept Overview

In modern motorsport, the gap between P1 and the midfield is often **driver consistency**, not just outright pace. Teams already collect huge amounts of telemetry, but turning that firehose into **clear coaching feedback** is still hard and time-consuming.

**LapLens** is a lightweight analytics tool that:

- Rebuilds each stint from **TRD GR86 telemetry** and **official COTA Race 1 lap times**
- Computes lap-level metrics (speed, throttle, brake usage, sample quality)
- Derives a **LapLens performance score (0–100)** for every lap
- Highlights **best vs worst laps** and generates short, **plain-English coaching notes**

The goal is a **post-event debrief assistant** that a driver or engineer can open, select a car, and – in under a minute – understand:

> “Where was this driver strong? Where are the easiest wins next time we come to COTA?”

Unlike a generic lap-time dashboard, LapLens focuses on **turning telemetry into coaching language**: 
instead of just numbers, it gives a ranked list of laps plus specific notes a driver can act on before 
the next session.

---

## 2. What LapLens Delivers

### Key Questions It Answers

For each selected car & outing:

- **Pace & execution**
  - How does average speed evolve lap-by-lap?
  - Which laps combine both **high LapLens score** and **fast official lap time**?

- **Driver technique**
  - Is the driver **over-driving** (high throttle, heavy brakes, mediocre lap time)?
  - Are they too conservative early in the stint?

- **Consistency**
  - How stable are LapLens scores across laps?
  - Which laps are outliers that deserve video / onboard review?

---

## 3. Main Features

### ✅ Streamlit App (Primary Deliverable)

`streamlit_app.py` provides an interactive dashboard for **any car in the COTA Race 1 dataset**:

1. **Configuration sidebar**
   - Select **vehicle_id** (any car present in the TRD COTA Race 1 telemetry, e.g. `GR86-006-7`)
   - Select **outing** (e.g. `0.0`; multiple outings are supported where available)
   - “Reload data (clear cache)” to rebuild from CSVs

2. **Headline KPIs**
   - **Best Lap** (by LapLens score)
   - **Fastest Lap Time** (official timing)
   - **Worst Lap** (by LapLens score)
   - **Driver Consistency Index (0–100)**

3. **Lap Table with LapLens Score & Coaching Notes**
   - Per-lap:
     - Lap time (s)
     - Avg speed (km/h)
     - Avg throttle (%)
     - Avg front brake pressure (bar)
     - LapLens Score (0–100)
     - **Auto-generated coaching note** (plain English)

4. **Performance Trends per Lap**
   - **Average Speed per Lap** (line plot)
   - **Driver Inputs per Lap**:
     - Throttle trace
     - Front brake trace

5. **LapLens Score vs Official Lap Time**
   - Scatter plot: LapLens Score (x) vs Official Lap Time (y)
   - Optional linear regression fit and rough interpretation:
     - Left & high → conservative/slower laps  
     - Right & low → strong, efficient laps

The UI is intentionally **simple and dark-themed**, optimized for use in a garage or at-track environment.

### 3.1 Screenshots (Example Views)

- **Main dashboard & KPIs**  
  <img width="1388" height="488" alt="laplens overview" src="https://github.com/user-attachments/assets/55c26756-f851-4c01-98fd-c57f9dd2a1ba" />

- **Lap table with LapLens score & coaching notes**  
  <img width="1763" height="477" alt="laplens score and coaching notes table" src="https://github.com/user-attachments/assets/1671ccda-afb9-4c2c-a8b0-f2d6e8fc9abd" />

- **LapLens score vs official lap time scatter**  
  <img width="1236" height="939" alt="average time per lap" src="https://github.com/user-attachments/assets/368bdf32-5ee0-4cdc-b1f2-76f8fd5e5ede" />

---

## 4. Data & Methodology

### 4.1 TRD Datasets Used

From `data/COTA_Race1`:

- `R1_cota_telemetry_data.csv`
- `COTA_lap_start_time_R1.csv`
- `COTA_lap_end_time_R1.csv`
- `COTA_lap_time_R1.csv`

(Additional CSVs like `00_Results`, `23_AnalysisEndurance…`, and weather can be added later.)

### 4.2 Processing Pipeline (in `src/preprocess.py`)

1. **Load raw CSVs** – `load_race1_data(base_path)`
   - Reads telemetry + lap timing files with `pandas.read_csv`.

2. **Build lap windows** – `build_lap_windows(lap_start, lap_end)`
   - Groups start/end timestamps by `(vehicle_id, outing, lap)`  
   - Produces `start_time` and `end_time` per lap.

3. **Pivot telemetry long → wide** – `pivot_telemetry_long_to_wide(telemetry)`
   - Raw telemetry is long-format:
     - `telemetry_name`, `telemetry_value`
   - Pivot to wide by timestamp:
     - Columns like `speed`, `ath` (throttle), `pbrake_f`, `pbrake_r`, `accx_can`, `accy_can`, `Steering_Angle`, etc.

4. **Align telemetry timestamps** – `align_timestamps(telem_wide, lap_windows)`
   - ECU clock and timing system can differ.
   - Compute offset between earliest telemetry timestamp and first lap `start_time`.
   - Shift telemetry by this offset so telemetry and lap windows are on the same clock.

5. **Assign laps to telemetry** – `assign_laps_to_telemetry(telem_wide_aligned, lap_windows)`
   - For each `(vehicle_id, outing)`:
     - Use `merge_asof` to map each telemetry row to the most recent lap start.
     - Keep only rows inside `[start_time, end_time]` for that lap.

6. **Lap-level aggregates** – `build_lap_aggregates(telemetry_with_laps)`
   - Group by `(vehicle_id, outing, lap)` and compute:
     - `samples` (row count)
     - `max_speed`, `avg_speed`
     - `avg_throttle`
     - `avg_brake_f`, `avg_brake_r`

7. **High-level orchestrator** – `build_pipeline_outputs(base_path)`
   - Returns a dict:
     - `lap_windows`
     - `telemetry_wide`
     - `telemetry_with_laps`
     - `lap_aggregates`
     - `lap_time_raw`

These functions are reused **both** in the Jupyter notebook (`notebooks/visualizations.ipynb`) and in the Streamlit app, keeping logic consistent.

### 4.3 LapLens Score & Consistency Index

**LapLens Performance Score (0–100)**

For each lap:

1. Normalize:
   - `avg_speed_norm` ∈ [0,1]
   - `avg_throttle_norm` ∈ [0,1]
   - `avg_brake_f_norm` ∈ [0,1] (used as a proxy for how “hard” the lap was)

2. Weighted combination (in `streamlit_app.py`):

   ```python
   WEIGHT_SPEED    = 0.55
   WEIGHT_THROTTLE = 0.30
   WEIGHT_BRAKE    = 0.15

   performance_score = (
       avg_speed_norm    * WEIGHT_SPEED +
       avg_throttle_norm * WEIGHT_THROTTLE +
       avg_brake_f_norm  * WEIGHT_BRAKE
   ) * 100
   ```

This gives a **driver-legible 0–100** score per lap, balancing pace, commitment, and braking effort.

Driver Consistency Index (DCI)
	•	Based on the standard deviation of LapLens scores across laps.
	•	Lower variance → higher DCI:

   ```python
   std  = df["performance_score"].std()
   scale = 20.0  # typical std range
   dci = 100 * max(0.0, 1.0 - (std / scale))
   ```

---

## 5. Repository Structure
```text
laplens/
├─ data/
│  └─ COTA_Race1/
│     ├─ 00_Results GR Cup Race 1 Official_Anonymized.csv
│     ├─ 03_Provisional Results_Race 1_Anonymized.csv
│     ├─ 05_Provisional Results by Class_Race 1_Anonymized.csv
│     ├─ 23_AnalysisEnduranceWithSections_Race 1_Anonymized.csv
│     ├─ 26_Weather_Race 1_Anonymized.csv
│     ├─ 99_Best 10 Laps By Driver_Race 1_Anonymized.csv
│     ├─ COTA_lap_end_time_R1.csv
│     ├─ COTA_lap_start_time_R1.csv
│     ├─ COTA_lap_time_R1.csv
│     └─ R1_cota_telemetry_data.csv
├─ notebooks/
│  ├─ test_loader.ipynb        # sanity checks for loading CSVs
│  ├─ test.ipynb               # scratch work / quick experiments
│  └─ visualizations.ipynb     # “analysis notebook” version of LapLens
├─ src/
│  ├─ __init__.py
│  └─ preprocess.py            # core data pipeline used by both notebook and app
├─ streamlit_app.py            # main app entrypoint
├─ requirements.txt
└─ README.md
```
> The notebook (visualizations.ipynb) acts as a transparent analysis appendix: judges can see exactly how the metrics and LapLens score were derived.

---

## 6. Getting Started

### 6.1 GitHub Codespaces

This project is designed to run smoothly in **GitHub Codespaces**.

1. Open the repo in Codespaces.
2. Make sure the virtual environment is active (e.g. `.venv`).
3. Install dependencies (if not done automatically):
  ```bash
  pip install -r requirements.txt
  ```

4. Launch the app:
  ```bash
  streamlit run streamlit_app.py
  ```

5.	A browser tab (or Codespaces port) will open with the LapLens dashboard.

### 6.2 Running Locally (Desktop)

  1. Clone the repo:
  ```bash
  git clone https://github.com/mneang/laplens.git
  cd laplens
  ```

  2. Create & activate a virtual environment (optional but recommended):
  ```bash
  python -m venv .venv
  source .venv/bin/activate      # macOS / Linux
  # .venv\Scripts\activate       # Windows
  ```

  3.	Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

  4. Ensure the TRD CSVs are present under data/COTA_Race1/.
  5. Run the Streamlit app:
  ```bash
  streamlit run streamlit_app.py
  ```

### 6.3 Notebooks

To inspect or extend the analysis:
	1.	Open notebooks/visualizations.ipynb in VS Code Jupyter or Jupyter Lab.
	2.	Make sure the kernel uses the same environment as the app (.venv).
	3.	Run the cells top-to-bottom.

---

## 7. How to Use LapLens (for Judges & Engineers)

1. **Launch the app** – `streamlit run streamlit_app.py`.
2. **Pick a car + outing** – use the sidebar to select `vehicle_id` and `outing`.
3. **Scan the headline KPIs** – best lap, fastest lap time, worst lap, and Driver Consistency Index.
4. **Read the coaching notes** – start with the lowest LapLens scores to find “problem laps”.
5. **Check the trends** – confirm how speed, throttle, and brake evolve across the stint.
6. **Validate LapLens vs lap time** – use the scatter plot to see if higher scores actually map to faster laps, and flag any outliers for deeper review.

This flow is designed to match how real teams debrief: **top-down → identify issues → drill into specific laps**.

---

## 8. Limitations & Future Ideas (Short)

Current demo focus:

- COTA Race 1 only (GR86 GR Cup dataset).
- LapLens score uses **speed, throttle, and front brake** only – no tires, fuel, or sector times yet.

Possible next steps:

- Extend to **other races / tracks** in the TRD dataset.
- Add **corner profiles** (entry / mid / exit) using `Steering_Angle`, `accx_can`, `accy_can`.
- Integrate **video or sector overlays** so engineers can jump directly from a flagged lap to footage.
- Allow exporting per-lap LapLens reports as **PDF** for sharing with drivers after the event.

