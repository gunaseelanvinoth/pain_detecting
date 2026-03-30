from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DATA_DIR = Path(__file__).resolve().parent / "sample_data"


@st.cache_data
def list_sessions() -> list[Path]:
    return sorted(DATA_DIR.glob("pain_session_*.csv"), reverse=True)


@st.cache_data
def load_session(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    numeric_columns = [
        "pain_score_0_10",
        "current_episode_duration_s",
        "total_pain_duration_s",
        "eye_closure",
        "brow_tension",
        "mouth_tension",
        "smile_absence",
        "motion_score",
        "pain_active",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _find_episode_events_for_session(session_file: Path) -> Path | None:
    suffix = session_file.name.replace("pain_session_", "")
    candidate = DATA_DIR / f"pain_episode_events_{suffix}"
    return candidate if candidate.exists() else None


@st.cache_data
def load_episode_events(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_episodes(data: pd.DataFrame) -> pd.DataFrame:
    if "active_episode_id" not in data.columns or data.empty:
        return pd.DataFrame(columns=["episode_id", "start", "end", "duration_s", "max_score", "avg_score"])

    episodes = data[data["active_episode_id"].notna()].copy()
    if episodes.empty:
        return pd.DataFrame(columns=["episode_id", "start", "end", "duration_s", "max_score", "avg_score"])

    grouped = episodes.groupby("active_episode_id", as_index=False).agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        max_score=("pain_score_0_10", "max"),
        avg_score=("pain_score_0_10", "mean"),
    )
    grouped = grouped.rename(columns={"active_episode_id": "episode_id"})
    grouped["duration_s"] = (grouped["end"] - grouped["start"]).dt.total_seconds().fillna(0.0)
    return grouped[["episode_id", "start", "end", "duration_s", "max_score", "avg_score"]]


def main() -> None:
    st.set_page_config(page_title="Pain Monitoring Dashboard", layout="wide")
    st.title("Patient Pain Duration Dashboard")
    st.caption("Visualize pain score, pain episodes, and total pain duration from live logs.")

    auto_refresh = st.sidebar.checkbox("Auto refresh", value=False)
    refresh_seconds = st.sidebar.slider("Refresh every (seconds)", min_value=2, max_value=30, value=5)
    if auto_refresh:
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_seconds * 1000});
            </script>
            """,
            height=0,
        )

    sessions = list_sessions()
    if not sessions:
        st.warning("No pain_session_*.csv files found. Run `python pain_main.py live` first.")
        return

    selected_name = st.sidebar.selectbox("Session file", [path.name for path in sessions])
    selected_path = next(path for path in sessions if path.name == selected_name)
    data = load_session(str(selected_path))

    if data.empty:
        st.error("Selected file is empty.")
        return

    if "pain_level" in data.columns:
        levels = sorted(data["pain_level"].dropna().unique())
        picked = st.sidebar.multiselect("Pain level", levels, default=levels)
        if picked:
            data = data[data["pain_level"].isin(picked)]

    avg_score = float(data["pain_score_0_10"].mean()) if "pain_score_0_10" in data.columns else 0.0
    max_score = float(data["pain_score_0_10"].max()) if "pain_score_0_10" in data.columns else 0.0
    latest_total = float(data["total_pain_duration_s"].dropna().iloc[-1]) if "total_pain_duration_s" in data.columns and not data["total_pain_duration_s"].dropna().empty else 0.0
    active_ratio = float(data["pain_active"].mean() * 100.0) if "pain_active" in data.columns else 0.0

    top = st.columns(4)
    top[0].metric("Average pain", f"{avg_score:.2f}/10")
    top[1].metric("Peak pain", f"{max_score:.2f}/10")
    top[2].metric("Total pain duration", f"{latest_total:.1f} sec")
    top[3].metric("Pain-active frames", f"{active_ratio:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Pain Score Over Time")
        if {"timestamp", "pain_score_0_10"}.issubset(data.columns):
            chart = data.sort_values("timestamp")[ ["timestamp", "pain_score_0_10"] ]
            st.line_chart(chart, x="timestamp", y="pain_score_0_10")

    with right:
        st.subheader("Total Pain Duration Over Time")
        if {"timestamp", "total_pain_duration_s"}.issubset(data.columns):
            chart = data.sort_values("timestamp")[ ["timestamp", "total_pain_duration_s"] ]
            st.line_chart(chart, x="timestamp", y="total_pain_duration_s")

    st.subheader("Pain Level Distribution")
    if "pain_level" in data.columns:
        st.bar_chart(data["pain_level"].value_counts())

    episodes = summarize_episodes(data)
    st.subheader("Episode Summary")
    st.dataframe(episodes, use_container_width=True, height=220)

    event_file = _find_episode_events_for_session(selected_path)
    if event_file is not None:
        st.subheader("Episode Events")
        st.dataframe(load_episode_events(str(event_file)), use_container_width=True, height=180)

    st.subheader("Frame Log")
    st.dataframe(data, use_container_width=True, height=300)

    st.download_button(
        label="Download session CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=selected_name,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
