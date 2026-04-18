from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.decision import pain_detected_from_face, pain_status_text
from pain_monitoring.types import FramePainFeatures

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
        "wheeze_probability",
        "respiratory_motion",
        "pain_active",
        "eye_symmetry",
        "brow_energy",
        "mouth_opening",
        "lower_face_motion",
        "face_edge_density",
        "nasal_tension",
        "face_detected",
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
        return pd.DataFrame(columns=["episode_id", "start", "end", "duration_s", "max_score", "avg_score", "peak_wheeze"])

    episodes = data[data["active_episode_id"].notna()].copy()
    if episodes.empty:
        return pd.DataFrame(columns=["episode_id", "start", "end", "duration_s", "max_score", "avg_score", "peak_wheeze"])

    grouped = episodes.groupby("active_episode_id", as_index=False).agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        max_score=("pain_score_0_10", "max"),
        avg_score=("pain_score_0_10", "mean"),
        peak_wheeze=("wheeze_probability", "max") if "wheeze_probability" in episodes.columns else ("pain_score_0_10", "size"),
    )
    grouped = grouped.rename(columns={"active_episode_id": "episode_id"})
    grouped["duration_s"] = (grouped["end"] - grouped["start"]).dt.total_seconds().fillna(0.0)
    return grouped[["episode_id", "start", "end", "duration_s", "max_score", "avg_score", "peak_wheeze"]]


def _frame_features_from_row(row: pd.Series) -> FramePainFeatures:
    face_detected_value = row.get("face_detected", 0)
    face_detected = bool(face_detected_value) if pd.notna(face_detected_value) else False
    return FramePainFeatures(
        face_detected=face_detected,
        face_box=None,
        eye_closure=float(row.get("eye_closure", 0.0) or 0.0),
        brow_tension=float(row.get("brow_tension", 0.0) or 0.0),
        mouth_tension=float(row.get("mouth_tension", 0.0) or 0.0),
        smile_absence=float(row.get("smile_absence", 0.0) or 0.0),
        motion_score=float(row.get("motion_score", 0.0) or 0.0),
        eye_symmetry=float(row.get("eye_symmetry", 0.0) or 0.0),
        brow_energy=float(row.get("brow_energy", 0.0) or 0.0),
        mouth_opening=float(row.get("mouth_opening", 0.0) or 0.0),
        lower_face_motion=float(row.get("lower_face_motion", 0.0) or 0.0),
        face_edge_density=float(row.get("face_edge_density", 0.0) or 0.0),
        nasal_tension=float(row.get("nasal_tension", 0.0) or 0.0),
        respiratory_motion=float(row.get("respiratory_motion", 0.0) or 0.0),
        wheeze_probability=float(row.get("wheeze_probability", 0.0) or 0.0),
    )


def classify_pain_status(score: float, row: pd.Series | None = None) -> str:
    if row is None:
        return pain_status_text(score >= 2.5)
    config = PainMonitoringConfig()
    return pain_status_text(pain_detected_from_face(_frame_features_from_row(row), score, config))


def classify_wheeze_status(probability: float) -> str:
    if probability >= 0.2:
        return "Wheezing Detected"
    return "No Wheezing"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(41, 163, 255, 0.18), transparent 26%),
                linear-gradient(180deg, #f6fbff 0%, #eef4ff 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #083358 0%, #11698e 60%, #38b6a8 100%);
            color: #ffffff;
            border-radius: 24px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 24px 50px rgba(8, 51, 88, 0.18);
        }
        .hero-eyebrow {
            display: inline-block;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1rem;
            align-items: center;
        }
        .hero-card h1 {
            margin: 0.7rem 0 0.45rem;
            font-size: 2.4rem;
            line-height: 1.05;
        }
        .hero-card p {
            margin: 0;
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.9);
        }
        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            justify-content: flex-end;
        }
        .hero-badge {
            min-width: 124px;
            padding: 0.85rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.12);
            text-align: center;
        }
        .hero-badge strong {
            display: block;
            font-size: 1.1rem;
        }
        .status-card {
            border-radius: 20px;
            padding: 1rem 1.1rem;
            min-height: 148px;
            box-shadow: 0 16px 34px rgba(9, 43, 79, 0.1);
        }
        .status-card h4, .quick-card h4 {
            margin: 0 0 0.4rem;
        }
        .status-card p, .quick-card p {
            margin: 0;
            color: #425466;
        }
        .status-pain {
            background: linear-gradient(180deg, #fff4e5 0%, #ffe6d5 100%);
        }
        .status-wheeze {
            background: linear-gradient(180deg, #eef9ff 0%, #dff0ff 100%);
        }
        .status-file {
            background: linear-gradient(180deg, #f4f7fb 0%, #e7eef8 100%);
        }
        .quick-card {
            background: #ffffff;
            border: 1px solid rgba(17, 105, 142, 0.12);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            height: 100%;
        }
        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .hero-badges {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.warning("No session logs found yet.")
    st.info("Run the live monitor first. After that, refresh this page and your latest session data will appear here.")

    quick_left, quick_right = st.columns(2)
    with quick_left:
        st.code("python pain_main.py live", language="bash")
    with quick_right:
        st.code("python -m streamlit run streamlit_pain_app.py", language="bash")


def render_overview(data: pd.DataFrame, selected_name: str, sessions_count: int) -> None:
    if data.empty:
        max_score = 0.0
        latest_total = 0.0
        latest_score = 0.0
        latest_wheeze = 0.0
        avg_score = 0.0
        episode_count = 0
        latest_time_label = "Not available"
    else:
        latest_row = data.iloc[-1]
        latest_score = float(data["pain_score_0_10"].dropna().iloc[-1]) if "pain_score_0_10" in data.columns and not data["pain_score_0_10"].dropna().empty else 0.0
        latest_wheeze = float(data["wheeze_probability"].dropna().iloc[-1]) if "wheeze_probability" in data.columns and not data["wheeze_probability"].dropna().empty else 0.0
        avg_score = float(data["pain_score_0_10"].mean()) if "pain_score_0_10" in data.columns else 0.0
        max_score = float(data["pain_score_0_10"].max()) if "pain_score_0_10" in data.columns else 0.0
        latest_total = (
            float(data["total_pain_duration_s"].dropna().iloc[-1])
            if "total_pain_duration_s" in data.columns and not data["total_pain_duration_s"].dropna().empty
            else 0.0
        )
        episode_count = len(summarize_episodes(data))
        latest_time = data["timestamp"].dropna().iloc[-1] if "timestamp" in data.columns and not data["timestamp"].dropna().empty else None
        latest_time_label = latest_time.strftime("%Y-%m-%d %H:%M:%S") if latest_time is not None else "Not available"
        latest_pain_status = classify_pain_status(latest_score, latest_row)
    if data.empty:
        latest_pain_status = classify_pain_status(0.0)

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <span class="hero-eyebrow">Live Clinical Dashboard</span>
                    <h1>Pain and wheeze details appear on the first page itself.</h1>
                    <p>Open the latest session, check the current patient status quickly, and review trends only when you need more detail.</p>
                </div>
                <div class="hero-badges">
                    <div class="hero-badge"><strong>{sessions_count}</strong><span>Sessions</span></div>
                    <div class="hero-badge"><strong>{max_score:.1f}/10</strong><span>Peak pain</span></div>
                    <div class="hero-badge"><strong>{latest_total:.1f}s</strong><span>Total pain time</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top = st.columns(3)
    top[0].markdown(
        f"""
        <div class="status-card status-pain">
            <h4>Current Pain Status</h4>
            <h2>{latest_pain_status}</h2>
            <p>Latest score: {latest_score:.2f}/10</p>
            <p>Average score: {avg_score:.2f}/10</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top[1].markdown(
        f"""
        <div class="status-card status-wheeze">
            <h4>Current Wheeze Status</h4>
            <h2>{classify_wheeze_status(latest_wheeze)}</h2>
            <p>Latest probability: {latest_wheeze:.2f}</p>
            <p>Detected episodes: {episode_count}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top[2].markdown(
        f"""
        <div class="status-card status-file">
            <h4>Active Session</h4>
            <h2>{selected_name}</h2>
            <p>Last update: {latest_time_label}</p>
            <p>Total pain duration: {latest_total:.1f}s</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick = st.columns(3)
    quick[0].markdown(
        """
        <div class="quick-card">
            <h4>How to use</h4>
            <p>Start the camera monitor, then open this dashboard to view the current session, pain trend, and wheeze trend in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quick[1].markdown(
        """
        <div class="quick-card">
            <h4>What changed</h4>
            <p>The camera side now shows clear pain and wheezing detection labels, and the first page highlights the main status before the detailed charts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quick[2].markdown(
        """
        <div class="quick-card">
            <h4>Best workflow</h4>
            <p>Use Overview for quick checks, Trends for charts, Episodes for summaries, and Data Export for logs or CSV downloads.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Pain Monitoring Dashboard", layout="wide")
    inject_styles()

    st.sidebar.title("Dashboard Controls")
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
        render_overview(pd.DataFrame(), "No session selected", 0)
        render_empty_state()
        return

    selected_name = st.sidebar.selectbox("Session file", [path.name for path in sessions])
    selected_path = next(path for path in sessions if path.name == selected_name)
    data = load_session(str(selected_path))

    if data.empty:
        st.error("Selected file is empty.")
        return

    render_overview(data, selected_name, len(sessions))

    overview_tab, trends_tab, episodes_tab, data_tab = st.tabs(["Overview", "Trends", "Episodes", "Data Export"])

    with overview_tab:
        st.markdown("### Current Snapshot")
        latest = data.iloc[-1]
        current = st.columns(4)
        current[0].metric("Latest pain", f"{float(latest.get('pain_score_0_10', 0.0)):.2f}/10")
        current[1].metric("Latest wheeze", f"{float(latest.get('wheeze_probability', 0.0)):.2f}")
        current[2].metric("Episode duration", f"{float(latest.get('current_episode_duration_s', 0.0)):.1f} sec")
        current[3].metric("Respiratory motion", f"{float(latest.get('respiratory_motion', 0.0)):.2f}")

        st.markdown("### Latest Records")
        preview_columns = [
            column
            for column in ["timestamp", "pain_score_0_10", "wheeze_probability", "pain_active", "current_episode_duration_s"]
            if column in data.columns
        ]
        if preview_columns:
            st.dataframe(data[preview_columns].tail(12), use_container_width=True, height=320)

    with trends_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("### Pain Score Over Time")
            if {"timestamp", "pain_score_0_10"}.issubset(data.columns):
                chart = data.sort_values("timestamp")[["timestamp", "pain_score_0_10"]]
                st.line_chart(chart, x="timestamp", y="pain_score_0_10", use_container_width=True)

        with right:
            st.markdown("### Wheeze Probability Over Time")
            if {"timestamp", "wheeze_probability"}.issubset(data.columns):
                chart = data.sort_values("timestamp")[["timestamp", "wheeze_probability"]]
                st.line_chart(chart, x="timestamp", y="wheeze_probability", use_container_width=True)

    with episodes_tab:
        st.markdown("### Episode Summary")
        st.dataframe(summarize_episodes(data), use_container_width=True, height=240)

        event_file = _find_episode_events_for_session(selected_path)
        if event_file is not None:
            st.markdown("### Episode Events")
            st.dataframe(load_episode_events(str(event_file)), use_container_width=True, height=220)
        else:
            st.info("No episode event file found for this session yet.")

    with data_tab:
        st.markdown("### Full Frame Log")
        st.dataframe(data, use_container_width=True, height=360)
        st.download_button(
            label="Download session CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=selected_name,
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
