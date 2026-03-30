from __future__ import annotations

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.types import DurationStatus, EpisodeState, EpisodeSummary, RuntimeState


def _finalize_episode(runtime: RuntimeState, end_time: float) -> EpisodeSummary:
    assert runtime.active_episode is not None
    episode = runtime.active_episode
    duration = max(0.0, end_time - episode.start_time)
    avg_score = episode.score_sum / max(1, episode.frame_count)
    summary = EpisodeSummary(
        episode_id=episode.episode_id,
        start_time=episode.start_time,
        end_time=end_time,
        duration_seconds=duration,
        max_score=episode.max_score,
        avg_score=avg_score,
    )
    runtime.closed_duration_s += duration
    runtime.closed_episodes.append(summary)
    runtime.active_episode = None
    runtime.time_below_end = 0.0
    return summary


def update_duration_state(
    runtime: RuntimeState,
    score_0_10: float,
    timestamp: float,
    config: PainMonitoringConfig,
) -> DurationStatus:
    if runtime.previous_timestamp is None:
        dt = 0.0
    else:
        dt = max(0.0, timestamp - runtime.previous_timestamp)
    runtime.previous_timestamp = timestamp

    runtime.smoothed_score = (
        score_0_10
        if runtime.smoothed_score <= 0.0
        else config.smoothing_alpha * score_0_10 + (1.0 - config.smoothing_alpha) * runtime.smoothed_score
    )

    started_now = False
    ended_now = False
    finished_episode = None

    if runtime.active_episode is None:
        if runtime.smoothed_score >= config.pain_start_threshold:
            runtime.time_above_start += dt
        else:
            runtime.time_above_start = 0.0

        if runtime.time_above_start >= config.start_hold_seconds:
            runtime.episode_counter += 1
            start_time = max(timestamp - runtime.time_above_start, 0.0)
            runtime.active_episode = EpisodeState(
                episode_id=runtime.episode_counter,
                start_time=start_time,
                last_time=timestamp,
                max_score=runtime.smoothed_score,
                score_sum=runtime.smoothed_score,
                frame_count=1,
            )
            runtime.time_above_start = 0.0
            started_now = True
    else:
        episode = runtime.active_episode
        episode.last_time = timestamp
        episode.max_score = max(episode.max_score, runtime.smoothed_score)
        episode.score_sum += runtime.smoothed_score
        episode.frame_count += 1

        if runtime.smoothed_score <= config.pain_end_threshold:
            runtime.time_below_end += dt
        else:
            runtime.time_below_end = 0.0

        if runtime.time_below_end >= config.end_hold_seconds:
            end_time = max(timestamp - runtime.time_below_end, episode.start_time)
            finished_episode = _finalize_episode(runtime, end_time)
            ended_now = True

    active_duration = 0.0
    active_episode_id = None
    if runtime.active_episode is not None:
        active_duration = max(0.0, timestamp - runtime.active_episode.start_time)
        active_episode_id = runtime.active_episode.episode_id

    total_duration = runtime.closed_duration_s + active_duration

    return DurationStatus(
        pain_active=runtime.active_episode is not None,
        active_episode_id=active_episode_id,
        current_episode_duration_s=active_duration,
        total_pain_duration_s=total_duration,
        started_now=started_now,
        ended_now=ended_now,
        finished_episode=finished_episode,
    )
