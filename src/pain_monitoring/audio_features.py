from __future__ import annotations

import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass
class AudioFeatureSnapshot:
    respiratory_motion: float = 0.0
    wheeze_tonality: float = 0.0
    wheeze_band_energy: float = 0.0
    wheeze_entropy: float = 0.0
    wheeze_probability: float = 0.0


def _spectral_entropy(power: np.ndarray) -> float:
    if power.size == 0 or float(power.sum()) <= 1e-9:
        return 0.0
    distribution = power / power.sum()
    entropy = -np.sum(distribution * np.log(distribution + 1e-12)) / np.log(power.size)
    return float(1.0 - entropy)


def _compute_snapshot(samples: np.ndarray, sample_rate: int) -> AudioFeatureSnapshot:
    if samples.size < max(64, sample_rate // 10):
        return AudioFeatureSnapshot()

    signal = samples.astype(float)
    signal = signal - float(np.mean(signal))
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 1e-6:
        signal = signal / peak

    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    total_power = float(power.sum()) + 1e-9

    respiratory_motion = _clip01(float(np.sqrt(np.mean(signal**2))) * 2.2)
    wheeze_band = power[(freqs >= 350.0) & (freqs <= 1800.0)]
    wheeze_band_energy = _clip01(float(wheeze_band.sum()) / total_power * 2.8)

    peak_idx = int(np.argmax(power)) if power.size else 0
    local_span = power[max(0, peak_idx - 3) : peak_idx + 4]
    peak_power = float(power[peak_idx]) if power.size else 0.0
    local_energy = float(local_span.sum()) + 1e-9
    wheeze_tonality = _clip01(peak_power / local_energy)
    wheeze_entropy = _clip01(_spectral_entropy(power))

    wheeze_probability = _clip01(
        0.38 * wheeze_tonality
        + 0.34 * wheeze_band_energy
        + 0.18 * wheeze_entropy
        + 0.10 * respiratory_motion
    )
    return AudioFeatureSnapshot(
        respiratory_motion=respiratory_motion,
        wheeze_tonality=wheeze_tonality,
        wheeze_band_energy=wheeze_band_energy,
        wheeze_entropy=wheeze_entropy,
        wheeze_probability=wheeze_probability,
    )


class FileAudioFeatureProvider:
    def __init__(self, audio_path: Path, window_seconds: float = 1.5) -> None:
        self.audio_path = audio_path
        self.window_seconds = window_seconds
        self.samples, self.sample_rate = self._load_wav(audio_path)

    @staticmethod
    def _load_wav(audio_path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(audio_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if sample_width not in dtype_map:
            raise ValueError("Only 8-bit, 16-bit, and 32-bit PCM WAV files are supported for wheeze analysis.")

        audio = np.frombuffer(frames, dtype=dtype_map[sample_width]).astype(np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        scale = max(1.0, float(np.max(np.abs(audio))))
        return audio / scale, sample_rate

    def get_snapshot(self, elapsed_seconds: float) -> AudioFeatureSnapshot:
        center = int(max(0.0, elapsed_seconds) * self.sample_rate)
        radius = int(self.window_seconds * self.sample_rate / 2.0)
        start = max(0, center - radius)
        end = min(self.samples.size, center + radius)
        return _compute_snapshot(self.samples[start:end], self.sample_rate)


class MicrophoneAudioFeatureProvider:
    def __init__(self, sample_rate: int = 16000, window_seconds: float = 1.5, channels: int = 1) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for live microphone wheeze monitoring.") from exc

        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.channels = channels
        self._sd = sd
        self._buffer = deque(maxlen=int(sample_rate * window_seconds * 2.5))
        self._stream = sd.InputStream(
            channels=channels,
            samplerate=sample_rate,
            callback=self._callback,
            blocksize=max(256, sample_rate // 8),
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info, status
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata
        self._buffer.extend(float(item) for item in mono)

    def get_snapshot(self, elapsed_seconds: float | None = None) -> AudioFeatureSnapshot:
        del elapsed_seconds
        if not self._buffer:
            return AudioFeatureSnapshot()
        samples = np.fromiter(self._buffer, dtype=np.float32)
        return _compute_snapshot(samples, self.sample_rate)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
