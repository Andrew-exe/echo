from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import audio_io
from audio_io import AudioDeviceError, AudioError, AudioIO
from config import AppConfig


class FakeDefault:
    device = (-1, -1)


class FakeSD:
    default = FakeDefault()

    def __init__(self, fail_record: bool = False):
        self.events: list[str] = []
        self.fail_record = fail_record

    def query_devices(self):
        return [
            {
                "name": "Fake Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
                "hostapi": 0,
            },
            {
                "name": "Fake Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000.0,
                "hostapi": 0,
            },
        ]

    def check_input_settings(self, **kwargs):
        if int(kwargs.get("samplerate", 0)) not in (44_100, 48_000):
            raise RuntimeError("unsupported")
        return None

    def check_output_settings(self, **kwargs):
        if int(kwargs.get("samplerate", 0)) not in (44_100, 48_000):
            raise RuntimeError("unsupported")
        return None

    def play(self, data, **kwargs):
        self.events.append("play")
        assert kwargs.get("device") in (None, 0, 1)
        assert len(data) > 0
        return None

    def rec(self, frames, **kwargs):
        self.events.append("rec")
        if self.fail_record:
            raise RuntimeError("permission denied")
        return np.zeros((int(frames), 1), dtype=np.float32)

    def wait(self):
        self.events.append("wait")
        return None


def test_record_note_writes_wav_and_cues_surround_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data", encoder_backend="classic",
        show_preview=False, beep_start_delay=0.0,
    )
    fake = FakeSD()
    monkeypatch.setattr(audio_io, "_load_sounddevice", lambda: fake)
    audio = AudioIO(config)
    destination = tmp_path / "note.wav"

    duration = audio.record_note(destination, 0.15, start_cue=True, end_cue=True)

    assert destination.exists()
    assert 0.14 <= duration <= 0.16
    import soundfile as sf
    data, sr = sf.read(str(destination))
    assert sr == 48_000
    assert len(data) == int(0.15 * 48_000)
    # The last cue play must happen after microphone capture has stopped.
    capture_index = fake.events.index("rec")
    wait_index = fake.events.index("wait")
    assert fake.events[capture_index:wait_index + 1] == ["rec", "wait"]
    play_indices = [i for i, event in enumerate(fake.events) if event == "play"]
    assert len(play_indices) >= 3  # start cue (2 tones), end cue (2 tones)
    assert play_indices[0] < capture_index
    assert play_indices[-1] > wait_index


def test_record_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data", encoder_backend="classic",
        show_preview=False, beep_start_delay=0.0,
    )
    fake = FakeSD(fail_record=True)
    monkeypatch.setattr(audio_io, "_load_sounddevice", lambda: fake)
    audio = AudioIO(config)
    with pytest.raises(AudioDeviceError, match="Microphone recording failed"):
        audio.record_note(tmp_path / "fail.wav", 0.05, start_cue=False, end_cue=False)
    assert not (tmp_path / "fail.wav").exists()


def test_play_wav_missing_file_does_not_need_audio_device(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data", encoder_backend="classic",
        show_preview=False,
    )
    audio = AudioIO(config)
    with pytest.raises(AudioError, match="not found"):
        audio.play_wav(tmp_path / "does-not-exist.wav")


def test_play_wav_uses_output_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data", encoder_backend="classic",
        show_preview=False,
    )
    fake = FakeSD()
    monkeypatch.setattr(audio_io, "_load_sounddevice", lambda: fake)
    import soundfile as sf
    path = tmp_path / "tone.wav"
    sf.write(str(path), np.zeros(4_800, dtype=np.float32), 48_000, subtype="PCM_16")
    audio = AudioIO(config)
    duration = audio.play_wav(path)
    assert 0.09 <= duration <= 0.11
    assert "play" in fake.events


def test_tone_window_has_fades_and_expected_length() -> None:
    tone = AudioIO._tone_samples(880, 0.1, 48_000, 0.2)
    assert tone.dtype == np.float32
    assert len(tone) == 4_800
    assert abs(float(tone[0])) < 1e-6
    assert abs(float(tone[-1])) < 1e-6
    assert np.max(np.abs(tone)) <= 0.22
