"""Microphone recording, WAV playback, audible cues, and optional TTS.

``sounddevice`` is imported lazily. This matters on Linux systems without
PortAudio and lets object registration/storage tests run on headless machines.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from config import AppConfig

_SD: Any = None


class AudioError(RuntimeError):
    """Base error with a message suitable for showing to a user."""


class AudioUnavailableError(AudioError):
    pass


class AudioDeviceError(AudioError):
    pass


def _load_sounddevice() -> Any:
    global _SD
    if _SD is not None:
        return _SD
    try:
        import sounddevice as sd  # type: ignore
    except (ImportError, OSError) as exc:
        detail = str(exc)
        if "PortAudio" in detail or "library" in detail.lower():
            hint = (
                "PortAudio could not be loaded. On Windows, reinstall with "
                "`pip install --force-reinstall sounddevice`; on Linux install "
                "`libportaudio2` (for example `sudo apt install libportaudio2`)."
            )
        else:
            hint = "Install audio support with `pip install sounddevice`."
        raise AudioUnavailableError(f"{hint} Original error: {detail}") from exc
    _SD = sd
    return sd


class AudioIO:
    """Small, defensive wrapper around sounddevice + soundfile."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.input_device = config.input_device
        self.output_device = config.output_device
        self._requested_sr = int(config.sample_rate)
        self._io_lock = threading.RLock()
        self._input_warned = False
        self._output_warned = False

    # --------------------------------------------------------------- devices
    @staticmethod
    def print_devices() -> None:
        try:
            sd = _load_sounddevice()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as exc:
            print(f"Audio unavailable: {exc}", file=sys.stderr)
            return
        try:
            default_in, default_out = sd.default.device
        except Exception:
            default_in, default_out = -1, -1
        print("Audio devices (use the numeric index with --input-device/--output-device):")
        for index, dev in enumerate(devices):
            try:
                host = hostapis[int(dev["hostapi"])]["name"]
            except Exception:
                host = "?"
            marks = []
            if index == default_in:
                marks.append("default input")
            if index == default_out:
                marks.append("default output")
            mark = f"  [{', '.join(marks)}]" if marks else ""
            print(
                f"  {index:2d}: {dev['name']}\n"
                f"      in={int(dev['max_input_channels'])}, "
                f"out={int(dev['max_output_channels'])}, "
                f"default_sr={float(dev['default_samplerate']):.0f}, host={host}{mark}"
            )
        if len(devices) == 0:
            print("  (no devices reported)")
        print("If a device is missing, check Windows microphone/camera privacy settings.")

    def _device_list(self, sd: Any) -> Any:
        try:
            return sd.query_devices()
        except Exception as exc:
            raise AudioDeviceError(f"Could not enumerate audio devices: {exc}") from exc

    def _resolve_device(self, spec: int | str | None, kind: str) -> int | None:
        """Resolve an index or a case-insensitive name substring."""
        sd = _load_sounddevice()
        devices = self._device_list(sd)
        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        if spec is None:
            try:
                default = sd.default.device[0 if kind == "input" else 1]
                if default is None or int(default) < 0:
                    return None
                return int(default)
            except Exception:
                return None
        if isinstance(spec, int):
            if not 0 <= spec < len(devices):
                raise AudioDeviceError(
                    f"{kind} device index {spec} does not exist. "
                    "Run `python main.py --list-devices`."
                )
            if int(devices[spec][channel_key]) < 1:
                raise AudioDeviceError(
                    f"Device {spec} ({devices[spec]['name']}) has no {kind} channels. "
                    "Run `python main.py --list-devices`."
                )
            return spec
        needle = str(spec).casefold()
        matches = [
            (i, d) for i, d in enumerate(devices)
            if int(d[channel_key]) >= 1 and needle in str(d["name"]).casefold()
        ]
        if not matches:
            raise AudioDeviceError(
                f"No {kind} device matches '{spec}'. "
                "Run `python main.py --list-devices`."
            )
        # Prefer an exact (case-insensitive) name, otherwise the first match.
        exact = [m for m in matches if str(m[1]["name"]).casefold() == needle]
        index = (exact or matches)[0][0]
        return int(index)

    @staticmethod
    def _safe_name(devices: Any, index: int | None) -> str:
        if index is None:
            return "system default"
        try:
            return f"{index} ({devices[index]['name']})"
        except Exception:
            return str(index)

    def _choose_rate(self, sd: Any, device: int | None, kind: str) -> int:
        devices = self._device_list(sd)
        candidates: list[int] = [self._requested_sr]
        try:
            default_rate = int(round(float(devices[device]["default_samplerate"])))
        except Exception:
            default_rate = 48_000
        for rate in (default_rate, 48_000, 44_100):
            if rate not in candidates:
                candidates.append(rate)
        check = sd.check_input_settings if kind == "input" else sd.check_output_settings
        errors = []
        for rate in candidates:
            try:
                check(device=device, channels=1, samplerate=rate, dtype="float32")
                return rate
            except Exception as exc:
                errors.append(f"{rate} Hz: {exc}")
        raise AudioDeviceError(
            f"Could not find a working sample rate for {kind} device "
            f"{self._safe_name(devices, device)}. Tried " + "; ".join(errors)
        )

    # ------------------------------------------------------------------ tone
    @staticmethod
    def _tone_samples(freq: float, duration: float, sr: int, volume: float) -> np.ndarray:
        n = max(1, int(round(duration * sr)))
        t = np.arange(n, dtype=np.float32) / float(sr)
        wave = np.sin(2.0 * np.pi * float(freq) * t, dtype=np.float32)
        # Short fades suppress clicks at tone boundaries.
        fade_n = min(int(0.008 * sr), n // 2)
        if fade_n > 0:
            fade = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            wave[:fade_n] *= fade
            wave[-fade_n:] *= fade[::-1]
        return np.clip(wave * float(volume), -1.0, 1.0)

    def _play_tone(self, freq: float, duration: float) -> None:
        sd = _load_sounddevice()
        with self._io_lock:
            out_dev = self._resolve_device(self.output_device, "output")
            sr = self._choose_rate(sd, out_dev, "output")
            tone = self._tone_samples(freq, duration, sr, self.config.beep_volume)
            sd.play(tone, samplerate=sr, device=out_dev, blocking=True)

    def _safe_tone_sequence(self, sequence: Iterable[tuple[float, float, float]]) -> bool:
        """Play (frequency, duration, following_pause) triples; never crash the app."""
        try:
            for freq, duration, pause in sequence:
                self._play_tone(freq, duration)
                if pause > 0:
                    time.sleep(pause)
            return True
        except AudioError as exc:
            if not self._output_warned:
                print(
                    f"[audio] Audible cues disabled: {exc}",
                    file=sys.stderr,
                )
                self._output_warned = True
            return False
        except Exception as exc:  # PortAudio can raise backend-specific errors.
            if not self._output_warned:
                print(f"[audio] Audible cue failed: {exc}", file=sys.stderr)
                self._output_warned = True
            return False

    def beep_start(self) -> bool:
        return self._safe_tone_sequence([(740, 0.09, 0.05), (1040, 0.12, 0.0)])

    def beep_end(self) -> bool:
        return self._safe_tone_sequence([(1040, 0.08, 0.04), (740, 0.14, 0.0)])

    def beep_success(self) -> bool:
        return self._safe_tone_sequence([(1200, 0.09, 0.0)])

    def beep_ambiguous(self) -> bool:
        return self._safe_tone_sequence([(620, 0.11, 0.08), (620, 0.11, 0.0)])

    def beep_no_match(self) -> bool:
        return self._safe_tone_sequence(
            [(300, 0.12, 0.06), (240, 0.12, 0.06), (180, 0.18, 0.0)]
        )

    # ------------------------------------------------------------- recording
    def record_note(
        self,
        path: Path | str,
        seconds: float | None = None,
        *,
        start_cue: bool = True,
        end_cue: bool = True,
    ) -> float:
        """Record a mono WAV and return its actual duration.

        The start cue completes before the microphone stream opens, and the end
        cue is played after it closes, so cue tones are not part of the note.
        """
        if seconds is None:
            seconds = self.config.record_seconds
        if seconds <= 0:
            raise ValueError("Recording duration must be positive")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(destination.stem + ".tmp.wav")
        try:
            sd = _load_sounddevice()
            with self._io_lock:
                in_dev = self._resolve_device(self.input_device, "input")
                sr = self._choose_rate(sd, in_dev, "input")
                if start_cue:
                    # Use a separate call so a missing speaker does not prevent recording.
                    try:
                        out_dev = self._resolve_device(self.output_device, "output")
                        out_sr = self._choose_rate(sd, out_dev, "output")
                        for freq, duration in ((740, 0.09), (1040, 0.12)):
                            tone = self._tone_samples(
                                freq, duration, out_sr, self.config.beep_volume
                            )
                            sd.play(tone, samplerate=out_sr, device=out_dev, blocking=True)
                    except Exception as exc:
                        if not self._output_warned:
                            print(f"[audio] Start cue unavailable: {exc}", file=sys.stderr)
                            self._output_warned = True
                if start_cue and self.config.beep_start_delay > 0:
                    time.sleep(self.config.beep_start_delay)

                frames = max(1, int(round(float(seconds) * sr)))
                try:
                    recording = sd.rec(
                        frames,
                        samplerate=sr,
                        channels=1,
                        dtype="float32",
                        device=in_dev,
                    )
                    sd.wait()
                except Exception as exc:
                    device_name = self._safe_name(self._device_list(sd), in_dev)
                    raise AudioDeviceError(
                        f"Microphone recording failed on {device_name}: {exc}. "
                        "Check Windows microphone privacy/permissions and "
                        "`python main.py --list-devices`."
                    ) from exc

            recording = np.asarray(recording, dtype=np.float32)
            if recording.ndim > 1:
                recording = recording[:, 0]
            if recording.size == 0:
                raise AudioDeviceError("Microphone returned no samples")
            if not np.all(np.isfinite(recording)):
                recording = np.nan_to_num(recording, nan=0.0, posinf=0.0, neginf=0.0)
            recording = np.clip(recording, -1.0, 1.0)
            try:
                import soundfile as sf
                sf.write(str(temp_path), recording, sr, subtype="PCM_16", format="WAV")
            except Exception as exc:
                raise AudioError(f"Could not write WAV file {destination}: {exc}") from exc
            os.replace(temp_path, destination)
            duration = float(recording.shape[0]) / float(sr)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
        if end_cue:
            self.beep_end()
        return duration

    # -------------------------------------------------------------- playback
    def play_wav(self, path: Path | str) -> float:
        """Play a WAV file through the selected output device; return its duration."""
        audio_path = Path(path)
        if not audio_path.exists():
            raise AudioError(f"Audio file not found: {audio_path}")
        try:
            import soundfile as sf
            data, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise AudioError(f"Could not read WAV {audio_path}: {exc}") from exc
        data = np.asarray(data, dtype=np.float32)
        if data.size == 0:
            raise AudioError(f"WAV file is empty: {audio_path}")
        actual_duration = float(data.shape[0]) / float(file_sr)
        try:
            sd = _load_sounddevice()
            with self._io_lock:
                out_dev = self._resolve_device(self.output_device, "output")
                out_sr = self._choose_rate(sd, out_dev, "output")
                if out_sr != file_sr:
                    # Lightweight linear resampling avoids failing solely because
                    # the configured sample rate is unsupported by this device.
                    new_length = max(1, int(round(data.shape[0] * out_sr / file_sr)))
                    old_positions = np.arange(data.shape[0], dtype=np.float64)
                    new_positions = np.linspace(
                        0.0, float(data.shape[0] - 1), new_length, dtype=np.float64
                    )
                    if data.ndim == 1:
                        data = np.interp(new_positions, old_positions, data).astype(np.float32)
                    else:
                        data = np.stack(
                            [
                                np.interp(new_positions, old_positions, data[:, ch])
                                for ch in range(data.shape[1])
                            ],
                            axis=1,
                        ).astype(np.float32)
                sd.play(data, samplerate=out_sr, device=out_dev, blocking=True)
        except AudioError:
            raise
        except Exception as exc:
            raise AudioDeviceError(
                f"Audio playback failed: {exc}. Check the output device with "
                "`python main.py --list-devices`."
            ) from exc
        return actual_duration

    def play_files(self, paths: Iterable[Path | str]) -> int:
        played = 0
        for path in paths:
            self.play_wav(path)
            played += 1
        return played

    # ------------------------------------------------------------- convenience
    def probe(self) -> tuple[bool, str]:
        """Try to initialize input and output without recording/playing."""
        try:
            sd = _load_sounddevice()
            in_dev = self._resolve_device(self.input_device, "input")
            out_dev = self._resolve_device(self.output_device, "output")
            self._choose_rate(sd, in_dev, "input")
            self._choose_rate(sd, out_dev, "output")
            return True, (
                f"audio ready (input={self._safe_name(self._device_list(sd), in_dev)}, "
                f"output={self._safe_name(self._device_list(sd), out_dev)})"
            )
        except AudioError as exc:
            return False, str(exc)


class TTS:
    """Optional offline TTS. The app remains fully usable when it is unavailable."""

    def __init__(self, enabled: bool = False, rate: int = 175):
        self.enabled = bool(enabled)
        self.rate = int(rate)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._warned = False
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()

    def _ensure_worker(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts")
        self._thread.start()

    def _worker(self) -> None:
        engine = None
        while True:
            text = self._queue.get()
            if text is None:
                break
            if not text:
                continue
            try:
                if engine is None:
                    import pyttsx3  # type: ignore
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.rate)
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                with self._lock:
                    if not self._warned:
                        print(
                            f"[tts] Offline TTS unavailable ({exc}). "
                            "Install optional support: pip install pyttsx3",
                            file=sys.stderr,
                        )
                        self._warned = True
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass
                engine = None
            finally:
                with self._lock:
                    self._pending = max(0, self._pending - 1)
                    if self._pending == 0:
                        self._idle.set()

    def speak(self, text: str) -> None:
        if not self.enabled or not text:
            return
        try:
            self._ensure_worker()
            with self._lock:
                self._pending += 1
                self._idle.clear()
                self._queue.put(text)
        except Exception as exc:
            if not self._warned:
                print(f"[tts] Could not queue speech: {exc}", file=sys.stderr)
                self._warned = True

    def wait(self, timeout: float = 10.0) -> bool:
        """Wait for queued speech so it does not overlap recorded/played audio."""
        if not self.enabled:
            return True
        return bool(self._idle.wait(max(0.0, timeout)))

    def close(self) -> None:
        if self.enabled and self._thread and self._thread.is_alive():
            self._queue.put(None)
