"""Simulated UI workflow tests with fake camera frames and fake audio I/O.

These exercise the app's key-driven attach/retrieve logic without requiring a
webcam or microphone. Physical hardware still needs the manual smoke test in
README.md.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import make_object_image, perturbed, register_view
from config import AppConfig
from main import ObjectEchoApp
from vision import ClassicGlobalEncoder, LocalFeatureExtractor


def make_frame(image: np.ndarray) -> np.ndarray:
    """A 700x700 camera frame whose 60% center crop is exactly ``image``."""
    assert image.shape[:2] == (420, 420)
    frame = np.full((700, 700, 3), 238, dtype=np.uint8)
    frame[140:560, 140:560] = image
    return frame


def write_wav(path: Path) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(800, dtype=np.float32), 8_000, subtype="PCM_16")


@pytest.fixture
def app(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> ObjectEchoApp:
    monkeypatch.setattr(
        "audio_io.AudioIO.probe", lambda self: (True, "audio ready (fake)")
    )
    instance = ObjectEchoApp(config)
    # Never touch physical audio in these simulated UI tests.
    for cue in (
        "beep_start", "beep_end", "beep_success", "beep_ambiguous", "beep_no_match"
    ):
        monkeypatch.setattr(instance.audio, cue, lambda: True)
    instance.play_calls: list[Path] = []  # type: ignore[attr-defined]
    instance.recorded_seconds: list[float] = []  # type: ignore[attr-defined]

    def fake_record(path, seconds=None, **kwargs):
        write_wav(Path(path))
        instance.recorded_seconds.append(float(seconds))  # type: ignore[attr-defined]
        return 0.1

    def fake_play(path):
        instance.play_calls.append(Path(path))  # type: ignore[attr-defined]
        return 0.1

    monkeypatch.setattr(instance.audio, "record_note", fake_record)
    monkeypatch.setattr(instance.audio, "play_wav", fake_play)
    yield instance
    instance.storage.close()


def register_object(
    app: ObjectEchoApp, image: np.ndarray, name: str
) -> str:
    encoder = ClassicGlobalEncoder()
    extractor = LocalFeatureExtractor(app.config)
    obj = app.storage.create_object(name=name)
    register_view(
        app.config, app.storage, encoder, extractor, obj.object_id, image
    )
    app._reload_index()
    return obj.object_id


def test_new_object_attach_then_retrieve_same_app(app: ObjectEchoApp) -> None:
    image = make_object_image(0)
    app.camera.last_frame = make_frame(image)
    assert app.storage.list_objects() == []

    app.handle_key("a")
    objects = app.storage.list_objects()
    assert len(objects) == 1
    notes = app.storage.list_notes(objects[0].object_id)
    assert len(notes) == 1
    assert app.recorded_seconds == [app.config.record_seconds]
    path = app.storage.resolve_path(notes[0].audio_path)
    assert path.exists()
    assert app.index is not None and app.index.observation_count == 1

    app.camera.last_frame = make_frame(perturbed(image, angle=4, scale=0.97))
    app.handle_key("p")
    assert app.pending is None
    assert app.play_calls == [path]


def test_attach_to_existing_does_not_create_duplicate(app: ObjectEchoApp) -> None:
    image = make_object_image(1)
    object_id = register_object(app, image, "backpack")
    app.camera.last_frame = make_frame(image.copy())

    app.handle_key("a")
    assert len(app.storage.list_objects()) == 1
    notes = app.storage.list_notes(object_id)
    assert len(notes) == 1
    assert app.storage.count_observations(object_id) == 2


def test_ambiguous_attach_waits_for_explicit_selection(app: ObjectEchoApp) -> None:
    image = make_object_image(2)
    first = register_object(app, image, "same-one")
    second = register_object(app, image, "same-two")
    app.camera.last_frame = make_frame(image.copy())

    app.handle_key("a")
    assert app.pending is not None
    assert app.pending.kind == "attach"
    assert app.storage.list_notes(first) == []
    assert app.storage.list_notes(second) == []
    assert app.recorded_seconds == []

    chosen = app.pending.result.candidates[0].object_id
    app.confirm_best_pending()
    assert app.pending is None
    assert len(app.storage.list_notes(chosen)) == 1
    other = second if chosen == first else first
    assert app.storage.list_notes(other) == []


def test_ambiguous_retrieve_does_not_play_until_selection(app: ObjectEchoApp) -> None:
    image = make_object_image(0)
    first = register_object(app, image, "duplicate-one")
    second = register_object(app, image, "duplicate-two")
    first_audio = app.config.recordings_dir / "first.wav"
    second_audio = app.config.recordings_dir / "second.wav"
    write_wav(first_audio)
    write_wav(second_audio)
    app.storage.add_note(first, first_audio, 0.1)
    app.storage.add_note(second, second_audio, 0.1)
    app._reload_index()
    app.camera.last_frame = make_frame(image.copy())

    app.handle_key("p")
    assert app.pending is not None
    assert app.pending.kind == "retrieve"
    assert app.play_calls == []

    chosen = app.pending.result.candidates[0].object_id
    app.confirm_best_pending()
    expected = first_audio if chosen == first else second_audio
    assert len(app.play_calls) == 1
    assert app.play_calls[0].resolve() == expected.resolve()


def test_no_confident_retrieve_plays_nothing(app: ObjectEchoApp) -> None:
    register_object(app, make_object_image(1), "only-object")
    blank = np.full((700, 700, 3), 238, dtype=np.uint8)
    cv2.putText(blank, "UNKNOWN", (180, 360), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), 4, cv2.LINE_AA)
    app.camera.last_frame = blank

    app.handle_key("p")
    assert app.play_calls == []
    assert app.pending is not None
    assert app.status.startswith("No confident match") or "Uncertain" in app.status


def test_matched_object_without_notes_is_audible_no_notes(app: ObjectEchoApp) -> None:
    image = make_object_image(0)
    register_object(app, image, "silent-object")
    app.camera.last_frame = make_frame(image.copy())
    app.handle_key("p")
    assert app.play_calls == []
    assert "No notes" in app.status


def test_cancel_pending_changes_nothing(app: ObjectEchoApp) -> None:
    image = make_object_image(2)
    first = register_object(app, image, "one")
    second = register_object(app, image, "two")
    app.camera.last_frame = make_frame(image.copy())
    app.handle_key("a")
    assert app.pending is not None
    app.handle_key("x")
    assert app.pending is None
    assert app.storage.list_notes(first) == []
    assert app.storage.list_notes(second) == []
