from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import AppConfig
from storage import Storage


def make_wav(path: Path, sample_rate: int = 8_000, seconds: float = 0.1) -> Path:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def test_objects_observations_notes_persist_across_reopen(config: AppConfig) -> None:
    db = config.data_dir / "objectecho.sqlite3"
    image = config.objects_dir / "o1" / "view.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpeg")
    features = config.objects_dir / "o1" / "view_orb.npz"
    features.write_bytes(b"fake-npz")
    wav = make_wav(config.recordings_dir / "note1.wav")
    embedding = np.linspace(-1.0, 1.0, 16, dtype=np.float32)

    with Storage(config.data_dir, db) as store:
        obj = store.create_object(name="red mug")
        obs = store.add_observation(
            obj.object_id, image, features, embedding,
            embedding_model="test-encoder", image_width=320, image_height=320,
            metadata={"camera": "synthetic"},
        )
        assert obs.observation_id
        note = store.add_note(obj.object_id, wav, 0.1, transcript=None)
        object_id = obj.object_id
        note_id = note.note_id

    # Reopen exactly like an app restart.
    with Storage(config.data_dir, db) as reopened:
        objects = reopened.list_objects()
        assert len(objects) == 1
        assert objects[0].object_id == object_id
        assert objects[0].name == "red mug"
        observations = reopened.list_observations(object_id)
        assert len(observations) == 1
        assert observations[0].embedding_model == "test-encoder"
        np.testing.assert_allclose(
            np.asarray(observations[0].embedding, dtype=np.float32),
            embedding, atol=1e-6,
        )
        notes = reopened.list_notes(object_id)
        assert [n.note_id for n in notes] == [note_id]
        assert reopened.resolve_path(notes[0].audio_path) == wav.resolve()
        assert reopened.resolve_path(observations[0].image_path) == image.resolve()


def test_additional_notes_never_overwrite_previous_notes(config: AppConfig) -> None:
    wav_one = make_wav(config.recordings_dir / "one.wav")
    wav_two = make_wav(config.recordings_dir / "two.wav")
    before = wav_one.read_bytes()

    with Storage(config.data_dir, config.db_path) as store:
        obj = store.create_object()
        first = store.add_note(obj.object_id, wav_one, 0.1)
        second = store.add_note(obj.object_id, wav_two, 0.1)
        notes = store.list_notes(obj.object_id)
        assert len(notes) == 2
        assert first.note_id != second.note_id
        assert [n.note_id for n in notes] == [first.note_id, second.note_id]
        assert wav_one.read_bytes() == before

    with Storage(config.data_dir, config.db_path) as reopened:
        assert len(reopened.list_notes(obj.object_id)) == 2
        assert reopened.count_notes(obj.object_id) == 2


def test_unknown_object_rejected_and_delete_cascades(config: AppConfig) -> None:
    wav = make_wav(config.recordings_dir / "orphan.wav")
    with Storage(config.data_dir, config.db_path) as store:
        with pytest.raises(ValueError):
            store.add_note("missing-id", wav, 0.1)
        store.create_object("fixed-id")
        with pytest.raises(ValueError):
            store.create_object("fixed-id")
        obj = store.create_object("to-delete")
        store.add_note(obj.object_id, wav, 0.1)
        store.delete_object(obj.object_id)
        assert store.get_object(obj.object_id) is None
        assert store.list_notes(obj.object_id) == []


def test_relative_paths_survive_data_dir_move(tmp_path: Path) -> None:
    data = tmp_path / "first"
    cfg = AppConfig(data_dir=data, encoder_backend="classic", show_preview=False)
    audio = make_wav(cfg.recordings_dir / "keep.wav")
    with Storage(cfg.data_dir, cfg.db_path) as store:
        obj = store.create_object(name="portable")
        note = store.add_note(obj.object_id, audio, 0.1)
        raw_path = note.audio_path
        assert not Path(raw_path).is_absolute()

    moved = tmp_path / "second"
    data.rename(moved)
    with Storage(moved, moved / "objectecho.sqlite3") as store:
        notes = store.list_notes()
        assert len(notes) == 1
        assert store.resolve_path(notes[0].audio_path).exists()
