"""End-to-end-ish recognition/storage workflows without physical hardware."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import (
    make_object_image,
    perturbed,
    register_view,
)
from config import AppConfig
from storage import Storage
from vision import ClassicGlobalEncoder, LocalFeatureExtractor, RecognitionIndex


def add_wav(path: Path, value: float = 0.0) -> Path:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.full(800, value, dtype=np.float32), 8_000, subtype="PCM_16")
    return path


def test_two_object_attach_and_retrieve_with_notes_persisted(
    config: AppConfig,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> None:
    with Storage(config.data_dir, config.db_path) as store:
        images: dict[str, np.ndarray] = {}
        notes: dict[str, Path] = {}
        for kind, (name, val) in enumerate((("bottle", 0.1), ("backpack", -0.1))):
            image = make_object_image(kind)
            obj = store.create_object(name=name)
            register_view(config, store, encoder, extractor, obj.object_id, image)
            note_path = add_wav(config.recordings_dir / f"{name}.wav", val)
            note = store.add_note(obj.object_id, note_path, 0.1)
            images[obj.object_id] = image
            notes[obj.object_id] = config.data_dir.resolve() / note.audio_path

    # Restart and use only persistence-loaded state.
    with Storage(config.data_dir, config.db_path) as store:
        index = RecognitionIndex.from_storage(store, config, encoder)
        for idx, (object_id, image) in enumerate(images.items()):
            query = perturbed(image, angle=9, scale=0.93, tx=10, ty=-5, seed=20 + idx)
            result = index.recognize(query)
            assert result.status == "match"
            assert result.best is not None and result.best.object_id == object_id
            stored_notes = store.list_notes(object_id)
            assert len(stored_notes) == 1
            assert store.resolve_path(stored_notes[0].audio_path).resolve() == notes[object_id].resolve()
            assert store.resolve_path(stored_notes[0].audio_path).exists()

        # A later note must be added without replacing the first.
        first_object = next(iter(images))
        second_audio = add_wav(config.recordings_dir / "second-note.wav", 0.2)
        store.add_note(first_object, second_audio, 0.1)
        assert store.count_notes(first_object) == 2
        assert len(store.list_notes(first_object)) == 2

    with Storage(config.data_dir, config.db_path) as restart:
        assert restart.count_notes(first_object) == 2
        index = RecognitionIndex.from_storage(restart, config, encoder)
        matched = index.recognize(perturbed(
            images[first_object], angle=5, scale=0.95, tx=4, ty=2, seed=33
        ))
        assert matched.status == "match"
        assert matched.best is not None
        assert matched.best.notes_count == 2


def test_near_duplicate_objects_are_ambiguous_not_silently_chosen(
    config: AppConfig,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> None:
    image = make_object_image(0)
    with Storage(config.data_dir, config.db_path) as store:
        first = store.create_object(name="copy-one")
        second = store.create_object(name="copy-two")
        register_view(config, store, encoder, extractor, first.object_id, image)
        register_view(config, store, encoder, extractor, second.object_id, image)
        index = RecognitionIndex.from_storage(store, config, encoder)

        query = perturbed(image, angle=3, scale=0.97, tx=3, ty=-2, seed=7)
        result = index.recognize(query)
        print(result.status, result.margin,
              [(c.object_id, c.fused_score) for c in result.candidates])
        assert {first.object_id, second.object_id} <= {
            c.object_id for c in result.candidates
        }
        # Identical visual evidence must never trigger an automatic association.
        assert result.status == "ambiguous"
        assert result.margin < config.min_margin
