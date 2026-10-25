from __future__ import annotations

from pathlib import Path

import numpy as np

from config import AppConfig
from main import run_reset_memory
from storage import Storage


def make_wav(path: Path) -> Path:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(400, dtype=np.float32), 8_000, subtype="PCM_16")
    return path


def test_reset_memory_deletes_database_reference_images_and_notes(
    config: AppConfig,
) -> None:
    image = config.objects_dir / "object-a" / "view.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpeg")
    features = image.with_name("view_orb.npz")
    features.write_bytes(b"fake-orb")
    wav = make_wav(config.recordings_dir / "object-a" / "note.wav")
    kept = config.data_dir / "unrelated.txt"
    kept.write_text("keep me", encoding="utf-8")

    with Storage(config.data_dir, config.db_path) as store:
        obj = store.create_object(name="Temporary")
        store.add_observation(obj.object_id, image, features, np.ones(8), "test", 420, 420)
        store.add_note(obj.object_id, wav, 0.05)
    assert config.db_path.exists()

    assert run_reset_memory(config, assume_yes=True) == 0
    assert not config.db_path.exists()
    assert not image.exists()
    assert not features.exists()
    assert not wav.exists()
    assert kept.exists(), "reset must only touch ObjectEcho runtime data"

    with Storage(config.data_dir, config.db_path) as reopened:
        assert reopened.list_objects() == []
        assert reopened.list_observations() == []
        assert reopened.list_notes() == []
