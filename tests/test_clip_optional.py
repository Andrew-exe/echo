"""Optional integration test for the CLIP global encoder.

Skipped by default so the core test suite does not download model weights.
Enable after installing requirements-vision.txt:
    OBJECTECHO_TEST_CLIP=1 python -m pytest -q tests/test_clip_optional.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import make_object_image, perturbed, register_view
from config import AppConfig
from storage import Storage
from vision import ClipEncoder, LocalFeatureExtractor, RecognitionIndex


pytestmark = pytest.mark.skipif(
    os.environ.get("OBJECTECHO_TEST_CLIP") != "1",
    reason="set OBJECTECHO_TEST_CLIP=1 and install optional CLIP dependencies",
)


def test_clip_plus_orb_recognizes_after_restart(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    config = AppConfig(
        data_dir=tmp_path / "data",
        encoder_backend="clip",
        clip_device="cpu",
        show_preview=False,
    )
    encoder = ClipEncoder(config)
    extractor = LocalFeatureExtractor(config)
    with Storage(config.data_dir, config.db_path) as store:
        images = {}
        for kind, name in enumerate(("bottle", "backpack")):
            image = make_object_image(kind)
            obj = store.create_object(name=name)
            register_view(config, store, encoder, extractor, obj.object_id, image)
            images[obj.object_id] = image

    with Storage(config.data_dir, config.db_path) as reopened:
        index = RecognitionIndex.from_storage(reopened, config, encoder)
        query = perturbed(images[next(iter(images))], angle=7, scale=0.94, seed=99)
        result = index.recognize(query)
        assert result.status == "match"
        assert result.best is not None
        assert result.best.object_id == next(iter(images))
        assert result.best.global_cosine > 0.7
        assert result.best.local_score >= config.min_local_for_match
