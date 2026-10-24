from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from conftest import (
    make_object_image,
    perturbed,
    register_view,
)
from config import AppConfig
from storage import Storage
from vision import (
    ClassicGlobalEncoder,
    LocalFeatureExtractor,
    MatchCandidate,
    ObservationData,
    ObjectData,
    RecognitionIndex,
    center_crop,
    classify_candidates,
    cosine_similarity,
    global_evidence,
    load_local_features,
    match_local_features,
    normalize_embedding,
    save_local_features,
)


def populate_three_objects(
    config: AppConfig,
    storage: Storage,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for kind, name in enumerate(("bottle", "backpack", "lamp")):
        image = make_object_image(kind)
        obj = storage.create_object(name=name)
        register_view(config, storage, encoder, extractor, obj.object_id, image)
        images[obj.object_id] = image
    return images


def test_center_crop_is_centered_square() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop, (x, y, w, h) = center_crop(frame, 0.6)
    assert (w, h) == (432, 432)
    assert (x, y) == ((1280 - 432) // 2, (720 - 432) // 2)
    assert crop.shape == (432, 432, 3)
    assert crop is not frame


def test_embedding_normalization_and_cosine(
    encoder: ClassicGlobalEncoder,
) -> None:
    base = make_object_image(0)
    other = make_object_image(1)
    a = encoder.encode(base)
    b = encoder.encode(base)
    c = encoder.encode(other)
    assert a.shape == (encoder.dimension,)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-6)
    assert cosine_similarity(a, b) > 0.999
    assert cosine_similarity(a, c) < cosine_similarity(a, b)
    zero = normalize_embedding(np.zeros(8, dtype=np.float32))
    assert np.all(zero == 0)
    assert global_evidence(1.0, "classic", AppConfig()) == 1.0
    assert global_evidence(0.0, "clip", AppConfig()) == 0.0


def test_orb_features_are_rotation_and_noise_tolerant(
    config: AppConfig, extractor: LocalFeatureExtractor
) -> None:
    base = make_object_image(1)
    query = perturbed(base, angle=10, scale=0.92, tx=10, ty=6, seed=2)
    ref_features = extractor.extract(base)
    query_features = extractor.extract(query)
    assert ref_features.count > 100
    assert query_features.count > 100
    evidence = match_local_features(query_features, ref_features, config)
    assert evidence.good_matches >= 20
    assert evidence.inliers >= 20
    assert evidence.score > 0.60
    assert evidence.score >= config.min_local_for_match


def test_orb_rejects_different_object(
    config: AppConfig, extractor: LocalFeatureExtractor
) -> None:
    a = extractor.extract(make_object_image(0))
    b = extractor.extract(make_object_image(2))
    evidence = match_local_features(a, b, config)
    assert evidence.score < config.min_local_for_match


def test_local_feature_roundtrip(
    config: AppConfig, extractor: LocalFeatureExtractor, tmp_path: Path
) -> None:
    features = extractor.extract(make_object_image(2))
    path = tmp_path / "features.npz"
    save_local_features(path, features)
    loaded = load_local_features(path)
    assert loaded is not None
    assert loaded.count == features.count
    np.testing.assert_allclose(loaded.keypoints, features.keypoints)
    np.testing.assert_array_equal(loaded.descriptors, features.descriptors)
    assert (loaded.image_width, loaded.image_height) == (
        features.image_width, features.image_height
    )
    path.write_bytes(b"not an npz")
    assert load_local_features(path, warn=False) is None


def test_conservative_decision_logic(config: AppConfig) -> None:
    def candidate(fused: float, local: float) -> MatchCandidate:
        return MatchCandidate(
            object_id="abc", name=None, fused_score=fused, local_score=local
        )

    good = candidate(0.85, 0.80)
    status, _ = classify_candidates([good], margin=0.50, config=config)
    assert status == "match"

    semantic_only = candidate(0.80, 0.10)
    status, reason = classify_candidates([semantic_only], margin=0.50, config=config)
    assert status == "ambiguous"
    assert "ORB" in reason

    close = candidate(0.80, 0.70)
    status, reason = classify_candidates([close], margin=0.01, config=config)
    assert status == "ambiguous"
    assert "margin" in reason

    low = candidate(0.20, 0.10)
    status, _ = classify_candidates([low], margin=0.20, config=config)
    assert status == "unknown"
    assert classify_candidates([], margin=0.0, config=config)[0] == "unknown"


def test_index_recognizes_correct_object_after_storage_reload(
    config: AppConfig,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> None:
    with Storage(config.data_dir, config.db_path) as storage:
        images = populate_three_objects(config, storage, encoder, extractor)

    # Simulate a restart: new Storage connection, embeddings/features reloaded.
    with Storage(config.data_dir, config.db_path) as reopened:
        index = RecognitionIndex.from_storage(reopened, config, encoder)
        assert index.object_count == 3
        assert index.observation_count == 3
        object_items = list(images.items())
        for idx, (object_id, base) in enumerate(object_items):
            query = perturbed(
                base, angle=6 + idx * 3, scale=0.94, tx=8 - idx * 4,
                ty=-5 + idx * 2, seed=10 + idx,
            )
            result = index.recognize(query)
            print(idx, result.status, result.margin, [
                (c.object_id, c.fused_score, c.local_score) for c in result.candidates
            ])
            assert result.status == "match"
            assert result.best is not None
            assert result.best.object_id == object_id
            assert result.margin >= config.min_margin
            assert result.best.local_score >= config.min_local_for_match


def test_index_rejects_unrelated_scene_as_automatic_match(
    config: AppConfig,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> None:
    with Storage(config.data_dir, config.db_path) as storage:
        populate_three_objects(config, storage, encoder, extractor)
        index = RecognitionIndex.from_storage(storage, config, encoder)

    scene = np.full((420, 420, 3), 238, dtype=np.uint8)
    cv2.putText(scene, "HELLO", (70, 230), cv2.FONT_HERSHEY_SIMPLEX,
                2.5, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(scene, (40, 40), (380, 380), (0, 80, 240), 5)
    result = index.recognize(scene)
    print("unrelated:", result.status, result.reason,
          [(c.fused_score, c.local_score) for c in result.candidates])
    assert result.status != "match"
    assert result.best is None or result.best.local_score < config.min_local_for_match


def test_multiple_reference_views_are_loaded(
    config: AppConfig,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> None:
    base = make_object_image(0)
    second = perturbed(base, angle=-14, scale=0.88, tx=-14, ty=10, seed=5)
    with Storage(config.data_dir, config.db_path) as storage:
        obj = storage.create_object(name="multi-view")
        register_view(config, storage, encoder, extractor, obj.object_id, base)
        register_view(config, storage, encoder, extractor, obj.object_id, second)
        assert storage.count_observations(obj.object_id) == 2
        index = RecognitionIndex.from_storage(storage, config, encoder)
        assert index.observation_count == 2
        query = perturbed(base, angle=4, scale=0.95, tx=6, ty=-4, seed=9)
        result = index.recognize(query)
        assert result.status == "match"
        assert result.best is not None and result.best.object_id == obj.object_id


def test_orb_only_fallback_when_embedding_backend_changed(
    config: AppConfig,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> None:
    base = make_object_image(1)
    query_image = perturbed(base, angle=6, scale=0.95, seed=13)
    # Simulate references saved by a different global encoder. The local ORB
    # channel must still support complete recognition.
    observation = ObservationData(
        observation_id="obs",
        object_id="object",
        image_path=Path("unused.jpg"),
        embedding=encoder.encode(base),
        embedding_model="some-other-encoder",
        local=extractor.extract(base),
    )
    obj = ObjectData("object", "fallback", [observation], notes_count=0)
    index = RecognitionIndex(config, encoder, [obj])
    result = index.recognize(query_image)
    assert result.status == "match"
    assert result.best is not None
    assert result.best.global_score == 0.0
    assert result.best.local_score >= config.min_local_for_match


def _textured_background(seed: int, size: int = 420) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 180, dtype=np.uint8)
    for _ in range(100):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        radius = int(rng.integers(2, 18))
        color = tuple(int(c) for c in rng.integers(30, 220, 3))
        cv2.circle(image, (x, y), radius, color, -1)
    return cv2.GaussianBlur(image, (9, 9), 1.5)


def _composite_on_background(
    obj: np.ndarray, background: np.ndarray, indent: int = 0
) -> np.ndarray:
    """Paste a synthetic object's central region over a textured background."""
    h, w = obj.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.37), int(h * 0.45)),
                0, 0, 360, 255, -1)
    scene = background.copy()
    scene[mask > 0] = obj[mask > 0]
    if indent:
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 0.95)
        matrix[0, 2] += indent
        scene = cv2.warpAffine(scene, matrix, (w, h),
                              borderMode=cv2.BORDER_REFLECT_101)
    return scene


def test_static_background_does_not_create_false_instance_match(
    config: AppConfig,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
) -> None:
    background = _textured_background(77)
    registered = _composite_on_background(make_object_image(0), background)
    obs = ObservationData(
        observation_id="only",
        object_id="only",
        image_path=Path("unused.jpg"),
        embedding=encoder.encode(registered),
        embedding_model=encoder.name,
        local=extractor.extract(registered),
    )
    index = RecognitionIndex(
        config, encoder, [ObjectData("only", "only", [obs], notes_count=1)]
    )
    same_result = index.recognize(
        _composite_on_background(make_object_image(0), background, indent=3)
    )
    assert same_result.status == "match"

    # The desk/background is identical, but the centered object is not. ORB is
    # masked to the central object region, so this must not auto-match.
    wrong_result = index.recognize(
        _composite_on_background(make_object_image(1), background, indent=3)
    )
    assert wrong_result.status != "match"
    assert wrong_result.best is not None
    assert wrong_result.best.local_score < config.min_local_for_match
