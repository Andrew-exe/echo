"""Shared fixtures and synthetic visual data (no camera required)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig  # noqa: E402
from storage import Storage  # noqa: E402
from vision import (  # noqa: E402
    ClassicGlobalEncoder,
    LocalFeatureExtractor,
    QueryFeatures,
    save_local_features,
    save_reference_crop,
)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        encoder_backend="classic",
        show_preview=False,
        record_seconds=1.0,
        beep_start_delay=0.0,
    )


@pytest.fixture
def encoder() -> ClassicGlobalEncoder:
    return ClassicGlobalEncoder()


@pytest.fixture
def extractor(config: AppConfig) -> LocalFeatureExtractor:
    return LocalFeatureExtractor(config)


def make_object_image(kind: int, size: int = 420) -> np.ndarray:
    """Three deliberately distinct, texture-rich synthetic objects."""
    rng = np.random.default_rng(100 + kind)
    image = np.full((size, size, 3), 238, dtype=np.uint8)
    cx = size // 2
    if kind == 0:
        cv2.rectangle(image, (cx - 75, 95), (cx + 75, size - 45), (25, 25, 215), -1)
        cv2.rectangle(image, (cx - 35, 55), (cx + 35, 120), (25, 25, 215), -1)
        cv2.circle(image, (cx, 230), 38, (20, 220, 245), -1)
        cv2.putText(image, "AQUA", (cx - 65, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (255, 255, 255), 3, cv2.LINE_AA)
        for _ in range(18):
            point = (int(rng.integers(cx - 65, cx + 65)),
                     int(rng.integers(120, size - 70)))
            cv2.circle(image, point, 3, (0, 0, 0), -1)
    elif kind == 1:
        cv2.ellipse(image, (cx, 220), (115, 160), 0, 0, 360, (185, 75, 20), -1)
        cv2.rectangle(image, (cx - 70, 80), (cx + 70, 155), (115, 40, 10), -1)
        cv2.line(image, (cx - 100, 235), (cx + 100, 235), (235, 235, 235), 9)
        cv2.putText(image, "PACK", (cx - 72, 315), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (40, 230, 250), 3, cv2.LINE_AA)
        for _ in range(16):
            x = int(rng.integers(cx - 95, cx + 80))
            y = int(rng.integers(160, 350))
            cv2.line(image, (x, y), (x + 18, y + 18), (255, 175, 35), 2)
    else:
        cv2.rectangle(image, (cx - 32, 200), (cx + 32, size - 30), (45, 185, 45), -1)
        polygon = np.array([
            [cx - 95, 205], [cx + 95, 205], [cx + 55, 85], [cx - 55, 85]
        ], np.int32)
        cv2.fillPoly(image, [polygon], (70, 225, 70))
        cv2.putText(image, "LAMP", (cx - 70, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 0, 0), 2, cv2.LINE_AA)
        for _ in range(22):
            point = (int(rng.integers(cx - 28, cx + 28)),
                     int(rng.integers(120, 370)))
            cv2.circle(image, point, 2, (255, 255, 255), -1)
    return image


def perturbed(
    image: np.ndarray,
    angle: float = 7.0,
    scale: float = 0.93,
    tx: float = 12.0,
    ty: float = -7.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped = cv2.warpAffine(
        image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
        borderValue=(238, 238, 238),
    )
    warped = cv2.GaussianBlur(warped, (3, 3), 0.6)
    noise = rng.integers(-4, 5, warped.shape, dtype=np.int16)
    return np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def register_view(
    config: AppConfig,
    storage: Storage,
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
    object_id: str,
    image: np.ndarray,
) -> str:
    image_path = save_reference_crop(config.objects_dir, object_id, image)
    features_path = image_path.with_name(image_path.stem + "_orb.npz")
    local = extractor.extract(image)
    save_local_features(features_path, local)
    record = storage.add_observation(
        object_id=object_id,
        image_path=image_path,
        features_path=features_path,
        embedding=encoder.encode(image),
        embedding_model=encoder.name,
        image_width=image.shape[1],
        image_height=image.shape[0],
    )
    return record.observation_id


def make_query(
    encoder: ClassicGlobalEncoder,
    extractor: LocalFeatureExtractor,
    image: np.ndarray,
) -> QueryFeatures:
    return QueryFeatures(
        crop=image,
        embedding=encoder.encode(image),
        local=extractor.extract(image),
    )
