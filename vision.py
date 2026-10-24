"""Object instance recognition: global image embeddings + ORB local evidence.

CLIP is useful for semantic similarity but *not* sufficient for instance
identity. This module therefore fuses a normalized global cosine score with
ORB keypoint matches verified by RANSAC, and only auto-accepts when the local
evidence is strong and the top two objects are well separated.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from config import AppConfig
from storage import Storage

EPS = 1e-9


class VisionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Feature data
# ---------------------------------------------------------------------------
@dataclass
class LocalFeatures:
    keypoints: np.ndarray  # (N, 2) float32, image pixel coordinates
    descriptors: np.ndarray  # (N, 32) uint8
    image_width: int
    image_height: int

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])


@dataclass
class ObservationData:
    observation_id: str
    object_id: str
    image_path: Path
    embedding: np.ndarray | None
    embedding_model: str | None
    local: LocalFeatures | None


@dataclass
class ObjectData:
    object_id: str
    name: str | None
    observations: list[ObservationData] = field(default_factory=list)
    notes_count: int = 0

    @property
    def display_name(self) -> str:
        return self.name or f"object-{self.object_id[:8]}"


@dataclass
class QueryFeatures:
    crop: np.ndarray
    embedding: np.ndarray | None
    local: LocalFeatures


@dataclass
class MatchCandidate:
    object_id: str
    name: str | None
    fused_score: float = 0.0
    global_cosine: float = 0.0
    global_score: float = 0.0
    local_score: float = 0.0
    inliers: int = 0
    good_matches: int = 0
    ref_keypoints: int = 0
    observation_id: str | None = None
    observations: int = 0
    notes_count: int = 0
    geometry_coverage: float = 0.0

    @property
    def display_name(self) -> str:
        return self.name or f"object-{self.object_id[:8]}"


@dataclass
class RecognitionResult:
    status: Literal["match", "ambiguous", "unknown"]
    candidates: list[MatchCandidate]
    margin: float
    reason: str
    elapsed_ms: float
    query: QueryFeatures

    @property
    def best(self) -> MatchCandidate | None:
        return self.candidates[0] if self.candidates else None

    def candidates_for_manual_choice(self, limit: int = 9) -> list[MatchCandidate]:
        return self.candidates[: max(1, limit)]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def normalize_embedding(vector: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size and not np.all(np.isfinite(arr)):
        raise VisionError("Embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(arr))
    if norm < EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < EPS:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def global_evidence(cosine: float, encoder_kind: str, config: AppConfig) -> float:
    """Map raw cosine similarity into a calibrated 0..1 evidence score."""
    floor = (
        config.clip_global_floor
        if encoder_kind == "clip"
        else config.classic_global_floor
    )
    return float(np.clip((cosine - floor) / max(EPS, 1.0 - floor), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Global image encoders
# ---------------------------------------------------------------------------
class ImageEncoder:
    name: str = "abstract"
    kind: str = "classic"
    dimension: int = 0

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ClassicGlobalEncoder(ImageEncoder):
    """Fast local color/texture descriptor used when CLIP is unavailable.

    It is not pretrained and not semantic. It is explicitly only supporting
    evidence; ORB carries most instance-specific weight.
    """

    name = "classic-hsv-hog-v1"
    kind = "classic"
    dimension = 288

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise VisionError("Cannot encode an empty image")
        image = cv2.resize(image_bgr, (160, 160), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Hellinger-normalized 3-D HSV histogram (color distribution).
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]
        ).reshape(-1).astype(np.float32)
        hist = np.sqrt(np.maximum(hist, 0.0))
        hist /= max(EPS, float(np.linalg.norm(hist)))

        # Coarse HOG (shape/texture; 2x2 cells, 8 orientations).
        hog = np.zeros((8 * 4,), dtype=np.float32)
        small_gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        angle = (np.arctan2(gy, gx) + np.pi) % np.pi
        bin_idx = np.minimum((angle / np.pi * 8).astype(np.int32), 7)
        cell_i = 0
        for y0 in (0, 32):
            for x0 in (0, 32):
                m = magnitude[y0:y0 + 32, x0:x0 + 32].reshape(-1)
                b = bin_idx[y0:y0 + 32, x0:x0 + 32].reshape(-1)
                cell_hist = np.bincount(b, weights=m, minlength=8).astype(np.float32)
                cell_hist /= max(EPS, float(np.linalg.norm(cell_hist)))
                hog[cell_i * 8:(cell_i + 1) * 8] = cell_hist
                cell_i += 1

        # Coarse spatial color moments: circular hue mean + S + V per cell.
        small_hsv = cv2.resize(hsv, (4, 4), interpolation=cv2.INTER_AREA)
        cells = small_hsv.reshape(-1, 3).astype(np.float32)
        hue = cells[:, 0] / 180.0 * (2.0 * np.pi)
        sat = cells[:, 1] / 255.0
        val = cells[:, 2] / 255.0
        color_moments = np.stack(
            [np.sin(hue), np.cos(hue), sat, val], axis=1
        ).reshape(-1).astype(np.float32)
        color_moments /= max(EPS, float(np.linalg.norm(color_moments)))

        vector = np.concatenate(
            [0.55 * hist, 0.30 * hog, 0.15 * color_moments]
        ).astype(np.float32)
        vector = normalize_embedding(vector)
        if vector.size != self.dimension:
            # Keep this explicit so a code change does not silently break old data.
            raise VisionError(
                f"Classic encoder dimension mismatch: {vector.size} != {self.dimension}"
            )
        return vector


class ClipEncoder(ImageEncoder):
    """open_clip image encoder (CPU or CUDA), loaded once and cached in memory."""

    kind = "clip"

    def __init__(self, config: AppConfig):
        self.config = config
        try:
            import torch  # type: ignore
            import open_clip  # type: ignore
            from PIL import Image  # noqa: F401  (required by preprocessing)
        except Exception as exc:
            raise VisionError(
                "CLIP backend unavailable. Install optional dependencies with "
                "`pip install -r requirements-vision.txt`. Original error: "
                f"{exc}"
            ) from exc

        requested_device = config.clip_device
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            print("[vision] CUDA requested but unavailable; using CPU.", flush=True)
            device = "cpu"
        else:
            device = requested_device
        self._torch = torch
        self._open_clip = open_clip
        self.device = device
        print(
            f"[vision] Loading CLIP {config.clip_model}/{config.clip_pretrained} "
            f"on {device}. First run may download model weights...",
            flush=True,
        )
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                config.clip_model,
                pretrained=config.clip_pretrained,
                device=device,
                # The OpenAI ViT-B/32 checkpoint was trained with QuickGELU;
                # open_clip warns on the tag/config mismatch without this.
                force_quick_gelu=(config.clip_pretrained == "openai"),
            )
            model.eval()
        except Exception as exc:
            raise VisionError(
                f"Could not load CLIP {config.clip_model}/{config.clip_pretrained}: "
                f"{exc}. Try `--encoder classic` for the offline fallback."
            ) from exc
        self._model = model
        self._preprocess = preprocess
        self.name = f"clip:{config.clip_model}/{config.clip_pretrained}"
        with torch.inference_mode():
            dummy = torch.zeros((1, 3, 224, 224), device=device)
            try:
                out = model.encode_image(dummy)
                self.dimension = int(out.shape[-1])
            except Exception:
                # Some architectures prefer encode_image's default input size;
                # the real encode path below will surface any true failure.
                self.dimension = 512
        print(
            f"[vision] CLIP ready ({self.dimension}-d embeddings, device={device}).",
            flush=True,
        )

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise VisionError("Cannot encode an empty image")
        torch = self._torch
        try:
            from PIL import Image
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                features = self._model.encode_image(tensor)
                features = features.float()
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(EPS)
            return features[0].detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            raise VisionError(f"CLIP inference failed: {exc}") from exc


def create_encoder(config: AppConfig) -> ImageEncoder:
    if config.encoder_backend == "classic":
        encoder: ImageEncoder = ClassicGlobalEncoder()
        print(f"[vision] Image encoder: {encoder.name} (offline fallback)", flush=True)
        return encoder
    if config.encoder_backend == "clip":
        return ClipEncoder(config)

    # auto: prefer CLIP, but never leave the app unusable if installation or
    # first-run model download fails.
    try:
        return ClipEncoder(config)
    except VisionError as exc:
        print(f"[vision] {exc}", flush=True)
        print(
            "[vision] Falling back to the fully offline classical descriptor. "
            "ORB still provides instance matching.",
            flush=True,
        )
        return ClassicGlobalEncoder()


# ---------------------------------------------------------------------------
# ORB local features
# ---------------------------------------------------------------------------
class LocalFeatureExtractor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.orb = cv2.ORB_create(
            nfeatures=int(config.orb_features),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            fastThreshold=12,
        )

    def extract(self, image_bgr: np.ndarray) -> LocalFeatures:
        if image_bgr is None or image_bgr.size == 0:
            raise VisionError("Cannot extract features from an empty image")
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Ignore the extreme corners: they are almost always background and can
        # create false "matches" across different objects on the same desk.
        mask = np.zeros((h, w), dtype=np.uint8)
        # Keep keypoints in the centered object region, not on the background
        # around it. The mask dimensions are diameters as fractions of the crop.
        cv2.ellipse(
            mask,
            (w // 2, h // 2),
            (
                max(4, int(w * self.config.local_mask_width / 2.0)),
                max(4, int(h * self.config.local_mask_height / 2.0)),
            ),
            0, 0, 360, 255, -1,
        )
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) == 0:
            return LocalFeatures(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 32), dtype=np.uint8),
                w, h,
            )
        pts = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
        descriptors = np.asarray(descriptors, dtype=np.uint8)
        return LocalFeatures(pts, descriptors, w, h)


@dataclass
class LocalMatchEvidence:
    good_matches: int = 0
    inliers: int = 0
    score: float = 0.0
    coverage: float = 0.0
    ref_keypoints: int = 0


def match_local_features(
    query: LocalFeatures | None,
    reference: LocalFeatures | None,
    config: AppConfig,
) -> LocalMatchEvidence:
    evidence = LocalMatchEvidence(
        ref_keypoints=reference.count if reference else 0
    )
    if query is None or reference is None:
        return evidence
    if query.count < config.orb_min_good or reference.count < config.orb_min_good:
        return evidence

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(query.descriptors, reference.descriptors, k=2)
    except cv2.error:
        return evidence
    good = [pair[0] for pair in knn if len(pair) == 2
            and pair[0].distance < config.orb_ratio_test * pair[1].distance]
    evidence.good_matches = len(good)
    if len(good) < config.orb_min_good:
        return evidence

    # Keep RANSAC cheap while retaining the strongest matches.
    good.sort(key=lambda m: m.distance)
    good = good[:250]
    src = np.asarray(
        [reference.keypoints[m.trainIdx] for m in good], dtype=np.float32
    ).reshape(-1, 1, 2)
    dst = np.asarray(
        [query.keypoints[m.queryIdx] for m in good], dtype=np.float32
    ).reshape(-1, 1, 2)
    try:
        homography, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, float(config.orb_ransac_px),
            maxIters=2_000, confidence=0.995,
        )
    except cv2.error:
        return evidence
    if homography is None or mask is None:
        return evidence
    if not np.all(np.isfinite(homography)):
        return evidence
    mask = mask.reshape(-1).astype(bool)
    inlier_count = int(mask.sum())
    evidence.inliers = inlier_count
    if inlier_count == 0:
        return evidence

    # Spatial spread guards against a handful of collinear/background matches.
    inlier_src = src.reshape(-1, 2)[mask]
    if len(inlier_src) >= 3:
        hull = cv2.convexHull(inlier_src.reshape(-1, 1, 2))
        area = float(cv2.contourArea(hull))
        ref_area = max(1.0, float(reference.image_width * reference.image_height))
        evidence.coverage = float(np.clip(area / ref_area, 0.0, 1.0))

    if evidence.good_matches < config.orb_min_good:
        return evidence
    if evidence.inliers < config.orb_min_inliers:
        return evidence

    inlier_strength = min(1.0, evidence.inliers / float(config.orb_target_inliers))
    inlier_ratio = evidence.inliers / max(1, evidence.good_matches)
    evidence.score = float(np.clip(
        0.70 * inlier_strength + 0.30 * inlier_ratio, 0.0, 1.0
    ))
    return evidence


# ---------------------------------------------------------------------------
# Feature persistence
# ---------------------------------------------------------------------------
def save_local_features(path: Path | str, features: LocalFeatures) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    try:
        with open(temp, "wb") as handle:
            np.savez_compressed(
                handle,
                keypoints=features.keypoints.astype(np.float32),
                descriptors=features.descriptors.astype(np.uint8),
                image_width=np.int32(features.image_width),
                image_height=np.int32(features.image_height),
                format_version=np.int32(1),
            )
        os.replace(temp, destination)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    return destination


def load_local_features(path: Path | str, warn: bool = True) -> LocalFeatures | None:
    source = Path(path)
    if not source.exists():
        if warn:
            print(f"[vision] Missing ORB feature file: {source}", flush=True)
        return None
    try:
        with np.load(source, allow_pickle=False) as data:
            keypoints = np.asarray(data["keypoints"], dtype=np.float32).reshape(-1, 2)
            descriptors = np.asarray(data["descriptors"], dtype=np.uint8)
            w = int(data["image_width"])
            h = int(data["image_height"])
    except Exception as exc:
        if warn:
            print(f"[vision] Could not load ORB features {source}: {exc}", flush=True)
        return None
    if descriptors.ndim != 2 or descriptors.shape[1] != 32:
        if warn:
            print(f"[vision] Invalid ORB descriptor shape in {source}", flush=True)
        return None
    if descriptors.shape[0] != keypoints.shape[0]:
        if warn:
            print(f"[vision] ORB keypoint/descriptor count mismatch in {source}", flush=True)
        return None
    return LocalFeatures(keypoints, descriptors, w, h)


def save_reference_crop(
    objects_dir: Path | str, object_id: str, crop: np.ndarray
) -> Path:
    folder = Path(objects_dir) / object_id
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"obs_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    path = folder / filename
    ok = cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise VisionError(f"Could not save reference image: {path}")
    return path


# ---------------------------------------------------------------------------
# Recognition index and decision logic
# ---------------------------------------------------------------------------
def classify_candidates(
    candidates: list[MatchCandidate], margin: float, config: AppConfig
) -> tuple[Literal["match", "ambiguous", "unknown"], str]:
    if not candidates:
        return "unknown", "no registered objects"
    top = candidates[0]
    if top.fused_score < config.min_candidate_score:
        return "unknown", (
            f"best score {top.fused_score:.3f} < candidate threshold "
            f"{config.min_candidate_score:.3f}"
        )
    if margin < config.min_margin:
        return "ambiguous", (
            f"margin {margin:.3f} < required {config.min_margin:.3f}"
        )
    if top.fused_score < config.min_score:
        return "ambiguous", (
            f"score {top.fused_score:.3f} < required {config.min_score:.3f}"
        )
    if top.local_score < config.min_local_for_match:
        return "ambiguous", (
            "global similarity is high but ORB local evidence is weak "
            f"({top.local_score:.3f} < {config.min_local_for_match:.3f}); "
            "confirm to avoid a wrong association"
        )
    return "match", "confident local + global match"


class RecognitionIndex:
    """In-memory, precomputed recognition data loaded from SQLite once."""

    def __init__(
        self,
        config: AppConfig,
        encoder: ImageEncoder,
        objects: list[ObjectData],
    ):
        self.config = config
        self.encoder = encoder
        self.objects = objects
        self.extractor = LocalFeatureExtractor(config)

    @classmethod
    def from_storage(
        cls, storage: Storage, config: AppConfig, encoder: ImageEncoder
    ) -> "RecognitionIndex":
        notes_by_object = storage.notes_by_object()
        grouped: dict[str, ObjectData] = {}
        for obj in storage.list_objects():
            grouped[obj.object_id] = ObjectData(
                object_id=obj.object_id,
                name=obj.name,
                notes_count=len(notes_by_object.get(obj.object_id, [])),
            )
        stale_global = 0
        missing_features = 0
        for obs in storage.list_observations():
            target = grouped.get(obs.object_id)
            if target is None:
                continue
            embedding: np.ndarray | None = None
            if obs.embedding:
                arr = np.asarray(obs.embedding, dtype=np.float32)
                if arr.size and np.all(np.isfinite(arr)):
                    embedding = normalize_embedding(arr)
            if obs.embedding_model != encoder.name:
                stale_global += 1
                embedding_model = obs.embedding_model
                # Keep the vector for diagnostics, but recognize() will not use it
                # across encoder families.
            else:
                embedding_model = obs.embedding_model
            local = load_local_features(storage.resolve_path(obs.features_path), warn=False)
            if local is None:
                missing_features += 1
            target.observations.append(
                ObservationData(
                    observation_id=obs.observation_id,
                    object_id=obs.object_id,
                    image_path=storage.resolve_path(obs.image_path),
                    embedding=embedding,
                    embedding_model=embedding_model,
                    local=local,
                )
            )
        if stale_global:
            print(
                f"[vision] Warning: {stale_global} saved embedding(s) were made with "
                "a different encoder. ORB still works; re-register references or "
                "run with the original encoder for full hybrid matching.",
                flush=True,
            )
        if missing_features:
            print(
                f"[vision] Warning: {missing_features} observation(s) have no usable "
                "ORB feature file.",
                flush=True,
            )
        return cls(config, encoder, list(grouped.values()))

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def observation_count(self) -> int:
        return sum(len(obj.observations) for obj in self.objects)

    def extract_query(self, crop_bgr: np.ndarray) -> QueryFeatures:
        embedding = self.encoder.encode(crop_bgr)
        local = self.extractor.extract(crop_bgr)
        return QueryFeatures(crop_bgr, embedding, local)

    def recognize(self, crop_bgr: np.ndarray) -> RecognitionResult:
        started = time.perf_counter()
        query = self.extract_query(crop_bgr)
        candidates: list[MatchCandidate] = []
        weights_total = self.config.normalized_global_weight + self.config.normalized_local_weight
        # normalized_* already sums to 1 unless zero; guard for clarity.
        w_global = self.config.normalized_global_weight / max(EPS, weights_total)
        w_local = self.config.normalized_local_weight / max(EPS, weights_total)

        for obj in self.objects:
            candidate = MatchCandidate(
                object_id=obj.object_id,
                name=obj.name,
                observations=len(obj.observations),
                notes_count=obj.notes_count,
            )
            for obs in obj.observations:
                global_cos = 0.0
                has_global = (
                    query.embedding is not None
                    and obs.embedding is not None
                    and obs.embedding_model == self.encoder.name
                    and query.embedding.shape == obs.embedding.shape
                )
                if has_global:
                    global_cos = cosine_similarity(query.embedding, obs.embedding)
                    global_score = global_evidence(global_cos, self.encoder.kind, self.config)
                else:
                    global_score = 0.0

                local_ev = match_local_features(query.local, obs.local, self.config)
                if has_global:
                    fused = w_global * global_score + w_local * local_ev.score
                else:
                    # ORB-only fallback: e.g. the optional CLIP backend is not
                    # installed now but the references were saved with it. Never
                    # let a semantic-only channel pass on its own.
                    fused = local_ev.score
                if fused > candidate.fused_score + EPS:
                    candidate.fused_score = fused
                    candidate.global_cosine = global_cos
                    candidate.global_score = global_score
                    candidate.local_score = local_ev.score
                    candidate.inliers = local_ev.inliers
                    candidate.good_matches = local_ev.good_matches
                    candidate.ref_keypoints = local_ev.ref_keypoints
                    candidate.observation_id = obs.observation_id
                    candidate.geometry_coverage = local_ev.coverage
            candidates.append(candidate)

        candidates.sort(key=lambda c: (-c.fused_score, c.object_id))
        margin = (
            candidates[0].fused_score - candidates[1].fused_score
            if len(candidates) >= 2
            else (candidates[0].fused_score if candidates else 0.0)
        )
        status, reason = classify_candidates(candidates, margin, self.config)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RecognitionResult(
            status=status,
            candidates=candidates,
            margin=float(margin),
            reason=reason,
            elapsed_ms=elapsed_ms,
            query=query,
        )


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def center_crop(frame: np.ndarray, ratio: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a central square and return it with (x, y, w, h)."""
    if frame is None or frame.size == 0:
        raise VisionError("Empty camera frame")
    h, w = frame.shape[:2]
    side = int(round(min(h, w) * float(np.clip(ratio, 0.2, 1.0))))
    side = max(2, min(side, min(h, w)))
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return frame[y0:y0 + side, x0:x0 + side].copy(), (x0, y0, side, side)
