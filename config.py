"""Configuration for ObjectEcho.

Everything is deliberately flat and JSON-friendly so the demo can be tuned
without touching code. Command-line flags override values from ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass
class AppConfig:
    # Storage
    data_dir: Path = PROJECT_ROOT / "data"

    # Camera / framing
    camera_index: int = 0
    camera_backend: str = "auto"  # auto, dshow, msmf, any (passed to OpenCV otherwise)
    frame_width: int = 1280
    frame_height: int = 720
    crop_ratio: float = 0.60
    # ORB is restricted to the central object area. The object should fill the
    # reticle; this prevents static desk/background texture from becoming a
    # false instance match.
    local_mask_width: float = 0.70
    local_mask_height: float = 0.82
    show_preview: bool = True
    mirror_preview: bool = True
    preview_width: int = 960

    # Audio
    record_seconds: float = 6.0
    sample_rate: int = 48_000
    input_device: int | str | None = None
    output_device: int | str | None = None
    beep_volume: float = 0.22
    beep_start_delay: float = 0.25
    tts_enabled: bool = False
    tts_rate: int = 175

    # Global image encoder: auto -> CLIP if installed, otherwise classic OpenCV
    encoder_backend: str = "auto"  # auto, clip, classic
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_device: str = "auto"  # auto, cpu, cuda

    # ORB local features / hybrid scoring
    orb_features: int = 1_200
    orb_ratio_test: float = 0.75
    orb_min_good: int = 10
    orb_min_inliers: int = 10
    orb_ransac_px: float = 5.0
    orb_target_inliers: int = 24.0
    global_weight: float = 0.40
    local_weight: float = 0.60
    clip_global_floor: float = 0.45
    classic_global_floor: float = 0.18
    # Strict by default. Tighten further if a demo environment produces
    # false positives; loosen only after testing on the actual objects.
    min_score: float = 0.68
    min_margin: float = 0.15
    min_candidate_score: float = 0.40
    min_local_for_match: float = 0.55
    max_observations_per_object: int = 12

    # Demo / UX
    debug: bool = False
    max_manual_candidates: int = 9
    speak_saved_notes: bool = True
    window_title: str = "ObjectEcho - DEBUG PREVIEW (not the final UI)"

    # Paths derived after loading, not read from JSON.
    db_path: Path = field(init=False)
    objects_dir: Path = field(init=False)
    recordings_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        self.db_path = self.data_dir / "objectecho.sqlite3"
        self.objects_dir = self.data_dir / "objects"
        self.recordings_dir = self.data_dir / "recordings"
        self.validate()

    def validate(self) -> None:
        if not 0.20 <= self.crop_ratio <= 1.0:
            raise ValueError("crop_ratio must be between 0.20 and 1.0")
        if not 0.20 <= self.local_mask_width <= 1.0:
            raise ValueError("local_mask_width must be between 0.20 and 1.0")
        if not 0.20 <= self.local_mask_height <= 1.0:
            raise ValueError("local_mask_height must be between 0.20 and 1.0")
        if self.record_seconds <= 0 or self.record_seconds > 120:
            raise ValueError("record_seconds must be in (0, 120]")
        if self.sample_rate < 8_000 or self.sample_rate > 192_000:
            raise ValueError("sample_rate must be between 8000 and 192000")
        if self.encoder_backend not in {"auto", "clip", "classic"}:
            raise ValueError("encoder_backend must be auto, clip, or classic")
        if self.camera_backend not in {"auto", "dshow", "msmf"}:
            raise ValueError("camera_backend must be auto, dshow, or msmf")
        for name in (
            "min_score",
            "min_margin",
            "min_candidate_score",
            "min_local_for_match",
            "clip_global_floor",
            "classic_global_floor",
            "orb_ratio_test",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.global_weight < 0 or self.local_weight < 0:
            raise ValueError("global_weight and local_weight must be non-negative")
        if self.global_weight + self.local_weight <= 0:
            raise ValueError("at least one scoring weight must be positive")
        if self.orb_target_inliers <= 0:
            raise ValueError("orb_target_inliers must be positive")
        if self.max_observations_per_object < 1:
            raise ValueError("max_observations_per_object must be >= 1")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    @property
    def normalized_global_weight(self) -> float:
        total = self.global_weight + self.local_weight
        return self.global_weight / total

    @property
    def normalized_local_weight(self) -> float:
        total = self.global_weight + self.local_weight
        return self.local_weight / total


# Fields accepted from JSON. Derived path fields are excluded.
CONFIG_FIELDS = {
    f.name
    for f in fields(AppConfig)
    if f.name not in {"db_path", "objects_dir", "recordings_dir"}
}


def parse_device(value: Any) -> int | str | None:
    """Accept None, numeric indexes, or device-name substrings."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "default", "auto"}:
        return None
    if text.isdigit():
        return int(text)
    return text


def load_config(path: Path | None = None) -> AppConfig:
    """Load config.json if present; missing file means defaults."""
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    values: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read config file {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SystemExit(f"Config file {config_path} must contain a JSON object")
        unknown = sorted(set(raw) - CONFIG_FIELDS)
        if unknown:
            print(
                f"[config] Ignoring unknown keys in {config_path}: {', '.join(unknown)}",
                file=sys.stderr,
            )
        values = {k: v for k, v in raw.items() if k in CONFIG_FIELDS}

    if "data_dir" in values:
        values["data_dir"] = Path(values["data_dir"]).expanduser()
        if not values["data_dir"].is_absolute():
            values["data_dir"] = (PROJECT_ROOT / values["data_dir"]).resolve()
    for key in ("input_device", "output_device"):
        if key in values:
            values[key] = parse_device(values[key])

    try:
        return AppConfig(**values)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid configuration: {exc}") from exc


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI overrides. Only explicitly supplied flags override JSON/defaults."""
    g = parser.add_argument_group("configuration")
    g.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                   help="JSON configuration file (default: ./config.json if present)")
    g.add_argument("--data-dir", type=Path,
                   help="directory for SQLite DB, reference images, and WAV files")
    g.add_argument("--camera", type=int, dest="camera_index",
                   help="webcam index, e.g. 0 or 1")
    g.add_argument("--camera-backend", choices=["auto", "dshow", "msmf"])
    g.add_argument("--crop", type=float, dest="crop_ratio",
                   help="central square crop as a fraction of the shorter frame side")
    g.add_argument("--width", type=int, dest="frame_width")
    g.add_argument("--height", type=int, dest="frame_height")
    g.add_argument("--record-seconds", type=float,
                   help="audio note length in seconds (default 6)")
    g.add_argument("--sample-rate", type=int)
    g.add_argument("--input-device", type=parse_device,
                   help="microphone index or name substring")
    g.add_argument("--output-device", type=parse_device,
                   help="speaker index or name substring")
    g.add_argument("--encoder", choices=["auto", "clip", "classic"],
                   dest="encoder_backend",
                   help="image encoder; auto prefers CLIP when installed")
    g.add_argument("--clip-model", default=None,
                   help="open_clip architecture (default ViT-B-32)")
    g.add_argument("--clip-pretrained", default=None,
                   help="open_clip pretrained tag (default openai)")
    g.add_argument("--clip-device", choices=["auto", "cpu", "cuda"], default=None)
    g.add_argument("--min-score", type=float)
    g.add_argument("--min-margin", type=float)
    g.add_argument("--min-candidate-score", type=float)
    g.add_argument("--min-local-for-match", type=float)
    g.add_argument("--tts", action=argparse.BooleanOptionalAction, default=None,
                   dest="tts_enabled",
                   help="enable offline text-to-speech announcements")
    preview = g.add_mutually_exclusive_group()
    preview.add_argument("--preview", action="store_true", default=None,
                         help="show the optional developer preview (default)")
    preview.add_argument("--no-preview", action="store_false", dest="preview",
                         help="run without a window; use terminal keys")
    g.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--list-devices", action="store_true",
                   help="list audio devices and exit")
    g.add_argument("--list-objects", action="store_true",
                   help="list stored objects and notes, then exit")
    g.add_argument("--record-test", type=Path, metavar="WAV",
                   help="record a test WAV using the configured microphone, then exit")
    g.add_argument("--play-test", type=Path, metavar="WAV",
                   help="play a WAV using the configured speakers, then exit")
    g.add_argument("--camera-test", action="store_true",
                   help="open the camera preview only (Q to quit)")


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    values = config.__dict__.copy()
    simple = {
        "data_dir": "data_dir",
        "camera_index": "camera_index",
        "camera_backend": "camera_backend",
        "crop_ratio": "crop_ratio",
        "frame_width": "frame_width",
        "frame_height": "frame_height",
        "record_seconds": "record_seconds",
        "sample_rate": "sample_rate",
        "input_device": "input_device",
        "output_device": "output_device",
        "encoder_backend": "encoder_backend",
        "clip_model": "clip_model",
        "clip_pretrained": "clip_pretrained",
        "clip_device": "clip_device",
        "min_score": "min_score",
        "min_margin": "min_margin",
        "min_candidate_score": "min_candidate_score",
        "min_local_for_match": "min_local_for_match",
        "tts_enabled": "tts_enabled",
        "debug": "debug",
    }
    for arg_name, config_name in simple.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            values[config_name] = value
    if getattr(args, "preview", None) is not None:
        values["show_preview"] = args.preview
    # Remove derived fields so __post_init__ recomputes paths from new data_dir.
    for derived in ("db_path", "objects_dir", "recordings_dir"):
        values.pop(derived, None)
    try:
        return AppConfig(**values)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid configuration: {exc}") from exc
