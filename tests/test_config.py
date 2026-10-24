from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from config import (
    AppConfig,
    apply_cli_overrides,
    load_config,
    parse_device,
)


def test_defaults_and_validated_directories(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path / "data")
    assert cfg.data_dir.is_absolute()
    assert cfg.objects_dir.exists()
    assert cfg.recordings_dir.exists()
    assert cfg.db_path.name == "objectecho.sqlite3"
    assert cfg.normalized_global_weight + cfg.normalized_local_weight == pytest.approx(1.0)


def test_load_json_and_ignore_unknown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "data_dir": str(tmp_path / "custom-data"),
        "crop_ratio": 0.55,
        "record_seconds": 2.5,
        "min_score": 0.61,
        "show_preview": False,
        "mystery_setting": 42,
    }), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.crop_ratio == 0.55
    assert cfg.record_seconds == 2.5
    assert cfg.min_score == 0.61
    assert cfg.show_preview is False
    assert "mystery_setting" in capsys.readouterr().err


def test_invalid_json_values_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"crop_ratio": 5.0}), encoding="utf-8")
    with pytest.raises(SystemExit, match="Invalid configuration"):
        load_config(path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="Could not read config"):
        load_config(path)


def test_device_parser() -> None:
    assert parse_device(None) is None
    assert parse_device("default") is None
    assert parse_device("2") == 2
    assert parse_device(3) == 3
    assert parse_device("USB Microphone") == "USB Microphone"


def test_cli_overrides(tmp_path: Path) -> None:
    base = AppConfig(data_dir=tmp_path / "base", show_preview=True)
    args = Namespace(
        data_dir=tmp_path / "overridden",
        camera_index=1,
        camera_backend=None,
        crop_ratio=0.5,
        frame_width=None,
        frame_height=None,
        record_seconds=3.0,
        sample_rate=None,
        input_device=None,
        output_device=None,
        encoder_backend=None,
        clip_model=None,
        clip_pretrained=None,
        clip_device=None,
        min_score=None,
        min_margin=None,
        min_candidate_score=None,
        min_local_for_match=None,
        tts_enabled=None,
        preview=False,
        debug=None,
        list_devices=False,
        list_objects=False,
        record_test=None,
        play_test=None,
        camera_test=False,
    )
    cfg = apply_cli_overrides(base, args)
    assert cfg.data_dir == (tmp_path / "overridden").resolve()
    assert cfg.camera_index == 1
    assert cfg.crop_ratio == 0.5
    assert cfg.record_seconds == 3.0
    assert cfg.show_preview is False
