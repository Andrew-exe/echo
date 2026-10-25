"""Flask API tests with synthetic frames and fake hardware."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import web_server
from conftest import make_object_image
from config import AppConfig


def make_frame(image: np.ndarray) -> np.ndarray:
    assert image.shape[:2] == (420, 420)
    frame = np.full((700, 700, 3), 238, dtype=np.uint8)
    frame[140:560, 140:560] = image
    return frame


class FakeCameraStream:
    def __init__(self, config: AppConfig):
        self.width = 700
        self.height = 700
        self.error = None
        self.config = config
        self.frame = make_frame(make_object_image(0))
        self.jpeg = cv2.imencode(".jpg", self.frame)[1].tobytes()
        self.stopped = False

    def start(self) -> None:
        pass

    def latest_frame(self) -> np.ndarray:
        return self.frame.copy()

    def latest_jpeg(self) -> bytes:
        return self.jpeg

    def is_online(self) -> bool:
        return True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch, config: AppConfig):
    monkeypatch.setattr("audio_io.AudioIO.probe", lambda self: (True, "fake audio ready"))
    monkeypatch.setattr(web_server, "CameraStream", FakeCameraStream)
    service = web_server.WebService(config)
    recorded: list[Path] = []

    def fake_record(path, seconds=None, **kwargs):
        import soundfile as sf
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), np.zeros(800, dtype=np.float32), 8_000, subtype="PCM_16")
        recorded.append(destination)
        return 0.1

    for cue in ("beep_start", "beep_end", "beep_success", "beep_ambiguous", "beep_no_match"):
        monkeypatch.setattr(service.audio, cue, lambda: True)
    monkeypatch.setattr(service.audio, "record_note", fake_record)
    service.recorded = recorded  # type: ignore[attr-defined]
    app = web_server.create_app(service)
    app.config.update(TESTING=True)
    client = app.test_client()
    yield service, client
    service.shutdown()


def test_index_and_empty_state(web) -> None:
    service, client = web
    response = client.get("/")
    assert response.status_code == 200
    assert b"ObjectEcho" in response.data
    state = client.get("/api/state").get_json()
    assert state["ok"] is True
    assert state["object_count"] == 0
    assert state["audio_ready"] is True
    assert state["live"]["status"] in {"unknown", "error"}


def test_attach_playback_media_and_live_recognition(web) -> None:
    service, client = web
    response = client.post("/api/attach", json={})
    payload = response.get_json()
    assert payload["ok"] is True, payload
    assert payload["outcome"] == "saved"
    assert payload["object_count"] == 1
    assert payload["note_count"] == 1
    assert service.recorded and service.recorded[0].exists()

    obj = payload["objects"][0]
    assert obj["image_url"]
    assert len(obj["notes"]) == 1
    image_response = client.get(obj["image_url"])
    assert image_response.status_code == 200
    assert image_response.data[:2] == b"\xff\xd8"  # JPEG magic
    note_response = client.get(obj["notes"][0]["url"])
    assert note_response.status_code == 200
    assert note_response.data[:4] == b"RIFF"  # WAV magic

    # Live recognition worker should notice the registered object in the frame.
    deadline = time.time() + 4.0
    live = None
    while time.time() < deadline:
        live = client.get("/api/state").get_json()["live"]
        if live.get("status") == "match":
            break
        time.sleep(0.15)
    assert live is not None
    assert live["status"] == "match", live
    assert live["candidates"][0]["object_id"] == obj["object_id"]

    played = client.post("/api/play", json={}).get_json()
    assert played["ok"] is True
    assert played["outcome"] == "play"
    assert len(played["notes"]) == 1
    assert played["notes"][0]["url"].startswith("/api/media/note/")

    reference = client.post("/api/reference", json={}).get_json()
    assert reference["ok"] is True
    assert reference["outcome"] == "reference_added"
    assert reference["objects"][0]["observations"] == 2


def test_ambiguous_play_requires_explicit_selection(web) -> None:
    service, client = web
    first = client.post("/api/attach", json={}).get_json()
    first_id = first["saved"]["object_id"]
    # Create a second, visually identical object from the same captured view.
    with service._vision_lock:
        index = service.index
        frame = service.camera_stream.latest_frame()
        from vision import center_crop
        crop, _ = center_crop(frame, service.config.crop_ratio)
        result = index.recognize(crop)
    second = service.storage.create_object()
    service._persist_observation(second.object_id, result.query)
    service._reload_index()

    response = client.post("/api/play", json={}).get_json()
    assert response["ok"] is True
    assert response["outcome"] == "pending"
    assert response["pending"]["kind"] == "retrieve"
    assert response["pending"]["image_url"].startswith("/api/pending/image")
    assert not response.get("notes")

    pending_image = client.get(response["pending"]["image_url"])
    assert pending_image.status_code == 200
    assert pending_image.data[:2] == b"\xff\xd8"

    chosen = client.post(
        "/api/pending/select", json={"object_id": first_id}
    ).get_json()
    assert chosen["ok"] is True
    assert chosen["outcome"] == "play"
    assert chosen["pending"] is None
    assert len(chosen["notes"]) == 1
