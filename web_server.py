"""Local web interface for ObjectEcho.

The browser is only a presentation layer: Windows camera, microphone, recognition,
SQLite data, and saved WAV notes stay in this Python process. That keeps the web
MVP close to the tested desktop pipeline and avoids reimplementing audio capture
in JavaScript.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from audio_io import AudioIO
from camera import Camera, CameraOpenError
from config import AppConfig
from storage import NoteRecord, ObservationRecord, Storage
from vision import (
    MatchCandidate,
    QueryFeatures,
    RecognitionIndex,
    RecognitionResult,
    center_crop,
    create_encoder,
    save_local_features,
    save_reference_crop,
)

log = logging.getLogger("objectecho.web")


class ActionError(RuntimeError):
    """An expected user-facing failure while handling a web action."""


def friendly_label(object_id: str, name: str | None) -> str:
    return name or f"Object {object_id[:6].upper()}"


@dataclass
class PendingChoice:
    kind: Literal["attach", "retrieve", "reference"]
    result: RecognitionResult
    query: QueryFeatures
    image_jpeg: bytes
    created_at: float


# ---------------------------------------------------------------------------
# Camera streaming
# ---------------------------------------------------------------------------
class CameraStream:
    """Read the webcam once and share frames/JPEGs with recognition + browser."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.camera = Camera(config)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._last_frame_at = 0.0
        self._last_jpeg_at = 0.0
        self.width = 0
        self.height = 0
        self.error: str | None = None

    def start(self) -> None:
        self.camera.open()
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.camera.close()
            raise CameraOpenError("Camera opened but did not return an initial frame")
        self._publish_frame(frame)
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def _publish_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        with self._lock:
            self._frame = frame
            self.width, self.height = w, h
            self._last_frame_at = time.time()

    def _run(self) -> None:
        last_encode = 0.0
        failures = 0
        while not self._stop.is_set():
            try:
                ok, frame = self.camera.read()
            except Exception as exc:  # camera may disappear mid-demo
                self.error = str(exc)
                time.sleep(0.2)
                continue
            if not ok or frame is None:
                failures += 1
                if failures == 30:
                    self.error = "Camera frame read is failing; check the camera connection."
                time.sleep(0.03)
                continue
            failures = 0
            self.error = None
            self._publish_frame(frame)
            now = time.time()
            if now - last_encode >= 1.0 / 15.0:
                display = frame
                if frame.shape[1] > 960:
                    scale = 960.0 / float(frame.shape[1])
                    display = cv2.resize(
                        frame, (960, max(1, int(round(frame.shape[0] * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok_jpeg, encoded = cv2.imencode(
                    ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                if ok_jpeg:
                    with self._lock:
                        self._jpeg = encoded.tobytes()
                        self._last_jpeg_at = time.time()
                last_encode = now

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def is_online(self) -> bool:
        with self._lock:
            return bool(self._frame is not None and time.time() - self._last_frame_at < 2.5)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.camera.close()


# ---------------------------------------------------------------------------
# Application service
# ---------------------------------------------------------------------------
class WebService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.storage = Storage(config.data_dir, config.db_path)
        self.audio = AudioIO(config)
        self.audio_ready, self.audio_status = self.audio.probe()
        log.info("Audio: %s", self.audio_status)
        self.encoder = create_encoder(config)
        self.index = RecognitionIndex.from_storage(self.storage, config, self.encoder)
        self.camera_stream = CameraStream(config)
        self.camera_stream.start()

        self._vision_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._action_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._busy = False
        self._busy_action: str | None = None
        self._busy_started = 0.0
        self._busy_seconds = 0.0
        self._message = "Center an object in the guides to begin."
        self._pending: PendingChoice | None = None
        self._live: dict[str, Any] = {
            "status": "unknown",
            "reason": "Starting recognition...",
            "margin": 0.0,
            "candidates": [],
            "updated_at": time.time(),
        }
        self._recognition_thread = threading.Thread(
            target=self._recognition_loop, daemon=True, name="live-recognition"
        )
        self._recognition_thread.start()

    # ------------------------------------------------------------- lifecycle
    def shutdown(self) -> None:
        self._shutdown.set()
        if self._recognition_thread.is_alive():
            self._recognition_thread.join(timeout=2.0)
        self.camera_stream.stop()
        self.storage.close()

    # --------------------------------------------------------------- helpers
    def _set_message(self, message: str) -> None:
        with self._state_lock:
            self._message = message

    def _set_busy(
        self, action: str, message: str, *, seconds: float = 0.0
    ) -> None:
        with self._state_lock:
            self._busy = True
            self._busy_action = action
            self._busy_started = time.time()
            self._busy_seconds = seconds
            self._message = message

    def begin_action(self, action: str = "scanning") -> bool:
        if not self._action_lock.acquire(blocking=False):
            return False
        self._set_busy(action, "Capturing the object...")
        return True

    def end_action(self) -> None:
        with self._state_lock:
            self._busy = False
            self._busy_action = None
            self._busy_seconds = 0.0
            if self._action_lock.locked():
                try:
                    self._action_lock.release()
                except RuntimeError:
                    pass

    def _reload_index(self) -> None:
        with self._vision_lock:
            self.index = RecognitionIndex.from_storage(
                self.storage, self.config, self.encoder
            )

    def _capture_and_recognize(self) -> tuple[QueryFeatures, RecognitionResult]:
        frame = self.camera_stream.latest_frame()
        if frame is None:
            raise ActionError("No camera frame available yet")
        crop, _ = center_crop(frame, self.config.crop_ratio)
        with self._vision_lock:
            index = self.index
            result = index.recognize(crop)
        self._set_live(result)
        return result.query, result

    def _set_live(self, result: RecognitionResult) -> None:
        candidates = [
            self._candidate_payload(c)
            for c in result.candidates[: self.config.max_manual_candidates]
        ]
        with self._state_lock:
            self._live = {
                "status": result.status,
                "reason": result.reason,
                "margin": round(float(result.margin), 4),
                "candidates": candidates,
                "updated_at": time.time(),
                "elapsed_ms": round(result.elapsed_ms, 1),
            }

    @staticmethod
    def _candidate_payload(candidate: MatchCandidate) -> dict[str, Any]:
        return {
            "object_id": candidate.object_id,
            "label": friendly_label(candidate.object_id, candidate.name),
            "name": candidate.name,
            "score": round(float(candidate.fused_score), 4),
            "global_score": round(float(candidate.global_score), 4),
            "global_cosine": round(float(candidate.global_cosine), 4),
            "local_score": round(float(candidate.local_score), 4),
            "inliers": candidate.inliers,
            "good_matches": candidate.good_matches,
            "notes_count": candidate.notes_count,
            "observations": candidate.observations,
        }

    def _recognition_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._state_lock:
                paused = self._busy or self._pending is not None
            if paused:
                time.sleep(0.15)
                continue
            frame = self.camera_stream.latest_frame()
            if frame is None:
                time.sleep(0.3)
                continue
            started = time.perf_counter()
            try:
                with self._vision_lock:
                    index = self.index
                    if index.object_count == 0:
                        result = None
                    else:
                        crop, _ = center_crop(frame, self.config.crop_ratio)
                        result = index.recognize(crop)
                if result is not None:
                    self._set_live(result)
                else:
                    with self._state_lock:
                        self._live = {
                            "status": "unknown",
                            "reason": "No objects registered yet.",
                            "margin": 0.0,
                            "candidates": [],
                            "updated_at": time.time(),
                            "elapsed_ms": 0.0,
                        }
            except Exception as exc:
                log.exception("Live recognition failed")
                with self._state_lock:
                    self._live = {
                        "status": "error",
                        "reason": str(exc),
                        "margin": 0.0,
                        "candidates": [],
                        "updated_at": time.time(),
                        "elapsed_ms": 0.0,
                    }
            # Keep a smooth live indicator without saturating the CPU.
            elapsed = time.perf_counter() - started
            time.sleep(max(0.30, 0.65 - elapsed))

    # ----------------------------------------------------------- serialization
    def _observation_url(self, observation: ObservationRecord | None) -> str | None:
        if observation is None:
            return None
        return f"/api/media/observation/{observation.observation_id}"

    @staticmethod
    def _note_url(note: NoteRecord) -> str:
        return f"/api/media/note/{note.note_id}"

    def _objects_payload(self) -> list[dict[str, Any]]:
        objects = self.storage.list_objects()
        result: list[dict[str, Any]] = []
        for obj in reversed(objects):  # newest memory first
            observations = self.storage.list_observations(obj.object_id)
            notes = self.storage.list_notes(obj.object_id)
            latest = observations[-1] if observations else None
            result.append(
                {
                    "object_id": obj.object_id,
                    "short_id": obj.object_id[:6].upper(),
                    "label": friendly_label(obj.object_id, obj.name),
                    "name": obj.name,
                    "created_at": obj.created_at,
                    "image_url": self._observation_url(latest),
                    "observations": len(observations),
                    "notes": [
                        {
                            "note_id": note.note_id,
                            "url": self._note_url(note),
                            "duration": round(float(note.duration), 2),
                            "created_at": note.created_at,
                        }
                        for note in notes
                    ],
                }
            )
        return result

    def _pending_payload(self) -> dict[str, Any] | None:
        with self._state_lock:
            pending = self._pending
            if pending is None:
                return None
            return {
                "kind": pending.kind,
                "status": pending.result.status,
                "reason": pending.result.reason,
                "margin": round(float(pending.result.margin), 4),
                "candidates": [
                    self._candidate_payload(c)
                    for c in pending.result.candidates_for_manual_choice(
                        self.config.max_manual_candidates
                    )
                ],
                "image_url": f"/api/pending/image?v={int(pending.created_at * 1000)}",
                "created_at": pending.created_at,
            }

    def state_payload(self) -> dict[str, Any]:
        objects = self._objects_payload()
        pending = self._pending_payload()
        with self._state_lock:
            live = self._live
            busy = self._busy
            busy_action = self._busy_action
            busy_started = self._busy_started
            busy_seconds = self._busy_seconds
            message = self._message
        now = time.time()
        return {
            "objects": objects,
            "object_count": len(objects),
            "note_count": sum(len(obj["notes"]) for obj in objects),
            "live": live,
            "pending": pending,
            "busy": busy,
            "busy_action": busy_action,
            "busy_elapsed": round(max(0.0, now - busy_started), 2) if busy else 0.0,
            "busy_seconds": busy_seconds,
            "message": message,
            "audio_ready": self.audio_ready,
            "audio_status": self.audio_status,
            "encoder": self.encoder.name,
            "camera_online": self.camera_stream.is_online(),
            "camera": {
                "width": self.camera_stream.width,
                "height": self.camera_stream.height,
                "crop_ratio": self.config.crop_ratio,
                "camera_error": self.camera_stream.error,
            },
            "record_seconds": self.config.record_seconds,
            "timestamp": now,
        }

    # ------------------------------------------------------------- persistence
    def _persist_observation(
        self, object_id: str, query: QueryFeatures
    ) -> str | None:
        if self.storage.count_observations(object_id) >= self.config.max_observations_per_object:
            return None
        if query.embedding is None:
            raise ActionError("The visual encoder did not produce an embedding")
        image_path = save_reference_crop(self.config.objects_dir, object_id, query.crop)
        features_path = image_path.with_name(image_path.stem + "_orb.npz")
        try:
            save_local_features(features_path, query.local)
            record = self.storage.add_observation(
                object_id=object_id,
                image_path=image_path,
                features_path=features_path,
                embedding=query.embedding,
                embedding_model=self.encoder.name,
                image_width=query.crop.shape[1],
                image_height=query.crop.shape[0],
                metadata={
                    "local_features": "orb",
                    "crop": "center",
                    "encoder_backend": self.encoder.kind,
                    "via": "web",
                },
            )
        except Exception:
            for path in (image_path, features_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
        return record.observation_id

    def _save_note_for(
        self, object_id: str | None, query: QueryFeatures, *, is_new: bool
    ) -> dict[str, Any]:
        target_id = object_id or str(uuid.uuid4())
        label = friendly_label(target_id, None)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = (
            self.config.recordings_dir
            / target_id
            / f"{stamp}_{uuid.uuid4().hex[:8]}.wav"
        )
        self._set_busy(
            "recording",
            f"Recording {self.config.record_seconds:.0f}s for {label}. Speak now.",
            seconds=self.config.record_seconds,
        )
        log.info("Recording %s for %s", audio_path, target_id)
        try:
            duration = self.audio.record_note(audio_path, self.config.record_seconds)
        except Exception as exc:
            raise ActionError(f"Recording failed: {exc}") from exc

        created = False
        try:
            if is_new:
                self.storage.create_object(target_id)
                created = True
            elif self.storage.get_object(target_id) is None:
                raise ActionError("That object no longer exists")
            obs_id = self._persist_observation(target_id, query)
            note = self.storage.add_note(target_id, audio_path, duration)
            self._reload_index()
        except Exception as exc:
            if created:
                try:
                    self.storage.delete_object(target_id)
                except Exception:
                    pass
            raise ActionError(
                f"Note recording succeeded but saving failed: {exc}. "
                f"The WAV was preserved at {audio_path}."
            ) from exc

        obj = self.storage.get_object(target_id)
        label = friendly_label(target_id, obj.name if obj else None)
        total = self.storage.count_notes(target_id)
        self._set_message(
            f"Saved note on {label}. Center it again and press Play, or press Attach to add another."
        )
        log.info(
            "Saved object=%s note=%s obs=%s duration=%.2fs path=%s",
            target_id, note.note_id, obs_id, duration, audio_path,
        )
        return {
            "object_id": target_id,
            "label": label,
            "note_id": note.note_id,
            "duration": round(float(duration), 2),
            "notes_total": total,
            "observation_id": obs_id,
            "audio_path": str(audio_path),
        }

    def _add_reference_for(self, object_id: str, query: QueryFeatures) -> dict[str, Any]:
        if self.storage.count_observations(object_id) >= self.config.max_observations_per_object:
            raise ActionError("This object already has the maximum number of reference views.")
        obs_id = self._persist_observation(object_id, query)
        if obs_id is None:
            raise ActionError("Reference view was not saved (limit reached).")
        self._reload_index()
        self.audio.beep_success()
        obj = self.storage.get_object(object_id)
        label = friendly_label(object_id, obj.name if obj else None)
        self._set_message(f"Added a new reference view for {label}.")
        return {"object_id": object_id, "label": label, "observation_id": obs_id}

    def _set_pending(
        self, kind: Literal["attach", "retrieve", "reference"], query: QueryFeatures,
        result: RecognitionResult,
    ) -> None:
        ok, jpeg = cv2.imencode(".jpg", query.crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise ActionError("Could not encode the captured object image")
        with self._state_lock:
            self._pending = PendingChoice(
                kind=kind,
                result=result,
                query=query,
                image_jpeg=jpeg.tobytes(),
                created_at=time.time(),
            )
            if kind == "attach":
                self._message = "That match is uncertain. Choose an object or create a new one."
            elif kind == "retrieve":
                self._message = "That match is uncertain. Nothing plays until you choose an object."
            else:
                self._message = "That match is uncertain. Choose which object this view belongs to."

    def _clear_pending(self) -> None:
        with self._state_lock:
            self._pending = None

    def _require_pending(self) -> PendingChoice:
        with self._state_lock:
            pending = self._pending
        if pending is None:
            raise ActionError("There is no pending selection")
        return pending

    def _notes_payload(self, object_id: str) -> dict[str, Any]:
        obj = self.storage.get_object(object_id)
        if obj is None:
            raise ActionError("Object not found")
        notes = self.storage.list_notes(object_id)
        label = friendly_label(object_id, obj.name)
        return {
            "object_id": object_id,
            "label": label,
            "notes": [
                {
                    "note_id": note.note_id,
                    "url": self._note_url(note),
                    "duration": round(float(note.duration), 2),
                    "created_at": note.created_at,
                }
                for note in notes
            ],
        }

    # ---------------------------------------------------------------- actions
    def attach(self) -> dict[str, Any]:
        query, result = self._capture_and_recognize()
        self._clear_pending()
        if result.status == "match" and result.best is not None:
            saved = self._save_note_for(
                result.best.object_id, query, is_new=False
            )
            self._set_message(
                f"Saved note #{saved['notes_total']} on {saved['label']}."
            )
            return {"outcome": "saved", "saved": saved}

        if not result.candidates:
            saved = self._save_note_for(None, query, is_new=True)
            self._set_message(f"Registered {saved['label']} and saved its first note.")
            return {"outcome": "saved", "saved": saved}

        self._set_pending("attach", query, result)
        return {"outcome": "pending"}

    def play(self) -> dict[str, Any]:
        query, result = self._capture_and_recognize()
        self._clear_pending()
        if result.status == "match" and result.best is not None:
            payload = self._notes_payload(result.best.object_id)
            if not payload["notes"]:
                self.audio.beep_no_match()
                self._set_message(
                    f"{payload['label']} is recognized, but it has no notes yet."
                )
                return {"outcome": "no_notes", "object": payload}
            self._set_message(f"Playing {len(payload['notes'])} note(s) for {payload['label']}.")
            return {"outcome": "play", **payload}

        if result.candidates:
            self._set_pending("retrieve", query, result)
            return {"outcome": "pending"}
        self.audio.beep_no_match()
        self._set_message("No matching object found. No note was played.")
        return {"outcome": "no_match"}

    def add_reference(self) -> dict[str, Any]:
        query, result = self._capture_and_recognize()
        self._clear_pending()
        if result.status == "match" and result.best is not None:
            added = self._add_reference_for(result.best.object_id, query)
            return {"outcome": "reference_added", "added": added}
        if result.candidates:
            self._set_pending("reference", query, result)
            return {"outcome": "pending"}
        self.audio.beep_no_match()
        self._set_message("No object recognized. Use Attach to create a new object.")
        return {"outcome": "no_match"}

    def choose_pending(
        self, object_id: str | None = None, *, create_new: bool = False
    ) -> dict[str, Any]:
        pending = self._require_pending()
        if pending.kind == "attach" and create_new:
            saved = self._save_note_for(None, pending.query, is_new=True)
            self._clear_pending()
            self._set_message(f"Registered {saved['label']} and saved a note.")
            return {"outcome": "saved", "saved": saved}
        if not object_id:
            raise ActionError("Choose an object")
        if self.storage.get_object(object_id) is None:
            raise ActionError("That object no longer exists")

        if pending.kind == "attach":
            saved = self._save_note_for(object_id, pending.query, is_new=False)
            self._clear_pending()
            self._set_message(f"Saved note #{saved['notes_total']} on {saved['label']}.")
            return {"outcome": "saved", "saved": saved}
        if pending.kind == "reference":
            added = self._add_reference_for(object_id, pending.query)
            self._clear_pending()
            return {"outcome": "reference_added", "added": added}
        payload = self._notes_payload(object_id)
        self._clear_pending()
        if not payload["notes"]:
            self.audio.beep_no_match()
            self._set_message(f"{payload['label']} has no notes.")
            return {"outcome": "no_notes", "object": payload}
        self._set_message(f"Playing {len(payload['notes'])} note(s) for {payload['label']}.")
        return {"outcome": "play", **payload}

    def cancel_pending(self) -> None:
        self._clear_pending()
        self._set_message("Selection cancelled. Nothing was changed.")

    def pending_image(self) -> bytes:
        pending = self._require_pending()
        return pending.image_jpeg

    def find_observation(self, observation_id: str) -> ObservationRecord | None:
        for obs in self.storage.list_observations():
            if obs.observation_id == observation_id:
                return obs
        return None

    def find_note(self, note_id: str) -> NoteRecord | None:
        for note in self.storage.list_notes():
            if note.note_id == note_id:
                return note
        return None


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
def create_app(service: WebService) -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False

    def action_response(result: dict[str, Any]) -> Response:
        payload = service.state_payload()
        payload.update(result)
        payload["ok"] = True
        return jsonify(payload)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/state")
    def api_state() -> Response:
        return jsonify({"ok": True, **service.state_payload()})

    @app.get("/api/pending/image")
    def api_pending_image() -> Response:
        try:
            data = service.pending_image()
        except ActionError:
            abort(404)
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/attach")
    def api_attach() -> Response:
        if not service.begin_action("scanning"):
            return jsonify({"ok": False, "error": "Another action is already running."}), 409
        try:
            result = service.attach()
            return action_response(result)
        except ActionError as exc:
            service._set_message(str(exc))
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()})
        except Exception as exc:
            log.exception("Attach failed")
            service._set_message(f"Unexpected error: {exc}")
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()}), 500
        finally:
            service.end_action()

    @app.post("/api/play")
    def api_play() -> Response:
        if not service.begin_action("scanning"):
            return jsonify({"ok": False, "error": "Another action is already running."}), 409
        try:
            result = service.play()
            return action_response(result)
        except ActionError as exc:
            service._set_message(str(exc))
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()})
        except Exception as exc:
            log.exception("Play failed")
            service._set_message(f"Unexpected error: {exc}")
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()}), 500
        finally:
            service.end_action()

    @app.post("/api/reference")
    def api_reference() -> Response:
        if not service.begin_action("scanning"):
            return jsonify({"ok": False, "error": "Another action is already running."}), 409
        try:
            result = service.add_reference()
            return action_response(result)
        except ActionError as exc:
            service._set_message(str(exc))
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()})
        except Exception as exc:
            log.exception("Add reference failed")
            service._set_message(f"Unexpected error: {exc}")
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()}), 500
        finally:
            service.end_action()

    @app.post("/api/pending/select")
    def api_pending_select() -> Response:
        body = request.get_json(silent=True) or {}
        if not service.begin_action("saving"):
            return jsonify({"ok": False, "error": "Another action is already running."}), 409
        try:
            result = service.choose_pending(
                body.get("object_id"),
                create_new=bool(body.get("create_new")),
            )
            return action_response(result)
        except ActionError as exc:
            service._set_message(str(exc))
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()})
        except Exception as exc:
            log.exception("Pending selection failed")
            service._set_message(f"Unexpected error: {exc}")
            return jsonify({"ok": False, "error": str(exc), **service.state_payload()}), 500
        finally:
            service.end_action()

    @app.post("/api/pending/cancel")
    def api_pending_cancel() -> Response:
        if not service.begin_action("cancelling"):
            return jsonify({"ok": False, "error": "Another action is already running."}), 409
        try:
            service.cancel_pending()
            return action_response({"outcome": "cancelled"})
        finally:
            service.end_action()

    @app.get("/api/media/observation/<observation_id>")
    def api_observation(observation_id: str) -> Response:
        obs = service.find_observation(observation_id)
        if obs is None:
            abort(404)
        path = service.storage.resolve_path(obs.image_path)
        if not path.exists():
            abort(404)
        return send_file(path, mimetype="image/jpeg", conditional=True, max_age=3600)

    @app.get("/api/media/note/<note_id>")
    def api_note(note_id: str) -> Response:
        note = service.find_note(note_id)
        if note is None:
            abort(404)
        path = service.storage.resolve_path(note.audio_path)
        if not path.exists():
            abort(404)
        return send_file(path, mimetype="audio/wav", conditional=True, max_age=3600)

    @app.get("/api/frame")
    def api_frame() -> Response:
        jpeg = service.camera_stream.latest_jpeg()
        if jpeg is None:
            abort(503)
        return Response(
            jpeg, mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/video_feed")
    def video_feed() -> Response:
        def generate():
            while True:
                jpeg = service.camera_stream.latest_jpeg()
                if jpeg is not None:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode("ascii")
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                time.sleep(0.05)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"ok": True, "camera": service.camera_stream.is_online()})

    return app


def run_server(
    config: AppConfig,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> int:
    try:
        service = WebService(config)
    except CameraOpenError as exc:
        print(f"[web] Camera could not start: {exc}")
        return 2
    except Exception as exc:
        print(f"[web] Startup failed: {exc}")
        return 2
    app = create_app(service)
    # Keep the demo terminal calm despite the UI polling state a few times/sec.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}"
    print("\n" + "=" * 68)
    print("ObjectEcho web interface")
    print(f"Open: {url}")
    print(f"Data: {config.data_dir}")
    print(f"Encoder: {service.encoder.name}")
    print(f"Audio: {service.audio_status}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 68 + "\n", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)
    except OSError as exc:
        print(f"[web] Could not start server on {host}:{port}: {exc}")
        return 2
    finally:
        service.shutdown()
    return 0
