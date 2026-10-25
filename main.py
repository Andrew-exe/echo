"""ObjectEcho - attach audio notes to real-world objects.

Default run:
    python main.py

The webcam is an MVP stand-in for a future wearable camera. The OpenCV window
is an optional developer preview; use ``--no-preview`` for a screenless run with
terminal keyboard controls.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from audio_io import AudioError, AudioIO, TTS
from camera import Camera, CameraOpenError
from config import (
    AppConfig,
    add_config_arguments,
    apply_cli_overrides,
    load_config,
)
from keyboard import TerminalKeys
from storage import ObjectRecord, Storage
from vision import (
    MatchCandidate,
    QueryFeatures,
    RecognitionIndex,
    RecognitionResult,
    VisionError,
    center_crop,
    save_local_features,
    save_reference_crop,
)

CONTROLS = "A attach | P play | R add view | L list | C confirm | N new | X cancel | Q quit"


@dataclass
class PendingChoice:
    kind: Literal["attach", "retrieve", "reference"]
    result: RecognitionResult
    created_at: float


def short_id(object_id: str, length: int = 8) -> str:
    return object_id[:length]


def format_object_label(obj: ObjectRecord | MatchCandidate) -> str:
    name = obj.name
    oid = obj.object_id
    return f"{name} ({short_id(oid)})" if name else f"object-{short_id(oid)}"


def print_match_report(result: RecognitionResult, limit: int | None = None) -> None:
    print(
        f"[recognize] status={result.status.upper()} "
        f"margin={result.margin:.3f} elapsed={result.elapsed_ms:.0f}ms "
        f"reason={result.reason}",
        flush=True,
    )
    shown = result.candidates if limit is None else result.candidates[:limit]
    if not shown:
        print("  (no registered objects)", flush=True)
    for idx, cand in enumerate(shown, start=1):
        print(
            f"  {idx}. id={cand.object_id} name={cand.name or '-'} "
            f"score={cand.fused_score:.3f} "
            f"global_cos={cand.global_cosine:.3f} "
            f"global={cand.global_score:.3f} local={cand.local_score:.3f} "
            f"inliers={cand.inliers}/{cand.good_matches} "
            f"obs={cand.observations} notes={cand.notes_count}",
            flush=True,
        )


def print_objects(storage: Storage) -> None:
    objects = storage.list_objects()
    if not objects:
        print("[storage] No objects registered yet. Press A in the app to create one.")
        return
    print(f"[storage] {len(objects)} object(s) in {storage.db_path}")
    for index, obj in enumerate(objects, start=1):
        observations = storage.list_observations(obj.object_id)
        notes = storage.list_notes(obj.object_id)
        print(
            f"  {index}. {format_object_label(obj)}  id={obj.object_id}\n"
            f"      created={obj.created_at} observations={len(observations)} "
            f"notes={len(notes)}"
        )
        for note in notes:
            print(
                f"      - note {note.note_id[:8]} | {note.duration:.1f}s | "
                f"{note.created_at} | {storage.resolve_path(note.audio_path)}"
            )
        for obs in observations:
            print(f"      * view {obs.observation_id[:8]} | {storage.resolve_path(obs.image_path)}")


class ObjectEchoApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.storage = Storage(config.data_dir, config.db_path)
        self.audio = AudioIO(config)
        self.tts = TTS(config.tts_enabled, config.tts_rate)
        self.audio_ready, self.audio_status = self.audio.probe()
        print(f"[audio] {self.audio_status}", flush=True)
        if not self.audio_ready:
            print(
                "[audio] The app will still start, but recording/playback need a "
                "working microphone and speaker. Run `python main.py --list-devices`.",
                flush=True,
            )

        self.encoder = None
        self.index: RecognitionIndex | None = None
        self._load_vision()
        self.camera = Camera(config)
        self.keys = TerminalKeys()
        self.pending: PendingChoice | None = None
        self.status = "Ready. Center an object; press A to attach a note or P to play."
        self.last_status = self.status
        self.running = True
        self.preview_ok = bool(config.show_preview)
        self._window_started = False
        self._last_camera_warning = 0.0
        self._keys_hint_printed = False

    # ------------------------------------------------------------------ setup
    def _load_vision(self) -> None:
        from vision import create_encoder

        self.encoder = create_encoder(self.config)
        self.index = RecognitionIndex.from_storage(
            self.storage, self.config, self.encoder
        )
        print(
            f"[vision] Index loaded: {self.index.object_count} object(s), "
            f"{self.index.observation_count} reference view(s). "
            f"Encoder={self.encoder.name}.",
            flush=True,
        )
        stale = [
            obj.display_name for obj in self.storage.list_objects()
            if self.storage.count_observations(obj.object_id) == 0
        ]
        if stale:
            print(
                "[vision] Objects without usable views: " + ", ".join(stale),
                flush=True,
            )

    def _reload_index(self) -> None:
        assert self.encoder is not None
        self.index = RecognitionIndex.from_storage(
            self.storage, self.config, self.encoder
        )

    def run(self) -> int:
        try:
            self.camera.open()
        except CameraOpenError as exc:
            print(f"[camera] {exc}", file=sys.stderr, flush=True)
            return 2
        print("\n" + "=" * 72)
        print("ObjectEcho ready")
        print("=" * 72)
        print(f"Data:    {self.config.data_dir}")
        print(f"Encoder: {self.encoder.name if self.encoder else '?'}")
        print(f"Audio:   {self.audio_status}")
        print(
            f"Objects: {self.index.object_count if self.index else 0} | "
            f"Notes: {len(self.storage.list_notes())}"
        )
        print(f"Controls: {CONTROLS}")
        if self.config.show_preview:
            print("Click the debug preview so keyboard input goes to the window.")
        print("=" * 72 + "\n", flush=True)

        self.keys.start()
        if not self.keys.enabled and not self.config.show_preview:
            print(
                "[input] stdin is not a TTY and preview is disabled; use a terminal "
                "with interactive input or enable --preview.",
                file=sys.stderr,
            )

        failures = 0
        last_status_printed = ""
        try:
            while self.running:
                terminal_keys = self.keys.poll()
                for key in terminal_keys:
                    self.handle_key(key)

                ok, frame = self.camera.read()
                if not ok or frame is None:
                    failures += 1
                    if failures in (1, 30) or time.monotonic() - self._last_camera_warning > 5:
                        print(
                            f"[camera] Dropped frame (consecutive failures={failures}); "
                            "trying to continue...",
                            file=sys.stderr,
                        )
                        self._last_camera_warning = time.monotonic()
                    if failures % 150 == 0:
                        try:
                            print("[camera] Attempting to reopen camera...", flush=True)
                            self.camera.reopen()
                        except CameraOpenError as exc:
                            print(f"[camera] Reopen failed: {exc}", file=sys.stderr)
                    time.sleep(0.02)
                    if self.config.show_preview and not self._pump_window_key():
                        continue
                    continue
                failures = 0

                if self.config.show_preview:
                    rendered = self.render_debug_frame(frame)
                    try:
                        if not self._window_started:
                            cv2.namedWindow(self.config.window_title, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(
                                self.config.window_title,
                                int(self.config.preview_width),
                                int(rendered.shape[0]),
                            )
                            self._window_started = True
                        cv2.imshow(self.config.window_title, rendered)
                    except cv2.error as exc:
                        print(
                            f"[preview] Could not show preview ({exc}). Continuing with "
                            "terminal controls only.",
                            file=sys.stderr,
                        )
                        self.config.show_preview = False
                        self.preview_ok = False
                    if self.config.show_preview:
                        if not self._pump_window_key():
                            break
                else:
                    # Avoid a busy loop when the camera is fast and there is no UI.
                    time.sleep(0.005)

                if self.pending and time.monotonic() - self.pending.created_at > 120:
                    self.pending = None
                    self.status = "Selection timed out; nothing was changed."
                    print(f"[input] {self.status}", flush=True)

                if self.status != last_status_printed and self.config.debug:
                    print(f"[status] {self.status}", flush=True)
                    last_status_printed = self.status
        except KeyboardInterrupt:
            print("\n[app] Interrupted.")
        finally:
            self.shutdown()
        return 0

    def _pump_window_key(self) -> bool:
        """Process one cv2 window key. Returns False if the window was closed."""
        try:
            if self._window_started and cv2.getWindowProperty(
                self.config.window_title, cv2.WND_PROP_VISIBLE
            ) < 1:
                self.running = False
                return False
            raw = cv2.waitKey(1) & 0xFF
        except cv2.error:
            return True
        if raw == 27:
            self.handle_key("x")
        elif raw != 255:
            char = chr(raw)
            if 32 <= raw < 127:
                self.handle_key(char.lower())
        return True

    def shutdown(self) -> None:
        self.running = False
        try:
            self.keys.close()
        finally:
            self.camera.close()
            self.tts.close()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self.storage.close()
        print("[app] Closed.", flush=True)

    # --------------------------------------------------------------- rendering
    def render_debug_frame(self, frame: np.ndarray) -> np.ndarray:
        display = cv2.flip(frame, 1) if self.config.mirror_preview else frame.copy()
        target_width = max(480, int(self.config.preview_width))
        scale = target_width / float(display.shape[1])
        display = cv2.resize(
            display,
            (target_width, max(1, int(round(display.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = display.shape[:2]
        box_size = min(w, h) * self.config.crop_ratio
        x0 = int((w - box_size) / 2)
        y0 = int((h - box_size) / 2)
        x1 = int(x0 + box_size)
        y1 = int(y0 + box_size)
        active = self.pending is not None
        color = (0, 170, 255) if active else (70, 220, 70)

        # Corner brackets around the central recognition crop.
        arm = max(14, int(box_size * 0.10))
        for x, y, dx, dy in (
            (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)
        ):
            cv2.line(display, (x, y), (x + dx * arm, y), color, 2, cv2.LINE_AA)
            cv2.line(display, (x, y), (x, y + dy * arm), color, 2, cv2.LINE_AA)
        cx, cy = w // 2, h // 2
        cv2.circle(display, (cx, cy), 10, color, 1, cv2.LINE_AA)
        cv2.line(display, (cx - 18, cy), (cx + 18, cy), color, 1, cv2.LINE_AA)
        cv2.line(display, (cx, cy - 18), (cx, cy + 18), color, 1, cv2.LINE_AA)
        cv2.putText(
            display, "Center object inside brackets - DEBUG PREVIEW", (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"objects={self.index.object_count if self.index else 0} "
            f"views={self.index.observation_count if self.index else 0}",
            (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1,
            cv2.LINE_AA,
        )

        lines: list[tuple[str, tuple[int, int, int]]] = []
        if self.pending:
            p = self.pending
            lines.append(
                (f"SELECTION REQUIRED ({p.kind.upper()})", (0, 200, 255))
            )
            for idx, cand in enumerate(
                p.result.candidates_for_manual_choice(self.config.max_manual_candidates),
                start=1,
            ):
                lines.append(
                    (
                        f"{idx}. {cand.display_name}  score {cand.fused_score:.2f}  "
                        f"local {cand.local_score:.2f}  notes {cand.notes_count}",
                        (235, 235, 235),
                    )
                )
            if p.kind == "attach":
                instructions = "C=best  N=new object  1-9=existing  X=cancel"
            else:
                instructions = "C=best  1-9=choose  X=cancel"
            lines.append((instructions, (135, 230, 255)))
        else:
            lines.append((self.status, (255, 255, 255)))
            lines.append((CONTROLS, (190, 220, 190)))

        line_height = 22
        panel_height = min(h // 2, line_height * len(lines) + 14)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)
        y = h - panel_height + 19
        for text, text_color in lines:
            cv2.putText(
                display, text[:110], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                text_color, 1, cv2.LINE_AA,
            )
            y += line_height
        return display

    def _paint_hold_status(self, text: str) -> None:
        """Paint a frozen status frame before a blocking audio operation."""
        if not self.config.show_preview or not self._window_started:
            return
        frame = self.camera.last_frame
        if frame is None:
            return
        previous = self.status
        self.status = text
        try:
            rendered = self.render_debug_frame(frame)
            cv2.imshow(self.config.window_title, rendered)
            cv2.waitKey(1)
        except cv2.error:
            pass
        finally:
            self.status = previous

    # --------------------------------------------------------------- key events
    def handle_key(self, key: str) -> None:
        key = key.lower()
        try:
            if key == "q":
                self.running = False
            elif key == "a":
                self.pending = None
                self.action_attach()
            elif key == "p":
                self.pending = None
                self.action_play()
            elif key == "r":
                self.pending = None
                self.action_add_reference()
            elif key == "l":
                print_objects(self.storage)
                self.set_status("Object list printed in the terminal.")
            elif key == "c":
                self.confirm_best_pending()
            elif key == "n":
                self.create_new_pending()
            elif key == "x":
                if self.pending:
                    self.pending = None
                    self.set_status("Selection cancelled. Nothing changed.")
                    print("[input] Selection cancelled.", flush=True)
            elif key == "t":
                self.speak_status()
            elif key.isdigit() and key != "0":
                self.select_pending(int(key))
        except (AudioError, VisionError, ValueError, OSError) as exc:
            print(f"[error] {exc}", file=sys.stderr, flush=True)
            self.set_status(f"ERROR: {exc}")
            self.audio.beep_no_match()
            if self.config.debug:
                traceback.print_exc()
        except Exception as exc:  # Keep the interactive demo alive.
            print(f"[error] Unexpected failure: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            self.set_status(f"Unexpected error: {exc}")
            self.audio.beep_no_match()

    def set_status(self, text: str) -> None:
        self.status = text
        self.last_status = text

    def speak_status(self) -> None:
        if not self.tts.enabled:
            print(
                "[tts] TTS is disabled. Start with --tts to enable offline speech. "
                f"Status: {self.status}"
            )
            return
        self.tts.speak(self.status)

    def _require_frame_and_recognize(self) -> RecognitionResult:
        frame = self.camera.last_frame
        if frame is None:
            raise VisionError("No camera frame is available yet")
        crop, _ = center_crop(frame, self.config.crop_ratio)
        assert self.index is not None
        if self.config.debug:
            print(f"[scan] crop={crop.shape[1]}x{crop.shape[0]}", flush=True)
        result = self.index.recognize(crop)
        print_match_report(result)
        return result

    # ------------------------------------------------------------------ attach
    def action_attach(self) -> None:
        try:
            result = self._require_frame_and_recognize()
        except (VisionError, AudioError) as exc:
            print(f"[attach] Scan failed: {exc}", file=sys.stderr, flush=True)
            self.audio.beep_no_match()
            self.set_status(f"Scan failed: {exc}")
            return

        if result.status == "match" and result.best is not None:
            self._record_note_for_object(result.best.object_id, result.query, is_new=False)
            return

        if not result.candidates:
            # No prior objects: creating one cannot silently mis-associate.
            self._record_note_for_object(None, result.query, is_new=True)
            return

        self.pending = PendingChoice("attach", result, time.monotonic())
        self.audio.beep_ambiguous() if result.status == "ambiguous" else self.audio.beep_no_match()
        if result.status == "ambiguous":
            message = (
                "Uncertain: ORB/embedding evidence is ambiguous. Choose the object "
                "to attach to."
            )
            speak = "Uncertain match. Choose the best candidate or an existing object."
        else:
            message = (
                "No confident match. You can attach to an existing object by number "
                "or press N to create a new object."
            )
            speak = "No confident match. Choose an existing object or create a new one."
        self.set_status(message)
        print(f"[attach] {message}", flush=True)
        self._print_pending_help()
        self.tts.speak(speak)

    def _print_pending_help(self) -> None:
        assert self.pending is not None
        print(
            "[input] C=confirm best candidate, N=new object (attach only), "
            "1-9=choose candidate, X=cancel. Capture is held while you choose.",
            flush=True,
        )

    def _record_note_for_object(
        self, object_id: str | None, query: QueryFeatures, *, is_new: bool
    ) -> None:
        """Record first, then commit an observation and note.

        For a new object, files are written before DB rows. If recording fails,
        no half-registered object is left behind.
        """
        target_id = object_id or str(uuid.uuid4())
        stamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = (
            self.config.recordings_dir
            / target_id
            / f"{stamp}_{uuid.uuid4().hex[:8]}.wav"
        )
        target_label = (
            f"new object {short_id(target_id)}"
            if is_new else f"object {short_id(target_id)}"
        )
        # Let any previous spoken confirmation finish before opening the mic.
        self.tts.wait(10.0)
        print(
            f"[audio] Recording {self.config.record_seconds:.1f}s note for "
            f"{target_label}. Speak after the start tone...",
            flush=True,
        )
        self.set_status(
            f"RECORDING {self.config.record_seconds:.0f}s for {target_label} - speak now"
        )
        self._paint_hold_status(
            f"RECORDING {self.config.record_seconds:.0f}s - speak now..."
        )
        try:
            duration = self.audio.record_note(
                audio_path, self.config.record_seconds
            )
        except AudioError as exc:
            print(f"[audio] Recording failed: {exc}", file=sys.stderr, flush=True)
            self.set_status(f"RECORDING FAILED: {exc}")
            self.audio.beep_no_match()
            self.tts.speak("Recording failed.")
            return
        except Exception as exc:
            print(f"[audio] Recording failed: {exc}", file=sys.stderr, flush=True)
            self.set_status(f"RECORDING FAILED: {exc}")
            self.audio.beep_no_match()
            return

        try:
            if is_new:
                self.storage.create_object(target_id)
            else:
                if self.storage.get_object(target_id) is None:
                    raise ValueError(f"Object disappeared from database: {target_id}")
            obs_id = self._persist_observation(target_id, query, enforce_limit=not is_new)
            note = self.storage.add_note(target_id, audio_path, duration)
        except Exception as exc:
            print(f"[storage] Could not save note: {exc}", file=sys.stderr, flush=True)
            print(
                f"[audio] The original recording was preserved at {audio_path}",
                file=sys.stderr,
            )
            self.set_status(f"SAVE FAILED: {exc}")
            return
        try:
            self._reload_index()
        except Exception as exc:
            print(
                f"[vision] Note was saved, but the in-memory index refresh failed: {exc}. "
                "Restart the app to reload it.",
                file=sys.stderr,
                flush=True,
            )

        total_notes = self.storage.count_notes(target_id)
        obs_count = self.storage.count_observations(target_id)
        print("\n[attach] SAVED", flush=True)
        print("  status=OK", flush=True)
        print(f"  object_id={target_id}", flush=True)
        print(f"  note_id={note.note_id}", flush=True)
        print(f"  audio_path={audio_path}", flush=True)
        print(f"  duration={duration:.2f}s", flush=True)
        print(
            f"  observation_id={obs_id or '(limit reached; no new view saved)'}",
            flush=True,
        )
        print(f"  object_notes={total_notes} object_views={obs_count}", flush=True)
        self.set_status(
            f"Note saved on {target_label}. Press P on this object to play it."
        )
        if self.config.speak_saved_notes:
            suffix = f" {total_notes} notes now." if total_notes > 1 else ""
            self.tts.speak(f"Note saved.{suffix}")

    def _persist_observation(
        self, object_id: str, query: QueryFeatures, *, enforce_limit: bool
    ) -> str | None:
        assert self.encoder is not None
        if enforce_limit:
            count = self.storage.count_observations(object_id)
            if count >= self.config.max_observations_per_object:
                print(
                    f"[vision] Observation limit "
                    f"({self.config.max_observations_per_object}) reached for "
                    f"object {object_id}; note is still saved.",
                    flush=True,
                )
                return None
        image_path = save_reference_crop(self.config.objects_dir, object_id, query.crop)
        features_path = image_path.with_name(image_path.stem + "_orb.npz")
        try:
            save_local_features(features_path, query.local)
            assert query.embedding is not None
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

    # ------------------------------------------------------------------ play
    def action_play(self) -> None:
        try:
            result = self._require_frame_and_recognize()
        except (VisionError, AudioError) as exc:
            print(f"[play] Scan failed: {exc}", file=sys.stderr, flush=True)
            self.audio.beep_no_match()
            self.set_status(f"Scan failed: {exc}")
            return
        if result.status == "match" and result.best is not None:
            self._play_for_candidate(result.best)
            return
        if not result.candidates:
            self.audio.beep_no_match()
            self.tts.speak("No notes found. No objects are registered.")
            self.set_status("No match and no registered objects. Nothing to play.")
            print("[play] No registered objects; nothing to play.", flush=True)
            return

        self.pending = PendingChoice("retrieve", result, time.monotonic())
        self.audio.beep_ambiguous() if result.status == "ambiguous" else self.audio.beep_no_match()
        if result.status == "ambiguous":
            message = (
                "Uncertain match. No audio will play until you confirm C or choose 1-9."
            )
            speak = "Uncertain match. No audio played. Choose a candidate."
        else:
            message = (
                "No confident match. Nothing played. Choose an existing object by "
                "number if you are sure, or X to cancel."
            )
            speak = "No matching object found. No note played."
        self.set_status(message)
        print(f"[play] {message}", flush=True)
        self._print_pending_help()
        self.tts.speak(speak)

    def _play_for_candidate(self, candidate: MatchCandidate) -> None:
        notes = self.storage.list_notes(candidate.object_id)
        obj = self.storage.get_object(candidate.object_id)
        label = obj.display_name if obj else f"object-{short_id(candidate.object_id)}"
        print(
            f"[play] Selected object id={candidate.object_id} label={label} "
            f"score={candidate.fused_score:.3f} local={candidate.local_score:.3f} "
            f"notes={len(notes)}",
            flush=True,
        )
        if not notes:
            self.audio.beep_no_match()
            self.tts.speak("No notes found for this object.")
            self.set_status(f"No notes found for {label}.")
            print(f"[play] No notes attached to object {candidate.object_id}.", flush=True)
            return

        self.tts.speak(
            f"Playing {len(notes)} note{'s' if len(notes) != 1 else ''} for {label}."
        )
        # With optional TTS enabled, finish the announcement before the note.
        self.tts.wait(10.0)
        self.audio.beep_success()
        played = 0
        for note in notes:
            path = self.storage.resolve_path(note.audio_path)
            print(
                f"[play] Playing note {note.note_id} ({note.duration:.1f}s): {path}",
                flush=True,
            )
            try:
                self.audio.play_wav(path)
                played += 1
            except AudioError as exc:
                print(f"[audio] Playback failed: {exc}", file=sys.stderr, flush=True)
                self.set_status(f"PLAYBACK FAILED: {exc}")
                self.tts.speak("Playback failed.")
                return
        self.set_status(f"Played {played} note(s) for {label}.")
        print(f"[play] Finished {played} note(s) for {candidate.object_id}.", flush=True)

    # -------------------------------------------------------------- references
    def action_add_reference(self) -> None:
        try:
            result = self._require_frame_and_recognize()
        except (VisionError, AudioError) as exc:
            print(f"[reference] Scan failed: {exc}", file=sys.stderr, flush=True)
            self.audio.beep_no_match()
            self.set_status(f"Scan failed: {exc}")
            return
        if not result.candidates:
            self.audio.beep_no_match()
            self.set_status("No objects exist yet. Press A to register one.")
            print("[reference] No objects; use A to create a new object.", flush=True)
            return
        if result.status == "match" and result.best is not None:
            self._add_reference_for(result.best, result.query)
            return

        self.pending = PendingChoice("reference", result, time.monotonic())
        self.audio.beep_ambiguous() if result.status == "ambiguous" else self.audio.beep_no_match()
        self.set_status(
            "Uncertain match for add-view. Choose the object (C=best, 1-9) or X."
        )
        print(
            "[reference] Uncertain match. Choose an object to add this view to; "
            "no new object will be created by R.",
            flush=True,
        )
        self._print_pending_help()
        self.tts.speak("Uncertain match. Choose an object to add a reference view.")

    def _add_reference_for(self, candidate: MatchCandidate, query: QueryFeatures) -> None:
        if self.storage.count_observations(candidate.object_id) >= self.config.max_observations_per_object:
            self.audio.beep_no_match()
            self.set_status("Reference view limit reached; nothing added.")
            print(
                f"[reference] Observation limit reached for {candidate.object_id}.",
                flush=True,
            )
            return
        obs_id = self._persist_observation(candidate.object_id, query, enforce_limit=True)
        if obs_id is None:
            self.audio.beep_no_match()
            self.set_status("Reference view not added (limit reached).")
            return
        self._reload_index()
        self.audio.beep_success()
        self.set_status("Reference view added.")
        print(
            f"[reference] Added view observation_id={obs_id} to "
            f"object_id={candidate.object_id}.",
            flush=True,
        )
        self.tts.speak("Reference view added.")

    # -------------------------------------------------------- uncertain choices
    def confirm_best_pending(self) -> None:
        if not self.pending:
            print("[input] No uncertain match is pending.", flush=True)
            self.set_status("No uncertain match is pending.")
            return
        if self.pending.result.status != "ambiguous" or not self.pending.result.best:
            print(
                "[input] C only confirms an ambiguous match. Use 1-9 to choose an "
                "existing object for a low-confidence result.",
                flush=True,
            )
            self.set_status("Use 1-9 to choose an existing object for this result.")
            return
        candidate = self.pending.result.best
        query = self.pending.result.query
        kind = self.pending.kind
        self.pending = None
        self._execute_pending_choice(kind, candidate, query, confirmed=True)

    def create_new_pending(self) -> None:
        if not self.pending or self.pending.kind != "attach":
            print("[input] N is only available while attaching a note.", flush=True)
            self.set_status("N is only available while attaching a note.")
            return
        query = self.pending.result.query
        self.pending = None
        self._record_note_for_object(None, query, is_new=True)

    def select_pending(self, number: int) -> None:
        if not self.pending:
            print("[input] No candidate selection is pending.", flush=True)
            return
        candidates = self.pending.result.candidates_for_manual_choice(
            self.config.max_manual_candidates
        )
        if number < 1 or number > len(candidates):
            print(
                f"[input] Candidate {number} does not exist. Available: 1-{len(candidates)}.",
                flush=True,
            )
            self.set_status(f"Choose 1-{len(candidates)} or X.")
            return
        candidate = candidates[number - 1]
        query = self.pending.result.query
        kind = self.pending.kind
        print(
            f"[input] Manual selection: {number} -> id={candidate.object_id} "
            f"score={candidate.fused_score:.3f}. This was an explicit user choice.",
            flush=True,
        )
        self.pending = None
        self._execute_pending_choice(kind, candidate, query, confirmed=False)

    def _execute_pending_choice(
        self,
        kind: Literal["attach", "retrieve", "reference"],
        candidate: MatchCandidate,
        query: QueryFeatures,
        *,
        confirmed: bool,
    ) -> None:
        if kind == "attach":
            self._record_note_for_object(candidate.object_id, query, is_new=False)
        elif kind == "retrieve":
            self._play_for_candidate(candidate)
        elif kind == "reference":
            self._add_reference_for(candidate, query)


# ---------------------------------------------------------------------------
# Standalone modes
# ---------------------------------------------------------------------------
def run_camera_test(config: AppConfig) -> int:
    camera = Camera(config)
    try:
        camera.open()
    except CameraOpenError as exc:
        print(f"[camera] {exc}", file=sys.stderr)
        return 2
    print("[camera-test] Press Q or Escape in the preview window to quit.", flush=True)
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            display = cv2.flip(frame, 1) if config.mirror_preview else frame
            h, w = display.shape[:2]
            side = int(min(h, w) * config.crop_ratio)
            x0, y0 = (w - side) // 2, (h - side) // 2
            cv2.rectangle(display, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
            cv2.putText(
                display, "Camera OK - center object in square - Q to quit",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                cv2.LINE_AA,
            )
            cv2.imshow("ObjectEcho camera test", display)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return 0


def run_list_objects(config: AppConfig) -> int:
    storage = Storage(config.data_dir, config.db_path)
    try:
        print_objects(storage)
    finally:
        storage.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="objectecho",
        description="Attach and retrieve audio notes on physical objects using a webcam.",
    )
    add_config_arguments(parser)
    web = parser.add_argument_group("web interface")
    web.add_argument("--web", action="store_true",
                     help="serve the polished local browser interface instead of the OpenCV app")
    web.add_argument("--host", default="127.0.0.1",
                     help="web server bind address (default: 127.0.0.1)")
    web.add_argument("--port", type=int, default=8000,
                     help="web server port (default: 8000)")
    web.add_argument("--no-browser", action="store_true",
                     help="do not automatically open the browser")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)

    if args.web:
        from web_server import run_server
        return run_server(config, args.host, args.port, not args.no_browser)
    if args.list_devices:
        AudioIO.print_devices()
        return 0
    if args.list_objects:
        return run_list_objects(config)
    if args.camera_test:
        return run_camera_test(config)

    if args.record_test:
        audio = AudioIO(config)
        destination = args.record_test.expanduser().resolve()
        print(
            f"[audio-test] Recording {config.record_seconds:.1f}s to {destination}. "
            "Use a start/end beep to time your speech.",
            flush=True,
        )
        try:
            duration = audio.record_note(destination, config.record_seconds)
        except AudioError as exc:
            print(f"[audio-test] FAILED: {exc}", file=sys.stderr)
            return 2
        print(
            f"[audio-test] OK: {destination} ({duration:.2f}s). "
            f"Play it with: python main.py --play-test \"{destination}\"",
            flush=True,
        )
        return 0
    if args.play_test:
        audio = AudioIO(config)
        path = args.play_test.expanduser().resolve()
        print(f"[audio-test] Playing {path}", flush=True)
        try:
            duration = audio.play_wav(path)
        except AudioError as exc:
            print(f"[audio-test] FAILED: {exc}", file=sys.stderr)
            return 2
        print(f"[audio-test] Played {duration:.2f}s.", flush=True)
        return 0

    try:
        app = ObjectEchoApp(config)
    except (VisionError, CameraOpenError) as exc:
        print(f"[startup] FAILED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[startup] Unexpected failure: {exc}", file=sys.stderr)
        if config.debug:
            traceback.print_exc()
        return 2
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
