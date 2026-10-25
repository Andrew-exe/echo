# ObjectEcho

**Attach audio notes to real-world objects.**

ObjectEcho is a screenless-interface prototype: point a camera at an ordinary
object, leave a voice note, and later bring the same object back to hear that
note. This repository is a working hackathon MVP for the **No App** track. The
webcam and keyboard stand in for a future wearable camera and voice interface.

The primary demo is now a polished **local web interface** with a live camera
feed, continuous recognition status, captured reference images, and playable
voice notes. The original keyboard-first OpenCV application remains available as
a screenless fallback.

![ObjectEcho web interface](docs/web-preview.png)

*Web UI preview. The camera feed and objects shown in this documentation image
are synthetic test data, not a claim about a physical demo run.*

## The problem

Physical objects are full of useful context that is invisible to software:
*"This is the spare key for the garage,"* *"The charger only works in this
outlet,"* *"Add the blue filter to this coffee."* Phones require an app, a
login, a screen, and a deliberate search. ObjectEcho asks a different question:
what if an object itself could carry a voice memory?

## What works

- Local browser UI (`python main.py --web`) with a live camera feed, confidence
  badge, object cards, captured images, and playable saved notes.
- Continuous live recognition: when a known object returns to view, its card
  lights up and the recognition banner identifies it.
- Live webcam loop with a configurable central recognition crop.
- Attach a real WAV voice recording (default 6 seconds) to an object.
- Register new objects and add multiple notes to existing objects.
- Instance re-identification from a new image using **CLIP global embeddings +
  ORB local features + geometric verification**.
- Multiple reference views per object.
- Conservative automatic matching with score, local-evidence, and top-two
  margin gates.
- Explicit confirmation for ambiguous matches; automatic playback is never
  based on semantic similarity alone.
- Persistent SQLite storage across restarts: objects, reference views,
  embeddings, ORB features, and audio notes.
- Audible start/end tones, no-match and ambiguity cues, optional offline TTS.
- Optional OpenCV preview with a central reticle. `--no-preview` runs the legacy
  app without any window and accepts terminal keys.
- Human-confirmation cards in the web UI whenever visual identity is ambiguous.
- Offline-first operation. No cloud API or paid service is required.

## Quick start (Windows 11 PowerShell)

Prerequisites: Python 3.11, 3.12, or 3.13 (64-bit), webcam, microphone,
speakers or headphones, and Windows camera/microphone permissions enabled. The
core suite was tested with Python 3.12 on Linux and Python 3.13 on Windows 11.

```powershell
# From the repository directory
py -3.12 -m venv .venv   # use py -3.11 or py -3.13 if that is what you have
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the web interface:

```powershell
python main.py --web
```

The app opens `http://127.0.0.1:8000` automatically. Camera capture, microphone
recording, recognition, SQLite data, and WAV files stay in the Python process;
the browser is the visual interface. If the browser does not open, visit the URL
printed in the terminal.

The original keyboard-first OpenCV interface is still available:

```powershell
python main.py
```

On startup the app loads the image encoder. With the core requirements above, it
uses the fully offline classical/ORB fallback and needs no download. For the
strongest semantic evidence, install the optional CLIP backend:

```powershell
# CPU-only PyTorch wheels (recommended for a demo laptop)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "open_clip_torch>=2.24,<4" Pillow
```

If you have a supported NVIDIA GPU, install PyTorch with the CUDA command from
pytorch.org instead. `--clip-device auto` uses CUDA when available, otherwise
CPU.

### Running in WSL2

The code, tests, storage tooling, and CLIP encoder work inside WSL2, but stock
WSL does not expose the laptop webcam, microphone, or speakers to Linux. WSLg
can display the OpenCV preview window on the Windows desktop, but the live A/P
demo still needs camera and audio devices. The current WSL environment here has
no `/dev/video*`; PortAudio's ALSA backend also reports no devices.

From this repository in WSL:

```bash
cd /home/andre/projects/Echo
source .venv/bin/activate          # this workspace already has dependencies
python -m pytest -q                # 37 passed, 1 optional CLIP test skipped
python main.py --list-objects
python main.py --web               # browser UI; exits if no camera is attached
```

#### Windows hardware with the source staying in WSL

This is the recommended setup for this workspace: keep the one checkout in WSL,
but use a Windows Python environment and a Windows-local data directory. The
Linux `.venv` cannot be reused by Windows Python. Open **Windows PowerShell**
(not a WSL shell) and use the WSL network path:

```powershell
# Open PowerShell on Windows, then access the WSL checkout.
cd \\wsl.localhost\Ubuntu\home\andre\projects\Echo

# One-time Windows environment (already created on the current machine)
py -3.13 -m venv "$env:USERPROFILE\objectecho-win\.venv"
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# Optional CLIP backend
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" -m pip install "open_clip_torch>=2.24,<4" Pillow

# Run the WSL-hosted source with Windows camera/audio + web UI
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" main.py --web --data-dir "$env:USERPROFILE\objectecho-data"
```

Why the explicit data directory: SQLite and rapid image/WAV writes should live on
the Windows filesystem. Pointing `--data-dir` at `\\wsl.localhost\...` puts the
SQLite database on WSL's network filesystem and is not recommended. Code changes
made in WSL are picked up immediately by the Windows run.

For a fresh WSL checkout:

```bash
sudo apt update
sudo apt install -y python3-venv libgl1 libportaudio2 v4l-utils
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

An external **USB** webcam can sometimes be forwarded with `usbipd-win` from an
elevated Windows PowerShell, then attached with `usbipd attach --wsl --busid
<BUSID>` and checked with `v4l2-ctl --list-devices`. Internal laptop cameras
usually cannot be detached this way. Microphone/speaker forwarding into WSL is
less reliable than camera forwarding. For the hackathon demo and video, run the
app natively in Windows; use WSL for development and tests. If `python main.py
--list-devices` reports no devices after forwarding, that is a WSL/PortAudio
limitation, not an ObjectEcho recognition issue.

### First-run model download warning

When CLIP is installed and `encoder_backend` is `auto` (the default), the first
run downloads the `ViT-B-32/openai` weights (roughly 350 MB) from the
open_clip/Hugging Face cache. This can take a minute or more. The weights are
cached for later runs; ObjectEcho does not download them repeatedly. To skip
CLIP entirely, run with `python main.py --encoder classic`. CUDA is optional;
CPU inference is fast enough for the button-driven web UI and legacy A/P workflow.

### Optional spoken status

```powershell
pip install -r requirements-tts.txt
python main.py --tts
```

TTS uses Windows SAPI through `pyttsx3`. It is optional and applies to the
legacy OpenCV interface; the web UI provides visual + audible recording cues.

## Exact command to run

```powershell
python main.py
```

Useful variants:

```powershell
python main.py --no-preview          # no window; use terminal keys
python main.py --tts                 # optional offline announcements
python main.py --encoder classic     # never attempt a CLIP model load
python main.py --camera 1            # second camera
python main.py --crop 0.55           # tighter central crop
python main.py --record-seconds 8
python main.py --input-device 1 --output-device 3
python main.py --list-devices        # microphone/speaker indexes
python main.py --list-objects        # inspect the persisted registry
python main.py --camera-test         # camera preview only
python main.py --min-score 0.62 --min-margin 0.12
```

All settings can also live in `config.json`; copy `config.example.json` as a
starting point. The file is optional and is ignored by git. Command-line flags
override it.

## Using the web interface

- **Attach a note** captures the centered object and records six seconds of the
  laptop microphone. A card appears immediately with the captured reference
  image and an audio player for the note.
- **Play its notes** recognizes the centered object and plays its saved WAVs in
  the browser.
- **Add another reference view** teaches an existing object another angle.
- A green **IN VIEW** badge and matching object card appear continuously whenever
  the camera sees a known object. No button press is needed for this status.
- If the model is uncertain, the UI shows **Confirm identity** with the captured
  image and candidate cards. Nothing is attached or played until you make an
  explicit choice. Choose the correct object or **Create new object** /
  **Cancel**.
- Startup, live recognition, and all object memories survive a server restart.

The web server binds to `127.0.0.1` by default. It is a local demo, not a public
multi-user service. Browser audio playback starts after your click on **Play its
notes** or a note player. Recordings always use the microphone attached to the
machine running Python, not necessarily the viewer's browser device.

## Keyboard controls (legacy OpenCV interface)

| Key | Action |
| --- | --- |
| **A** | Attach a note to the centered object. Records immediately when the match is confident; asks for a decision when uncertain. |
| **P** | Play the attached note(s) for the centered object. |
| **R** | Add the current view as another reference image for an existing object. |
| **L** | List all registered objects, reference views, notes, and paths in the terminal. |
| **Q** | Quit. |
| **C** | Confirm the best candidate when a match is ambiguous. |
| **N** | Create a new object when attach matching is uncertain. |
| **1-9** | Explicitly select an existing candidate, including for a low-confidence match. |
| **X** | Cancel a pending uncertain choice. |
| **T** | Speak the current status (requires `--tts`). |

This is the legacy screenless fallback. The A/P/R/L/Q workflow requires no
mouse. With no preview, all keys work in the terminal. With the preview open,
click it first if terminal keys are not reaching the app.

The OpenCV preview displays the target reticle, match candidates, and
diagnostics for development. The future product would replace keyboard triggers
with wearable gestures and spoken commands.

## Three-object web demo walkthrough

Recommended objects: a colorful labeled bottle, a patterned backpack, and a
printed mug or book with cover art. Pick objects with **asymmetric texture,
logos, seams, or patterns**. Avoid plain white/black textureless objects for
the demo.

Use a quiet room, center each object so it fills most of the bracket area, and
avoid dramatic perspective changes during the first take. ORB deliberately
extracts features only from the central 70% x 82% ellipse of the recognition
crop; if the object is much smaller than the brackets, background keypoints can
still get into that region. Fill the reticle for best results.

1. Start `python main.py --web`. The browser opens the live interface.
2. **Object A - bottle:** hold it inside the guides and click **Attach a note**.
   Speak after the rising tone. The ending tone and the new object card confirm
   that the WAV and captured image were saved.
3. **Object B - backpack:** center it and click **Attach a note**. If the app
   says the match is uncertain, click **Create new object**, then speak.
4. **Object C - mug/book:** repeat, using **Create new object** if asked.
5. Move objects or the camera. Center A again: the recognition banner and its
   card should turn green with **IN VIEW**. Click **Play its notes** to hear A.
6. Center B and click **Play its notes**. Verify B's distinct recording plays.
7. Optional: center A and click **Attach a note** again. A second WAV is added;
   **Play its notes** plays both. The card shows each note's own player.
8. Stop the server with Ctrl+C, start `python main.py --web` again, and center A.
   The same images, notes, and live recognition return from SQLite.

For the legacy keyboard demo, start `python main.py`, click the preview, and use
this sequence (press **N** only when the uncertainty panel appears):

```text
A  [speak]  A  N  [speak]  A  N  [speak]  P  P  A  [speak]  P  L  Q
```

When an existing match is uncertain in the legacy app, **C** confirms the best
candidate, **1-9** picks a specific candidate, **N** creates a new object, and
**X** cancels.

## How recognition works

Every action takes a central square crop. Recognition is intentionally not
"CLIP says both images contain a bottle."

1. **Global channel:** normalize a CLIP image embedding, or use the offline
   HSV/HOG classic descriptor when CLIP is not installed. Compare it with every
   saved reference view using cosine similarity and a calibrated evidence
   mapping.
2. **Local instance channel:** detect up to 1,200 ORB keypoints inside a central
   elliptical mask (70% x 82% of the crop by default), apply Lowe ratio
   filtering, and verify matches with a RANSAC homography. Inlier count and
   inlier ratio produce the local score. The tight mask keeps static desk and
   background texture from becoming a false object match.
3. **Fusion:** per reference view, `score = 0.40 * global + 0.60 * local`.
   An object's best view wins. If no compatible global embedding exists, the
   complete ORB-only fallback score is used.
4. **Conservative acceptance:** automatic match requires all of:
   - fused score >= `min_score` (default 0.58),
   - top score minus second object's score >= `min_margin` (default 0.10),
   - ORB local score >= `min_local_for_match` (default 0.42).
5. **Ambiguity:** anything plausible but unsafe becomes an explicit choice. It
   never silently attaches a note or plays one. A recognized object with no
   notes produces an audible "no notes" response rather than playing another
   object.

The console prints raw cosine, calibrated global score, local score, RANSAC
inlier/good-match counts, observation count, note count, and full object IDs.

Important: high CLIP similarity describes *semantics*, not instance identity.
Two identical bottles may be impossible to distinguish. ObjectEcho states this
limitation and provides manual selection. Add multiple reference views with
**R** for angle and lighting changes.

Thresholds, crop size, ORB settings, `local_mask_width`/`local_mask_height`, and
weights are exposed in `config.py` / `config.json`; the most useful thresholds
and crop size are also CLI flags. Lowering `min_score` or `min_margin` makes the
demo more permissive and increases the chance of a wrong association.

## Architecture

```text
main.py         CLI entry, legacy OpenCV app, and --web switch
web_server.py   Threaded Flask server, live camera/recognition service, JSON API
templates/      Polished single-page web interface (HTML/CSS/JS)
camera.py       Persistent OpenCV VideoCapture with Windows backend fallback
vision.py       CLIP/classic global encoder, ORB extractor/matcher,
                feature persistence, hybrid ranking and decision gates
storage.py      SQLite persistence and relative-path handling
audio_io.py     Lazy sounddevice recording/playback, beeps, optional TTS
config.py       JSON configuration, CLI overrides, validation
keyboard.py     Non-blocking terminal keys (Windows POSIX-compatible)
tests/          Storage, ranking, persistence, audio, Flask API, simulated UI
data/           Runtime-only SQLite DB, reference images, and WAV recordings
```

Data model:

- **Object**: UUID, creation timestamp, optional name, metadata.
- **Observation**: object UUID, reference crop path, normalized embedding,
  embedding-model identifier, ORB keypoints/descriptors path, dimensions.
- **Note**: UUID, object UUID, timestamp, WAV path, duration.

The database uses foreign keys and WAL mode. File paths are stored relative to
`data/` where possible. The model is loaded once at startup, embeddings and ORB
features are precomputed at registration, and all saved data is loaded when the
app starts. The camera and model are never reopened per action.

The web mode runs one background camera reader, one recognition loop (about
once per second), and a threaded Flask API. The browser never receives the
webcam directly; it receives an MJPEG preview and JSON state from the local
server. That keeps the tested Python recognition/audio pipeline as the single
source of truth.

## Data and privacy

Everything is local under `data/`:

```text
data/
  objectecho.sqlite3
  objects/<object-id>/obs_*.jpg
  objects/<object-id>/obs_*_orb.npz
  recordings/<object-id>/*.wav
```

Back up the entire `data/` directory to preserve the registry. Recordings are
not encrypted. This prototype always starts the microphone only after the start
tone, and the end tone is played after capture stops. Obtain consent before
recording people in the environment.

## Testing and verification

From the repository root:

```powershell
python -m pytest -q
```

The default suite uses synthetic object images, synthetic embeddings, temporary
SQLite databases, temp WAV files, and simulated app keyboard flows. It verifies:

- object and observation persistence across database reopen,
- multiple notes without overwriting,
- recognition ranking, thresholds, margins, and ambiguity decisions,
- ORB matching under rotation/scale/noise and rejection of different objects,
- local-feature reloading and the ORB-only fallback,
- simulated new-object attach, existing-object attach, ambiguous attach/retrieve,
  exactly-one selection, no-note behavior, and playback routing,
- audio WAV writing against a fake sound device, plus graceful missing-device
  errors,
- the Flask web API: empty state, live recognition, attaching and persisting a
  note, media serving, object cards, and explicit ambiguous playback selection.

To run the optional CLIP+ORB integration test after installing the CLIP extras:

```powershell
$env:OBJECTECHO_TEST_CLIP="1"
python -m pytest -q tests/test_clip_optional.py
```

### Hardware test status and manual smoke test

The automated suite passed on both the Linux development environment and a
Windows 11 host with Python 3.13 (`37 passed, 1 skipped` on each; the skipped test
is the opt-in CLIP test). The Windows host hardware smoke test also succeeded:

- DirectShow opened the integrated webcam at 1280x720 and returned a valid frame.
- The default microphone recorded a 1.5-second, 48 kHz mono WAV with the start
  and end cue sequence.
- The default speakers played that WAV back.
- The optional CLIP+ORB integration test passed on Windows after downloading the
  model weights.
- The browser UI was exercised headlessly against a synthetic camera/server:
  live recognition badge, green object-card highlight, audio buttons, note
  saving, pending candidate cards, and zero browser console errors.
- The real Windows `--web` server started, served the page, reported live camera
  state, and returned 960x540 JPEG frames.

The Linux WSL environment had no camera/audio devices and correctly produced
actionable errors instead of crashing.

**Not verified automatically:** a human holding two physical objects, leaving
spoken notes, moving away, returning, and hearing the intended recording. That
still requires the manual smoke test and the actual demo. Run this on the Windows
demo laptop before recording the demo video:

1. `python main.py --list-devices` should list the laptop microphone and speakers.
2. `python main.py --record-test data\recordings\smoke.wav --record-seconds 3`
   should beep, record, save, and report a duration.
3. `python main.py --play-test data\recordings\smoke.wav` should play that voice
   back.
4. `python main.py --camera-test` should show a live, correctly exposed camera
   feed with a green square; press Q.
5. Start `python main.py --web` and run the three-object walkthrough above.
   Confirm that captured images and WAV players appear on the object cards,
   live recognition highlights the correct card, and both A and B play back.
6. Stop the server, restart it, and confirm the same cards, images, notes, and
   live recognition return from SQLite. Watch the first-run CLIP download before
   the live demo.

If `--record-test` or `--play-test` fails, no recognition code needs to be
debugged; fix device selection or permissions first.

## Troubleshooting

**Camera will not open.** Close Teams/Zoom/OBS/browser camera tabs. Enable
Settings > Privacy & security > Camera > "Let desktop apps access your camera".
Try `python main.py --camera 1` or `--camera-backend msmf`. WSL does not expose
a Windows webcam by default; run the demo natively in Windows.

**Audio device missing, PortAudio error, or no microphone samples.** Run
`python main.py --list-devices`, then pass indexes or unique name substrings:
`python main.py --input-device 1 --output-device 2`. Check Windows microphone
privacy. On Linux install `libportaudio2`. A Bluetooth headset can have separate
input/output indexes.

**Web page does not open or says the site cannot be reached.** Keep the
PowerShell running; the Flask process must stay alive. Check the printed URL and
port, or use `--port 8081`. Open `http://127.0.0.1:<port>` manually.

**Web record/play buttons do nothing.** The microphone and speaker belong to the
Python server machine, not to the browser. Check `python main.py --list-devices`
and verify the server terminal for an error. Browser note players need a click
before audio can start.

**Preview receives no keys in the legacy app.** Click the OpenCV preview, or use
the terminal; terminal key input works while the terminal has focus. For a
completely windowless test, use `--no-preview` from an interactive PowerShell
terminal.

**"No confident match" for a known object.** It is intentionally conservative.
Press **C** if it is the top candidate, **1-9** if another candidate is correct,
or **R** when the correct object is known to add the current view. Add reference
views from several angles. Make sure the object fills the central square and
lighting is similar.

**Model download is slow/fails.** The app automatically falls back to the
offline classic/ORB path unless you explicitly pass `--encoder clip`. Run with
`--encoder classic` for the demo if the network is unreliable. Once downloaded,
model weights are cached across runs.

**Inference feels slow.** CUDA is used automatically when installed. CPU
inference after startup is roughly tens of milliseconds for a frame and only
runs on A/P/R, not continuously. Keep the preview optional for the final
screenless experience.

**Two identical objects are confused.** This is a fundamental limitation of
visual instance re-identification and is not hidden. Use distinct objects for
the demo and use manual candidate selection for similar ones.

## Known limitations

- Visually identical objects cannot be reliably distinguished.
- Textureless, symmetric, transparent, or reflective objects are difficult for
  ORB and may require the semantic channel plus explicit confirmation.
- Large viewpoint/lighting changes can drop ORB matches; add reference views.
- The central-crop assumption means object detection, segmentation, hand
  tracking, and gaze tracking are intentionally absent. The object must be
  centered and reasonably fill the reticle; a tiny object leaves background
  texture inside the local-feature mask.
- The web UI is a local single-user server; it has no accounts, TLS, or remote
  multi-user sync. Recording uses the Python host's microphone and playback uses
  either the browser or the server speakers depending on the action.
- The keyboard is a prototype trigger; there is no voice-command interface yet.
- Recording length is fixed per invocation; there is no tap-to-stop.
- Single-user local database; no encryption, synchronization, or access control.
- Match thresholds were calibrated on synthetic and expected demo conditions;
  adjust with `config.json` and verify on the actual demo objects.
- TTS output uses the operating system's default voice output, which may differ
  from `--output-device`.

## Future vision

The webcam and keyboard are only a stand-in. The intended product is a wearable
camera with a microphone and bone-conduction speaker: look at an object and
speak to leave a note; look at it again and hear it. Eventual work includes:

- Wearable camera, glance/hand gesture triggers, and wake-word voice commands.
- On-device instance re-identification across days, lighting, and viewpoints.
- Shared object memory with explicit consent and per-user access.
- Multi-user synchronization and conflict handling for notes.
- Opt-in public object stories, provenance, and accessibility annotations.
- Privacy-preserving embeddings, encryption, and clear recording indicators.

ObjectEcho is not claiming solved visual identity. It is a working, honest
prototype of an object-centric, screenless audio memory.

## License

MIT. See `LICENSE`.
