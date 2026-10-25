# ObjectEcho

**Attach audio notes to real-world objects.**

ObjectEcho is a working hackathon MVP for the **No App** track. Point a camera
at an everyday object, leave a voice note, and bring the same object back later
to hear it again.

## The idea

Objects carry useful context that software usually cannot see: the spare key,
the charger that only works in one outlet, the special coffee filter. Today we
store this knowledge in apps, notes, or memory. ObjectEcho lets an object hold
that context itself through a real voice recording.

The current prototype uses a laptop webcam, microphone, and browser. It is not a
finished product—it is a practical first step toward an object-centric,
screenless interface.

## What it does

- Shows a live camera feed with a guide for centering an object.
- Records a real audio note and attaches it to that object.
- Saves a reference image, a visual identity, and the original WAV.
- Recognizes the object again from a new camera view.
- Lights up the object's card when it comes back into view.
- Shows every saved object with its captured images and playable notes.
- Supports multiple objects and multiple notes per object.
- Preserves everything across restarts with a local SQLite database.
- Asks for confirmation when identity is ambiguous instead of guessing.

There is **no object-detection model** or bounding-box detector. Recognition is
based on a CLIP image embedding plus ORB local features, with strict matching
gates. The displayed percentage is a match score, not a calibrated probability.

## Vision

The webcam and keyboard are a stand-in for a future wearable camera with a
microphone and speaker. Instead of opening an app, a person could look at an
object, speak a note, and later look at it again to hear the memory.

Longer term, ObjectEcho could support shared object memories, voice commands,
better long-term re-identification, and opt-in collaboration between users. The
goal is simple: make physical objects easier to understand, remember, and share.

## Requirements

- Windows 11 (recommended for camera and audio access).
- Python 3.11, 3.12, or 3.13, 64-bit.
- Webcam, microphone, and speakers or headphones.
- Camera and microphone permissions enabled in Windows.
- Internet only for the first optional CLIP model download.

## Windows setup

Open **Windows PowerShell** from the repository root:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core installation includes OpenCV, NumPy, audio recording, Flask, and the
web interface. It works without downloading any AI model.

### Optional CLIP model

CLIP improves the semantic evidence used for matching. Install the CPU version
of PyTorch first:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install "open_clip_torch>=2.24,<4" Pillow
```

For an NVIDIA GPU, install the matching CUDA version of `torch` and `torchvision`
from [pytorch.org](https://pytorch.org/get-started/locally/) instead.

On the first CLIP-enabled start, ObjectEcho downloads the
`ViT-B-32/openai` model (roughly 350 MB). The weights are cached for later runs:

- Windows: `%USERPROFILE%\.cache\huggingface\hub\models--timm--vit_base_patch32_clip_224.openai`
- Linux: `~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai`

Optional Hugging Face settings:

```powershell
$env:HF_TOKEN = "hf_your_token_here"          # only if downloads are rate-limited
$env:HF_HOME = "D:\objectecho-hf-cache"       # optional custom cache location
$env:HF_HUB_OFFLINE = "1"                     # after weights are cached
```

To download the weights before a demo:

```powershell
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', force_quick_gelu=True)"
```

To skip CLIP and its download entirely:

```powershell
python main.py --web --encoder classic
```

If CLIP is installed but cannot load, the default `auto` mode falls back to the
offline classical/ORB path.

## Run

Start the web app:

```powershell
python main.py --web
```

The browser opens at **http://127.0.0.1:8000**.

Then:

1. Center an object inside the camera guides.
2. Click **Attach a note** and speak for six seconds.
3. Move the object away and bring it back—its card should light up.
4. Click **Play its notes** to hear the recording.

The original keyboard-based OpenCV interface is still available:

```powershell
python main.py
```

## Running from WSL

WSL is great for editing and tests, but Windows normally owns the camera and
microphone. If the repository lives in WSL, run the app with Windows Python from
**Windows PowerShell**:

```powershell
cd \\wsl.localhost\<Distro>\home\<user>\path\to\ObjectEcho

py -3.13 -m venv "$env:USERPROFILE\objectecho-win\.venv"
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# Optional CLIP
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" -m pip install "open_clip_torch>=2.24,<4" Pillow

& "$env:USERPROFILE\objectecho-win\.venv\Scripts\python.exe" main.py --web `
  --data-dir "$env:USERPROFILE\objectecho-data"
```

Use a Windows-local data directory so SQLite does not run over the WSL network
filesystem.

## License

MIT. See [LICENSE](LICENSE).
