# videoTranslator

Open-source video dubbing pipeline that translates speech, synthesizes new voice audio, preserves background sound effects/music, and muxes a final translated video.

This project has grown from a simple script into a full stack workflow:
- Python backend pipeline (`translator.py`)
- FastAPI + WebSocket service (`server.py`)
- Next.js frontend (`video-translator/`)
- Hardware-aware model selection + config overrides (`config.yaml`)
- Multi-engine TTS (`Qwen3-TTS` as primary, `MeloTTS` optional, `gTTS` fallback)

---

## What’s implemented now

### Core translation pipeline
- **ASR + alignment**: WhisperX with faster-whisper backend + forced alignment for tighter word timing.
- **Translation**: NLLB-200 model family with hardware-tiered model selection.
- **Speech synthesis**: Unified `TTSEngine` routing between:
  - `qwen3` — primary engine with **voice cloning** support
  - `melo` — optional alternative engine
  - `gtts` (fallback)
- **Audio quality controls**:
  - Demucs vocal/background separation
  - Rubber Band time-stretch via `pyrubberband`
  - Segment timing budget + translation compression pass for overflow segments
  - Crossfades and ducking to reduce artifacts/pops
- **Final render**: FFmpeg stream-copy video + AAC audio muxing.

### Runtime and platform improvements
- **GPU-aware optimization** via `gpu_config.py`.
- **Sequential model lifecycle** to reduce VRAM spikes.
- **Performance telemetry** via `performance_monitor.py`.
- **Config-driven behavior** through `config.yaml`.

### Interfaces
- **CLI pipeline execution** through `translator.py`.
- **API + WebSocket orchestration** through `server.py`.
- **Web UI** through Next.js app in `video-translator/`.

---

## Supported target languages (current map)

`en`, `es`, `fr`, `zh`, `ja`, `ko`, `de`, `pt`

See `LANGUAGE_MODEL_MAP` in `translator.py` for active speaker and language mappings.

---

## 🎬 Demo: English to French Translation

Here's a sample of the VideoTranslator in action!

I translated the first 5 minutes of this video by Fern:
- **Original Video (English):** [The Hunt for America's Smartest Killer](https://youtu.be/wkVygetgeRY?si=hKF2XqJD3jZU3KIL)

The full 28-minute video wasn't translated as it would take a significant amount of time (likely well over an hour) on Kaggle T4 GPU at 360p resolution. This 5-minute clip demonstrates the translation quality and process.

- **Translated Output (First 5 mins, French):** [View Translated Sample (translated_fern_eng.mp4)](./translated_fern_eng.mp4)

---

## Repository layout

```text
videoTranslator/
├── translator.py           # End-to-end dubbing pipeline
├── tts_engine.py           # Unified TTS backend (Qwen3/Melo/gTTS)
├── server.py               # FastAPI upload + WebSocket translation endpoints
├── run.py                  # Starts backend + frontend together
├── setup.py                # Bootstrap helper (venv, deps, frontend)
├── gpu_config.py           # Hardware tier detection / optimizer
├── performance_monitor.py  # Timing and resource tracking
├── config.yaml             # Runtime overrides
├── requirements.txt
└── video-translator/       # Next.js frontend
```

---

## Prerequisites

- Python **3.11+** (tested on 3.11 & 3.12)
- Node.js + npm (for frontend)
- FFmpeg in `PATH`
- `rubberband-cli` installed (required by `pyrubberband`)
- CUDA GPU recommended for speed (CPU works, slower)
- Qwen3-TTS source: `https://github.com/QwenLM/Qwen3-TTS`
- Melo backend source (optional): `https://github.com/Felixdiamond/MeloTTS` (modified for python 3.12 support)

Linux helper packages (example):

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg rubberband-cli mecab libmecab-dev mecab-ipadic-utf8 libavcodec-extra
```

---

## Installation

### Option A — automated setup

```bash
git clone https://github.com/Felixdiamond/videoTranslator.git
cd videoTranslator
python setup.py
```

This installs base Python dependencies, whisperX, Qwen3-TTS (primary TTS), and the frontend. MeloTTS is optional.

| Flag | Effect |
|------|--------|
| *(default)* | whisperX + Qwen3-TTS |
| `--melo` | also install MeloTTS |
| `--no-qwen3` | skip Qwen3-TTS |
| `--no-qwen3 --melo` | MeloTTS-only install |

Examples:

```bash
# default — whisperX + Qwen3-TTS
python setup.py

# Qwen3-TTS + MeloTTS
python setup.py --melo

# MeloTTS only (skip Qwen3-TTS)
python setup.py --no-qwen3 --melo
```

> whisperX is always installed — it is required for ASR alignment.

### Option B — manual setup

```bash
git clone https://github.com/Felixdiamond/videoTranslator.git
cd videoTranslator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then install the frontend (optional if CLI or Server only):

```bash
cd video-translator && npm install && cd ..
```

#### Qwen3-TTS (primary TTS engine)

Install this for the recommended default path with voice cloning support.

```bash
sudo apt-get install sox
pip install wheel packaging
pip install -U flash-attn --no-build-isolation
git clone https://github.com/QwenLM/Qwen3-TTS.git /tmp/qwen3tts
sed -i 's/transformers==4.57.3/transformers>=4.47.1/' /tmp/qwen3tts/pyproject.toml
pip install -e /tmp/qwen3tts
```

#### MeloTTS (optional alternative engine)

```bash
pip install git+https://github.com/Felixdiamond/MeloTTS.git
python -m unidic download
```

Then in `config.yaml`:

```yaml
tts_mode: qwen3
enable_voice_cloning: true
```

Without Qwen3-TTS the pipeline can use `melo` or `gtts`, but voice cloning is unavailable.

---

## Kaggle

Use [This notebook](https://www.kaggle.com/code/felixdiamond/videotranslator)

Kaggle session settings:
- `accelerator: T4 x 2`
- `environment: Always use latest`

Minimal Kaggle run flow:
1. Open `videotranslator.ipynb` in Kaggle.
2. Run cells in order.
3. Keep TTS mode on `melo`.
4. Run translation cell with your input video path and target language.

The notebook includes Kaggle-specific setup quirks handling (dependency pin relaxations and install order) to reduce environment breakage on latest images.

---

## Run modes

### 1) Full app (backend + frontend)

```bash
source venv/bin/activate
python run.py
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

### 2) Backend only

```bash
source venv/bin/activate
python server.py
```

Optional backend environment variables (for hosted deployments):

- `CORS_ORIGINS`: comma-separated allowed origins (example: `https://app.example.com,https://staging.example.com`)
- `FRONTEND_URL`: single frontend origin to append to CORS allow-list
- `CORS_ALLOW_ALL`: `true/1` to allow all origins (development/trusted environments only)

### 3) CLI only

```bash
source venv/bin/activate
python translator.py <video_path> <target_language_code>
```

Example:

```bash
python translator.py ./sample.mp4 fr
```

Outputs are created in `*_translated_output/` directories.

### Frontend API/WS base URLs

The Next.js frontend defaults to local development URLs:

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`
- `NEXT_PUBLIC_WS_BASE_URL=ws://localhost:8000`

For hosted deployments, set these in `video-translator/.env.local` (or your hosting env config):

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.example.com
NEXT_PUBLIC_WS_BASE_URL=wss://api.example.com
```

---

## Configuration (`config.yaml`)

`translator.py` loads `config.yaml` at startup and applies overrides for model/runtime selection.

Useful keys:
- `hardware_tier`: `auto | cpu_low | cpu_high | gpu_low | gpu_medium | gpu_high`
- `whisper_model`
- `translation_model`
- `tts_mode`: `qwen3 | melo | gtts`
- `qwen3_model_size`: `0.6B | 1.7B`
- `enable_voice_cloning`: `true | false`
- `melo_speaker_id`
- `demucs_model`

Melo-only timing note:
- `CPS_MAP` in `translator.py` is used only for Melo pacing heuristics.
- `estimate_ideal_tts_speed(...)` is Melo-only logic and is ignored when `tts_mode` is `qwen3` or `gtts`.

Use this as the recommended default in `config.yaml`:

```yaml
tts_mode: qwen3
enable_voice_cloning: true
qwen3_model_size: "1.7B"
```

---

## API surface (current)

### Health endpoint

- `GET /health`
- Returns basic service status (`{"ok": true}`)

### Frontend options endpoint

- `GET /options`
- Returns frontend configuration payload:
  - available languages + labels
  - TTS modes
  - Qwen3 model sizes
  - Melo speaker IDs by language
  - API defaults used by frontend controls

### Upload endpoint
- `POST /upload`
- Receives a file and stores it in `uploaded_files/`
- Returns project-relative `filePath`

### Output file endpoint

- `GET /files/{file_path}`
- Serves project files by relative path (used by frontend download link)

### Translation WebSocket
- `WS /translate/{video_path}/{target_language}`
- Query params:
  - `tts_mode` (default `qwen3`)
  - `qwen3_model_size` (`0.6B` or `1.7B`, default `1.7B`)
  - `speaker_id` (Melo speaker)
  - `enable_voice_cloning` (bool, default `true`)

The frontend now exposes and sends these controls directly.

---

## Troubleshooting

- **`ffmpeg` not found**: install FFmpeg and ensure it is in `PATH`.
- **`pyrubberband` errors**: install `rubberband-cli`.
- **MeloTTS import/setup issues**: install `Felixdiamond/MeloTTS` manually and confirm environment activation.
- **Qwen3 unavailable**: install `qwen-tts` or source package; fallback to `melo`/`gtts`.
- **OOM on GPU**: lower `hardware_tier`, switch `tts_mode`, or reduce workload length.

Logs are written to `logs/video_translator.log`.

---

## Status and next focus

Major quality/stability milestones from the implementation plan are completed (ASR alignment, Demucs mixing, NLLB migration, config-driven model control, FFmpeg final mux).

Remaining focus is mostly validation and iterative quality tuning (long-form tests, edge-case handling, voice naturalness).

---

## Contributing

Issues and PRs are welcome. If you propose a quality/performance change, include:
- a short reproducible test clip
- hardware info (CPU/GPU + VRAM)
- before/after observations (timing, quality, artifacts)
