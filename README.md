# videoTranslator

Open-source video dubbing pipeline that translates speech, synthesizes new voice audio, preserves background sound effects/music, and muxes a final translated video.

This project has grown from a simple script into a full stack workflow:
- Python backend pipeline (`translator.py`)
- FastAPI + WebSocket service (`server.py`)
- Next.js frontend (`video-translator/`)
- Hardware-aware model selection + config overrides (`config.yaml`)
- Multi-engine TTS (`MeloTTS` as primary, `Qwen3-TTS` for **voice cloning**, `gTTS` fallback)

---

## What’s implemented now

### Core translation pipeline
- **ASR + alignment**: WhisperX with faster-whisper backend + forced alignment for tighter word timing.
- **Translation**: NLLB-200 model family with hardware-tiered model selection.
- **Speech synthesis**: Unified `TTSEngine` routing between:
  - `melo` — default, good multilingual quality
  - `qwen3` — **voice cloning**: clones the original speaker's voice into the target language
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

## Repository layout

```text
videoTranslator/
├── translator.py           # End-to-end dubbing pipeline
├── tts_engine.py           # Unified TTS backend (Melo/Qwen3/gTTS)
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

- Python **3.12**
- Node.js + npm (for frontend)
- FFmpeg in `PATH`
- `rubberband-cli` installed (required by `pyrubberband`)
- CUDA GPU recommended for speed (CPU works, slower)
- Melo backend source: `https://github.com/Felixdiamond/MeloTTS` (modified for python 3.12 support)

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

This installs base Python dependencies, whisperX, MeloTTS (default TTS), and the frontend. TTS selection is opt-in:

| Flag | Effect |
|------|--------|
| *(default)* | whisperX + MeloTTS |
| `--no-melo` | skip MeloTTS (use `gTTS` fallback only) |
| `--qwen3` | also install Qwen3-TTS (adds **voice cloning**) |
| `--no-melo --qwen3` | whisperX + Qwen3-TTS only |

Examples:

```bash
# default — whisperX + MeloTTS
python setup.py

# MeloTTS + voice cloning
python setup.py --qwen3

# Qwen3-TTS only (skip MeloTTS)
python setup.py --no-melo --qwen3
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

#### MeloTTS (default TTS engine)

```bash
pip install git+https://github.com/Felixdiamond/MeloTTS.git
python -m unidic download
```

#### Qwen3-TTS (optional — required for voice cloning)

Install this if you want the pipeline to clone the original speaker's voice into the target language.

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git /tmp/qwen3tts
sed -i 's/transformers==4.57.3/transformers>=4.47.1/' /tmp/qwen3tts/pyproject.toml
pip install -e /tmp/qwen3tts
```

Then in `config.yaml`:

```yaml
tts_mode: qwen3
enable_voice_cloning: true
```

Without Qwen3-TTS the pipeline stays on `melo` or `gtts` and voice cloning is unavailable.

---

## Kaggle (recommended for new users)

Use **only** `videotranslator.ipynb` in this repository. Other older Kaggle notebooks/scripts are considered outdated.

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

---

## Configuration (`config.yaml`)

`translator.py` loads `config.yaml` at startup and applies overrides for model/runtime selection.

Useful keys:
- `hardware_tier`: `auto | cpu_low | cpu_high | gpu_low | gpu_medium | gpu_high`
- `whisper_model`
- `translation_model`
- `tts_mode`: `melo | qwen3 | gtts`
- `qwen3_model_size`: `0.6B | 1.7B`
- `enable_voice_cloning`: `true | false`
- `melo_speaker_id`
- `demucs_model`

If you want Qwen3 voice cloning by default, set:

```yaml
tts_mode: qwen3
enable_voice_cloning: true
qwen3_model_size: "1.7B"
```

---

## API surface (current)

### Upload endpoint
- `POST /upload`
- Receives a file and stores it in `uploaded_files/`

### Translation WebSocket
- `WS /translate/{video_path}/{target_language}`
- Query params:
  - `tts_mode` (default `melo`)
  - `speaker_id` (Melo speaker)
  - `enable_voice_cloning` (bool)

Frontend currently uses the default WebSocket parameters unless customized.

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
