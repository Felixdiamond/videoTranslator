import os
import sys
import logging
import subprocess
import tempfile
import shutil
import re
from typing import List, Tuple, Dict, Optional
import numpy as np
import gc # For garbage collection

import whisperx
import pyrubberband as pyrb
import torchaudio
from demucs.pretrained import get_model as demucs_get_model
from demucs.apply import apply_model as demucs_apply_model
from moviepy import VideoFileClip
# CompositeAudioClip might not be directly used if pydub handles all composition
from gtts import gTTS
from pydub import AudioSegment, effects as pydub_effects # Renamed to avoid conflict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
# from pydub.playback import play # Not used in the new script's core logic

import torch

import yaml

# Import from new local modules
from gpu_config import gpu_optimizer, detect_hardware_tier
from performance_monitor import performance_monitor

# TTS engine — TTSEngine, MELO_ALL_SPEAKERS, get_all_speaker_ids are defined in tts_engine.py
from tts_engine import TTSEngine


# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filepath = os.path.join(log_dir, "video_translator.log")

# Remove existing handlers before adding new ones to prevent duplicate logs in notebooks
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), # Ensure logs go to stdout for CLI/FastAPI
        logging.FileHandler(log_filepath)
    ]
)
logging.info("Logging system initialized for translator.py.")

# --- Optional config.yaml override ---
def _load_app_config(path: str = "config.yaml") -> dict:
    """Load optional config.yaml.  Returns {} if the file is absent or invalid."""
    try:
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logging.warning(f"config.yaml load error ({path}): {exc}")
        return {}

APP_CONFIG: dict = _load_app_config()
if APP_CONFIG:
    logging.info(f"Loaded config.yaml overrides: {APP_CONFIG}")


# TTS engine lifecycle is managed per-call by TTSEngine (see tts_engine.py)

# Use global GPU optimizer
accelerator = gpu_optimizer.accelerator
device = gpu_optimizer.device

# --- Device announcement ---
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(device)
    vram_gb  = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    logging.info(f"[DEVICE] Running on GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
else:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    logging.info(f"[DEVICE] Running on CPU (no CUDA device found). System RAM: {ram_gb:.1f} GB")

# Language mapping for MeloTTS and fallback gTTS language codes
LANGUAGE_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "en": {"melo_language": "EN", "speaker_id": "EN-US", "gtts_lang": "en"},
    "es": {"melo_language": "ES", "speaker_id": "ES", "gtts_lang": "es"},
    "fr": {"melo_language": "FR", "speaker_id": "FR", "gtts_lang": "fr"},
    "zh": {"melo_language": "ZH", "speaker_id": "ZH", "gtts_lang": "zh-CN"},
    "ja": {"melo_language": "JP", "speaker_id": "JP", "gtts_lang": "ja"},
    "ko": {"melo_language": "KR", "speaker_id": "KR", "gtts_lang": "ko"},
    # DE and PT: gTTS-only — MeloTTS has no model for these languages.
    # Do NOT add melo_language or speaker_id here; process_video gates on supported set.
    "de": {"gtts_lang": "de"},
    "pt": {"gtts_lang": "pt"},
}

DUCKING_GAIN_DB = -18  # How much to reduce original audio volume during translated speech
CROSSFADE_MS = 50     # Crossfade duration for audio segments
MIN_SEGMENT_MS = 800

# --- Hardware-tiered Whisper model sizes ---
WHISPER_MODEL_TIERS: Dict[str, str] = {
    "cpu_low":    "base",               # 74M params, CPU-friendly
    "cpu_high":   "small",              # 244M params
    "gpu_low":    "medium",             # 769M params, <4GB VRAM
    "gpu_medium": "large-v3-turbo",     # 809M params, 5x faster than large-v3
    "gpu_high":   "large-v3",           # 1.54B params, max accuracy
}

# --- Empirical chars/sec per language at MeloTTS speed=1.0 ---
# NOTE: These values should be re-calibrated after any MeloTTS update.
# Run:  python calibrate_cps.py  and paste the output here.
# Only MeloTTS-supported languages are listed. DE/PT use gTTS which has no speed control.
CPS_MAP: Dict[str, float] = {
    "en": 13.8,
    "fr": 18.3,
    "es": 13.1,
    "zh": 6.1,
    "ja": 6.2,
    "ko": 5.7,
}

# --- NLLB-200 translation model tiers ---
TRANSLATION_MODEL_TIERS: Dict[str, Tuple[str, str, str]] = {
    "cpu_low":    ("facebook/nllb-200-distilled-600M", "cpu",  "float32"),
    "cpu_high":   ("facebook/nllb-200-1.3B",           "cpu",  "float32"),
    "gpu_low":    ("facebook/nllb-200-distilled-600M", "cuda", "float16"),
    "gpu_medium": ("facebook/nllb-200-1.3B",           "cuda", "float16"),
    "gpu_high":   ("facebook/nllb-200-3.3B",           "cuda", "float16"),
}

# NLLB-200 BCP-47 language codes
NLLB_LANG_CODES: Dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
}


def get_whisper_model_size() -> str:
    """Pick a Whisper model size based on available hardware (or config.yaml override)."""
    # config.yaml takes priority
    override = APP_CONFIG.get("whisper_model")
    if override:
        logging.info(f"[CONFIG] whisper_model override: {override!r}")
        return override

    hw_tier = APP_CONFIG.get("hardware_tier", "auto")
    if hw_tier == "auto" or hw_tier is None:
        if device.type != "cuda":
            tier = "cpu_high" if __import__("psutil").virtual_memory().total / 1024**3 >= 16 else "cpu_low"
        else:
            try:
                vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            except Exception:
                vram_gb = 8
            if vram_gb < 6:
                tier = "gpu_low"
            elif vram_gb < 12:
                tier = "gpu_medium"
            else:
                tier = "gpu_high"
    else:
        tier = hw_tier
    return WHISPER_MODEL_TIERS.get(tier, WHISPER_MODEL_TIERS["gpu_medium"])

def create_project_structure(video_path: str, target_language: str) -> str:
    logging.info("Creating project structure...")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Use a simpler project directory name, target_language will be in filenames
    project_dir_name = f"{base_name}_translated_output"
    project_dir = os.path.join(os.getcwd(), project_dir_name)
    os.makedirs(project_dir, exist_ok=True)

    for subdir in ['audio', 'transcripts', 'translations', 'translated_segments', 'audio_debug']:
        os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
    
    logging.info(f"Project directory: {project_dir}")
    return project_dir

def extract_audio(video_path: str, output_path: str) -> str:
    logging.info(f"Extracting audio from video: {video_path} to {output_path}")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_path, codec='pcm_s16le', logger=None)
        logging.info(f"Audio extracted successfully to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to extract audio: {e}", exc_info=True)
        raise
    return output_path

def transcribe_with_whisperx(audio_path: str) -> dict:
    """
    Transcribe audio using faster-whisper via WhisperX, then run forced phoneme
    alignment to get precise word-level start/end timestamps (±30ms accuracy).
    Falls back to segment-level timestamps if alignment fails.
    """
    logging.info("Transcribing audio with WhisperX (faster-whisper backend)...")
    model_size = get_whisper_model_size()
    compute_type = "float16" if device.type == "cuda" else "int8"
    device_str = str(device).split(":")[0]  # "cuda" or "cpu"

    if device.type == "cuda":
        logging.info(f"[WHISPERX] Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logging.info("[WHISPERX] Using CPU (no GPU available)")
    logging.info(f"Loading WhisperX model: {model_size!r}, compute={compute_type!r}, device={device_str!r}")
    try:
        wx_model = whisperx.load_model(
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            device_str,
            compute_type=compute_type,
            asr_options={"word_timestamps": True},
        )
        audio = whisperx.load_audio(audio_path)
        batch_size = 16 if device.type == "cuda" else 4
        result = wx_model.transcribe(audio, batch_size=batch_size)
        detected_lang = result.get("language", "en")
        logging.info(f"Transcription complete. Detected language: {detected_lang}")
    except Exception as e:
        logging.error(f"WhisperX transcription failed: {e}", exc_info=True)
        raise
    finally:
        # Free Whisper model before loading alignment model
        try:
            del wx_model
        except NameError:
            pass
        gc.collect()
        gpu_optimizer.clear_cache()
        logging.info("WhisperX transcription model freed from memory.")

    # Forced phoneme alignment for precise word-level timestamps
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device_str,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device_str,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()
        gpu_optimizer.clear_cache()
        logging.info("WhisperX forced alignment complete; word-level timestamps available.")
    except Exception as e:
        logging.warning(
            f"WhisperX alignment failed for lang '{detected_lang}': {e}. "
            f"Falling back to Whisper segment timestamps."
        )

    if "text" not in result:
        segments = result.get("segments") or []
        result["text"] = " ".join(
            (seg.get("text") or "").strip()
            for seg in segments
            if isinstance(seg, dict) and (seg.get("text") or "").strip()
        )

    return result


# Keep old name as alias so nothing breaks during refactor transition
transcribe_with_whisper = transcribe_with_whisperx

def text_to_speech(
    text_to_synthesize: str,
    target_language_code: str,
    output_filepath: str,
    tts_engine: Optional["TTSEngine"] = None,
    speed: float = 1.0,
    speaker_id: Optional[str] = None,
) -> None:
    """Synthesise *text_to_synthesize* and write audio to *output_filepath*.

    Delegates to *tts_engine* (MeloTTS or Qwen3-TTS) when available,
    otherwise falls back to gTTS directly.

    Args:
        text_to_synthesize: Text to speak.
        target_language_code: Short language code (``"en"``, ``"fr"`` …).
        output_filepath: Destination WAV path.
        tts_engine: Loaded :class:`tts_engine.TTSEngine` instance, or ``None``
                    to force a direct gTTS fallback.
        speed: TTS playback-speed multiplier (MeloTTS only; ignored by Qwen3/gTTS).
        speaker_id: Explicit MeloTTS speaker ID (e.g. ``"EN-BR"``).
    """
    if tts_engine is not None:
        try:
            tts_engine.synthesize(
                text_to_synthesize,
                target_language_code,
                output_filepath,
                speed=speed,
                speaker_id=speaker_id,
            )
            return
        except Exception as e:
            logging.error(
                f"text_to_speech: TTSEngine failed for lang='{target_language_code}': {e}. "
                "Falling back to gTTS.",
                exc_info=True,
            )

    # Direct gTTS fallback when no tts_engine provided or on failure
    lang_config = LANGUAGE_MODEL_MAP.get(target_language_code)
    if not lang_config:
        logging.error(
            f"text_to_speech: language '{target_language_code}' not in LANGUAGE_MODEL_MAP."
        )
        AudioSegment.silent(duration=10).export(output_filepath, format="wav")
        return

    gtts_lang_code = lang_config.get("gtts_lang", "en")
    try:
        logging.info(
            f"text_to_speech: gTTS '{text_to_synthesize[:30]}...' lang={gtts_lang_code}"
        )
        gTTS(text=text_to_synthesize, lang=gtts_lang_code, slow=False).save(output_filepath)
    except Exception as e:
        logging.error(
            f"text_to_speech: gTTS failed for '{text_to_synthesize[:30]}...': {e}. "
            "Writing silent placeholder.",
            exc_info=True,
        )
        AudioSegment.silent(duration=10, frame_rate=22050).export(output_filepath, format="wav")


def extract_voice_reference(
    audio_path: str,
    transcript: dict,
    project_dir: str,
    fallback_audio_path: Optional[str] = None,

) -> Optional[Dict[str, object]]:
    """Extract a clean 4–10 s speech segment from the source audio for voice cloning.

    Scans transcript segments for a clip with duration 4–10 s and RMS above
    -30 dBFS (i.e. clearly audible speech, not background noise). Prefers
    segments with usable transcript text and durations near 7 s. Exports the
    chosen clip as ``<project_dir>/audio/voice_reference.wav``.

    Args:
        audio_path: Preferred audio source for reference extraction.
        transcript: WhisperX transcript dict with a ``"segments"`` list.
        project_dir: Root project directory (``audio/`` sub-folder must exist).
        fallback_audio_path: Optional secondary audio source to try if the
            preferred one yields no usable reference segment.

    Returns:
        Metadata for the exported reference clip, including the clip path and
        matched transcript text, or ``None`` if no suitable segment was found
        (pipeline will proceed without voice cloning).
    """
    ref_path = os.path.join(project_dir, "audio", "voice_reference.wav")
    segments = transcript.get("segments", [])

    candidate_audio_paths = []
    for path in (audio_path, fallback_audio_path):
        if path and path not in candidate_audio_paths and os.path.exists(path):
            candidate_audio_paths.append(path)

    for candidate_audio_path in candidate_audio_paths:
        try:
            audio = AudioSegment.from_file(candidate_audio_path)
        except Exception as e:
            logging.warning(
                f"extract_voice_reference: could not load audio '{candidate_audio_path}': {e}"
            )
            continue

        best_candidate = None
        for seg in segments:
            if not isinstance(seg, dict):
                continue

            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                continue

            dur = end - start
            if not (4.0 <= dur <= 10.0):
                continue

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            clip = audio[start_ms:end_ms]
            if clip.dBFS <= -30:
                continue

            text = (seg.get("text") or "").strip()
            text_bonus = 10.0 if text else 0.0
            duration_penalty = abs(dur - 7.0)
            score = clip.dBFS + text_bonus - duration_penalty
            candidate = {
                "path": ref_path,
                "text": text,
                "start": start,
                "end": end,
                "duration": dur,
                "dbfs": clip.dBFS,
                "clip": clip,
                "score": score,
                "source_audio": candidate_audio_path,
            }
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

        if best_candidate:
            best_candidate["clip"].export(ref_path, format="wav")
            logging.info(
                "extract_voice_reference: exported "
                f"{best_candidate['start']:.1f}s–{best_candidate['end']:.1f}s "
                f"({best_candidate['duration']:.1f}s, {best_candidate['dbfs']:.1f} dBFS) "
                f"from='{best_candidate['source_audio']}' "
                f"text='{str(best_candidate['text'])[:80]}' → {ref_path}"
            )
            best_candidate.pop("clip", None)
            best_candidate.pop("score", None)
            return best_candidate

    logging.warning(
        "extract_voice_reference: no suitable segment found (need 4–10 s, dBFS > -30). "
        "Proceeding without voice cloning."
    )
    return None


def _translate_text_nllb(
    text: str,
    source_language: str,
    target_language: str,
    model,
    tokenizer,
    num_beams: int = 4,
    length_penalty: float = 1.0,
) -> str:
    """Translate text using an NLLB-200 model ([PLAN_h.md L787–L807](PLAN_h.md))."""
    if not text.strip():
        return ""
    if source_language == target_language:
        return text

    src_nllb = NLLB_LANG_CODES.get(source_language, f"{source_language}_Latn")
    tgt_nllb = NLLB_LANG_CODES.get(target_language, f"{target_language}_Latn")

    logging.info(f"Translating ({src_nllb}→{tgt_nllb}): '{text[:60]}…'")

    tokenizer.src_lang = src_nllb
    raw_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in raw_inputs.items()}
    target_lang_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_new_tokens=512,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    del inputs, raw_inputs, outputs
    gc.collect()
    return translated


def translate_with_length_target(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model,
    tokenizer,
    target_length_ratio: float = 1.0,
) -> str:
    """Translate encouraging shorter output when target_length_ratio < 0.85 ([PLAN_h.md L890–L915](PLAN_h.md))."""
    if not text.strip() or src_lang == tgt_lang:
        return text
    length_penalty = 0.6 if target_length_ratio < 0.85 else 1.0
    return _translate_text_nllb(
        text, src_lang, tgt_lang, model, tokenizer,
        num_beams=4, length_penalty=length_penalty,
    )


def batch_translate_segments(
    segments: List[dict],
    src_lang: str,
    tgt_lang: str,
    model,
    tokenizer,
    batch_size: int = 8,
) -> List[dict]:
    """Translate all transcript segments in batches before TTS begins ([PLAN_h.md L1238–L1282](PLAN_h.md)).

    Returns copies of the segment dicts with a ``'translated_text'`` key added.
    """
    src_nllb = NLLB_LANG_CODES.get(src_lang, f"{src_lang}_Latn")
    tgt_nllb = NLLB_LANG_CODES.get(tgt_lang, f"{tgt_lang}_Latn")
    target_lang_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

    texts = [seg["text"].strip() for seg in segments]
    translated_texts: List[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokenizer.src_lang = src_nllb
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            out_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_new_tokens=512,
                num_beams=4,
                length_penalty=1.0,
            )

        batch_translations = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
        translated_texts.extend([t.strip() for t in batch_translations])
        del inputs, out_tokens
        gc.collect()
        logging.info(f"Batch-translated segments {i}–{min(i + batch_size, len(texts)) - 1}")

    results = []
    for seg, trans in zip(segments, translated_texts):
        results.append({**seg, "translated_text": trans})
    return results


def analyze_segment_timing_budget(
    segments: List[dict],
    translations: List[str],
    target_lang: str,
) -> List[Dict]:
    """Pre-screen every segment for timing difficulty before TTS synthesis ([PLAN_h.md L950–L975](PLAN_h.md)).

    Returns list of dicts with keys:
      segment, translation, available_ms, estimated_ms, ratio, action

    Action values: ``"natural"`` | ``"speed_up_tts"`` | ``"slow_down_tts"`` | ``"compress_translation"``
    """
    cps = CPS_MAP.get(target_lang, 14.0)
    results = []
    for seg, trans in zip(segments, translations):
        available_ms = int((seg["end"] - seg["start"]) * 1000)
        estimated_ms = int((len(trans) / cps) * 1000) if cps > 0 else available_ms
        ratio = estimated_ms / available_ms if available_ms > 0 else 1.0
        if ratio > 1.8:
            action = "compress_translation"
        elif ratio > 1.3:
            action = "speed_up_tts"
        elif ratio < 0.6:
            action = "slow_down_tts"
        else:
            action = "natural"
        results.append({
            "segment": seg,
            "translation": trans,
            "available_ms": available_ms,
            "estimated_ms": estimated_ms,
            "ratio": ratio,
            "action": action,
        })
    return results


def merge_short_segments(segments: List[dict], min_ms: int = MIN_SEGMENT_MS) -> List[dict]:
    """Merge very short consecutive segments into a neighbor before translation/TTS."""
    if not segments:
        return segments

    merged: List[dict] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        duration_ms = int(max(0.0, seg_end - seg_start) * 1000)

        if duration_ms < min_ms and i + 1 < len(segments):
            next_seg = segments[i + 1]
            next_start = float(next_seg.get("start", seg_end) or seg_end)
            next_end = float(next_seg.get("end", next_start) or next_start)
            combined_text = " ".join(
                part for part in [str(seg.get("text", "")).strip(), str(next_seg.get("text", "")).strip()] if part
            )
            combined = {
                **next_seg,
                "start": min(seg_start, next_start),
                "end": max(seg_end, next_end),
                "text": combined_text,
            }
            merged.append(combined)
            logging.info(
                f"[MERGE] merged seg {i} ({duration_ms}ms) + seg {i+1} "
                f"({int(max(0.0, next_end - next_start) * 1000)}ms) -> "
                f"{int(max(0.0, combined['end'] - combined['start']) * 1000)}ms"
            )
            i += 2
        else:
            merged.append(seg)
            i += 1

    return merged


def split_translation_at_clauses(text: str) -> List[str]:
    """Split translated text at sentence boundaries only (not commas)."""
    # Only split at strong boundaries: period, exclamation, question mark
    clauses = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short clauses (< 4 words) — merge back with previous
    merged = []
    for clause in clauses:
        if clause.strip():
            if merged and len(clause.split()) < 4:
                merged[-1] = merged[-1] + " " + clause
            else:
                merged.append(clause.strip())
    return merged if merged else [text.strip()]


def synthesise_with_pauses(
    text: str,
    available_ms: int,
    target_language_code: str,
    tts_engine: Optional["TTSEngine"],
    speed: float,
    speaker_id: Optional[str],
    segment_index: int,
) -> AudioSegment:
    """Synthesize clause-by-clause and interleave silence to reduce overflow pressure."""
    clauses = split_translation_at_clauses(text)
    if len(clauses) <= 1:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        try:
            text_to_speech(
                text,
                target_language_code,
                out_path,
                tts_engine=tts_engine,
                speed=speed,
                speaker_id=speaker_id,
            )
            return AudioSegment.from_file(out_path)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    clause_audio: List[AudioSegment] = []
    total_speech_ms = 0
    temp_paths: List[str] = []
    try:
        for clause in clauses:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                clause_path = tmp.name
            temp_paths.append(clause_path)
            text_to_speech(
                clause,
                target_language_code,
                clause_path,
                tts_engine=tts_engine,
                speed=speed,
                speaker_id=speaker_id,
            )
            audio = AudioSegment.from_file(clause_path)
            clause_audio.append(audio)
            total_speech_ms += len(audio)

        gap_count = max(0, len(clauses) - 1)
        remaining_ms = max(0, available_ms - total_speech_ms)
        gap_ms = min(400, (remaining_ms // gap_count) if gap_count > 0 else 0)

        result = clause_audio[0]
        for audio in clause_audio[1:]:
            if gap_ms > 0:
                result += AudioSegment.silent(duration=gap_ms, frame_rate=audio.frame_rate)
            result += audio

        logging.info(
            f"[PROSODIC_PAUSE] seg {segment_index}: {len(clauses)} clauses, "
            f"speech={total_speech_ms}ms, gaps={gap_ms}ms x {gap_count}, total={len(result)}ms"
        )
        return result
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)


def load_translation_model(tier: str) -> Tuple:
    """Load the NLLB-200 translation model and tokenizer for the given hardware tier ([PLAN_h.md L762–L770](PLAN_h.md)).

    Respects ``config.yaml`` ``translation_model`` key to override the tier default.
    """
    model_id, model_device, dtype_str = TRANSLATION_MODEL_TIERS.get(
        tier, TRANSLATION_MODEL_TIERS["gpu_medium"]
    )
    # Allow config.yaml to override the default model choice.
    cfg_model = APP_CONFIG.get("translation_model")
    if cfg_model:
        logging.info(f"[CONFIG] translation_model override: {cfg_model!r}")
        model_id = cfg_model
    if model_device == "cuda" and torch.cuda.is_available():
        logging.info(f"[TRANSLATION] Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        model_device = "cpu"  # fall back gracefully if CUDA was requested but unavailable
        logging.info("[TRANSLATION] Using CPU")
    logging.info(f"Loading translation model: {model_id} on {model_device} ({dtype_str})")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    trans_model = trans_model.to(model_device)
    trans_model.eval()
    logging.info(f"Translation model loaded: {model_id}")
    return trans_model, tokenizer

def enhance_voice(audio: AudioSegment) -> AudioSegment:
    """Enhance voice with EQ, compression, and normalization."""
    enhanced = audio.high_pass_filter(85) # Cut sub-bass rumble
    # Optional: slight boost in presence/clarity range, e.g., 2-5kHz, but be careful.
    # enhanced = enhanced.low_pass_filter(10000) # Cut very high hiss if present

    # Compressor: threshold, ratio, attack, release. Tune these.
    enhanced = pydub_effects.compress_dynamic_range(enhanced, threshold=-18.0, ratio=3.0, attack=5.0, release=100.0)
    
    # Normalize to a target peak level (e.g., -1.0 dBFS for headroom)
    enhanced = pydub_effects.normalize(enhanced, headroom=1.0)
    return enhanced

def advanced_time_stretch(audio: AudioSegment, target_duration_ms: int) -> AudioSegment:
    """
    High-quality time-stretch using Rubber Band Library v4 via pyrubberband.
    Handles ratios from 0.25–4.0 with minimal artifacts on speech.
    Requires: sudo apt-get install rubberband-cli
    """
    if len(audio) == 0 or target_duration_ms <= 0:
        return audio if len(audio) > 0 else AudioSegment.silent(duration=10)

    current_duration_ms = len(audio)
    rate = current_duration_ms / target_duration_ms  # >1 = speed up, <1 = slow down

    RATE_MIN, RATE_MAX = 0.65, 1.50
    if rate < RATE_MIN or rate > RATE_MAX:
        logging.warning(
            f"Time stretch rate {rate:.2f} outside natural range [{RATE_MIN}, {RATE_MAX}]. "
            f"Clamping and padding/trimming to compensate."
        )
    rate = float(np.clip(rate, RATE_MIN, RATE_MAX))
    logging.info(f"[RUBBERBAND] stretch rate={rate:.3f} ({current_duration_ms}ms → {target_duration_ms}ms)")

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.sample_width == 2:
        samples /= 32768.0
    elif audio.sample_width == 1:
        samples = (samples - 128) / 128.0

    sr = audio.frame_rate

    if audio.channels == 2:
        samples_2ch = samples.reshape(-1, 2).T  # (2, n_samples)
        stretched = pyrb.time_stretch(samples_2ch, sr, rate)
        stretched = stretched.T.flatten()
    else:
        stretched = pyrb.time_stretch(samples, sr, rate)

    # Precise sample-count correction to avoid off-by-one drift
    target_samples = int(target_duration_ms / 1000 * sr) * audio.channels
    if len(stretched) > target_samples:
        stretched = stretched[:target_samples]
    elif len(stretched) < target_samples:
        stretched = np.concatenate([stretched, np.zeros(target_samples - len(stretched), dtype=np.float32)])

    if audio.sample_width == 2:
        stretched = (stretched * 32768.0).clip(-32768, 32767).astype(np.int16)
    elif audio.sample_width == 1:
        stretched = ((stretched * 128) + 128).clip(0, 255).astype(np.uint8)

    return AudioSegment(
        stretched.tobytes(),
        frame_rate=sr,
        sample_width=audio.sample_width,
        channels=audio.channels,
    )


def estimate_ideal_tts_speed(
    text: str,
    original_duration_ms: int,
    lang_code: str,
    tts_mode: str = "melo",
) -> float:
    """
    Pre-TTS speed estimation (MeloTTS only).
    For Qwen3/gTTS, returns 1.0 immediately — speed is not a numeric parameter there.
    Requires the language to be in CPS_MAP (MeloTTS-supported set).
    """
    if tts_mode != "melo":
        return 1.0
    if lang_code not in CPS_MAP:
        return 1.0  # language not supported by MeloTTS, no CPS data
    cps = CPS_MAP[lang_code]
    if cps <= 0 or original_duration_ms <= 0 or not text.strip():
        return 1.0
    estimated_duration_s = len(text) / cps
    target_duration_s = original_duration_ms / 1000.0
    ideal_speed = estimated_duration_s / target_duration_s
    ideal_speed = float(np.clip(ideal_speed, 0.7, 1.4))
    logging.info(
        f"[TTS speed est] lang={lang_code!r} chars={len(text)} "
        f"est={estimated_duration_s:.2f}s target={target_duration_s:.2f}s speed={ideal_speed:.2f}"
    )
    return ideal_speed


def fit_tts_to_slot(
    tts_audio: AudioSegment,
    available_ms: int,
    segment_index: int,
    next_gap_ms: int = 0,
) -> AudioSegment:
    """Fit synthesized speech to a segment slot with naturalness-first rules."""
    tts_ms = len(tts_audio)
    if available_ms <= 0:
        return tts_audio

    ratio = tts_ms / available_ms

    if 0.90 <= ratio <= 1.10:
        logging.info(f"[TTS_FIT] seg {segment_index}: natural fit ({ratio:.2f}), no adjustment")
        return tts_audio

    STRETCH_FLOOR = 0.65
    STRETCH_CEIL  = 1.50

    if ratio > 1.10:
        # Try borrowing from next gap first before stretching or trimming.
        # Cap the borrow so effective_ratio stays inside [STRETCH_FLOOR, STRETCH_CEIL]:
        # borrowing too much pushes rate below 0.65, rubberband clamps silently, leaving a silent hole.
        max_effective_ms = int(tts_ms / STRETCH_FLOOR)   # floor constraint: rate can't go below 0.65
        raw_effective_ms = available_ms + next_gap_ms
        effective_ms = min(raw_effective_ms, max_effective_ms)
        effective_ratio = tts_ms / effective_ms if effective_ms > 0 else ratio

        if effective_ratio <= STRETCH_CEIL:
            stretched = advanced_time_stretch(tts_audio, effective_ms)
            actual_borrow = effective_ms - available_ms
            logging.info(
                f"[TTS_FIT] seg {segment_index}: borrowed {actual_borrow}ms "
                f"from next gap (rate {ratio:.2f} → {effective_ratio:.2f})"
            )
            return stretched

        # Gap borrowing not enough — stretch to 1.5x hard cap, then trim
        safe_target_ms = int(tts_ms / 1.50)
        stretched = advanced_time_stretch(tts_audio, safe_target_ms)
        trimmed = stretched[:available_ms]
        logging.warning(
            f"[TTS_FIT] seg {segment_index}: overflow capped at 1.50x, "
            f"trimmed {len(stretched)}ms -> {available_ms}ms"
        )
        return trimmed

    if ratio < 0.75:
        silence_ms = max(0, available_ms - tts_ms)
        if silence_ms > 400:
            # Distribute: small lead-in silence + speech + remaining silence
            lead_ms = min(150, silence_ms // 4)
            tail_ms = silence_ms - lead_ms
            padded = (
                AudioSegment.silent(duration=lead_ms, frame_rate=tts_audio.frame_rate)
                + tts_audio
                + AudioSegment.silent(duration=tail_ms, frame_rate=tts_audio.frame_rate)
            )
        else:
            padded = tts_audio + AudioSegment.silent(
                duration=silence_ms, frame_rate=tts_audio.frame_rate
            )
        logging.info(
            f"[TTS_FIT] seg {segment_index}: underflow — "
            f"padding {silence_ms}ms silence (ratio={ratio:.2f})"
        )
        return padded

    # 0.75–0.90: gentle stretch
    return advanced_time_stretch(tts_audio, available_ms)


# --- Segment Processing and Synchronization ---
def process_segment(
    segment_info: Tuple[int, dict],
    target_language_code: str,
    segments_dir: str,
    tts_engine: Optional["TTSEngine"],
    speaker_id: Optional[str] = None,
    next_gap_ms: int = 0,
):
    """Synthesise and time-stretch one pre-translated segment.

    The segment dict is expected to have a ``'translated_text'`` key injected
    by ``batch_translate_segments`` before this function is called.

    """
    i, segment = segment_info
    start_time_ms = int(segment['start'] * 1000)
    end_time_ms = int(segment['end'] * 1000)
    original_duration_ms = end_time_ms - start_time_ms

    if original_duration_ms <= 0:
        logging.warning(f"Segment {i} has zero/negative duration ({original_duration_ms}ms). Skipping.")
        return None

    # Translation was done in a pre-pass by batch_translate_segments
    translated_text = segment.get("translated_text", "").strip()
    if not translated_text:
        logging.warning(f"Segment {i}: No translated_text on segment. Skipping TTS.")
        return None

    logging.info(f"Processing segment {i}: '{translated_text[:40]}...' (Orig dur: {original_duration_ms}ms)")

    tts_output_path = os.path.join(segments_dir, f"segment_{i:04d}.wav")


    # Use word-level start for more precise overlay position.
    words = segment.get("words", [])
    word_starts = [w["start"] for w in words if w.get("start") is not None]
    if word_starts:
        start_time_ms = max(0, int(word_starts[0] * 1000) - 30)  # 30ms pre-roll pad

    try:
        # CPS estimation only valid for MeloTTS — skip for Qwen3/gTTS
        tts_mode_active = tts_engine.mode if tts_engine is not None else "gtts"
        ideal_speed = estimate_ideal_tts_speed(
            translated_text, original_duration_ms, target_language_code, tts_mode=tts_mode_active
        )

        if tts_mode_active == "melo" and target_language_code in CPS_MAP:
            cps = CPS_MAP[target_language_code]
            estimated_ms = int((len(translated_text) / cps) * 1000)
            estimated_ratio = estimated_ms / original_duration_ms if original_duration_ms > 0 else 1.0
        else:
            estimated_ratio = 1.0  # no CPS estimate available — synthesise normally

        if estimated_ratio < 0.75 and original_duration_ms > 1200:
            # Underflow with enough room: use clause-level pauses to fill naturally
            generated_audio = synthesise_with_pauses(
                translated_text,
                original_duration_ms,
                target_language_code,
                tts_engine,
                ideal_speed,
                speaker_id,
                i,
            )
            generated_audio.export(tts_output_path, format="wav")
        else:
            # Overflow or natural: single synthesis call
            text_to_speech(translated_text, target_language_code, tts_output_path,
                           tts_engine=tts_engine, speed=ideal_speed, speaker_id=speaker_id)
            generated_audio = AudioSegment.from_file(tts_output_path)

        fitted_audio = fit_tts_to_slot(generated_audio, original_duration_ms, i, next_gap_ms=next_gap_ms)
        del generated_audio; gc.collect()

        fitted_audio.export(tts_output_path, format="wav")

        return (start_time_ms, tts_output_path)

    except Exception as e:
        logging.error(f"Error processing segment {i} ('{translated_text[:30]}...'): {e}", exc_info=True)
        gpu_optimizer.clear_cache()
        gc.collect()
        return None


def adaptive_segment_processing(
    segments_data: List[dict],
    target_language_code: str,
    project_dir: str,
    tts_engine: Optional["TTSEngine"],
    speaker_id: Optional[str] = None,
):
    """Synthesise all pre-translated segments sequentially.

    ``segments_data`` must be a list of dicts that each have a ``'translated_text'``
    key (injected by ``batch_translate_segments`` in ``process_video``).
    Computes ``next_gap_ms`` for each segment so ``fit_tts_to_slot`` can borrow
    time from inter-segment silence when needed.
    """
    segments_dir = os.path.join(project_dir, 'translated_segments')
    processed_segments_info = []

    # Pre-compute gap after each segment (time before next segment starts)
    next_gaps_ms = []
    for i, seg in enumerate(segments_data):
        if i + 1 < len(segments_data):
            gap = int((segments_data[i + 1]['start'] - seg['end']) * 1000)
            next_gaps_ms.append(max(0, gap))
        else:
            next_gaps_ms.append(0)

    for i, segment in enumerate(tqdm(segments_data, desc="Synthesising audio segments")):
        result = process_segment(
            (i, segment),
            target_language_code,
            segments_dir,
            tts_engine,
            speaker_id,
            next_gap_ms=next_gaps_ms[i],
        )
        if result:
            processed_segments_info.append(result)

    return processed_segments_info


# ---------------------------------------------------------------------------
# Word-level speech intervals
# ---------------------------------------------------------------------------

def build_speech_intervals(
    transcript: dict,
    pad_start_ms: int = 30,
    pad_end_ms: int = 50,
) -> List[Tuple[int, int]]:
    """
    Build precise speech intervals from WhisperX word-level timestamps.

    pad_start_ms: bring silence in EARLIER than actual word start (pre-roll removal)
    pad_end_ms:   keep original muted slightly LONGER after word ends (post-roll)

    Words within 300ms of each other are merged into a single contiguous speech block.
    Falls back to segment-level timestamps when no word-level data exists.
    """
    words: List[Tuple[int, int]] = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            s = w.get("start")
            e = w.get("end")
            if s is not None and e is not None:
                words.append((int(s * 1000), int(e * 1000)))

    if not words:
        # Fallback: segment-level timestamps with padding
        return sorted([
            (max(0, int(s["start"] * 1000) - pad_start_ms),
             int(s["end"] * 1000) + pad_end_ms)
            for s in transcript.get("segments", [])
            if s["end"] > s["start"]
        ])

    # Merge words into contiguous blocks (gap < 300ms = same block)
    intervals: List[Tuple[int, int]] = []
    block_start = words[0][0] - pad_start_ms
    block_end   = words[0][1] + pad_end_ms
    for start_ms, end_ms in words[1:]:
        if start_ms - block_end < 300:
            block_end = end_ms + pad_end_ms
        else:
            intervals.append((max(0, block_start), block_end))
            block_start = start_ms - pad_start_ms
            block_end   = end_ms + pad_end_ms
    intervals.append((max(0, block_start), block_end))
    return sorted(intervals)


# ---------------------------------------------------------------------------
# Demucs source separation
# ---------------------------------------------------------------------------

def separate_vocals_from_audio(
    audio_path: str,
    output_dir: str,
    device_obj,  # torch.device
    model_name: str = "htdemucs_ft",
) -> Tuple[str, str]:
    """
    Separate audio into vocals and background (all non-vocal stems summed).
    Returns (vocals_wav_path, background_wav_path).

    Uses htdemucs_ft (fine-tuned hybrid Transformer Demucs) — best quality.
    Requires ~3GB VRAM on GPU; automatically falls back to overlap-add on CPU.
    """
    if device_obj.type == "cuda":
        logging.info(f"[DEMUCS] Using GPU: {torch.cuda.get_device_name(device_obj)}")
    else:
        logging.info("[DEMUCS] Using CPU (no GPU available — separation will be slower)")
    logging.info(f"[DEMUCS] Separating vocals from {audio_path} using {model_name!r}...")
    demucs_model = demucs_get_model(model_name)
    device_str = str(device_obj).split(":")[0]
    demucs_model.to(device_obj)
    demucs_model.eval()

    wav, sr = torchaudio.load(audio_path)
    if sr != demucs_model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, demucs_model.samplerate)
        wav = resampler(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)          # mono → stereo
    wav = wav.unsqueeze(0).to(device_obj)  # (1, 2, samples)

    with torch.no_grad():
        sources = demucs_apply_model(demucs_model, wav, split=True, overlap=0.25)[0]
    # sources shape: (n_stems, 2, samples)  —  ['drums', 'bass', 'other', 'vocals']

    source_names = demucs_model.sources
    vocal_idx = source_names.index("vocals")

    vocals_wav     = sources[vocal_idx].cpu()
    background_wav = (
        sources[:vocal_idx].sum(0).cpu() + sources[vocal_idx + 1:].sum(0).cpu()
    )

    audio_out_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)
    vocals_path     = os.path.join(audio_out_dir, "demucs_vocals.wav")
    background_path = os.path.join(audio_out_dir, "demucs_background.wav")
    torchaudio.save(vocals_path,     vocals_wav,     demucs_model.samplerate)
    torchaudio.save(background_path, background_wav, demucs_model.samplerate)

    del demucs_model, sources, wav
    gc.collect()
    gpu_optimizer.clear_cache()
    logging.info(f"[DEMUCS] Saved vocals → {vocals_path}, background → {background_path}")
    return vocals_path, background_path


def preserve_sound_effects(
    original_audio: AudioSegment,
    synced_speech_track: AudioSegment,
    transcript: dict,
    project_dir: str,
    device_obj,  # torch.device
) -> AudioSegment:
    """
    Demucs-based audio mixer.

    Strategy:
    1. Separate original audio into vocals + background (Demucs htdemucs_ft)
    2. Silence ONLY the vocals stem during speech blocks → SFX/music fully preserved
    3. Apply gentle -8dB duck (not -18dB) with 80ms fade edges during speech windows
    4. Overlay translated TTS on the clean background track
    """
    debug_dir = os.path.join(project_dir, "audio_debug")
    extracted_audio_path = os.path.join(project_dir, "audio", "extracted_audio.wav")

    # A1: reuse pre-computed stems from step 2 of process_video so Demucs is never
    # loaded alongside Whisper, NLLB, or TTSEngine.  Fall back to on-demand separation
    # only if stems are missing (e.g. when this function is called standalone in tests).
    _pre_vocals = os.path.join(project_dir, "audio", "demucs_vocals.wav")
    _pre_bg     = os.path.join(project_dir, "audio", "demucs_background.wav")
    if os.path.exists(_pre_vocals) and os.path.exists(_pre_bg):
        vocals_path = _pre_vocals
        bg_path     = _pre_bg
        logging.info("[MIX] Reusing pre-computed Demucs stems (A1 lifecycle).")
    else:
        logging.warning(
            "[MIX] Demucs stems not found at expected paths — running separation now. "
            "This means Demucs is loading while other models may still occupy VRAM (A1 violated)."
        )
        vocals_path, bg_path = separate_vocals_from_audio(
            extracted_audio_path, project_dir, device_obj
        )

    background_seg = AudioSegment.from_file(bg_path)
    if background_seg.frame_rate != original_audio.frame_rate:
        background_seg = background_seg.set_frame_rate(original_audio.frame_rate)
    target_channels = max(original_audio.channels, synced_speech_track.channels)
    background_seg = background_seg.set_channels(target_channels)

    speech_intervals = build_speech_intervals(transcript)

    # Gentle duck (-8dB) with fade edges during speech windows
    DUCK_DB   = -8.0
    FADE_MS   = 80
    final_mix = background_seg
    for start_ms, end_ms in speech_intervals:
        duck_seg = final_mix[start_ms:end_ms].apply_gain(DUCK_DB)
        fade = min(FADE_MS, max(0, len(duck_seg) // 4))
        if fade > 0:
            duck_seg = duck_seg.fade_in(fade).fade_out(fade)
        final_mix = final_mix.overlay(duck_seg, position=start_ms)

    tts_track = synced_speech_track.set_channels(target_channels)
    final_mix = final_mix.overlay(tts_track, position=0)
    final_mix = final_mix[:len(original_audio)]

    final_mix.export(os.path.join(debug_dir, "2_final_mixed.wav"), format="wav")
    logging.info(f"[MIX] Final mix duration: {len(final_mix)/1000:.2f}s")
    return final_mix


def create_synced_audio_track(
    original_audio_ref: AudioSegment,
    translated_segments: List[dict],
    target_language_code: str,
    project_dir: str,
    tts_engine: Optional["TTSEngine"],
    speaker_id: Optional[str] = None,
) -> AudioSegment:
    """Creates a single audio track of the translated speech, synced to original timings.

    ``translated_segments`` must be pre-translated (each dict has ``'translated_text'``).
    """
    logging.info("Creating full synced translated speech track with segment fades...")
    target_channels = original_audio_ref.channels

    full_synced_speech_track = AudioSegment.silent(
        duration=len(original_audio_ref),
        frame_rate=original_audio_ref.frame_rate
    ).set_channels(target_channels)

    processed_segments_info = adaptive_segment_processing(
        translated_segments,
        target_language_code, project_dir,
        tts_engine, speaker_id,
    )

    for start_time_ms, audio_filepath in processed_segments_info:
        try:
            segment_audio = AudioSegment.from_file(audio_filepath).set_channels(target_channels) # Ensure channel match
            
            # Apply crossfades if segment is long enough
            if CROSSFADE_MS > 0:
                fade_len = min(CROSSFADE_MS, len(segment_audio) // 2 if len(segment_audio) > 0 else 0)
                if fade_len > 0:
                    segment_audio = segment_audio.fade_in(fade_len).fade_out(fade_len)
            
            full_synced_speech_track = full_synced_speech_track.overlay(segment_audio, position=start_time_ms)
            del segment_audio; gc.collect()
        except Exception as e:
            logging.error(f"Error overlaying segment from {audio_filepath}: {e}", exc_info=True)
            
    full_synced_speech_track = enhance_voice(full_synced_speech_track)
    logging.info("[ENHANCE] Applied once globally to synced speech track.")
    return full_synced_speech_track


def create_final_video(video_path: str, audio_path: str, output_path: str) -> None:
    """Replace audio track using FFmpeg stream-copy — 10–50× faster than moviepy.

    Video stream is copied without re-encoding (-c:v copy).  Audio is encoded
    to AAC at 192k to satisfy the MP4 container.

    Raises:
        RuntimeError: if ffmpeg is not in $PATH or if the FFmpeg process fails.
    ([PLAN_h.md L1013–L1031](PLAN_h.md))
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found in $PATH.  "
            "Install it with: sudo apt-get install ffmpeg  (or brew install ffmpeg)"
        )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",   # video from original file
        "-map", "1:a:0",   # audio from translated track
        "-c:v", "copy",    # no video re-encode (lossless, fast)
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",       # trim to the shorter of video/audio
        output_path,
    ]
    logging.info(f"[FFMPEG] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"[FFMPEG] stderr: {result.stderr}")
        raise RuntimeError(
            f"FFmpeg failed (rc={result.returncode}):\n{result.stderr[-500:]}"
        )
    logging.info(f"[FFMPEG] Final video created: {output_path}")


# --- Main Processing Function ---
def process_video(
    video_path: str,
    target_language_code: str,
    tts_mode: str = "melo",
    qwen3_model_size: str = "1.7B",
    melo_speaker_id: Optional[str] = None,
    enable_voice_cloning: bool = True,
) -> Optional[str]:
    """Main translation pipeline.

    Args:
        video_path: Path to the source video file.
        target_language_code: Short language code like ``"en"``, ``"fr"``.
        tts_mode: TTS backend — ``"melo"`` | ``"qwen3"`` | ``"gtts"``.
        qwen3_model_size: Qwen3-TTS model size ``"0.6B"`` or ``"1.7B"`` (only
                          used when *tts_mode* is ``"qwen3"``).
        melo_speaker_id: Explicit MeloTTS speaker ID (e.g. ``"EN-BR"``);
                         ``None`` auto-selects the default for the target language.
        enable_voice_cloning: When ``True`` and ``tts_mode="qwen3"``, extract a
                              voice-reference clip and pre-compute a voice-clone
                              embedding before synthesis begins.
    """

    performance_monitor.log_gpu_status_direct()
    output_video_path = None  # Initialize

    # Apply config.yaml parameter overrides (caller args take precedence
    # only when they differ from defaults; config provides non-default fallbacks).
    tts_mode = APP_CONFIG.get("tts_mode", tts_mode) or tts_mode
    qwen3_model_size = APP_CONFIG.get("qwen3_model_size", qwen3_model_size) or qwen3_model_size
    if APP_CONFIG.get("enable_voice_cloning") is not None:
        enable_voice_cloning = bool(APP_CONFIG["enable_voice_cloning"])
    if melo_speaker_id is None and APP_CONFIG.get("melo_speaker_id"):
        melo_speaker_id = APP_CONFIG["melo_speaker_id"]


    with performance_monitor.timer("total_video_processing_pipeline"):
        project_dir = create_project_structure(video_path, target_language_code)

        # Models managed explicitly so they can be sequentially unloaded before the next loads.
        translation_model, translation_tokenizer = None, None

        try:
            # 1. Audio Extraction
            with performance_monitor.timer("audio_extraction"):
                extracted_audio_path = os.path.join(project_dir, 'audio', 'extracted_audio.wav')
                extract_audio(video_path, extracted_audio_path)

            # 2. Demucs source separation — must run BEFORE transcription/translation/TTS so
            #    only one large model occupies VRAM at a time. (A1: sequential model lifecycle)
            with performance_monitor.timer("demucs_source_separation"):
                cfg_tier_early = APP_CONFIG.get("hardware_tier", "auto")
                hw_tier_early = (
                    cfg_tier_early
                    if (cfg_tier_early and cfg_tier_early != "auto")
                    else detect_hardware_tier()
                )
                _demucs_model_name = APP_CONFIG.get("demucs_model") or (
                    "htdemucs_ft" if hw_tier_early == "gpu_high" else "htdemucs"
                )
                logging.info(f"[LIFECYCLE] Hardware tier: {hw_tier_early} → Demucs model: {_demucs_model_name}")
                separate_vocals_from_audio(
                    extracted_audio_path, project_dir, device, model_name=_demucs_model_name
                )
                # separate_vocals_from_audio deletes the Demucs model internally and calls
                # gpu_optimizer.clear_cache() — VRAM is freed here before Whisper loads.
                logging.info("[LIFECYCLE] Demucs unloaded. VRAM freed before transcription.")

            # 3. Transcription
            with performance_monitor.timer("transcription"):
                transcript_data = transcribe_with_whisper(extracted_audio_path)
                source_language_detected = transcript_data.get('language', 'en')
                if source_language_detected and '-' in source_language_detected:
                    source_language_detected = source_language_detected.split('-')[0]
                logging.info(f"Detected source language: {source_language_detected}")

            transcript_text_path = os.path.join(project_dir, 'transcripts', 'transcript.txt')
            with open(transcript_text_path, 'w', encoding='utf-8') as f:
                fallback_text = " ".join(
                    (seg.get('text') or '').strip()
                    for seg in (transcript_data.get('segments') or [])
                    if isinstance(seg, dict) and (seg.get('text') or '').strip()
                )
                f.write(transcript_data.get('text') or fallback_text)

            # 4. Translation — NLLB-200 (load → batch translate all segments → unload before TTS)
            with performance_monitor.timer("translation_model_loading"):
                # Respect config.yaml hardware_tier override.
                # hw_tier_early was already resolved in step 2; reuse it.
                hw_tier = hw_tier_early
                logging.info(f"Hardware tier: {hw_tier}")
                translation_model, translation_tokenizer = load_translation_model(hw_tier)

            with performance_monitor.timer("batch_translation"):
                merged_input_segments = merge_short_segments(
                    transcript_data.get("segments", []),
                    min_ms=MIN_SEGMENT_MS,
                )
                logging.info(
                    f"[MERGE] {len(transcript_data.get('segments', []))} -> {len(merged_input_segments)} "
                    f"segments after short-segment merge"
                )

                # First pass: translate all segments at length_penalty=1.0
                translated_segments = batch_translate_segments(
                    merged_input_segments,
                    source_language_detected,
                    target_language_code,
                    translation_model,
                    translation_tokenizer,
                )

                # Save full translated text to file (join segment translations)
                full_translated_text = " ".join(s['translated_text'] for s in translated_segments)
                translated_text_path = os.path.join(project_dir, 'translations', 'translation.txt')
                with open(translated_text_path, 'w', encoding='utf-8') as f:
                    f.write(full_translated_text)

                # Timing budget pre-screen.
                translation_texts = [s['translated_text'] for s in translated_segments]
                timing_budget = analyze_segment_timing_budget(
                    merged_input_segments, translation_texts, target_language_code
                )

                # Second pass: re-translate overflow segments with compression bias.
                # Iterative schedule — tightens penalty each pass until the CPS estimate
                # fits or all passes are exhausted. Normal segments exit after pass 1.
                COMPRESSION_SCHEDULE = [
                    (1.8, 0.85),   # pass 1 — gentle   (ratio threshold, target_ratio)
                    (1.5, 0.70),   # pass 2 — moderate
                    (1.3, 0.60),   # pass 3 — aggressive (last resort)
                ]

                compress_indices = [idx for idx, tb in enumerate(timing_budget) if tb['action'] == 'compress_translation']
                if compress_indices:
                    logging.info(f"Compressing {len(compress_indices)} overflow translation(s) with iterative schedule")
                    _cps = CPS_MAP.get(target_language_code, 14.0)
                    for idx in compress_indices:
                        seg = merged_input_segments[idx]
                        available_ms = timing_budget[idx]['available_ms']
                        current_translation = translated_segments[idx]['translated_text']

                        for pass_num, (ratio_threshold, target_ratio) in enumerate(COMPRESSION_SCHEDULE, 1):
                            # Check if the current translation already fits
                            estimated_ms = int((len(current_translation) / _cps) * 1000) if _cps > 0 else available_ms
                            current_ratio = estimated_ms / available_ms if available_ms > 0 else 1.0

                            if current_ratio <= ratio_threshold:
                                logging.info(
                                    f"  Segment {idx}: ratio={current_ratio:.2f} fits after pass {pass_num - 1}, "
                                    f"no further compression needed"
                                )
                                break

                            compressed = translate_with_length_target(
                                seg['text'],
                                source_language_detected,
                                target_language_code,
                                translation_model,
                                translation_tokenizer,
                                target_length_ratio=target_ratio,
                            )
                            new_estimated_ms = int((len(compressed) / _cps) * 1000) if _cps > 0 else available_ms
                            new_ratio = new_estimated_ms / available_ms if available_ms > 0 else 1.0

                            if new_ratio < current_ratio:
                                current_translation = compressed
                                logging.info(
                                    f"  Segment {idx} pass {pass_num}: ratio {current_ratio:.2f} → {new_ratio:.2f} "
                                    f"(target_ratio={target_ratio}) -> '{compressed[:50]}…'"
                                )
                            else:
                                logging.info(
                                    f"  Segment {idx} pass {pass_num}: compression did not improve "
                                    f"({current_ratio:.2f} -> {new_ratio:.2f}), keeping previous"
                                )
                                break

                        translated_segments[idx]['translated_text'] = current_translation

            # Release translation model BEFORE TTS loads — sequential VRAM lifecycle (A1).
            # At this point: Demucs already unloaded (step 2), Whisper unloaded (step 3),
            # translation model about to be unloaded → VRAM clear for TTSEngine next.
            logging.info("Unloading translation model to free VRAM before TTS.")
            del translation_model, translation_tokenizer
            translation_model, translation_tokenizer = None, None
            gpu_optimizer.clear_cache()
            gc.collect()

            # 5. Audio Synthesis and Synchronization
            # TTSEngine context manager ensures model is unloaded before mixing step.
            original_audio_segment = AudioSegment.from_wav(extracted_audio_path)
            logging.info(f"Original audio duration: {len(original_audio_segment)/1000:.2f}s")

            with performance_monitor.timer("audio_synthesis_and_sync_pipeline"):
                with TTSEngine(
                    mode=tts_mode,
                    device=str(device),
                    model_size=qwen3_model_size,
                    language_model_map=LANGUAGE_MODEL_MAP,
                ) as tts_engine:
                    with performance_monitor.timer("tts_model_loading"):
                        if tts_mode == "qwen3":
                            tts_engine.load_qwen3(for_voice_cloning=enable_voice_cloning)
                            if tts_engine._qwen is None:
                                raise RuntimeError(
                                    "Qwen3-TTS is not installed. "
                                    "Install it with: pip install -U qwen-tts"
                                )
                            if enable_voice_cloning:
                                demucs_vocals_path = os.path.join(
                                    project_dir, "audio", "demucs_vocals.wav"
                                )
                                ref_sample = extract_voice_reference(
                                    demucs_vocals_path,
                                    transcript_data,
                                    project_dir,
                                    fallback_audio_path=extracted_audio_path,
                                )
                                if ref_sample:
                                    ref_audio = str(ref_sample["path"])
                                    ref_text = str(ref_sample.get("text") or "")
                                    tts_engine.extract_voice_embedding(ref_audio, ref_text)
                        else:  # "melo" (default) or unrecognised mode
                            MELO_SUPPORTED = {"en", "es", "fr", "zh", "ja", "ko"}
                            if target_language_code not in MELO_SUPPORTED:
                                logging.warning(
                                    f"[TTS] MeloTTS does not support '{target_language_code}'. "
                                    "Falling back to gTTS for this language."
                                )
                                tts_engine.mode = "gtts"
                            else:
                                melo_lang = LANGUAGE_MODEL_MAP.get(
                                    target_language_code, {}
                                ).get("melo_language", "EN")
                                tts_engine.load_melo(melo_lang)

                    synced_translated_speech_track = create_synced_audio_track(
                        original_audio_segment,
                        translated_segments,
                        target_language_code,
                        project_dir,
                        tts_engine,
                        melo_speaker_id,
                    )
                # tts_engine.__exit__ fires here: models unloaded, VRAM freed

            # 6. Preserve sound effects and mix audio (Demucs stem reuse)
            with performance_monitor.timer("audio_mixing_with_effects"):
                final_mixed_audio_segment = preserve_sound_effects(
                    original_audio_segment,
                    synced_translated_speech_track,
                    transcript_data,
                    project_dir,
                    device_obj=device,
                )

            final_audio_output_path = os.path.join(project_dir, 'audio', 'final_audio.wav')
            logging.info(f"Exporting final mixed audio to: {final_audio_output_path}")
            final_mixed_audio_segment.export(final_audio_output_path, format="wav")

            # 7. Create final video (FFmpeg stream-copy)
            with performance_monitor.timer("final_video_creation"):
                base_video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_video_path = os.path.join(project_dir, f"{base_video_name}_translated_{target_language_code}.mp4")
                create_final_video(video_path, final_audio_output_path, output_video_path)

            logging.info(f"Translation complete. Output video: {output_video_path}")

        except Exception as e:
            logging.error(f"Critical error in process_video: {e}", exc_info=True)
            # output_video_path remains None or its last value if error occurred mid-way
            raise # Re-raise the exception to be caught by the caller
        finally:
            # Cleanup models (translation model may already be None if successfully unloaded above)
            logging.info("Cleaning up models from process_video scope...")
            if translation_model: del translation_model
            if translation_tokenizer: del translation_tokenizer
            # TTSEngine lifecycle is managed by its context manager; no manual cleanup needed here.
            
            # Clean up intermediate audio segments from disk
            segments_dir_cleanup = os.path.join(project_dir, 'translated_segments')
            if os.path.exists(segments_dir_cleanup):
                try:
                    shutil.rmtree(segments_dir_cleanup)
                    logging.info(f"Cleaned up translated_segments directory: {segments_dir_cleanup}")
                except Exception as e_clean:
                    logging.error(f"Error cleaning up segments directory {segments_dir_cleanup}: {e_clean}")

            gpu_optimizer.clear_cache()
            gc.collect()
            logging.info("process_video cleanup complete.")
            logging.info(performance_monitor.get_summary()) # Log summary at the end

    return output_video_path


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]: # Optional project_dir
        print("Usage: python translator.py <video_path> <target_language_code> [custom_output_base_dir]")
        print("Example: python translator.py myvideo.mp4 fr")
        print("Example: python translator.py myvideo.mp4 fr /path/to/custom_outputs")
        sys.exit(1)

    video_file_path = sys.argv[1]
    target_lang = sys.argv[2]
    
    if not os.path.exists(video_file_path):
        logging.error(f"Video file not found: {video_file_path}")
        sys.exit(1)

    if target_lang not in LANGUAGE_MODEL_MAP:
        logging.error(f"Target language '{target_lang}' is not supported or not configured in LANGUAGE_MODEL_MAP.")
        logging.error(f"Available languages: {list(LANGUAGE_MODEL_MAP.keys())}")
        sys.exit(1)

    # Optional: Allow overriding the base output directory
    if len(sys.argv) == 4:
        custom_output_base = sys.argv[3]
        # Modify create_project_structure or pass base_dir to it if this is desired.
        # For now, project structure is created relative to CWD.
        # This example doesn't use custom_output_base directly, but shows how it could be passed.
        logging.info(f"Custom output base directory specified (not yet fully implemented in this example): {custom_output_base}")


    logging.info(f"Starting translation for: {video_file_path} to {target_lang}")
    
    try:
        final_output_video = process_video(video_file_path, target_lang)
        if final_output_video:
            print(f"\n✅ Translation successful!")
            print(f"🎞️  Output video saved to: {final_output_video}")
        else:
            print(f"\n⚠️ Translation process completed, but no output video path was returned.")

    except Exception as e:
        print(f"\n❌ An error occurred during the translation process:")
        # Log the full traceback for debugging
        logging.error("Main script execution error", exc_info=True) 
        # Print a simpler error to console
        print(f"Error details: {e}")
    finally:
        logging.info("Video translation script finished.")