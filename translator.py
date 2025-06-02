import os
import sys
import logging
import tempfile
import math
import shutil
import mmap
from typing import List, Tuple, Dict, Optional
import numpy as np
import concurrent.futures
import gc # For garbage collection

import whisper
from moviepy.editor import VideoFileClip, AudioFileClip
# CompositeAudioClip might not be directly used if pydub handles all composition
from gtts import gTTS
from pydub import AudioSegment, effects as pydub_effects # Renamed to avoid conflict
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
# from pydub.playback import play # Not used in the new script's core logic
from pydub.silence import detect_nonsilent

import torch
import torch.nn as nn # Not directly used in the final script but good for context
from torch.cuda.amp import autocast # GradScaler not used directly in this script part

import librosa
import soundfile as sf
import noisereduce as nr

# Import from new local modules
from gpu_config import gpu_optimizer
from performance_monitor import performance_monitor

# MeloTTS import
try:
    from melo.api import TTS as MeloTTS_API
except ImportError:
    logging.error("MeloTTS library not found. Please ensure it is installed. Falling back to gTTS for all TTS tasks.")
    MeloTTS_API = None


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

# Initialize melo TTS model globally (will be loaded properly in process_video)
global melo_tts_instance
melo_tts_instance = None

# Use global GPU optimizer
accelerator = gpu_optimizer.accelerator
device = gpu_optimizer.device

# Language mapping for MeloTTS and fallback gTTS language codes
LANGUAGE_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "en": {"melo_language": "EN", "speaker_id": "EN-US", "gtts_lang": "en"},
    "es": {"melo_language": "ES", "speaker_id": "ES", "gtts_lang": "es"},
    "fr": {"melo_language": "FR", "speaker_id": "FR", "gtts_lang": "fr"},
    "zh": {"melo_language": "ZH", "speaker_id": "ZH", "gtts_lang": "zh-CN"}, # gTTS uses zh-CN for Mandarin
    "ja": {"melo_language": "JP", "speaker_id": "JP", "gtts_lang": "ja"}, # Changed jp to ja for gTTS
    "ko": {"melo_language": "KR", "speaker_id": "KR", "gtts_lang": "ko"}, # Changed kr to ko for gTTS
    # Add other languages as needed, ensuring gTTS compatibility
    "de": {"melo_language": "DE", "speaker_id": "DE_FEMALE", "gtts_lang": "de"}, # Example, check Melo speaker IDs
    "pt": {"melo_language": "PT", "speaker_id": "PT_FEMALE", "gtts_lang": "pt"}, # Example
}

DUCKING_GAIN_DB = -18  # How much to reduce original audio volume during translated speech
CROSSFADE_MS = 50     # Crossfade duration for audio segments

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

def memory_mapped_audio_loader(audio_path: str) -> np.ndarray:
    logging.info(f"Loading audio with memory mapping attempt: {audio_path}")
    try:
        # Librosa handles resampling and mono conversion directly and is robust.
        # Whisper expects 16kHz mono float32 numpy array.
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        logging.info(f"Audio loaded via librosa: shape={audio_data.shape}, dtype={audio_data.dtype}, sr={sr}")
        return audio_data
    except Exception as e:
        logging.error(f"Librosa loading failed for {audio_path}: {e}", exc_info=True)
        raise

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

def transcribe_with_whisper(audio_path: str) -> dict:
    logging.info("Transcribing audio with Whisper...")
    model = None
    audio_data = None
    try:
        # Load model with GPU optimization
        # Using "medium" as a balance. Device is from gpu_optimizer.
        model = whisper.load_model("medium", device=device)
        model = gpu_optimizer.optimize_model(model)

        audio_data = memory_mapped_audio_loader(audio_path)
        
        logging.info(f"Audio loaded for Whisper: shape={audio_data.shape}, dtype={audio_data.dtype}")
        result = model.transcribe(audio_data, word_timestamps=True, fp16= (gpu_optimizer.mixed_precision == 'fp16' and device.type == 'cuda') )
        
        detected_lang = result.get('language', 'unknown')
        logging.info(f"Transcription complete. Detected language: {detected_lang}")
        return result
    except Exception as e:
        logging.error(f"Whisper transcription failed: {e}", exc_info=True)
        raise
    finally:
        logging.info("Attempting to unload Whisper model and free memory.")
        if model is not None:
            del model
        if audio_data is not None:
            del audio_data
        gc.collect()
        gpu_optimizer.clear_cache()
        logging.info("Whisper model and associated data unloaded; GPU cache cleared.")

def text_to_speech(text_to_synthesize: str, target_language_code: str, output_filepath: str,
                   melo_tts_global_instance, speed: float = 1.0):
    """
    Synthesizes text to speech, prioritizing MeloTTS if available and configured,
    otherwise falls back to gTTS.
    target_language_code: Short code like 'en', 'fr'.
    """
    global melo_tts_instance # Access the global instance

    lang_config = LANGUAGE_MODEL_MAP.get(target_language_code)
    if not lang_config:
        logging.error(f"Language code '{target_language_code}' not found in LANGUAGE_MODEL_MAP. Cannot perform TTS.")
        AudioSegment.silent(duration=10).export(output_filepath, format="wav") # Create dummy silent file
        return

    melo_lang = lang_config.get("melo_language")
    melo_spk_id = lang_config.get("speaker_id")
    gtts_lang_code = lang_config.get("gtts_lang")

    # Try MeloTTS first
    if melo_tts_instance and melo_lang and melo_spk_id and MeloTTS_API:
        try:
            # Ensure the global instance has the correct language loaded if it differs
            # MeloTTS instance might need re-initialization or a method to change language/speaker
            # For simplicity, assuming the global instance is re-usable or handles language switching.
            # If MeloTTS requires specific language loading per call, this needs adjustment.
            # The current MeloTTS API seems to load one language at init.
            # If the global instance's language doesn't match melo_lang, we might need to reload it.
            # This is a complex part if the global instance is strictly single-language.
            # For now, we assume the `tts_to_file` can handle different speaker IDs if the base language model supports them.
            
            # Check if speaker ID is valid for the currently loaded Melo model
            # This is tricky because the global melo_tts_instance is loaded with one language.
            # We'll assume for now that if melo_lang matches the instance's language, speaker_id is fine.
            # A more robust solution would involve a MeloTTS manager class.
            
            # If the melo_tts_instance's language (from its init) is not the target melo_lang,
            # gTTS fallback might be safer unless MeloTTS can dynamically switch.
            # Let's assume the `process_video` loaded MeloTTS with the `target_language`.

            logging.info(f"Attempting MeloTTS: lang='{melo_lang}', spk='{melo_spk_id}', text='{text_to_synthesize[:30]}...'")
            melo_tts_instance.tts_to_file(
                text_to_synthesize,
                melo_spk_id, # Pass the speaker ID string
                output_filepath,
                speed=speed
            )
            logging.info(f"MeloTTS generated audio to {output_filepath}")
            return
        except Exception as e:
            logging.error(f"MeloTTS generation failed for lang='{melo_lang}', spk='{melo_spk_id}': {e}. Falling back to gTTS.", exc_info=True)
            # Fall through to gTTS

    # Fallback to gTTS
    if not gtts_lang_code:
        logging.error(f"gTTS language code not configured for '{target_language_code}'. Cannot perform TTS.")
        AudioSegment.silent(duration=10).export(output_filepath, format="wav")
        return
        
    try:
        logging.info(f"Using gTTS for '{text_to_synthesize[:30]}...' (lang: {gtts_lang_code}) to {output_filepath}")
        tts = gTTS(text=text_to_synthesize, lang=gtts_lang_code, slow=False)
        tts.save(output_filepath)
    except Exception as e:
        logging.error(f"gTTS generation failed for '{text_to_synthesize[:30]}...': {e}. Skipping segment audio.", exc_info=True)
        AudioSegment.silent(duration=10, frame_rate=22050).export(output_filepath, format="wav")


def _translate_text_internal(text: str, source_language: str, target_language: str, translation_model, translation_tokenizer) -> str:
    if not text.strip():
        logging.info("Input text for translation is empty. Returning empty string.")
        return ""
    if source_language == target_language:
        return text

    logging.info(f"Translating chunk from {source_language} to {target_language}: '{text[:50]}...'")
    
    # mBART requires specific language codes (e.g., 'en_XX', 'fr_XX')
    # Helsinki models use short codes in their name (e.g., 'en', 'fr')
    # This logic assumes `source_language` and `target_language` are short codes (e.g., 'en')
    
    src_lang_for_tokenizer = source_language
    tgt_lang_for_tokenizer = target_language
    forced_bos_token_id = None

    # Handle mBART specific lang codes if mBART model is detected
    if "mbart" in translation_tokenizer.name_or_path.lower():
        # Convert 'en' to 'en_XX', etc.
        # This is a simplified mapping; mBART has a specific list.
        if len(source_language) == 2: src_lang_for_tokenizer = f"{source_language}_XX"
        if len(target_language) == 2: tgt_lang_for_tokenizer = f"{target_language}_XX"
        
        try:
            translation_tokenizer.src_lang = src_lang_for_tokenizer
            if hasattr(translation_tokenizer, 'lang_code_to_id'):
                 forced_bos_token_id = translation_tokenizer.lang_code_to_id.get(tgt_lang_for_tokenizer)
            elif hasattr(translation_tokenizer, 'get_lang_id'): # Fallback for older transformers
                 forced_bos_token_id = translation_tokenizer.get_lang_id(tgt_lang_for_tokenizer)

            if forced_bos_token_id is None:
                logging.warning(f"Could not determine forced_bos_token_id for mBART with target {tgt_lang_for_tokenizer}. Translation might be suboptimal.")
        except Exception as e:
            logging.warning(f"Error setting mBART language codes: {e}")


    raw_inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in raw_inputs.items()}

    with autocast(enabled=(accelerator.mixed_precision in ['fp16', 'bf16'])):
        gen_kwargs = {"max_new_tokens": 512}
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
        
        translated_tokens = translation_model.generate(**inputs, **gen_kwargs)
    
    translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    del inputs, translated_tokens, raw_inputs
    gc.collect()
    return translated_text


def _summarize_text_internal(text: str, summarizer_pipeline, max_length_ratio: float = 0.7, min_length_abs: int = 20) -> str:
    if not text.strip(): return ""
    
    text_len_chars = len(text)
    # If text is already very short, summarization might not be useful or could make it too short.
    if text_len_chars < 150: # Arbitrary threshold, adjust as needed
        logging.info(f"Text too short for summarization (length: {text_len_chars}). Returning original.")
        return text

    # Calculate max_length for summarizer (often in tokens, approximated by chars here)
    # Ensure max_length is less than original text length.
    calculated_max_length = int(text_len_chars * max_length_ratio)
    final_max_length = min(calculated_max_length, text_len_chars - 5) # Ensure it's shorter
    final_max_length = max(final_max_length, min_length_abs + 10) # Ensure it's not too small vs min_length

    # Ensure min_length is reasonable and less than max_length
    final_min_length = min(min_length_abs, final_max_length - 10)
    final_min_length = max(10, final_min_length) # Absolute minimum

    if final_min_length >= final_max_length:
        logging.warning(f"Cannot summarize: effective min_length ({final_min_length}) >= max_length ({final_max_length}). Returning original text.")
        return text

    logging.info(f"Summarizing text (len: {text_len_chars}) with min_length={final_min_length}, max_length={final_max_length}")
    
    with autocast(enabled=(accelerator.mixed_precision in ['fp16', 'bf16'])):
        try:
            summary_list = summarizer_pipeline(text, min_length=final_min_length, max_length=final_max_length, do_sample=False)
            summary = summary_list[0]['summary_text']
        except Exception as e:
            logging.error(f"Summarization failed: {e}. Returning original text.", exc_info=True)
            return text
    return summary

# --- Audio Processing Functions ---
def noise_reduction(audio: AudioSegment, reduction_amount: float = 0.6) -> AudioSegment:
    """Applies noise reduction. reduction_amount (0.0 to 1.0, lower is less reduction)."""
    if not (0.0 <= reduction_amount <= 1.0):
        reduction_amount = np.clip(reduction_amount, 0.0, 1.0)
        logging.warning(f"Noise reduction_amount clamped to {reduction_amount}")

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Normalize samples to [-1, 1] based on sample_width
    if audio.sample_width == 2: # 16-bit
        samples /= (2**15)
    elif audio.sample_width == 1: # 8-bit unsigned
        samples = (samples - 128) / 128.0
    # Add other sample widths if necessary, or raise error for unsupported.

    if audio.channels == 2:
        samples_mono_for_profile = samples.reshape((-1, 2)).mean(axis=1)
        # Create a noise profile from the mono version. `y_noise=None` lets `nr` estimate it.
        # `stationary=False` might be better for general background noise.
        # Using a portion of the audio if it's long, or assuming some silence, can improve profile.
        # For simplicity, using the whole mono track to estimate noise characteristics.
        # `prop_decrease=0` for `y_noise` means it's just estimating the noise profile.
        noise_profile_segment = nr.reduce_noise(y=samples_mono_for_profile, sr=audio.frame_rate, prop_decrease=0.0, stationary=False)
        
        reduced_channels = []
        for i in range(audio.channels):
            channel_samples = samples.reshape((-1, audio.channels))[:, i]
            # Apply reduction using the estimated noise profile (or characteristics from it)
            # `y_noise` here should ideally be the actual noise clip if known, or `nr` uses its internal estimate.
            # If `noise_profile_segment` is the noise itself, use it. If it's audio *with noise removed*, that's different.
            # The API of `noisereduce` can be a bit nuanced here.
            # Assuming `noise_profile_segment` is a representation of the noise to be reduced.
            # A common pattern is to find a silent part and use that as `y_noise`.
            # If `y_noise` is not provided, `nr` tries to estimate it from `y`.
            reduced_channel = nr.reduce_noise(y=channel_samples, sr=audio.frame_rate, prop_decrease=reduction_amount, stationary=False) # y_noise=noise_profile_segment if it's actual noise
            reduced_channels.append(reduced_channel)
        reduced_noise_samples = np.stack(reduced_channels, axis=-1).flatten()
    else: # Mono
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=reduction_amount, stationary=False)

    # Convert back to original integer type
    if audio.sample_width == 2:
        reduced_noise_samples = (reduced_noise_samples * (2**15)).astype(np.int16)
    elif audio.sample_width == 1:
        reduced_noise_samples = ((reduced_noise_samples * 128) + 128).astype(np.uint8)

    return AudioSegment(
        reduced_noise_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

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
    if len(audio) == 0 or target_duration_ms <= 0:
        logging.warning(f"Cannot time stretch: audio len {len(audio)}ms, target {target_duration_ms}ms.")
        return audio if len(audio) > 0 else AudioSegment.silent(duration=10)

    logging.info(f"Time stretching. Input: {len(audio)/1000:.2f}s, Target: {target_duration_ms/1000:.2f}s")

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.sample_width == 2: samples /= (2**15)
    elif audio.sample_width == 1: samples = (samples - 128) / 128.0

    if audio.channels == 2:
        samples_mono = samples.reshape((-1, 2)).mean(axis=1)
    else:
        samples_mono = samples

    # librosa.effects.time_stretch rate: > 1.0 speeds up, < 1.0 slows down
    # rate = current_duration / target_duration
    # If current is 10s, target 5s, rate = 2 (speed up)
    # If current is 5s, target 10s, rate = 0.5 (slow down)
    stretch_rate = len(audio) / target_duration_ms 
    stretch_rate = np.clip(stretch_rate, 0.5, 2.0) # Clamp to avoid extreme distortion

    logging.info(f"Librosa time_stretch rate: {stretch_rate:.2f}")
    
    stretched_mono = librosa.effects.time_stretch(samples_mono, rate=stretch_rate)

    if audio.channels == 2:
        # Duplicate mono to stereo
        stretched_samples = np.vstack((stretched_mono, stretched_mono)).T.flatten()
    else:
        stretched_samples = stretched_mono
    
    # Ensure exact duration by padding/truncating (librosa stretch is approximate)
    target_num_frames = int(target_duration_ms / 1000 * audio.frame_rate)
    target_num_samples_total = target_num_frames * audio.channels
    
    current_num_samples = len(stretched_samples)

    if current_num_samples > target_num_samples_total:
        stretched_samples = stretched_samples[:target_num_samples_total]
    elif current_num_samples < target_num_samples_total:
        padding = np.zeros(target_num_samples_total - current_num_samples, dtype=np.float32)
        stretched_samples = np.concatenate([stretched_samples, padding])

    if audio.sample_width == 2:
        stretched_samples = (stretched_samples * (2**15)).astype(np.int16)
    elif audio.sample_width == 1:
        stretched_samples = ((stretched_samples * 128) + 128).astype(np.uint8)
    
    return AudioSegment(
        stretched_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

# --- Segment Processing and Synchronization ---
def process_segment(segment_info: Tuple[int, dict], 
                    source_language: str, target_language_code: str, segments_dir: str,
                    translation_model, translation_tokenizer, summarizer_pipeline, 
                    melo_tts_global_instance):
    i, segment = segment_info
    start_time_ms = int(segment['start'] * 1000)
    end_time_ms = int(segment['end'] * 1000)
    original_duration_ms = end_time_ms - start_time_ms

    if original_duration_ms <= 0:
        logging.warning(f"Segment {i} has zero/negative duration ({original_duration_ms}ms). Skipping.")
        return None

    logging.info(f"Processing segment {i}: '{segment['text'][:30]}...' (Orig dur: {original_duration_ms}ms)")

    translated_text = _translate_text_internal(segment['text'], source_language, target_language_code, 
                                             translation_model, translation_tokenizer)
    if not translated_text.strip():
        logging.warning(f"Segment {i}: Empty translation for '{segment['text'][:30]}...'. Skipping TTS.")
        return None

    tts_output_path = os.path.join(segments_dir, f"segment_{i:04d}.wav")
    
    try:
        text_to_speech(translated_text, target_language_code, tts_output_path, melo_tts_global_instance)
        
        generated_audio = AudioSegment.from_file(tts_output_path)
        current_tts_duration_ms = len(generated_audio)

        # Summarization if TTS is much longer (e.g., > 1.6x original and > 2s absolute diff)
        if current_tts_duration_ms > original_duration_ms * 1.6 and \
           current_tts_duration_ms - original_duration_ms > 2000:
            logging.info(f"Segment {i}: TTS too long ({current_tts_duration_ms}ms vs {original_duration_ms}ms). Summarizing.")
            summarized_text = _summarize_text_internal(translated_text, summarizer_pipeline)
            if summarized_text.strip() and len(summarized_text) < len(translated_text):
                text_to_speech(summarized_text, target_language_code, tts_output_path, melo_tts_global_instance)
                generated_audio = AudioSegment.from_file(tts_output_path)
                logging.info(f"Segment {i}: Resynthesized with summary. New TTS dur: {len(generated_audio)}ms")
            else:
                logging.info(f"Segment {i}: Summarization didn't shorten. Using original TTS.")
        
        # Audio processing pipeline
        stretched_audio = advanced_time_stretch(generated_audio, original_duration_ms)
        del generated_audio; gc.collect()
        
        # Noise reduction can be aggressive, use sparingly or make configurable
        # noise_reduced_audio = noise_reduction(stretched_audio, reduction_amount=0.1)
        # del stretched_audio; gc.collect()
        # processed_audio = noise_reduced_audio
        processed_audio = stretched_audio # Skipping NR for now

        enhanced_audio = enhance_voice(processed_audio)
        if processed_audio is not enhanced_audio: # Check if enhance_voice returned a new object
             del processed_audio; gc.collect()
        
        enhanced_audio.export(tts_output_path, format="wav")
        del enhanced_audio; gc.collect()
        
        return (start_time_ms, tts_output_path)

    except Exception as e:
        logging.error(f"Error processing segment {i} ('{segment['text'][:30]}...'): {e}", exc_info=True)
        gpu_optimizer.clear_cache()
        gc.collect()
        return None


def adaptive_segment_processing(segments_data: List[dict], 
                                source_language: str, target_language_code: str, project_dir: str,
                                translation_model, translation_tokenizer, summarizer_pipeline, 
                                melo_tts_global_instance):
    segments_dir = os.path.join(project_dir, 'translated_segments')
    # No need to pass original_audio AudioSegment object here, durations come from transcript

    processed_segments_info = []
    
    # Sequential processing for stability, especially with GPU memory.
    # Could be parallelized with ThreadPoolExecutor if process_segment is thread-safe
    # and GPU resources are managed carefully (e.g., one model instance per thread or locking).
    tasks_args = [
        ((i, segment), source_language, target_language_code, segments_dir,
         translation_model, translation_tokenizer, summarizer_pipeline, melo_tts_global_instance)
        for i, segment in enumerate(segments_data)
    ]

    for args in tqdm(tasks_args, desc="Processing audio segments"):
        result = process_segment(*args) # Unpack arguments
        if result:
            processed_segments_info.append(result)
        # Optional: More aggressive cleanup if memory is an issue
        # gpu_optimizer.clear_cache(); gc.collect()
            
    return processed_segments_info


def preserve_sound_effects(original_audio: AudioSegment, 
                           synced_speech_track: AudioSegment, 
                           transcript: dict, 
                           project_dir: str, 
                           silence_original_speech_segments: bool = True,
                           duck_gain_db: Optional[float] = DUCKING_GAIN_DB) -> AudioSegment:
    
    debug_audio_dir = os.path.join(project_dir, 'audio_debug') # Already created by create_project_structure
    logging.info(f"Mixing audio. Silence original: {silence_original_speech_segments}. Duck gain: {duck_gain_db}dB")

    # Determine target channels (prefer stereo if either track is stereo)
    target_channels = 2 if original_audio.channels == 2 or synced_speech_track.channels == 2 else 1
    
    # Standardize channels
    working_original = original_audio.set_channels(target_channels)
    working_synced_speech = synced_speech_track.set_channels(target_channels)
    
    working_original.export(os.path.join(debug_audio_dir, "0_original_standardized.wav"), format="wav")
    working_synced_speech.export(os.path.join(debug_audio_dir, "0_synced_speech_standardized.wav"), format="wav")

    # Create the base for the original audio (background) track
    background_track = AudioSegment.empty()
    
    speech_intervals_ms = sorted(
        [(int(s['start']*1000), int(s['end']*1000)) for s in transcript['segments'] if int(s['end']*1000) > int(s['start']*1000)]
    )

    if silence_original_speech_segments:
        logging.info("Reconstructing original audio with speech segments silenced.")
        last_segment_end_ms = 0
        for start_ms, end_ms in speech_intervals_ms:
            if start_ms > last_segment_end_ms: # Non-speech part
                background_track += working_original[last_segment_end_ms:start_ms]
            
            duration_to_silence_ms = end_ms - start_ms
            if duration_to_silence_ms > 0:
                silence = AudioSegment.silent(duration=duration_to_silence_ms, frame_rate=working_original.frame_rate)
                background_track += silence.set_channels(working_original.channels)
            last_segment_end_ms = end_ms
        
        if last_segment_end_ms < len(working_original): # Remaining part
            background_track += working_original[last_segment_end_ms:]
        
        # Ensure correct length
        if len(background_track) < len(working_original):
            padding = AudioSegment.silent(duration=len(working_original) - len(background_track), frame_rate=working_original.frame_rate).set_channels(working_original.channels)
            background_track += padding
        elif len(background_track) > len(working_original):
            background_track = background_track[:len(working_original)]
        background_track.export(os.path.join(debug_audio_dir, "1_background_AFTER_silencing.wav"), format="wav")

    elif duck_gain_db is not None:
        logging.info(f"Applying ducking with gain {duck_gain_db}dB to original audio during speech.")
        background_track = working_original.dup() # Start with a copy
        for start_ms, end_ms in speech_intervals_ms:
            # Apply ducking with crossfades for smoother transitions
            # The segment to duck is from original audio
            segment_to_duck = background_track[start_ms:end_ms]
            ducked_segment = segment_to_duck.apply_gain(duck_gain_db)
            
            # Simple overlay for ducking (no crossfade here, pydub's gain is instant)
            # For crossfaded ducking, one would need to manage segments and fades more manually.
            background_track = background_track.overlay(ducked_segment, position=start_ms)
        background_track.export(os.path.join(debug_audio_dir, "1_background_AFTER_ducking.wav"), format="wav")
    else:
        background_track = working_original # No silencing or ducking
        background_track.export(os.path.join(debug_audio_dir, "1_background_NO_OP.wav"), format="wav")

    logging.info(f"Overlaying translated speech (len: {len(working_synced_speech)/1000:.2f}s) "
                 f"onto background (len: {len(background_track)/1000:.2f}s)")
    
    # Final mix: overlay translated speech onto the processed background track
    final_mix = background_track.overlay(working_synced_speech, position=0, loop=False, times=1) # position=0 assumes synced_speech_track is already timed correctly
    
    # Ensure final mix is not longer than original (can happen with slight timing issues)
    if len(final_mix) > len(working_original):
        final_mix = final_mix[:len(working_original)]

    final_mix.export(os.path.join(debug_audio_dir, "2_final_mixed_audio.wav"), format="wav")
    logging.info(f"Audio mixing complete. Final duration: {len(final_mix)/1000:.2f}s")
    return final_mix


def create_synced_audio_track(original_audio_ref: AudioSegment, transcript: dict, 
                              source_language: str, target_language_code: str, project_dir: str,
                              translation_model, translation_tokenizer, summarizer_pipeline, 
                              melo_tts_global_instance) -> AudioSegment:
    """Creates a single audio track of the translated speech, synced to original timings."""
    logging.info("Creating full synced translated speech track with segment fades...")
    
    # Determine target channels based on original audio (or default to stereo if original is mono for richer TTS)
    # Let's assume TTS output might be stereo, so aim for stereo if original is mono.
    # However, process_segment handles its own TTS output. This track is for combining them.
    # The channel handling should be consistent. If original_audio_ref is stereo, this should be stereo.
    target_channels = original_audio_ref.channels
    
    full_synced_speech_track = AudioSegment.silent(
        duration=len(original_audio_ref),
        frame_rate=original_audio_ref.frame_rate
    ).set_channels(target_channels)

    processed_segments_info = adaptive_segment_processing(
        transcript['segments'], 
        source_language, target_language_code, project_dir,
        translation_model, translation_tokenizer, summarizer_pipeline, melo_tts_global_instance
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
            
    return full_synced_speech_track


def create_final_video(video_path: str, final_audio_path: str, output_path: str):
    logging.info(f"Creating final video: {output_path}")
    video_clip = None
    audio_clip_obj = None
    final_video_clip = None
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip_obj = AudioFileClip(final_audio_path)
        final_video_clip = video_clip.set_audio(audio_clip_obj)
        # Use more threads for potentially faster writing, and a progress bar.
        final_video_clip.write_videofile(output_path, audio_codec='aac', threads=os.cpu_count() or 4, logger='bar')
    except Exception as e:
        logging.error(f"Failed to create final video: {e}", exc_info=True)
        raise
    finally:
        if video_clip: video_clip.close()
        if audio_clip_obj: audio_clip_obj.close()
        if final_video_clip: final_video_clip.close() # MoviePy clips often need explicit close
        del video_clip, audio_clip_obj, final_video_clip; gc.collect()


# --- Main Processing Function ---
def process_video(video_path: str, target_language_code: str) -> Optional[str]:
    """
    Main function to process the video translation.
    target_language_code: Short code like 'en', 'fr'.
    """
    global melo_tts_instance # Use the global instance

    performance_monitor.log_gpu_status_direct()
    output_video_path = None # Initialize

    with performance_monitor.timer("total_video_processing_pipeline"):
        project_dir = create_project_structure(video_path, target_language_code)
        
        # Initialize models to None; they will be loaded as needed
        translation_model, translation_tokenizer, summarizer_pipeline_obj = None, None, None

        try:
            # 1. Audio Extraction
            with performance_monitor.timer("audio_extraction"):
                extracted_audio_path = os.path.join(project_dir, 'audio', 'extracted_audio.wav')
                extract_audio(video_path, extracted_audio_path)

            # 2. Transcription
            with performance_monitor.timer("transcription"):
                transcript_data = transcribe_with_whisper(extracted_audio_path)
                source_language_detected = transcript_data.get('language', 'en') # Default to 'en' if not detected
                # Normalize detected language code (e.g., en-US -> en)
                if source_language_detected and '-' in source_language_detected:
                    source_language_detected = source_language_detected.split('-')[0]
                logging.info(f"Detected source language: {source_language_detected}")
            
            transcript_text_path = os.path.join(project_dir, 'transcripts', 'transcript.txt')
            with open(transcript_text_path, 'w', encoding='utf-8') as f:
                f.write(transcript_data['text'])

            # --- Load Models (once per type, within this process_video call) ---
            # Translation Model
            with performance_monitor.timer("translation_model_loading"):
                # Try Helsinki-NLP first, then mBART as fallback
                specific_helsinki_model_name = f"Helsinki-NLP/opus-mt-{source_language_detected}-{target_language_code}"
                try:
                    logging.info(f"Attempting to load translation model: {specific_helsinki_model_name}")
                    translation_tokenizer = AutoTokenizer.from_pretrained(specific_helsinki_model_name)
                    translation_model = AutoModelForSeq2SeqLM.from_pretrained(specific_helsinki_model_name)
                    translation_model = gpu_optimizer.optimize_model(translation_model) # Optimize after loading
                    logging.info(f"Loaded {specific_helsinki_model_name}")
                except Exception as e_helsinki:
                    logging.warning(f"Failed to load {specific_helsinki_model_name} ({e_helsinki}). Falling back to mBART.")
                    if translation_model: del translation_model; translation_model = None
                    if translation_tokenizer: del translation_tokenizer; translation_tokenizer = None
                    gpu_optimizer.clear_cache(); gc.collect()

                    mbart_model_name = "facebook/mbart-large-50-many-to-many-mmt"
                    logging.info(f"Loading mBART model: {mbart_model_name}")
                    # mBART requires specific lang codes for tokenizer, e.g., en_XX
                    src_lang_mbart = f"{source_language_detected}_XX" if len(source_language_detected) == 2 else source_language_detected
                    translation_tokenizer = AutoTokenizer.from_pretrained(mbart_model_name, src_lang=src_lang_mbart)
                    translation_model = AutoModelForSeq2SeqLM.from_pretrained(mbart_model_name)
                    translation_model = gpu_optimizer.optimize_model(translation_model)
                    logging.info(f"Loaded {mbart_model_name}")
            
            # Summarization Model
            with performance_monitor.timer("summarization_model_loading"):
                summarizer_model_name = "facebook/bart-large-cnn"
                logging.info(f"Loading summarization pipeline: {summarizer_model_name}")
                summarizer_device_idx = device.index if device.type == 'cuda' else -1
                summarizer_pipeline_obj = pipeline("summarization", model=summarizer_model_name, tokenizer=summarizer_model_name, device=summarizer_device_idx, framework="pt")
                # The model within the pipeline is what would be optimized if needed, but pipeline handles device placement.
                # If direct optimization: summarizer_pipeline_obj.model = gpu_optimizer.optimize_model(summarizer_pipeline_obj.model)
                logging.info("Summarization pipeline loaded.")

            # MeloTTS Model (Global Instance)
            with performance_monitor.timer("melo_tts_model_loading"):
                if melo_tts_instance is None and MeloTTS_API is not None:
                    melo_target_lang_config = LANGUAGE_MODEL_MAP.get(target_language_code)
                    if melo_target_lang_config and melo_target_lang_config.get("melo_language"):
                        melo_init_lang = melo_target_lang_config["melo_language"]
                        logging.info(f"Loading MeloTTS model globally for language: {melo_init_lang} on device: {str(device)}")
                        try:
                            melo_tts_instance = MeloTTS_API(language=melo_init_lang, device=str(device))
                            logging.info("MeloTTS model loaded globally.")
                        except Exception as e_melo_load:
                            logging.error(f"Failed to load MeloTTS model globally: {e_melo_load}. TTS will rely on gTTS.", exc_info=True)
                            melo_tts_instance = None # Ensure it's None
                    else:
                        logging.warning(f"MeloTTS configuration not found for target language {target_language_code}. MeloTTS will not be used.")
                        melo_tts_instance = None
                elif MeloTTS_API is None:
                     logging.warning("MeloTTS API could not be imported. MeloTTS will not be used.")
                     melo_tts_instance = None


            # 3. Full Text Translation (for saving to file)
            with performance_monitor.timer("full_text_translation"):
                full_translated_text = _translate_text_internal(transcript_data['text'], source_language_detected, target_language_code,
                                                              translation_model, translation_tokenizer)
            translated_text_path = os.path.join(project_dir, 'translations', 'translation.txt')
            with open(translated_text_path, 'w', encoding='utf-8') as f:
                f.write(full_translated_text)

            # 4. Audio Synthesis and Synchronization
            original_audio_segment = None
            synced_translated_speech_track = None
            final_mixed_audio_segment = None

            with performance_monitor.timer("audio_synthesis_and_sync_pipeline"):
                original_audio_segment = AudioSegment.from_wav(extracted_audio_path)
                logging.info(f"Original audio duration: {len(original_audio_segment)/1000:.2f}s")
                
                synced_translated_speech_track = create_synced_audio_track(
                    original_audio_segment, transcript_data, 
                    source_language_detected, target_language_code, project_dir,
                    translation_model, translation_tokenizer, summarizer_pipeline_obj, melo_tts_instance
                )
            
            # 5. Preserve Sound Effects & Mix Audio
            with performance_monitor.timer("audio_mixing_with_effects"):
                final_mixed_audio_segment = preserve_sound_effects(
                    original_audio_segment, 
                    synced_translated_speech_track, 
                    transcript_data, 
                    project_dir,
                    silence_original_speech_segments=True, # Or False to try ducking
                    duck_gain_db=DUCKING_GAIN_DB 
                )
            
            final_audio_output_path = os.path.join(project_dir, 'audio', 'final_audio.wav')
            logging.info(f"Exporting final mixed audio to: {final_audio_output_path}")
            final_mixed_audio_segment.export(final_audio_output_path, format="wav")

            # 6. Create Final Video
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
            # Cleanup models
            logging.info("Cleaning up models from process_video scope...")
            if translation_model: del translation_model
            if translation_tokenizer: del translation_tokenizer
            if summarizer_pipeline_obj:
                 if hasattr(summarizer_pipeline_obj, 'model'): del summarizer_pipeline_obj.model
                 del summarizer_pipeline_obj
            # Global melo_tts_instance is not deleted here; managed globally for notebook or app lifecycle.
            # If this were a one-shot script, melo_tts_instance would be cleared too.
            
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
        # Clean up global MeloTTS instance if it was loaded, for script-like behavior
        if melo_tts_instance is not None:
            logging.info("Cleaning up global MeloTTS instance.")
            del melo_tts_instance
            melo_tts_instance = None # Ensure it's reset
            gpu_optimizer.clear_cache() # Clear cache after model deletion
            gc.collect()

        logging.info("Video translation script finished.")