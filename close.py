import os
import sys
import logging
import tempfile
import math
import shutil
from typing import List, Tuple, Dict
import numpy as np
import concurrent.futures

import whisper
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from gtts import gTTS
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from pydub.playback import play
from pydub.silence import detect_nonsilent
from pydub import effects
import torch
import librosa
import soundfile as sf
import noisereduce as nr

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a dictionary mapping languages to their respective models
LANGUAGE_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "en": {"model_name": "tts_models/en/ljspeech/fast_pitch"},
    "es": {"model_name": "tts_models/es/mai/tacotron2-DDC"},
    "fr": {"model_name": "tts_models/fr/mai/tacotron2-DDC"},
    "de": {"model_name": "tts_models/de/thorsten/vits"},
    "it": {"model_name": "tts_models/it/mai_female/glow-tts"},
    "pt": {"model_name": "tts_models/pt/cv/vits"},
    "pl": {"model_name": "tts_models/pl/mai_female/vits"},
    "tr": {"model_name": "tts_models/tr/common-voice/glow-tts"},
    "ru": {"model_name": "tts_models/ru/multi-dataset/vits"},
    "nl": {"model_name": "tts_models/nl/mai/tacotron2-DDC"},
}

def create_project_structure(video_path: str, target_language: str) -> str:
    print("Creating project structure...")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    project_dir = os.path.join(os.getcwd(), f"{base_name}_{target_language}_translation")
    os.makedirs(project_dir, exist_ok=True)
    
    for subdir in ['audio', 'transcripts', 'translations', 'translated_segments']:
        os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
    
    print(f"Project directory: {project_dir}")
    return project_dir

def extract_audio(video_path: str, output_path: str) -> str:
    logging.info("Extracting audio from video...")
    print("Extracting audio from video...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, logger=None)
    print(f"Audio extracted to: {output_path}")
    return output_path

def transcribe_with_whisper(audio_path: str) -> dict:
    logging.info("Transcribing audio with Whisper...")
    print("Transcribing audio with Whisper...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small", device=device)
    result = model.transcribe(audio_path, word_timestamps=True)
    print(f"Transcription complete. Detected language: {result['language']}")
    return result

def translate_text(text: str, source_language: str, target_language: str) -> str:
    logging.info(f"Translating text from {source_language} to {target_language}...")
    if source_language == target_language:
        return text
    
    model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        logging.error(f"Error loading translation model: {str(e)}")
        logging.info("Attempting to use a multi-language model...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    # Split text into smaller chunks to avoid tokenizer length issues
    max_chunk_length = 512
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    translated_chunks = []
    for chunk in tqdm(chunks, desc="Translating chunks"):
        try:
            if 'facebook/mbart' in str(translator.model.name_or_path):
                translation = translator(chunk, src_lang=source_language, tgt_lang=target_language)[0]['translation_text']
            else:
                translation = translator(chunk)[0]['translation_text']
            translated_chunks.append(translation)
        except Exception as e:
            logging.error(f"Error translating chunk: {str(e)}")
            translated_chunks.append(chunk)  # Append original chunk if translation fails
    
    return " ".join(translated_chunks)

def text_to_speech(text: str, language: str, output_path: str, rate: float = 1.0) -> str:
    logging.info(f"Generating speech with gTTS (rate: {rate})...")
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(output_path)

    audio = AudioSegment.from_mp3(output_path)
    
    return output_path

def text_to_speech_coqui(text: str, language: str, output_path: str, rate: float = 1.0) -> str:
    if language not in LANGUAGE_MODEL_MAP:
        raise ValueError(f"Unsupported language: {language}")

    model_name = LANGUAGE_MODEL_MAP[language]["model_name"]
    
    # Initialize the ModelManager
    manager = ModelManager()
    
    # Download and load the model
    model_path, config_path, _ = manager.download_model(model_name)
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
    )
    
    # Generate speech
    wav = synthesizer.tts(text)
    
    # Save the generated audio
    synthesizer.save_wav(wav, output_path)
    
    return output_path

def summarize_text(text: str, max_length: int = 512) -> str:
    print("Summarizing text...")
    if len(text) <= max_length:
        return text
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    
    # Adjust max_length based on input length
    max_length = min(max_length, len(text) // 2)
    min_length = min(30, max_length - 1)
    
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    
    return summary

def process_segment(segment, original_audio, source_language, target_language, segments_dir, i):
    start_time = int(segment['start'] * 1000)
    end_time = int(segment['end'] * 1000)
    original_duration = end_time - start_time
    
    translated_segment = translate_text(segment['text'], source_language, target_language)
    
    if not translated_segment.strip():
        logging.warning(f"Skipping empty translated segment for: '{segment['text']}'")
        return None
    
    tts_output_path = os.path.join(segments_dir, f"segment_{i:04d}.wav")
    
    try:
        # Generate initial TTS audio
        try:
            text_to_speech_coqui(translated_segment, target_language, tts_output_path)
        except Exception as e:
            logging.error(f"Coqui TTS failed on segment {i}: {str(e)}")
            logging.info(f"Switching to gtts")
            text_to_speech(translated_segment, target_language, tts_output_path)
        translated_audio = AudioSegment.from_file(tts_output_path)
        
        # Check if summarization is needed
        if len(translated_audio) - original_duration >= 3000:
            logging.info(f"Summarizing segment {i} due to length difference")
            summarized_text = summarize_text(translated_segment, max_length=len(translated_segment) // 2)
            try:
                text_to_speech_coqui(summarized_text, target_language, tts_output_path)
            except Exception as e:
                logging.error(f"TTS failed on segment {i}: {str(e)}")
                logging.info(f"Switching to gtts")
                text_to_speech(summarized_text, target_language, tts_output_path)
            translated_audio = AudioSegment.from_file(tts_output_path)
        
        # Apply advanced time stretching
        stretched_audio = advanced_time_stretch(translated_audio, original_duration)
        
        # Apply audio normalization
        normalized_audio = effects.normalize(stretched_audio)
        
        # Apply subtle noise reduction
        noise_reduced_audio = noise_reduction(normalized_audio)
        
        # Apply voice enhancement
        enhanced_audio = enhance_voice(noise_reduced_audio)
        
        enhanced_audio.export(tts_output_path, format="wav")
        
        return (start_time, enhanced_audio)
        
    except Exception as e:
        logging.error(f"Error processing segment {i}: {str(e)}")
        return None

def adaptive_segment_processing(segments, original_audio, source_language, target_language, project_dir):
    segments_dir = os.path.join(project_dir, 'translated_segments')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_segment, segment, original_audio, source_language, target_language, segments_dir, i) 
                   for i, segment in enumerate(segments)]
        
        processed_segments = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                processed_segments.append(result)
    
    return processed_segments

def noise_reduction(audio: AudioSegment, reduction_amount=3):
    """
    Apply a simple noise reduction to the audio segment.
    """
    samples = np.array(audio.get_array_of_samples())
    reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate, prop_decrease=reduction_amount/10)
    return AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

def enhance_voice(audio: AudioSegment):
    """
    Enhance the voice in the audio segment by applying EQ and compression.
    """
    # Apply a mild EQ to enhance voice frequencies
    enhanced = audio.high_pass_filter(80)  # Remove low rumble
    enhanced = enhanced.low_pass_filter(8000)  # Remove high hiss
    
    # Apply compression to even out the volume
    enhanced = effects.compress_dynamic_range(enhanced, threshold=-20, ratio=4.0, attack=5, release=50)
    
    return enhanced

def advanced_time_stretch(audio: AudioSegment, target_duration: int) -> AudioSegment:
    logging.info(f"Performing advanced time stretch. Input duration: {len(audio) / 1000:.2f}s, Target duration: {target_duration / 1000:.2f}s")
    
    # Convert pydub AudioSegment to numpy array
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32767.0
    
    # If stereo, convert to mono
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    
    # Calculate the stretch factor
    stretch_factor = target_duration / len(audio)
    
    # Limit the stretch factor to avoid extreme stretching
    max_stretch = 2.0
    min_stretch = 0.5
    stretch_factor = max(min(stretch_factor, max_stretch), min_stretch)
    
    logging.info(f"Adjusted stretch factor: {stretch_factor:.2f}")
    
    # Use librosa for time stretching
    stretched_samples = librosa.effects.time_stretch(samples, rate=1/stretch_factor)
    
    # Ensure the stretched audio matches the target duration
    if len(stretched_samples) > target_duration * audio.frame_rate // 1000:
        stretched_samples = stretched_samples[:target_duration * audio.frame_rate // 1000]
    elif len(stretched_samples) < target_duration * audio.frame_rate // 1000:
        padding = np.zeros(target_duration * audio.frame_rate // 1000 - len(stretched_samples))
        stretched_samples = np.concatenate([stretched_samples, padding])
    
    # Convert back to int16 for pydub
    stretched_samples = (stretched_samples * 32767).astype(np.int16)
    
    # Convert back to pydub AudioSegment
    stretched_audio = AudioSegment(
        stretched_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=1
    )
    
    logging.info(f"Stretched audio duration: {len(stretched_audio) / 1000:.2f}s")
    
    return stretched_audio

def preserve_sound_effects(original_audio: AudioSegment, synced_speech: AudioSegment, transcript: dict) -> AudioSegment:
    logging.info("Preserving sound effects and music...")
    
    # Create a silent audio segment of the same length as the original
    final_audio = AudioSegment.silent(duration=len(original_audio))
    
    # Overlay the synced speech
    final_audio = final_audio.overlay(synced_speech)
    
    # Define speech segments
    speech_segments = [(int(seg['start'] * 1000), int(seg['end'] * 1000)) for seg in transcript['segments']]
    
    # Find non-speech segments (potential sound effects and music)
    non_speech_segments = []
    last_end = 0
    for start, end in speech_segments:
        if start > last_end:
            non_speech_segments.append((last_end, start))
        last_end = end
    if len(original_audio) > last_end:
        non_speech_segments.append((last_end, len(original_audio)))
    
    # Overlay non-speech segments from the original audio
    for start, end in non_speech_segments:
        non_speech_audio = original_audio[start:end]
        final_audio = final_audio.overlay(non_speech_audio, position=start)
    
    return final_audio

def create_synced_audio(original_audio: AudioSegment, transcript: dict, translated_text: str, source_language: str, target_language: str, project_dir: str) -> AudioSegment:
    logging.info("Creating synced translated audio...")
    
    final_audio = AudioSegment.silent(duration=len(original_audio))
    
    processed_segments = adaptive_segment_processing(transcript['segments'], original_audio, source_language, target_language, project_dir)
    
    for start_time, audio_segment in processed_segments:
        final_audio = final_audio.overlay(audio_segment, position=start_time)
    
    return final_audio

def create_final_video(video_path: str, final_audio_path: str, output_path: str):
    logging.info("Creating final video...")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(final_audio_path)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(output_path, audio_codec='aac', logger=None)

def process_video(video_path: str, target_language: str):
    project_dir = create_project_structure(video_path, target_language)
    
    try:
        # Extract audio
        audio_path = os.path.join(project_dir, 'audio', 'extracted_audio.wav')
        extract_audio(video_path, audio_path)
        
        # Transcribe with word timestamps and detect language
        transcript_result = transcribe_with_whisper(audio_path)
        source_language = transcript_result['language']
        print(f"Source language from Whisper: {source_language}, target language: {target_language}")

        # Save transcript
        transcript_path = os.path.join(project_dir, 'transcripts', 'transcript.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_result['text'])
        
        # Translate full text
        translated_text = translate_text(transcript_result['text'], source_language, target_language)
        
        # Save translation
        translation_path = os.path.join(project_dir, 'translations', 'translation.txt')
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        
        # Create synced translated audio
        original_audio = AudioSegment.from_wav(audio_path)
        print(f"Original audio duration: {len(original_audio)}")
        synced_speech = create_synced_audio(original_audio, transcript_result, translated_text, source_language, target_language, project_dir)
        
        # Preserve sound effects and music
        final_audio = preserve_sound_effects(original_audio, synced_speech, transcript_result)
        
        # Export final audio
        final_audio_path = os.path.join(project_dir, 'audio', 'final_audio.wav')
        final_audio.export(final_audio_path, format="wav")
        
        # Create final video
        output_video_path = os.path.join(project_dir, f"translated_{os.path.basename(video_path)}")
        create_final_video(video_path, final_audio_path, output_video_path)
        
        logging.info(f"Translation complete. Output video: {output_video_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python translate.py <video_path> <target_language>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    target_language = sys.argv[2]
    
    process_video(video_path, target_language)