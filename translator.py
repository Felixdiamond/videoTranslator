import os
import sys
import logging
import tempfile
import math
import shutil
from typing import List, Tuple

import whisper
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from gtts import gTTS
from pydub import AudioSegment
from transformers import pipeline
from tqdm import tqdm
from pydub.playback import play
from pydub.silence import detect_nonsilent
from pydub import effects
from TTS.api import TTS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, word_timestamps=True)
    print("Transcription complete.")
    return result

def translate_text(text: str, target_language: str) -> str:
    logging.info(f"Translating text to {target_language}...")
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}")
    
    # Split text into smaller chunks to avoid tokenizer length issues
    max_chunk_length = 512
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    translated_chunks = []
    for chunk in tqdm(chunks, desc="Translating chunks"):
        translation = translator(chunk)[0]['translation_text']
        translated_chunks.append(translation)
    
    return " ".join(translated_chunks)

def text_to_speech(text: str, language: str, output_path: str, rate: float = 1.0) -> str:
    logging.info(f"Generating speech with gTTS (rate: {rate})...")
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(output_path)

    audio = AudioSegment.from_mp3(output_path)
    
    # Adjust speech rate
    
    # adjusted_audio = audio.speedup(playback_speed=rate)
    # adjusted_audio.export(output_path, format="mp3")
    
    return output_path

def text_to_speech_coqui(text: str, language: str, output_path: str, rate: float = 1.0) -> str:
    logging.info(f"Generating speech with Coqui TTS (rate: {rate})...")
    
    # Map language codes to Coqui TTS model names
    language_model_map = {
        'en': 'tts_models/en/ljspeech/tacotron2-DDC',
        'es': 'tts_models/es/mai/tacotron2-DDC',
        'fr': 'tts_models/fr/css10/vits',
        'de': 'tts_models/de/thorsten/vits',
        'it': 'tts_models/it/mai_female/vits',
        # Add more languages as needed
    }
    
    # Select the appropriate model based on the target language
    model_name = language_model_map.get(language, 'tts_models/en/ljspeech/tacotron2-DDC')
    
    # Initialize TTS
    tts = TTS(model_name)
    
    # Generate speech
    tts.tts_to_file(text=text, file_path=output_path)
    
    # Load the generated audio
    audio = AudioSegment.from_file(output_path)
    
    # Adjust speech rate
    if rate != 1.0:
        adjusted_audio = audio.speedup(playback_speed=rate)
        adjusted_audio.export(output_path, format="mp3")
    
    return output_path

def fit_audio_to_duration(audio: AudioSegment, target_duration: int) -> AudioSegment:
    """
    Use advanced methods to fit the audio to the target duration.
    """
    # Method 1: Gradual speed-up
    current_speed = 1.0
    max_speed = 2.0
    step = 0.1
    
    while len(audio) > target_duration and current_speed < max_speed:
        current_speed += step
        audio = audio.speedup(playback_speed=current_speed)
    
    # Method 2: If still too long, remove silences
    if len(audio) > target_duration:
        audio = remove_silences(audio)
    
    # Method 3: If still too long, use audio compression
    if len(audio) > target_duration:
        audio = apply_audio_compression(audio, target_duration)
    
    # Final truncation if still necessary
    return audio[:target_duration]

def remove_silences(audio: AudioSegment, silence_thresh=-50.0, min_silence_len=100) -> AudioSegment:
    """
    Remove silences from the audio to shorten its duration.
    """
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return sum([audio[start:end] for start, end in non_silent_ranges])

def apply_audio_compression(audio: AudioSegment, target_duration: int) -> AudioSegment:
    """
    Apply audio compression to fit the audio within the target duration.
    """
    compression_ratio = len(audio) / target_duration
    return effects.compress_dynamic_range(audio, threshold=-20, ratio=compression_ratio, attack=5, release=50)

def create_synced_audio(original_audio: AudioSegment, transcript: dict, translated_text: str, target_language: str, project_dir: str) -> AudioSegment:
    logging.info("Creating synced translated audio...")
    
    final_audio = AudioSegment.silent(duration=len(original_audio))
    segments_dir = os.path.join(project_dir, 'translated_segments')
    
    for i, segment in enumerate(transcript['segments']):
        start_time = int(segment['start'] * 1000)
        end_time = int(segment['end'] * 1000)
        duration = end_time - start_time
        
        # Translate segment text
        translated_segment = translate_text(segment['text'], target_language)
        
        logging.info(f"Segment {i}: Original: '{segment['text']}', Translated: '{translated_segment}'")
        
        if not translated_segment.strip():
            logging.warning(f"Skipping empty translated segment for: '{segment['text']}'")
            continue
        
        # Generate speech for translated segment
        tts_output_path = os.path.join(segments_dir, f"segment_{i:04d}.mp3")
        
        try:
            text_to_speech(translated_segment, target_language, tts_output_path)
            translated_audio = AudioSegment.from_mp3(tts_output_path)
            
            # Check if the translated audio is longer than the original segment
            if len(translated_audio) > duration:
                # Try to speed up the audio to fit the duration
                speed_factor = len(translated_audio) / duration
                adjusted_audio = translated_audio.speedup(playback_speed=speed_factor)
                
                # If still too long, use advanced methods to fit the duration
                if len(adjusted_audio) > duration:
                    adjusted_audio = fit_audio_to_duration(adjusted_audio, duration)
            else:
                adjusted_audio = translated_audio
            
            # Ensure the adjusted audio is not longer than the segment duration
            adjusted_audio = adjusted_audio[:duration]
            
            # Save the adjusted audio segment
            adjusted_audio.export(tts_output_path, format="mp3")
            
            # Cut out the silent part from the final audio and insert the translated segment
            final_audio = final_audio[:start_time] + adjusted_audio + final_audio[end_time:]
        except Exception as e:
            logging.error(f"Error processing segment {i}: {str(e)}")
    
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
        
        # Transcribe with word timestamps
        transcript_result = transcribe_with_whisper(audio_path)
        # save the actual transcript result
        # transcript_result_path = os.path.join(project_dir, 'transcripts', 'transcript_result.json')
        # with open(transcript_result_path, 'w') as f:
        #     f.write(str(transcript_result['segments'][0]['text']))

        # Save transcript
        transcript_path = os.path.join(project_dir, 'transcripts', 'transcript.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_result['text'])
        
        # Translate full text
        translated_text = translate_text(transcript_result['text'], target_language)
        
        # Save translation
        translation_path = os.path.join(project_dir, 'translations', 'translation.txt')
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        
        # Create synced translated audio
        original_audio = AudioSegment.from_wav(audio_path)
        print(f"Original audio duration: {len(original_audio)}")
        synced_audio = create_synced_audio(original_audio, transcript_result, translated_text, target_language, project_dir)
        
        # Export synced audio
        final_audio_path = os.path.join(project_dir, 'audio', 'final_audio.wav')
        synced_audio.export(final_audio_path, format="wav")
        
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