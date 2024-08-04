import os
import sys
import time
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.AudioClip import AudioArrayClip
from gtts import gTTS
import whisper
from googletrans import Translator
from deep_translator import GoogleTranslator
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.effects import speedup

def get_language_code(language_name):
    language_map = {lang.lower(): lang for lang in GoogleTranslator().get_supported_languages()}
    language_name = language_name.lower()
    if language_name in language_map:
        return language_map[language_name]
    raise ValueError(f"Language '{language_name}' not found. Please check the spelling.")

def transcribe_audio_whisper(audio_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Attempting Whisper transcription (attempt {attempt + 1}/{max_retries})...")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            if result["text"]:
                return result["text"]
            else:
                print("Whisper returned an empty transcript.")
                return None
        except Exception as e:
            print(f"An error occurred during Whisper transcription: {e}")
            if "checksum does not match" in str(e):
                print("Attempting to re-download the Whisper model...")
                cache_dir = os.path.expanduser("~/.cache/whisper")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
            if attempt < max_retries - 1:
                print(f"Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
    
    print("All Whisper transcription attempts failed.")
    return None

def translate_text(text, target_language):
    if not text:
        print("Error: Empty text provided for translation.")
        return None

    translator = Translator()
    chunk_size = 5000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []

    for i, chunk in enumerate(chunks):
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                print(f"Translating chunk {i+1}/{len(chunks)} (length: {len(chunk)})...")
                translation = translator.translate(chunk, dest=target_language)
                if translation and translation.text:
                    translated_chunks.append(translation.text)
                    print(f"Chunk {i+1} translated successfully.")
                    break
                else:
                    print(f"Warning: Empty translation for chunk {i+1}.")
                    retry_count += 1
            except Exception as e:
                print(f"Error translating chunk {i+1}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in {2 ** retry_count} seconds...")
                    time.sleep(2 ** retry_count)
                else:
                    print(f"Failed to translate chunk {i+1} after {max_retries} attempts.")
                    return None

    if len(translated_chunks) == len(chunks):
        full_translation = ' '.join(translated_chunks)
        return full_translation
    else:
        print("Translation incomplete. Some chunks failed to translate.")
        return None

def extract_speech_segments(video_clip, min_silence_duration=0.3, silence_threshold=0.005):
    print("Extracting speech segments...")
    try:
        audio_array = video_clip.audio.to_soundarray(fps=22000)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # Convert to mono
    except Exception as e:
        print(f"Error using MoviePy's to_soundarray: {e}")
        print("Falling back to librosa for audio extraction...")
        temp_audio_path = "temp_audio.wav"
        video_clip.audio.write_audiofile(temp_audio_path)
        audio_array, _ = librosa.load(temp_audio_path, sr=22000)
        os.remove(temp_audio_path)

    # Detect speech segments
    is_speech = np.abs(audio_array) > silence_threshold
    speech_changes = np.diff(is_speech.astype(int))
    speech_starts = np.where(speech_changes == 1)[0] / 22000
    speech_ends = np.where(speech_changes == -1)[0] / 22000

    # Combine segments that are close together
    speech_segments = []
    for start, end in zip(speech_starts, speech_ends):
        if not speech_segments or start - speech_segments[-1][1] >= min_silence_duration:
            speech_segments.append([start, end])
        else:
            speech_segments[-1][1] = end

    # Ensure we have at least one segment and that segments don't overlap
    if not speech_segments:
        speech_segments.append([0, video_clip.duration])
    else:
        for i in range(1, len(speech_segments)):
            if speech_segments[i][0] < speech_segments[i-1][1]:
                speech_segments[i][0] = speech_segments[i-1][1]

    print(f"Detected {len(speech_segments)} speech segments")
    return speech_segments

def text_to_speech(text, language, output_path):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        print(f"Text-to-speech audio saved to {output_path}")
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        raise

def create_synchronized_translation(video_clip, transcript, translated_text, target_language):
    print("Creating synchronized translation...")
    
    speech_segments = extract_speech_segments(video_clip)
    original_sentences = transcript.split('. ')
    translated_sentences = translated_text.split('. ')
    
    min_count = min(len(original_sentences), len(translated_sentences), len(speech_segments))
    original_sentences = original_sentences[:min_count]
    translated_sentences = translated_sentences[:min_count]
    speech_segments = speech_segments[:min_count]

    final_audio = AudioSegment.silent(duration=int(video_clip.duration * 1000))
    sample_rate = 44100  # Standard sample rate

    for i, ((start, end), sentence) in enumerate(zip(speech_segments, translated_sentences)):
        print(f"Processing segment {i+1}/{min_count}")
        temp_path = f"temp_segment_{i}.mp3"
        text_to_speech(sentence.strip(), target_language, temp_path)
        
        segment = AudioSegment.from_mp3(temp_path)
        target_duration = int((end - start) * 1000)  # Convert to milliseconds
        
        # Adjust segment duration
        if len(segment) > target_duration:
            segment = segment.speedup(playback_speed=len(segment) / target_duration)
        elif len(segment) < target_duration:
            segment = segment + AudioSegment.silent(duration=target_duration - len(segment))
        
        # Ensure segment doesn't exceed the video duration
        if int(start * 1000) + len(segment) > len(final_audio):
            segment = segment[:len(final_audio) - int(start * 1000)]
        
        # Overlay the segment at the correct position
        final_audio = final_audio.overlay(segment, position=int(start * 1000))
        
        os.remove(temp_path)

    # Convert final_audio to numpy array
    samples = np.array(final_audio.get_array_of_samples())
    
    # Reshape the array to stereo if it's not already
    if samples.ndim == 1:
        samples = np.column_stack((samples, samples))
    
    # Normalize the samples to the range [-1, 1]
    samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max

    # Create an AudioArrayClip from the numpy array
    final_audio_clip = AudioArrayClip(samples, fps=sample_rate)
    final_audio_clip = final_audio_clip.set_duration(video_clip.duration)

    print(f"Final audio duration: {final_audio_clip.duration} seconds")
    return final_audio_clip

def process_video(video_path, target_language):
    target_language_code = get_language_code(target_language)

    video = VideoFileClip(video_path)
    print(f"Video duration: {video.duration} seconds")

    try:
        print("Transcribing audio...")
        transcript = transcribe_audio_whisper(video_path)
        if not transcript:
            raise Exception("Transcription failed or returned empty result")
        print("Transcription complete.")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Transcript (first 100 characters): {transcript[:100]}...")

        print("Translating text...")
        translated_text = translate_text(transcript, target_language)
        if not translated_text:
            raise Exception("Translation failed or returned empty result")
        print("Translation complete.")
        print(f"Translated text length: {len(translated_text)} characters")
        print(f"Translated text (first 100 characters): {translated_text[:100]}...")

        print("Generating synchronized translated audio...")
        new_audio = create_synchronized_translation(video, transcript, translated_text, target_language_code)
        print("Synchronized audio generation complete.")

        print("Combining new audio with video...")
        final_clip = video.set_audio(new_audio)

        output_path = f"translated_{os.path.basename(video_path)}"
        final_clip.write_videofile(output_path)
        print(f"Translated video saved as {output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        video.close()

def get_language_code(language_name):
    language_map = {lang.lower(): lang for lang in GoogleTranslator().get_supported_languages()}
    language_name = language_name.lower()
    if language_name in language_map:
        lang_code = language_map[language_name]
        
        # Map for gTTS language codes
        gtts_map = {
            'french': 'fr',
            'english': 'en',
            'spanish': 'es',
            'german': 'de',
            'italian': 'it',
            'japanese': 'ja',
        }
        
        return gtts_map.get(language_name, lang_code)
    raise ValueError(f"Language '{language_name}' not found. Please check the spelling.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ranslator.py <video_path> <target_language>")
        print("Example: python translator.py my_video.mp4 Spanish")
        sys.exit(1)
    
    video_path = sys.argv[1]
    target_language = sys.argv[2]
    
    try:
        process_video(video_path, target_language)
    except ValueError as e:
        print(f"Error: {e}")
        print("Supported languages:")
        for lang in GoogleTranslator().get_supported_languages():
            print(f"  - {lang}")
    except Exception as e:
        print(f"An error occurred: {e}")