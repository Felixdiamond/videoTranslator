import os
import sys
import tempfile
import time
import numpy as np
import torch
import torchaudio
import nltk
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from scipy.signal import correlate, find_peaks
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import whisper
from TTS.api import TTS
import re
import unicodedata
import essentia
import essentia.standard as es

nltk.download('punkt', quiet=True)

def transcribe_audio_whisper(audio_path):
    print("Transcribing audio with word-level timestamps...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    return result

def translate_text(text, target_language):
    if not text:
        print("Error: Empty text provided for translation.")
        return None

    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}")
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []

    for i, chunk in enumerate(chunks):
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                print(f"Translating chunk {i+1}/{len(chunks)} (length: {len(chunk)})...")
                translation = translator(chunk)[0]['translation_text']
                if translation:
                    translated_chunks.append(translation)
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

def text_to_speech_coqui(text, language, output_path):
    print("Generating speech using Coqui TTS...")
    language_to_model = {
        'en': 'tts_models/en/ljspeech/tacotron2-DDC',
        'fr': 'tts_models/fr/mai/tacotron2-DDC',
        'de': 'tts_models/de/thorsten/tacotron2-DDC',
        'es': 'tts_models/es/mai/tacotron2-DDC',
        # Add more language mappings as needed
    }

    if language not in language_to_model:
        raise ValueError(f"Unsupported language: {language}")

    model_name = language_to_model[language]

    # Initialize TTS
    print(f"Initializing TTS model: {model_name}")
    try:
        tts = TTS(model_name)
    except Exception as e:
        print(f"Error initializing TTS model: {e}")
        return

    # Clean the text
    print("Sanitizing text...")
    text = sanitize_text(text)

    # Generate speech
    print("Generating speech...")
    try:
        tts.tts_to_file(text=text, file_path=output_path)
    except Exception as e:
        print(f"Error generating speech: {e}")
        return

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise Exception(f"Failed to create audio file: {output_path}")

    print(f"Audio file created successfully: {output_path}")


def text_to_speech_gtts(text, language, output_path):
    mp3_path = output_path.replace('.wav', '.mp3')
    wav_path = output_path
    
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(mp3_path)
    
    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise Exception(f"Failed to create audio file: {mp3_path}")

    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

    os.remove(mp3_path)

    try:
        AudioSegment.from_wav(wav_path)
    except Exception as e:
        raise Exception(f"Invalid WAV file created: {wav_path}. Error: {str(e)}")

def analyze_audio_characteristics(audio_path):
    print("Analyzing audio characteristics...")
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Loaded audio file: {audio_path}, Sample rate: {sample_rate}")

    # Ensure waveform is mono
    if waveform.size(0) > 1:
        print("Converting to mono...")
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert waveform to numpy array
    waveform_np = waveform.squeeze().numpy()

    # Detect speech segments using VAD
    try:
        print("Applying VAD...")
        vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
        speech_mask = vad(waveform)
        print("VAD applied successfully.")
    except Exception as e:
        print(f"Error applying VAD: {e}")
        return None
    
    # Verify VAD output
    if speech_mask.size(0) == 0:
        print("VAD output is empty.")
        return None

    # Extract speech segments
    print("Segmenting speech...")
    try:
        speech_mask = speech_mask.squeeze().byte()
        speech_mask_np = speech_mask.numpy()
        speech_segments = []
        
        # Identify speech segments based on VAD mask
        in_speech = False
        start = 0
        for i in range(len(speech_mask_np)):
            if speech_mask_np[i] and not in_speech:
                start = i
                in_speech = True
            elif not speech_mask_np[i] and in_speech:
                end = i
                speech_segments.append((start / sample_rate, end / sample_rate))
                in_speech = False
        # Handle edge case if speech ends at the end of the file
        if in_speech:
            end = len(speech_mask_np)
            speech_segments.append((start / sample_rate, end / sample_rate))
        
        if len(speech_segments) == 0:
            print("No speech segments found.")
            avg_speech_duration = 0
        else:
            avg_speech_duration = np.mean([end - start for start, end in speech_segments])
        
        print(f"Speech segments identified: {len(speech_segments)}")
    except Exception as e:
        print(f"Error segmenting speech: {e}")
        return None
    
    # Estimate tempo using Essentia's RhythmExtractor2013
    print("Estimating tempo using Essentia...")
    try:
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(waveform_np)
        print(f"Estimated BPM: {bpm}")
    except Exception as e:
        print(f"Error estimating tempo: {e}")
        bpm = 0
    
    # Calculate RMS as a proxy for noise level
    print("Calculating noise level...")
    try:
        noise_level = np.sqrt(np.mean(waveform_np ** 2))
        print(f"Noise level: {noise_level}")
    except Exception as e:
        print(f"Error calculating noise level: {e}")
        noise_level = 0

    print("Audio analysis complete.")
    
    return {
        'avg_speech_duration': avg_speech_duration,
        'tempo': bpm,
        'noise_level': noise_level
    }


def adaptive_preprocessing(audio_path):
    print("Adaptive preprocessing...")
    characteristics = analyze_audio_characteristics(audio_path)
    
    # if characteristics is None:
    #     print("Error: Unable to analyze audio characteristics.")
    #     # Handle error or return default values
    #     return 0.1, 0.3, 1.0
    
    silence_threshold = min(0.1, characteristics['noise_level'] * 2)
    min_silence_duration = min(0.3, characteristics['avg_speech_duration'] / 2)
    speech_rate_factor = 1 + (characteristics['tempo'] - 120) / 120
    
    return silence_threshold, min_silence_duration, speech_rate_factor

def detect_speech_segments(audio_path):
    print("Detecting speech segments using Whisper...")

    # Load the Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")

    # Load the audio file
    print("Loading audio file...")
    audio = whisper.load_audio(audio_path)

    # Transcribe the audio with word-level timestamps
    print("Transcribing audio...")
    result = model.transcribe(audio, word_timestamps=True)

    # Extract the speech segments from the transcription
    speech_segments = []
    for segment in result["segments"]:
        for word in segment["words"]:
            start_time = word["start"]
            end_time = word["end"]
            speech_segments.append((start_time, end_time))

    print("Speech segments detected.")
    return speech_segments

def sanitize_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Normalize unicode characters (e.g., combining accents)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove known special characters and symbols
    text = re.sub(r'[^\w\s\']', '', text)
    
    # Replace accented characters with their base characters
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_mismatch(original_audio, translated_audio):
    if original_audio.channels > 1:
        original_audio = original_audio.set_channels(1)
    if translated_audio.channels > 1:
        translated_audio = translated_audio.set_channels(1)

    original_array = np.array(original_audio.get_array_of_samples())
    translated_array = np.array(translated_audio.get_array_of_samples())

    correlation = correlate(original_array, translated_array, mode='full')
    delay = np.argmax(correlation) - (len(translated_array) - 1)

    return delay

def adjust_audio(audio, delay):
    if delay > 0:
        silence = AudioSegment.silent(duration=delay)
        return silence + audio
    elif delay < 0:
        return audio[-delay:]
    else:
        return audio

def resynchronize(original_audio, adjusted_audio):
    original_mfcc = torchaudio.functional.mfcc(original_audio)
    adjusted_mfcc = torchaudio.functional.mfcc(adjusted_audio)

    _, wp = torchaudio.functional.dtw(original_mfcc, adjusted_mfcc, subseq=True)

    adjusted_array = np.array(adjusted_audio.get_array_of_samples())
    resynced_array = torchaudio.functional.time_stretch(adjusted_array, rate=len(wp)/len(adjusted_array))

    return AudioSegment(
        resynced_array.tobytes(),
        frame_rate=adjusted_audio.frame_rate,
        sample_width=adjusted_audio.sample_width,
        channels=1
    )

def assess_sync_quality(original_audio, synced_audio):
    if original_audio.channels > 1:
        original_audio = original_audio.set_channels(1)
    if synced_audio.channels > 1:
        synced_audio = synced_audio.set_channels(1)

    original_array = np.array(original_audio.get_array_of_samples())
    synced_array = np.array(synced_audio.get_array_of_samples())

    correlation = np.corrcoef(original_array, synced_array)[0, 1]

    return correlation

def iterative_synchronization(original_audio, translated_audio, max_iterations=5):
    best_sync = translated_audio
    best_score = float('-inf')

    for i in range(max_iterations):
        mismatch = analyze_mismatch(original_audio, best_sync)
        adjusted_audio = adjust_audio(best_sync, mismatch)
        resynced_audio = resynchronize(original_audio, adjusted_audio)
        sync_score = assess_sync_quality(original_audio, resynced_audio)

        if sync_score > best_score:
            best_sync = resynced_audio
            best_score = sync_score
        else:
            break

    return best_sync

def improve_synchronization(original_audio_path, translated_audio):
    # Ensure original_audio_path is a valid path to a .wav file
    original_audio = AudioSegment.from_wav(original_audio_path)

    # Check if translated_audio is an AudioSegment object
    if isinstance(translated_audio, AudioSegment):
        # If already an AudioSegment, use it directly
        pass
    else:
        # If it's a path, load it as AudioSegment
        translated_audio = AudioSegment.from_wav(translated_audio)

    # Debug: Confirm that both audio segments are loaded
    print(f"Original audio duration: {original_audio.duration_seconds} seconds")
    print(f"Translated audio duration: {translated_audio.duration_seconds} seconds")

    # Detect speech segments
    # original_segments = detect_speech_segments(original_audio_path)
    
    # # Save the translated audio to a temporary file
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
    #     translated_audio.export(tmp_file.name, format="wav")
    #     translated_segments = detect_speech_segments(tmp_file.name)

    # # Remove the temporary file
    # os.remove(tmp_file.name)

    # Synchronize audio
    final_synced_audio = iterative_synchronization(original_audio, translated_audio)

    # Debug: Confirm synchronization
    print(f"Final synced audio duration: {final_synced_audio.duration_seconds} seconds")

    return final_synced_audio

def create_synchronized_translation(video_path, transcription_result, translated_text, target_language):
    print("Creating synchronized translation...")
    
    video = VideoFileClip(video_path)
    
    silence_threshold, min_silence_duration, speech_rate_factor = adaptive_preprocessing(video.audio.filename)
    
    phrases = []
    current_phrase = []
    last_end_time = 0

    # print(f"Transcription result: {transcription_result}")
    # print(f"Translated text: {translated_text}")
    print("Transcription and translation results obtained.")

    for segment in transcription_result['segments']:
        for word in segment.get('words', []):
            start_time, end_time = word['start'], word['end']
            
            if start_time - last_end_time >= min_silence_duration:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []
            
            current_phrase.append((start_time, end_time, word['word']))
            last_end_time = end_time

    if current_phrase:
        phrases.append(current_phrase)
    
    # print(f"Detected phrases: {phrases}")
    print("Phrases detected.")

    translated_sentences = nltk.sent_tokenize(translated_text)
    
    final_audio = AudioSegment.silent(duration=int(video.duration * 1000))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, phrase in enumerate(phrases):
            if i >= len(translated_sentences):
                break
            
            start_time = phrase[0][0]
            end_time = phrase[-1][1]
            
            temp_path = os.path.join(temp_dir, f"temp_phrase_{i}.wav")
            
            try:
                text_to_speech_gtts(translated_sentences[i], target_language, temp_path)
                phrase_audio = AudioSegment.from_wav(temp_path)
                
                target_duration = (end_time - start_time) * 1000
                
                if len(phrase_audio) > target_duration:
                    phrase_audio = phrase_audio.speedup(playback_speed=len(phrase_audio) / target_duration)
                elif len(phrase_audio) < target_duration:
                    silence = AudioSegment.silent(duration=int(target_duration - len(phrase_audio)))
                    phrase_audio += silence
                
                fade_duration = min(100, len(phrase_audio) // 4)
                phrase_audio = phrase_audio.fade_in(fade_duration).fade_out(fade_duration)
                
                final_audio = final_audio.overlay(phrase_audio, position=int(start_time * 1000))
            except Exception as e:
                print(f"Error processing phrase {i}: {str(e)}")
                continue

    return final_audio

def process_video(video_path, target_language):
    print("Extracting audio from video...")
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    
    print("Transcribing audio...")
    transcription_result = transcribe_audio_whisper(audio_path)
    transcript = transcription_result["text"]
    print("Transcription complete.")

    print("Translating text...")
    translated_text = translate_text(transcript, target_language)
    print("Translation complete.")

    print("Generating synchronized translated audio...")
    initial_audio = create_synchronized_translation(video_path, transcription_result, translated_text, target_language)
    
    print("Improving synchronization...")
    final_audio = improve_synchronization(audio_path, initial_audio)
    print("Synchronized audio generation complete.")

    print("Combining new audio with video...")
    new_audio_path = "new_audio.wav"
    final_audio.export(new_audio_path, format="wav")
    new_audio_clip = AudioFileClip(new_audio_path)
    
    final_clip = video.set_audio(new_audio_clip)

    output_path = f"translated_{os.path.basename(video_path)}"
    final_clip.write_videofile(output_path)
    print(f"Translated video saved as {output_path}")

    video.close()
    new_audio_clip.close()
    os.remove(audio_path)
    os.remove(new_audio_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python translate.py <video_path> <target_language>")
        print("Example: python translate.py my_video.mp4 fr")
        sys.exit(1)
    
    video_path = sys.argv[1]
    target_language = sys.argv[2]
    
    process_video(video_path, target_language)
