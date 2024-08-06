import os
import sys
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import whisper
from googletrans import Translator
import numpy as np
from pydub import AudioSegment
import tempfile
import time
import nltk
from scipy.io import wavfile
from scipy import signal
# from TTS.api import TTS
import librosa
from gtts import gTTS

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

def calculate_adaptive_silence(transcription_result, percentile=75):
    silences = []
    last_end_time = 0
    for segment in transcription_result['segments']:
        for word in segment.get('words', []):
            start_time = word['start']
            silence_duration = start_time - last_end_time
            if silence_duration > 0:
                silences.append(silence_duration)
            last_end_time = word['end']
    if silences:
        return np.percentile(silences, percentile)
    return 0.5  # Default if no silences found

def group_words_by_silence(transcription_result, min_silence_duration):
    phrases = []
    current_phrase = []
    last_end_time = 0

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

    return phrases

def match_phrases_to_sentences(phrases, translated_sentences):
    matched_pairs = []
    phrase_index = 0
    sentence_index = 0

    while phrase_index < len(phrases) and sentence_index < len(translated_sentences):
        current_phrase = ' '.join(word for _, _, word in phrases[phrase_index])
        current_sentence = translated_sentences[sentence_index]

        if len(current_phrase.split()) <= len(current_sentence.split()):
            matched_pairs.append((phrases[phrase_index], current_sentence))
            phrase_index += 1
        else:
            if sentence_index + 1 < len(translated_sentences):
                current_sentence += ' ' + translated_sentences[sentence_index + 1]
                matched_pairs.append((phrases[phrase_index], current_sentence))
                sentence_index += 1
            phrase_index += 1
        sentence_index += 1

    return matched_pairs

def text_to_speech_gtts(text, language, output_path):
    mp3_path = output_path.replace('.wav', '.mp3')
    wav_path = output_path
    
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(mp3_path)
    
    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise Exception(f"Failed to create audio file: {mp3_path}")

    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

    # Clean up the MP3 file
    os.remove(mp3_path)

    # Verify the WAV file
    try:
        AudioSegment.from_wav(wav_path)
    except Exception as e:
        raise Exception(f"Invalid WAV file created: {wav_path}. Error: {str(e)}")
def text_to_speech_coqui(text, language, output_path):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=text, file_path=output_path, language=language)

def adjust_speech_rate(audio_segment, target_duration):
    current_duration = len(audio_segment)
    rate = current_duration / target_duration

    if rate > 1.5:
        audio_segment = audio_segment.speedup(playback_speed=1.5)
        remaining_duration = target_duration - len(audio_segment)
        if remaining_duration > 0:
            silence = AudioSegment.silent(duration=int(remaining_duration))
            audio_segment = audio_segment + silence
    elif rate < 0.8:
        silence_duration = int(target_duration - current_duration)
        silence = AudioSegment.silent(duration=silence_duration)
        audio_segment = audio_segment + silence
    else:
        audio_segment = audio_segment.speedup(playback_speed=rate)

    return audio_segment

def detect_sound_events(audio_path):
    y, sr = librosa.load(audio_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times

def create_synchronized_translation(video_path, transcription_result, translated_text, target_language):
    print("Creating synchronized translation...")
    
    video = VideoFileClip(video_path)
    
    adaptive_silence = calculate_adaptive_silence(transcription_result)
    phrases = group_words_by_silence(transcription_result, adaptive_silence)
    
    translated_sentences = nltk.sent_tokenize(translated_text)
    matched_pairs = match_phrases_to_sentences(phrases, translated_sentences)
    
    final_audio = AudioSegment.silent(duration=int(video.duration * 1000))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (phrase, translated_sentence) in enumerate(matched_pairs):
            start_time = phrase[0][0]
            end_time = phrase[-1][1]
            
            temp_path = os.path.join(temp_dir, f"temp_phrase_{i}.wav")
            
            try:
                text_to_speech_gtts(translated_sentence, target_language, temp_path)
                phrase_audio = AudioSegment.from_wav(temp_path)
                
                target_duration = (end_time - start_time) * 1000  # Convert to milliseconds
                
                phrase_audio = adjust_speech_rate(phrase_audio, target_duration)
                
                # Improved fade in/out
                fade_duration = min(100, len(phrase_audio) // 4)
                phrase_audio = phrase_audio.fade_in(fade_duration).fade_out(fade_duration)
                
                # Add a short pause between phrases
                pause_duration = min(200, (target_duration * 0.1))
                silence = AudioSegment.silent(duration=int(pause_duration))
                phrase_audio = silence + phrase_audio + silence
                
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
    new_audio = create_synchronized_translation(video_path, transcription_result, translated_text, target_language)
    print("Synchronized audio generation complete.")

    print("Combining new audio with video...")
    new_audio_path = "new_audio.wav"
    new_audio.export(new_audio_path, format="wav")
    new_audio_clip = AudioFileClip(new_audio_path)
    
    # Create a new video with only the translated audio
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
        print("Usage: python video_translator.py <video_path> <target_language>")
        print("Example: python video_translator.py my_video.mp4 fr")
        sys.exit(1)
    
    video_path = sys.argv[1]
    target_language = sys.argv[2]
    
    process_video(video_path, target_language)