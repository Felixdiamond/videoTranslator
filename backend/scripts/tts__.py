import os
import torch
from TTS.api import TTS

def text_to_speech(text, language, output_path):
    """
    Generates speech audio from the provided text using the TTS API.
    Args:
        text: The text to be converted to speech.
        language: The target language for the speech.
        output_path: The path to save the generated speech audio file.
    """
    try:
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Init TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        # Generate speech audio
        wav = tts.tts(text=text, language=language)
        tts.save_wav(wav, output_path)
        print(f"Text-to-speech audio saved to: {output_path}")
    except Exception as e:
        print(f"Error generating text-to-speech audio: {str(e)}")
        raise e
    
# Example usage
text_to_speech("Hello, how are you?", "en", "output.wav")