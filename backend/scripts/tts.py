import os
import json
import logging
from gtts import gTTS
from moviepy.editor import AudioFileClip
import pyttsx3

def text_to_speech(text, language, output_path, video_duration, isAudio):
    """
    Generates speech audio from the provided text.
    If isAudio is True, it uses gTTS without any adjustments.
    If isAudio is False, it uses pyttsx3 and adjusts the speech rate to match the video duration.

    Args:
        text: The text to be converted to speech.
        language: The target language for the speech.
        output_path: The path to save the generated speech audio file.
        video_duration: The duration of the original video in seconds.
        isAudio: Boolean indicating whether the input is an audio file or a video file.
    """
    try:
        if isAudio:
            # Use gTTS without any adjustments
            myobj = gTTS(text=text, lang=language, slow=False, tld='com.au')
            myobj.save(output_path)
            logging.info(f"Text-to-speech audio saved to: {output_path}")
        else:
            # Use pyttsx3 and adjust the speech rate
            engine = pyttsx3.init()

            # Set the language
            voices = engine.getProperty('voices')
            if language == 'en':
                engine.setProperty('voice', voices[0].id)  # Set English voice
            else:
                # Set the appropriate voice for the desired language
                pass

            # Calculate the desired speech rate
            num_words = len(text.split())
            desired_rate = num_words / video_duration

            # Set the speech rate
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate * desired_rate)

            # Save the speech audio to the output file
            engine.save_to_file(text, output_path)
            engine.runAndWait()

            logging.info(f"Text-to-speech audio saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error generating text-to-speech audio: {str(e)}")
        raise e

def main():
    # Get the directory of the current script file
    dir_name = os.path.dirname(os.path.realpath(__file__))

    # Load parameters from args.json
    with open(os.path.join(dir_name, "../temp/args.json"), "r") as f:
        args = json.load(f)

    # Call the text_to_speech function with the loaded parameters
    text_to_speech(
        args["text"], args["language"], args["output_path"], args["video_duration"], args["isAudio"]
    )


if __name__ == "__main__":
    main()
