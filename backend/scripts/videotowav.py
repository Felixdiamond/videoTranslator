import os
import sys
import moviepy.editor as mp

def extract_audio_video(video_path, audio_output_path, video_output_path):
    """
    Extracts audio and video streams from a video file and saves them to separate files.
    Args:
        video_path: Path to the video file.
        audio_output_path: Path to save the extracted audio file (WAV format).
        video_output_path: Path to save the extracted video file (MP4 format).
    """
    try:
        # Check if the output files already exist
        if os.path.exists(audio_output_path) and os.path.exists(video_output_path):
            print("Output files already exist, skipping extraction.")
            return audio_output_path, video_output_path

        print("Started process")
        clip = mp.VideoFileClip(video_path)
        print(f"Loaded video file: {video_path}")
        audio = clip.audio
        audio.write_audiofile(audio_output_path)
        print(f"Audio extracted and saved to: {audio_output_path}")
        clip.write_videofile(video_output_path, codec='libx264')
        print(f"Video extracted and saved to: {video_output_path}")
        clip.close()
        return audio_output_path, video_output_path
    except Exception as e:
        print(f"Error extracting audio and video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python videotoaudio.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../temp")
    audio_output_path = os.path.join(temp_dir, "audio.wav")
    video_output_path = os.path.join(temp_dir, "video.mp4")
    audio_path, video_path = extract_audio_video(video_path, audio_output_path, video_output_path)