import os
import sys
import moviepy.editor as mp

def merge_audio_video(video_path, audio_path, output_path):
    """
    Merges an audio file with a video file and saves the result to a new file.
    
    Args:
        video_path: Path to the video file.
        audio_path: Path to the audio file (WAV format).
        output_path: Path to save the merged video file (MP4 format).
    """
    try:
        print("Started merging process")
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        
        print(f"Loaded video file: {video_path}")
        print(f"Loaded audio file: {audio_path}")
        
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264')
        
        print(f"Merged video saved to: {output_path}")
        
        video_clip.close()
        audio_clip.close()
        final_clip.close()
    except Exception as e:
        print(f"Error merging audio and video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_audio_video.py <video_path> <audio_path> <output_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3]
    
    merge_audio_video(video_path, audio_path, output_path)