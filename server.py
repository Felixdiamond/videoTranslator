import os
import sys
import logging
import time
from typing import List, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from translator import (
    create_project_structure,
    extract_audio,
    transcribe_with_whisper,
    translate_text,
    text_to_speech_coqui,
    create_synced_audio,
    preserve_sound_effects,
    create_final_video,
)

app = FastAPI()

@app.websocket("/translate/{video_path}/{target_language}")
async def translate_video(websocket: WebSocket, video_path: str, target_language: str):
    await websocket.accept()

    try:
        # Create project structure
        project_dir = create_project_structure(video_path, target_language)

        # Extract audio
        audio_path = os.path.join(project_dir, 'audio', 'extracted_audio.wav')
        extract_audio(video_path, audio_path)
        await websocket.send_text("Audio extracted")

        # Transcribe with Whisper
        transcript_result = transcribe_with_whisper(audio_path)
        source_language = transcript_result['language']
        await websocket.send_text(f"Source language: {source_language}, Target language: {target_language}")

        # Translate full text
        translated_text = translate_text(transcript_result['text'], source_language, target_language)
        await websocket.send_text("Text translated")

        # Create synced translated audio
        original_audio = AudioSegment.from_wav(audio_path)
        synced_speech = create_synced_audio(original_audio, transcript_result, translated_text, source_language, target_language, project_dir)
        await websocket.send_text("Synced audio created")

        # Preserve sound effects and music
        final_audio = preserve_sound_effects(original_audio, synced_speech, transcript_result)
        await websocket.send_text("Sound effects preserved")

        # Export final audio
        final_audio_path = os.path.join(project_dir, 'audio', 'final_audio.wav')
        final_audio.export(final_audio_path, format="wav")
        await websocket.send_text("Final audio exported")

        # Create final video
        output_video_path = os.path.join(project_dir, f"translated_{os.path.basename(video_path)}")
        create_final_video(video_path, final_audio_path, output_video_path)
        await websocket.send_text(f"Translation complete. Output video: {output_video_path}")

        # Send the output video as a streaming response
        return StreamingResponse(open(output_video_path, 'rb'), media_type='video/mp4')

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        await websocket.send_text(f"Error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)