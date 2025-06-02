from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse # Keep for potential future use
from typing import Optional
from pathlib import Path
import os
import shutil
import logging
import asyncio # Added for running blocking IO in a thread

# Import the main processing function and language map from the updated translator
from translator import process_video, LANGUAGE_MODEL_MAP

app = FastAPI()

# Configure basic logging for the server
# translator.py now configures its own logger, so this will be for server-specific logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Allow CORS for specific origins
origins = [
    "http://localhost:3000", # Assuming your Next.js frontend runs on this port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = Path("uploaded_files")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Sanitize filename to prevent directory traversal or invalid characters
    filename = Path(file.filename).name # Basic sanitization
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_location = UPLOAD_DIRECTORY / filename
    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"File '{filename}' uploaded to '{file_location}'")
    except Exception as e:
        logger.error(f"Failed to save uploaded file '{filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # Return the path relative to the project root, as expected by the frontend and translator.py
    return {"filePath": str(file_location.relative_to(Path.cwd()))}


@app.websocket("/translate/{video_path:path}/{target_language}")
async def translate_video_ws(websocket: WebSocket, video_path: str, target_language: str):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for video: {video_path}, target language: {target_language}")

    if target_language not in LANGUAGE_MODEL_MAP:
        error_msg = f"Unsupported target language: {target_language}. Supported languages are: {list(LANGUAGE_MODEL_MAP.keys())}"
        logger.error(error_msg)
        await websocket.send_text(f"Error: {error_msg}")
        await websocket.close()
        return

    actual_video_path = Path.cwd() / video_path

    if not actual_video_path.exists():
        error_msg = f"Video file not found at resolved path: {actual_video_path} (original path: {video_path})"
        logger.error(error_msg)
        await websocket.send_text(f"Error: {error_msg}")
        await websocket.close()
        return

    try:
        await websocket.send_text(f"Translation process initiated for '{actual_video_path.name}' to '{target_language}'.")
        logger.info(f"Starting translation task for {actual_video_path} to {target_language}...")
        await websocket.send_text("Video processing in progress... This may take a while. Please check server logs for detailed progress.")

        output_video_file_path = await asyncio.to_thread(process_video, str(actual_video_path), target_language)

        if output_video_file_path:
            logger.info(f"Translation successful. Output video: {output_video_file_path}")
            relative_output_path = str(Path(output_video_file_path).relative_to(Path.cwd()))
            await websocket.send_text(f"Translation complete. Output video: {relative_output_path}")
        else:
            logger.error("Translation process completed but no output video path was returned.")
            await websocket.send_text("Error: Translation completed but no output file was generated. Check server logs.")

    except Exception as e:
        logger.error(f"An error occurred during translation for {actual_video_path}: {str(e)}", exc_info=True)
        await websocket.send_text(f"Error: An unexpected error occurred: {str(e)}")
    finally:
        logger.info(f"Closing WebSocket connection for {actual_video_path.name}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Video Translator server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)