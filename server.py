from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
from pathlib import Path
import shutil
import logging
import asyncio
import os

from translator import process_video, LANGUAGE_MODEL_MAP

app = FastAPI()

# Server logger setup.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd().resolve()

LANGUAGE_LABELS = {
    "en": "English (US Speaker)",
    "es": "Spanish (Spain Speaker)",
    "fr": "French (France Speaker)",
    "zh": "Chinese (Mandarin Speaker)",
    "ja": "Japanese (Japan Speaker)",
    "ko": "Korean (Korea Speaker)",
    "de": "German",
    "pt": "Portuguese",
}

MELO_SPEAKER_IDS_BY_LANGUAGE = {
    "en": ["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"],
    "es": ["ES"],
    "fr": ["FR"],
    "zh": ["ZH"],
    "ja": ["JP"],
    "ko": ["KR"],
}

VALID_TTS_MODES = {"melo", "qwen3", "gtts"}
VALID_QWEN3_MODEL_SIZES = {"0.6B", "1.7B"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_env_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        values.append(frontend_url)

    # Keep local DX friendly by default while allowing hosted URLs via env vars.
    defaults = ["http://localhost:3000", "http://127.0.0.1:3000"]
    merged = defaults + values

    seen = set()
    deduped = []
    for origin in merged:
        if origin not in seen:
            deduped.append(origin)
            seen.add(origin)
    return deduped


configured_origins = _parse_env_origins()
allow_all_origins = _env_flag("CORS_ALLOW_ALL", default=False) or "*" in configured_origins

cors_kwargs = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    # This API does not rely on cookies. Keeping credentials off avoids wildcard restrictions.
    "allow_credentials": False,
}

if allow_all_origins:
    cors_kwargs["allow_origins"] = ["*"]
    logger.warning("CORS configured for all origins. Use only in trusted environments.")
else:
    cors_kwargs["allow_origins"] = configured_origins
    # Also support localhost with any port for local-first workflows.
    cors_kwargs["allow_origin_regex"] = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    logger.info(f"CORS configured origins: {configured_origins}")

app.add_middleware(
    CORSMiddleware,
    **cors_kwargs,
)

UPLOAD_DIRECTORY = Path("uploaded_files")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def healthcheck():
    return {"ok": True}


@app.get("/options")
async def get_options():
    languages = []
    for code, cfg in LANGUAGE_MODEL_MAP.items():
        languages.append(
            {
                "code": code,
                "label": LANGUAGE_LABELS.get(code, code.upper()),
                "supportsMelo": "melo_language" in cfg,
                "defaultSpeakerId": cfg.get("speaker_id"),
            }
        )

    return {
        "languages": languages,
        "ttsModes": [
            {"value": "melo", "label": "MeloTTS"},
            {"value": "qwen3", "label": "Qwen3-TTS"},
            {"value": "gtts", "label": "gTTS"},
        ],
        "qwen3ModelSizes": sorted(VALID_QWEN3_MODEL_SIZES),
        "meloSpeakerIdsByLanguage": MELO_SPEAKER_IDS_BY_LANGUAGE,
        "defaults": {
            "ttsMode": "melo",
            "qwen3ModelSize": "1.7B",
            "enableVoiceCloning": True,
        },
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Basic filename sanitization.
    filename = Path(file.filename).name
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

    # Return a project-relative path.
    return {"filePath": str(file_location.relative_to(PROJECT_ROOT))}


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    resolved = (PROJECT_ROOT / file_path).resolve()
    if PROJECT_ROOT not in resolved.parents and resolved != PROJECT_ROOT:
        raise HTTPException(status_code=403, detail="Path is outside project root.")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=str(resolved), filename=resolved.name)


@app.websocket("/translate/{video_path:path}/{target_language}")
async def translate_video_ws(
    websocket: WebSocket,
    video_path: str,
    target_language: str,
    tts_mode: str = "melo",
    qwen3_model_size: str = "1.7B",
    speaker_id: Optional[str] = None,
    enable_voice_cloning: bool = True,
):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for video: {video_path}, target language: {target_language}")

    if target_language not in LANGUAGE_MODEL_MAP:
        error_msg = f"Unsupported target language: {target_language}. Supported languages are: {list(LANGUAGE_MODEL_MAP.keys())}"
        logger.error(error_msg)
        await websocket.send_text(f"Error: {error_msg}")
        await websocket.close()
        return

    if tts_mode not in VALID_TTS_MODES:
        error_msg = f"Unsupported tts_mode: {tts_mode}. Supported values are: {sorted(VALID_TTS_MODES)}"
        logger.error(error_msg)
        await websocket.send_text(f"Error: {error_msg}")
        await websocket.close()
        return

    if qwen3_model_size not in VALID_QWEN3_MODEL_SIZES:
        error_msg = (
            f"Unsupported qwen3_model_size: {qwen3_model_size}. "
            f"Supported values are: {sorted(VALID_QWEN3_MODEL_SIZES)}"
        )
        logger.error(error_msg)
        await websocket.send_text(f"Error: {error_msg}")
        await websocket.close()
        return

    actual_video_path = PROJECT_ROOT / video_path

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

        output_video_file_path = await asyncio.to_thread(
            process_video,
            str(actual_video_path),
            target_language,
            tts_mode=tts_mode,
            qwen3_model_size=qwen3_model_size,
            melo_speaker_id=speaker_id,
            enable_voice_cloning=enable_voice_cloning,
        )

        if output_video_file_path:
            logger.info(f"Translation successful. Output video: {output_video_file_path}")
            relative_output_path = str(Path(output_video_file_path).relative_to(PROJECT_ROOT))
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