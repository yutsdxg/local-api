from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.config import Settings
from app.services import whisper as whisper_service
from app.services import ytdlp as ytdlp_service

settings = Settings.load()

app = FastAPI(title="Local Utility API", description="Local API endpoints for n8n")


@app.post("/whisper")
async def whisper(
    file: UploadFile = File(...),
    language: str = Query("ja", description="Whisper transcription language code"),
) -> dict[str, str]:
    """Endpoint that proxies to the whisper transcription workflow."""

    try:
        text = await whisper_service.transcribe_upload(file, settings, language=language)
    except whisper_service.WhisperError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"text": text}


@app.post("/yt-dlp/audio")
async def download_youtube_audio(
    video_id: str = Query(..., description="YouTube video ID (e.g. oCVLb374gQ0)")
) -> dict[str, str]:
    """Endpoint that downloads a video's audio track as WAV via yt-dlp."""

    try:
        output_path, youtube_url = ytdlp_service.download_audio(video_id, settings)
    except ytdlp_service.VideoIdValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ytdlp_service.YtDlpError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"path": str(output_path), "url": youtube_url}
