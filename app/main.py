from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.config import Settings
from app.services import chord_melody as chord_melody_service
from app.services import obsidian_exports as obsidian_exports_service
from app.services import obsidian_google_docs as obsidian_google_docs_service
from app.services import rsync as rsync_service
from app.services import similar_tones as similar_tones_service
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


@app.post("/rsync")
async def sync_directories(
    src: str = Query(..., description="Source root directory"),
    dst: str = Query(..., description="Destination root directory"),
) -> dict[str, list[rsync_service.SyncResult]]:
    """Endpoint that performs rsync per first-level subdirectory."""

    try:
        synced = rsync_service.sync_directories(Path(src), Path(dst), settings.rsync_bin)
    except rsync_service.RsyncPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except rsync_service.RsyncExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"synced": synced}


@app.post("/audio/analyze/chord-melody")
async def analyze_chord_melody() -> dict[str, list[dict[str, str]]]:
    """Analyze chord/melody classification for .wav files and rename them."""

    try:
        results = chord_melody_service.analyze_chord_melody(settings)
    except chord_melody_service.ChordMelodyPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except chord_melody_service.ChordMelodyExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"results": results}


@app.post("/audio/similar-tones/index")
async def create_similar_tones_index(
    preset_dir: str = Query(..., description="Directory with preset audio files"),
    output_path: str = Query(..., description="Path to output index file"),
) -> dict[str, str | int]:
    """Create an index for similar-tones search."""

    try:
        summary = similar_tones_service.create_index(
            settings, Path(preset_dir), Path(output_path)
        )
    except similar_tones_service.SimilarTonesPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except similar_tones_service.SimilarTonesExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return summary


@app.post("/audio/similar-tones/search")
async def search_similar_tones(
    target_path: str = Query(..., description="Target audio file path"),
    index_path: str = Query(..., description="Index file path(s), comma-separated"),
    top_k: int = Query(10, ge=1, description="Number of similar presets to return"),
) -> dict[str, object]:
    """Search for similar preset tones based on a target audio file."""

    try:
        index_paths = [Path(path.strip()) for path in index_path.split(",") if path.strip()]
        response = similar_tones_service.search_similar(
            settings, Path(target_path), index_paths, top_k
        )
    except similar_tones_service.SimilarTonesPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except similar_tones_service.SimilarTonesExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return response


@app.post("/obsidian/merge")
async def merge_obsidian_exports() -> dict[str, object]:
    """Merge Obsidian markdown files into grouped export files."""

    try:
        summary = obsidian_exports_service.merge_exports(settings)
    except obsidian_exports_service.ObsidianExportPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except obsidian_exports_service.ObsidianFrontMatterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safeguard for unexpected failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return summary


@app.post("/obsidian/exports/google-docs")
async def export_obsidian_exports_to_google_docs(
    source_path: str = Query(..., description="Merged markdown file path"),
    title: str | None = Query(None, description="Google Docs title override"),
    folder_id: str | None = Query(None, description="Google Drive folder ID"),
) -> dict[str, object]:
    """Convert merged Obsidian markdown file(s) to Google Docs documents."""

    try:
        candidate = Path(source_path)
        if not candidate.is_absolute():
            candidate = settings.obsidian_export_dir / candidate
        if candidate.is_dir():
            results = obsidian_google_docs_service.export_markdown_directory_to_google_docs(
                settings, Path(source_path), folder_id=folder_id
            )
            return {
                "source_dir": source_path,
                "count": len(results),
                "documents": [result.__dict__ for result in results],
            }

        result = obsidian_google_docs_service.export_markdown_to_google_docs(
            settings, Path(source_path), title=title, folder_id=folder_id
        )
    except obsidian_google_docs_service.GoogleDocsPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except obsidian_google_docs_service.GoogleDocsDependencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except obsidian_google_docs_service.GoogleDocsAuthError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except obsidian_google_docs_service.GoogleDocsUploadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safeguard for unexpected failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "source_path": result.source_path,
        "document_id": result.document_id,
        "document_url": result.document_url,
        "title": result.title,
        "folder_id": result.folder_id,
        "skipped": result.skipped,
    }
