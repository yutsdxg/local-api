from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import Settings

try:
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError as exc:  # pragma: no cover - handled at runtime
    google = None  # type: ignore[assignment]
    service_account = None  # type: ignore[assignment]
    OAuthCredentials = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]
    HttpError = Exception  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


SCOPES = (
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
)


class GoogleDocsDependencyError(RuntimeError):
    """Raised when Google API dependencies are missing."""


class GoogleDocsPathError(ValueError):
    """Raised when source markdown path is invalid."""


class GoogleDocsAuthError(RuntimeError):
    """Raised when Google API authentication fails."""


class GoogleDocsUploadError(RuntimeError):
    """Raised when Google Docs API calls fail."""


@dataclass(frozen=True)
class GoogleDocsExportResult:
    source_path: str
    document_id: str
    document_url: str
    title: str
    folder_id: str | None


def export_markdown_to_google_docs(
    settings: Settings,
    source_path: Path,
    *,
    title: str | None = None,
    folder_id: str | None = None,
) -> GoogleDocsExportResult:
    docs_service, drive_service = _build_services(settings)
    return _export_markdown_with_services(
        settings,
        docs_service,
        drive_service,
        source_path,
        title=title,
        folder_id=folder_id,
    )


def export_markdown_directory_to_google_docs(
    settings: Settings,
    source_dir: Path,
    *,
    folder_id: str | None = None,
) -> list[GoogleDocsExportResult]:
    docs_service, drive_service = _build_services(settings)
    resolved_dir = _resolve_source_dir(settings.obsidian_export_dir, source_dir)
    results: list[GoogleDocsExportResult] = []
    for markdown_path in sorted(resolved_dir.rglob("*.md")):
        if not markdown_path.is_file():
            continue
        results.append(
            _export_markdown_with_services(
                settings,
                docs_service,
                drive_service,
                markdown_path,
                title=None,
                folder_id=folder_id,
            )
        )
    return results


def _export_markdown_with_services(
    settings: Settings,
    docs_service: object,
    drive_service: object,
    source_path: Path,
    *,
    title: str | None,
    folder_id: str | None,
) -> GoogleDocsExportResult:
    export_dir = settings.obsidian_export_dir
    resolved_path = _resolve_source_path(export_dir, source_path)
    markdown_text = resolved_path.read_text(encoding="utf-8")
    document_title = title or resolved_path.stem
    target_folder_id = folder_id or settings.google_docs_folder_id

    document_id = _create_or_replace_document(
        docs_service,
        drive_service,
        document_title,
        target_folder_id,
        markdown_text,
    )

    return GoogleDocsExportResult(
        source_path=str(resolved_path),
        document_id=document_id,
        document_url=f"https://docs.google.com/document/d/{document_id}/edit",
        title=document_title,
        folder_id=target_folder_id,
    )


def _resolve_source_path(export_dir: Path, source_path: Path) -> Path:
    if source_path.is_absolute():
        candidate = source_path
    else:
        candidate = export_dir / source_path

    candidate = candidate.expanduser().resolve()
    export_root = export_dir.expanduser().resolve()

    try:
        candidate.relative_to(export_root)
    except ValueError as exc:
        raise GoogleDocsPathError(
            f"Source path must be within export directory: {export_root}"
        ) from exc

    if not candidate.is_file():
        raise GoogleDocsPathError(f"Source markdown file not found: {candidate}")
    if candidate.suffix.lower() != ".md":
        raise GoogleDocsPathError(f"Source file must be a markdown file: {candidate}")

    return candidate


def _resolve_source_dir(export_dir: Path, source_dir: Path) -> Path:
    if source_dir.is_absolute():
        candidate = source_dir
    else:
        candidate = export_dir / source_dir

    candidate = candidate.expanduser().resolve()
    export_root = export_dir.expanduser().resolve()

    try:
        candidate.relative_to(export_root)
    except ValueError as exc:
        raise GoogleDocsPathError(
            f"Source directory must be within export directory: {export_root}"
        ) from exc

    if not candidate.is_dir():
        raise GoogleDocsPathError(f"Source directory not found: {candidate}")

    return candidate


def _build_services(settings: Settings) -> tuple[object, object]:
    if _IMPORT_ERROR is not None:
        raise GoogleDocsDependencyError(
            "Google API dependencies are missing. Install google-auth and google-api-python-client."
        )

    try:
        credentials = _load_credentials(settings)
    except Exception as exc:  # pragma: no cover - auth errors depend on environment
        raise GoogleDocsAuthError(f"Failed to load Google credentials: {exc}") from exc

    docs_service = build("docs", "v1", credentials=credentials, cache_discovery=False)
    drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return docs_service, drive_service


def _load_credentials(settings: Settings) -> object:
    if settings.google_oauth_refresh_token:
        if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
            raise GoogleDocsAuthError(
                "OAuth refresh token provided but client ID/secret are missing."
            )

        credentials = OAuthCredentials(
            None,
            refresh_token=settings.google_oauth_refresh_token,
            token_uri=settings.google_oauth_token_uri,
            client_id=settings.google_oauth_client_id,
            client_secret=settings.google_oauth_client_secret,
            scopes=SCOPES,
        )
        credentials.refresh(Request())
        return credentials

    if settings.google_docs_credentials_path:
        return service_account.Credentials.from_service_account_file(
            settings.google_docs_credentials_path, scopes=SCOPES
        )

    credentials, _ = google.auth.default(scopes=SCOPES)
    return credentials


def _create_or_replace_document(
    docs_service: object,
    drive_service: object,
    title: str,
    folder_id: str | None,
    text: str,
) -> str:
    try:
        if folder_id:
            existing_id = _find_existing_document(drive_service, title, folder_id)
            if existing_id:
                _clear_document(docs_service, existing_id)
                _insert_text(docs_service, existing_id, text)
                return existing_id

            file = (
                drive_service.files()
                .create(
                    body={
                        "name": title,
                        "mimeType": "application/vnd.google-apps.document",
                        "parents": [folder_id],
                    },
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            document_id = str(file.get("id"))
            _insert_text(docs_service, document_id, text)
            return document_id

        document = docs_service.documents().create(body={"title": title}).execute()
        document_id = str(document.get("documentId"))
        _insert_text(docs_service, document_id, text)
        return document_id
    except HttpError as exc:  # pragma: no cover - requires Google API
        raise GoogleDocsUploadError(f"Failed to create Google Doc: {exc}") from exc


def _insert_text(docs_service: object, document_id: str, text: str) -> None:
    if not text:
        return

    for batch in _build_insert_batches(text):
        try:
            docs_service.documents().batchUpdate(
                documentId=document_id, body={"requests": batch}
            ).execute()
        except HttpError as exc:  # pragma: no cover - requires Google API
            raise GoogleDocsUploadError(f"Failed to insert content: {exc}") from exc


def _build_insert_batches(text: str, *, chunk_size: int = 10000) -> Iterable[list[dict]]:
    index = 1
    batch: list[dict] = []
    for chunk in _chunk_text(text, chunk_size):
        batch.append({"insertText": {"location": {"index": index}, "text": chunk}})
        index += len(chunk)
        if len(batch) >= 100:
            yield batch
            batch = []
    if batch:
        yield batch


def _chunk_text(text: str, chunk_size: int) -> Iterable[str]:
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def _find_existing_document(drive_service: object, title: str, folder_id: str) -> str | None:
    safe_title = title.replace("'", "\\'")
    query = (
        "mimeType='application/vnd.google-apps.document' "
        f"and name='{safe_title}' "
        f"and '{folder_id}' in parents "
        "and trashed=false"
    )
    files = (
        drive_service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        .execute()
        .get("files", [])
    )
    if not files:
        return None
    return str(files[0].get("id"))


def _clear_document(docs_service: object, document_id: str) -> None:
    document = docs_service.documents().get(documentId=document_id).execute()
    body = document.get("body", {})
    content = body.get("content", [])
    if not content:
        return
    end_index = content[-1].get("endIndex", 1)
    if end_index <= 1:
        return
    docs_service.documents().batchUpdate(
        documentId=document_id,
        body={
            "requests": [
                {
                    "deleteContentRange": {
                        "range": {"startIndex": 1, "endIndex": end_index - 1}
                    }
                }
            ]
        },
    ).execute()
