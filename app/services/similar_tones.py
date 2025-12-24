from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from transformers import ClapModel as HFClapModel, ClapProcessor

from app.config import Settings

logger = logging.getLogger(__name__)


class SimilarTonesPathError(ValueError):
    """Raised when configured paths are invalid."""


class SimilarTonesExecutionError(RuntimeError):
    """Raised when similarity processing fails."""


def create_index(settings: Settings, preset_dir: Path, output_path: Path) -> dict[str, Any]:
    _ensure_directory(preset_dir, "Preset directory not found or not a directory")
    _configure_hf_cache(settings.similar_tones_cache_dir)

    try:
        service = SearchService(settings)
        summary = service.create_index(preset_dir, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create similar-tones index.")
        raise SimilarTonesExecutionError(str(exc)) from exc

    return summary


def search_similar(
    settings: Settings, target_path: Path, index_path: Path, top_k: int
) -> dict[str, Any]:
    _ensure_file(target_path, "Target audio file not found")
    _ensure_file(index_path, "Index file not found")
    if top_k < 1:
        raise SimilarTonesPathError("top_k must be at least 1")

    _configure_hf_cache(settings.similar_tones_cache_dir)

    try:
        service = SearchService(settings)
        results = service.find_similar(target_path, index_path, top_k=top_k)
        formatter = ResultFormatter()
        text = formatter.to_ranked_table(results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to search similar tones.")
        raise SimilarTonesExecutionError(str(exc)) from exc

    return {"text": text, "results": results}


def _configure_hf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))


def _ensure_directory(path: Path, message: str) -> None:
    if not path.is_dir():
        raise SimilarTonesPathError(f"{message}: {path}")


def _ensure_file(path: Path, message: str) -> None:
    if not path.exists():
        raise SimilarTonesPathError(f"{message}: {path}")
    if not path.is_file():
        raise SimilarTonesPathError(f"{message}: {path}")


class AudioLoader:
    TARGET_SAMPLE_RATE = 48000

    def __init__(self) -> None:
        logger.info("AudioLoader initialized")

    def load(self, file_path: Path) -> np.ndarray:
        if not file_path.exists():
            raise SimilarTonesPathError(f"Audio file not found: {file_path}")

        file_extension = file_path.suffix.lower()
        if file_extension == ".wav":
            audio_data, sample_rate = sf.read(str(file_path), dtype="float32")
        elif file_extension == ".ogg":
            audio_segment = AudioSegment.from_ogg(str(file_path))
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio_segment.frame_rate
            if audio_segment.channels == 2:
                audio_data = audio_data.reshape((-1, 2))
            audio_data = audio_data / 32768.0
        else:
            raise SimilarTonesPathError(
                f"Unsupported file format: {file_extension}. Supported: .wav, .ogg"
            )

        return self._convert_to_clap_format(audio_data, sample_rate)

    def _convert_to_clap_format(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)

        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()

        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=self.TARGET_SAMPLE_RATE
            )

        audio_data = audio_data.astype(np.float32)
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val

        return audio_data


class ClapModel:
    MODEL_NAME = "laion/clap-htsat-unfused"
    EMBEDDING_DIMENSION = 512

    def __init__(self, device: str) -> None:
        self.device = self._resolve_device(device)
        self.model: Optional[HFClapModel] = None
        self.processor: Optional[ClapProcessor] = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        requested = device.lower()
        if requested == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available; falling back to CPU.")
            return "cpu"
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return "cpu"

    def _load_model(self) -> None:
        try:
            self.model = HFClapModel.from_pretrained(self.MODEL_NAME)
            self.processor = ClapProcessor.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001
            raise SimilarTonesExecutionError(f"CLAP model loading failed: {exc}") from exc

    def get_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise SimilarTonesExecutionError("CLAP model not loaded")

        try:
            inputs = self.processor(
                audios=audio_data,
                sampling_rate=48000,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_audio_features(**inputs)
            embedding = outputs.cpu().numpy()
            if embedding.shape[0] == 1:
                embedding = embedding[0]
            return embedding.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            raise SimilarTonesExecutionError(f"Embedding generation failed: {exc}") from exc


class VectorStore:
    def save_index(self, vectors: np.ndarray, file_paths: list[str], output_path: Path) -> None:
        if len(vectors) != len(file_paths):
            raise SimilarTonesExecutionError("Vectors count must match file paths count")
        if len(vectors) == 0:
            raise SimilarTonesExecutionError("Cannot save empty index")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_data = {
            "vectors": vectors,
            "file_paths": file_paths,
            "metadata": {
                "count": len(vectors),
                "dimension": vectors.shape[1] if len(vectors.shape) > 1 else vectors.shape[0],
                "dtype": str(vectors.dtype),
            },
        }
        try:
            with output_path.open("wb") as output_file:
                pickle.dump(index_data, output_file)
        except Exception as exc:  # noqa: BLE001
            raise SimilarTonesExecutionError(f"Index saving failed: {exc}") from exc

    def load_index(self, index_path: Path) -> tuple[np.ndarray, list[str]]:
        if not index_path.exists():
            raise SimilarTonesPathError(f"Index file not found: {index_path}")

        try:
            with index_path.open("rb") as index_file:
                index_data = pickle.load(index_file)
        except Exception as exc:  # noqa: BLE001
            raise SimilarTonesExecutionError(f"Index loading failed: {exc}") from exc

        if not isinstance(index_data, dict):
            raise SimilarTonesExecutionError("Invalid index file format")

        for key in ("vectors", "file_paths", "metadata"):
            if key not in index_data:
                raise SimilarTonesExecutionError(f"Invalid index file format: missing {key}")

        vectors = index_data["vectors"]
        file_paths = index_data["file_paths"]
        if not isinstance(vectors, np.ndarray):
            raise SimilarTonesExecutionError("Vectors must be a numpy array")
        if not isinstance(file_paths, list):
            raise SimilarTonesExecutionError("File paths must be a list")
        if len(vectors) != len(file_paths):
            raise SimilarTonesExecutionError("Vectors count must match file paths count")

        return vectors, file_paths

    def search(
        self,
        query_vector: np.ndarray,
        index_vectors: np.ndarray,
        file_paths: list[str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if len(index_vectors) != len(file_paths):
            raise SimilarTonesExecutionError("Index vectors count must match file paths count")
        if len(index_vectors) == 0:
            return []

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise SimilarTonesExecutionError("Query vector cannot be zero vector")

        query_normalized = query_vector / query_norm
        index_norms = np.linalg.norm(index_vectors, axis=1)
        valid_indices = index_norms > 0
        if not np.any(valid_indices):
            return []

        valid_vectors = index_vectors[valid_indices]
        valid_file_paths = [file_paths[i] for i in range(len(file_paths)) if valid_indices[i]]
        valid_norms = index_norms[valid_indices]
        index_normalized = valid_vectors / valid_norms[:, np.newaxis]
        similarities = np.dot(index_normalized, query_normalized)
        sorted_indices = np.argsort(similarities)[::-1]
        top_k = min(top_k, len(sorted_indices))

        results = []
        for i in range(top_k):
            idx = sorted_indices[i]
            results.append(
                {
                    "file_path": valid_file_paths[idx],
                    "similarity_score": float(similarities[idx]),
                    "rank": i + 1,
                }
            )
        return results


class ResultFormatter:
    def to_ranked_table(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No similar presets found.\n"

        lines = []
        lines.append("Rank  Similarity  File Name  Path")
        lines.append("----  ----------  ---------  ----")
        for result in results:
            rank = result["rank"]
            score = result["similarity_score"]
            file_path = result["file_path"]
            file_name = Path(file_path).name
            lines.append(f"{rank:>4}  {score:>10.4f}  {file_name}  {file_path}")
        return "\n".join(lines) + "\n"


class SearchService:
    def __init__(self, settings: Settings) -> None:
        self.audio_loader = AudioLoader()
        self.embedding_model = ClapModel(device=settings.similar_tones_device)
        self.vector_store = VectorStore()

    def create_index(self, preset_dir: Path, output_path: Path) -> dict[str, Any]:
        audio_files = self._find_audio_files(preset_dir)
        if not audio_files:
            raise SimilarTonesExecutionError(f"No audio files found in {preset_dir}")

        embeddings = []
        file_paths = []
        skipped = 0

        for audio_file in audio_files:
            try:
                audio_data = self.audio_loader.load(audio_file)
                embedding = self.embedding_model.get_embedding(audio_data)
                embeddings.append(embedding)
                file_paths.append(str(audio_file))
            except Exception:  # noqa: BLE001
                skipped += 1
                logger.warning("Skipping file due to processing error: %s", audio_file)

        if not embeddings:
            raise SimilarTonesExecutionError("No valid embeddings generated")

        embeddings_array = np.array(embeddings)
        self.vector_store.save_index(embeddings_array, file_paths, output_path)

        return {
            "index_path": str(output_path),
            "indexed_count": len(embeddings_array),
            "skipped_count": skipped,
        }

    def find_similar(
        self, target_path: Path, index_path: Path, top_k: int = 10
    ) -> list[dict[str, Any]]:
        vectors, file_paths = self.vector_store.load_index(index_path)
        target_audio = self.audio_loader.load(target_path)
        target_embedding = self.embedding_model.get_embedding(target_audio)
        search_results = self.vector_store.search(
            target_embedding, vectors, file_paths, top_k=top_k
        )

        formatted_results = []
        for result in search_results:
            file_path = Path(result["file_path"])
            formatted_results.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "similarity_score": result["similarity_score"],
                    "rank": result["rank"],
                }
            )
        return formatted_results

    def _find_audio_files(self, directory: Path) -> list[Path]:
        supported_extensions = [".wav", ".ogg"]
        audio_files = []
        for ext in supported_extensions:
            audio_files.extend(directory.glob(f"**/*{ext}"))
        audio_files.sort(key=lambda path: str(path))
        return audio_files
