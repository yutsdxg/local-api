from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from typing_extensions import TypedDict

from app.config import Settings


class ExportSummary(TypedDict):
    vault_root: str
    output_dir: str
    generated: list["GeneratedFile"]
    skipped: "SkippedCounts"


class GeneratedFile(TypedDict):
    group: str
    output_path: str
    note_count: int


class SkippedCounts(TypedDict):
    excluded_tag: int
    no_target_tag: int
    parse_error: int


class ObsidianExportPathError(ValueError):
    """Raised when vault or output paths are invalid."""


class ObsidianFrontMatterError(ValueError):
    """Raised when front matter parsing fails."""


@dataclass(frozen=True)
class Note:
    source: Path
    title: str
    date: str
    tags: list[str]
    body: str

    @property
    def date_sort_key(self) -> str:
        return self.date or "0000-00-00"


@dataclass(frozen=True)
class ParseResult:
    front_matter: dict[str, object]
    body: str


def merge_exports(settings: Settings) -> ExportSummary:
    vault_root = settings.obsidian_vault_root
    output_dir = settings.obsidian_export_dir

    _ensure_directory(vault_root, "Vault root not found or not a directory")
    _ensure_target_directories(vault_root, settings.obsidian_target_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_directory(output_dir, "Output directory not found or not a directory")

    excluded_tags = set(settings.obsidian_exclude_tags)
    groups: dict[str, list[Note]] = {}
    skipped_excluded = 0
    skipped_no_target = 0
    skipped_parse_error = 0

    for note_path in _iter_markdown_files(vault_root, settings.obsidian_target_dirs):
        try:
            note = _load_note(note_path, vault_root)
        except ObsidianFrontMatterError:
            skipped_parse_error += 1
            continue

        if excluded_tags.intersection(note.tags):
            skipped_excluded += 1
            continue

        group_names = _determine_groups(note.tags, settings)
        if not group_names:
            skipped_no_target += 1
            continue

        for group in group_names:
            groups.setdefault(group, []).append(note)

    generated: list[GeneratedFile] = []
    for group, notes in sorted(groups.items(), key=lambda item: item[0]):
        ordered_notes = _order_notes(notes)
        output_path = output_dir / f"{_group_to_filename(group, settings)}.md"
        _write_group_file(output_path, ordered_notes)
        generated.append(
            {
                "group": group,
                "output_path": str(output_path),
                "note_count": len(notes),
            }
        )

    return {
        "vault_root": str(vault_root),
        "output_dir": str(output_dir),
        "generated": generated,
        "skipped": {
            "excluded_tag": skipped_excluded,
            "no_target_tag": skipped_no_target,
            "parse_error": skipped_parse_error,
        },
    }


def _iter_markdown_files(vault_root: Path, target_dirs: Iterable[str]) -> Iterable[Path]:
    for target_dir in target_dirs:
        base = vault_root / target_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            if path.is_file():
                yield path


def _ensure_target_directories(vault_root: Path, target_dirs: Iterable[str]) -> None:
    for target_dir in target_dirs:
        base = vault_root / target_dir
        if not base.is_dir():
            raise ObsidianExportPathError(
                f"Target directory not found or not a directory: {base}"
            )


def _load_note(path: Path, vault_root: Path) -> Note:
    content = path.read_text(encoding="utf-8")
    parse_result = _parse_front_matter(content)
    front_matter = parse_result.front_matter

    tags = _normalize_tags(front_matter.get("tags"))
    date_value = front_matter.get("date")
    date_str = "" if date_value is None else str(date_value)
    title = str(front_matter.get("title") or path.stem)

    return Note(
        source=path.relative_to(vault_root),
        title=title,
        date=date_str,
        tags=tags,
        body=parse_result.body,
    )


def _parse_front_matter(text: str) -> ParseResult:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ParseResult(front_matter={}, body=text.strip())

    end_index = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = idx
            break

    if end_index is None:
        return ParseResult(front_matter={}, body=text.strip())

    yaml_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")

    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise ObsidianFrontMatterError("Failed to parse front matter") from exc

    if not isinstance(data, dict):
        raise ObsidianFrontMatterError("Front matter must be a mapping")

    return ParseResult(front_matter=data, body=body)


def _normalize_tags(raw_tags: object) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        return [raw_tags]
    if isinstance(raw_tags, list):
        return [str(tag) for tag in raw_tags]
    return [str(raw_tags)]


def _determine_groups(tags: list[str], settings: Settings) -> list[str]:
    if settings.obsidian_journal_tag in tags:
        return [settings.obsidian_journal_tag]

    topic_groups = _extract_topic_groups(tags, settings.obsidian_topic_prefix)
    if topic_groups:
        return topic_groups

    return [settings.obsidian_others_group_name]


def _extract_topic_groups(tags: list[str], prefix: str) -> list[str]:
    groups: list[str] = []
    for tag in tags:
        if not tag.startswith(prefix):
            continue
        remainder = tag[len(prefix) :].lstrip("/")
        if not remainder:
            continue
        direct = remainder.split("/")[0]
        group = f"{prefix}{direct}"
        if group not in groups:
            groups.append(group)
    return groups


def _group_to_filename(group: str, settings: Settings) -> str:
    if group == settings.obsidian_journal_tag:
        base = "journal"
    else:
        base = group

    base = base.replace("/", "_")
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    return base or "exports"


def _order_notes(notes: list[Note]) -> list[Note]:
    ordered = sorted(notes, key=lambda note: note.source.as_posix())
    ordered = sorted(ordered, key=lambda note: note.date_sort_key, reverse=True)
    return ordered


def _write_group_file(output_path: Path, notes: list[Note]) -> None:
    sections: list[str] = []
    for note in notes:
        sections.append(_render_note(note))

    output_path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _render_note(note: Note) -> str:
    header_lines = ["---", f"source: {note.source.as_posix()}", f"title: {note.title}"]
    if note.date:
        header_lines.append(f"date: {note.date}")
    if note.tags:
        header_lines.append("tags:")
        header_lines.extend([f"  - {tag}" for tag in note.tags])
    header_lines.append("---")

    body = note.body.strip()
    if body:
        return "\n".join(header_lines + ["", body])
    return "\n".join(header_lines)


def _ensure_directory(path: Path, message: str) -> None:
    if not path.is_dir():
        raise ObsidianExportPathError(f"{message}: {path}")
