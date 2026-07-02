from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .course import DEFAULT_COURSE_TITLE
from .course_store import get_course
from .job import VideoJob
from .job_service import get_video_job
from .job_store import get_job
from .knowledge_card import KnowledgeCard
from .knowledge_card_store import list_cards, list_cards_for_job
from .markdown_export import (
    MarkdownCardSource,
    render_card_markdown,
    safe_markdown_filename,
    safe_path_component,
)
from .settings import get_app_path_settings


DEFAULT_EXPORT_DIR = Path(
    os.environ.get(
        "VCC_EXPORT_DIR",
        str(get_app_path_settings().export_dir),
    )
)
JOB_FOLDER_MANIFEST = ".vcc-job-export-manifest.json"
VAULT_FOLDER_MANIFEST = ".vcc-vault-export-manifest.json"


@dataclass(frozen=True)
class MarkdownArchive:
    filename: str
    content: bytes
    media_type: str = "application/zip"


@dataclass(frozen=True)
class SavedMarkdownArchive:
    filename: str
    path: str
    byte_count: int


@dataclass(frozen=True)
class SavedMarkdownFolder:
    root_path: str
    file_count: int
    files: list[str]


@dataclass(frozen=True)
class CardExportRecord:
    card: KnowledgeCard
    job: VideoJob
    course_title: str
    video_title: str


def export_job_cards_markdown(job_id: str) -> MarkdownArchive:
    job = get_video_job(job_id)
    records = [
        _build_record(card)
        for card in list_cards_for_job(job_id)
    ]

    root_name = safe_path_component(
        f"job-{job.id}-cards",
        fallback="job-cards",
    )
    entries = _job_archive_entries(records)

    return MarkdownArchive(
        filename=f"{root_name}.zip",
        content=_build_zip(
            entries=entries,
            readme=_build_readme(
                title="Job Card Export",
                description=(
                    "Markdown cards exported for one video processing job."
                ),
                card_count=len(records),
                scope=f"Job: {job.id}",
            ),
        ),
    )


def export_all_cards_markdown() -> MarkdownArchive:
    records = [
        _build_record(card)
        for card in list_cards()
        if get_job(card.job_id) is not None
    ]

    entries = _vault_archive_entries(records)

    return MarkdownArchive(
        filename="video-course-cards-vault.zip",
        content=_build_zip(
            entries=entries,
            readme=_build_readme(
                title="Video Course Cards Vault",
                description=(
                    "Obsidian-friendly Markdown vault exported from saved "
                    "knowledge cards."
                ),
                card_count=len(records),
                scope="All saved cards",
            ),
        ),
    )


def save_archive_to_disk(
    archive: MarkdownArchive,
    export_dir: Path | None = None,
) -> SavedMarkdownArchive:
    target_dir = export_dir or DEFAULT_EXPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = _unique_file_path(target_dir, archive.filename)
    target_path.write_bytes(archive.content)

    return SavedMarkdownArchive(
        filename=target_path.name,
        path=str(target_path),
        byte_count=len(archive.content),
    )


def save_job_cards_markdown_folder(
    job_id: str,
    export_dir: Path | None = None,
) -> SavedMarkdownFolder:
    job = get_video_job(job_id)
    records = [
        _build_record(card)
        for card in list_cards_for_job(job_id)
    ]
    target_root = export_dir or DEFAULT_EXPORT_DIR
    video_dir = safe_path_component(
        job.original_filename or job.stored_name or job.id,
        fallback=job.id,
    )
    target_dir = target_root / video_dir
    entries = _job_folder_entries(records)
    files = _write_folder_snapshot(
        target_dir=target_dir,
        entries=entries,
        manifest_name=JOB_FOLDER_MANIFEST,
    )

    return SavedMarkdownFolder(
        root_path=str(target_dir),
        file_count=len(files),
        files=files,
    )


def save_all_cards_markdown_folder(
    export_dir: Path | None = None,
) -> SavedMarkdownFolder:
    records = [
        _build_record(card)
        for card in list_cards()
        if get_job(card.job_id) is not None
    ]
    target_dir = export_dir or DEFAULT_EXPORT_DIR
    entries = _vault_archive_entries(records)
    files = _write_folder_snapshot(
        target_dir=target_dir,
        entries=entries,
        manifest_name=VAULT_FOLDER_MANIFEST,
    )

    return SavedMarkdownFolder(
        root_path=str(target_dir),
        file_count=len(files),
        files=files,
    )


def _build_record(card: KnowledgeCard) -> CardExportRecord:
    job = get_job(card.job_id)

    if job is None:
        raise ValueError(f"Card {card.id} has no matching job.")

    course = get_course(job.course_id)
    course_title = (
        course.title
        if course is not None
        else DEFAULT_COURSE_TITLE
    )
    video_title = job.original_filename or job.stored_name or job.id

    return CardExportRecord(
        card=card,
        job=job,
        course_title=course_title,
        video_title=video_title,
    )


def _job_archive_entries(
    records: list[CardExportRecord],
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    used_paths: set[str] = set()

    for index, record in enumerate(records, start=1):
        path = _unique_markdown_path(
            directory="cards",
            index=index,
            title=record.card.title,
            used_paths=used_paths,
        )
        entries.append((path, _render_record(record)))

    return entries


def _job_folder_entries(
    records: list[CardExportRecord],
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    used_paths: set[str] = set()

    for index, record in enumerate(records, start=1):
        path = _unique_markdown_path(
            directory="",
            index=index,
            title=record.card.title,
            used_paths=used_paths,
        )
        entries.append((path, _render_record(record)))

    return entries


def _vault_archive_entries(
    records: list[CardExportRecord],
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    used_paths: set[str] = set()

    for index, record in enumerate(records, start=1):
        course_dir = safe_path_component(
            record.course_title,
            fallback=DEFAULT_COURSE_TITLE,
        )
        video_dir = safe_path_component(
            record.video_title,
            fallback=record.job.id,
        )
        path = _unique_markdown_path(
            directory=f"{course_dir}/{video_dir}",
            index=index,
            title=record.card.title,
            used_paths=used_paths,
        )
        entries.append((path, _render_record(record)))

    return entries


def _render_record(record: CardExportRecord) -> str:
    return render_card_markdown(
        record.card,
        MarkdownCardSource(
            course_title=record.course_title,
            video_title=record.video_title,
            job_id=record.job.id,
        ),
    )


def _unique_markdown_path(
    *,
    directory: str,
    index: int,
    title: str,
    used_paths: set[str],
) -> str:
    filename = safe_markdown_filename(title, fallback="card")
    base_name = filename.removesuffix(".md")
    prefix = f"{directory}/" if directory else ""
    candidate = f"{prefix}{index:04d}-{filename}"
    suffix = 2

    while candidate in used_paths:
        candidate = f"{prefix}{index:04d}-{base_name}-{suffix}.md"
        suffix += 1

    used_paths.add(candidate)

    return candidate


def _build_zip(
    *,
    entries: list[tuple[str, str]],
    readme: str,
) -> bytes:
    buffer = BytesIO()

    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("README.md", readme)

        for path, content in entries:
            archive.writestr(path, content)

    return buffer.getvalue()


def _write_folder_snapshot(
    *,
    target_dir: Path,
    entries: list[tuple[str, str]],
    manifest_name: str,
) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()

    _delete_previous_snapshot_files(
        root=root,
        manifest_path=root / manifest_name,
    )

    files: list[str] = []

    for relative_path, content in entries:
        target_path = _resolve_export_path(root, relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        files.append(relative_path)

    _write_manifest(root / manifest_name, files)

    return files


def _delete_previous_snapshot_files(
    *,
    root: Path,
    manifest_path: Path,
) -> None:
    for relative_path in _read_manifest_files(manifest_path):
        target_path = _resolve_export_path(root, relative_path)

        if target_path.is_file():
            target_path.unlink()
            _remove_empty_parent_dirs(target_path.parent, stop_at=root)


def _read_manifest_files(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        return []

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    raw_files = payload.get("files") if isinstance(payload, dict) else None

    if not isinstance(raw_files, list):
        return []

    return [
        str(path)
        for path in raw_files
        if isinstance(path, str) and path.strip()
    ]


def _write_manifest(manifest_path: Path, files: list[str]) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "source": "video-course-cards",
                "mode": "sqlite-snapshot",
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_export_path(root: Path, relative_path: str) -> Path:
    target_path = root.joinpath(*relative_path.split("/")).resolve()
    target_path.relative_to(root)

    return target_path


def _remove_empty_parent_dirs(parent: Path, stop_at: Path) -> None:
    current = parent.resolve()
    root = stop_at.resolve()

    while current != root:
        current.relative_to(root)

        try:
            current.rmdir()
        except OSError:
            return

        current = current.parent


def _build_readme(
    *,
    title: str,
    description: str,
    card_count: int,
    scope: str,
) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            description,
            "",
            f"- Scope: {scope}",
            f"- Cards: {card_count}",
            "",
            "Each Markdown card includes source video metadata, timestamps, "
            "claims, evidence quotes, and active recall fields.",
            "",
        ]
    )


def _unique_file_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename

    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1

    while True:
        next_candidate = directory / f"{stem} ({index}){suffix}"

        if not next_candidate.exists():
            return next_candidate

        index += 1
