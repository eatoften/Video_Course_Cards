from __future__ import annotations

import hashlib
import importlib.metadata
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .metrics import normalize_ocr_text
from .schemas import (
    PageContent,
    PageContentBlock,
    PageReaderKind,
    StablePageReference,
)


class PageReadingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PageReadContext:
    reference: StablePageReference
    image_path: Path
    image_sha256: str


@runtime_checkable
class PageReader(Protocol):
    kind: PageReaderKind
    version: str
    preprocessing_version: str

    def read(self, context: PageReadContext) -> PageContent: ...


@runtime_checkable
class TextDetector(Protocol):
    name: str
    version: str

    def detect(self, image_path: Path) -> list[PageContentBlock]: ...


class SourceUnitLike(Protocol):
    id: str
    asset_id: str
    text: str
    locator: dict[str, object]


def prepare_page_context(
    reference: StablePageReference,
    *,
    image_root: str | Path,
) -> PageReadContext:
    configured_path = Path(reference.image_path)
    image_path = (
        configured_path
        if configured_path.is_absolute()
        else Path(image_root) / configured_path
    ).resolve()
    if not image_path.is_file():
        raise PageReadingError(f"Stable page image does not exist: {image_path}")
    return PageReadContext(
        reference=reference,
        image_path=image_path,
        image_sha256=sha256_file(image_path),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_page_reader_cache_key(
    *,
    image_sha256: str,
    reader: PageReaderKind,
    reader_version: str,
    preprocessing_version: str,
    source_fingerprint: str = "",
) -> str:
    payload = "\0".join(
        (
            image_sha256,
            reader.value,
            reader_version,
            preprocessing_version,
            source_fingerprint,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assemble_page_text(blocks: Sequence[PageContentBlock]) -> str:
    ordered = sorted(blocks, key=lambda block: block.order)
    expected_orders = list(range(len(ordered)))
    if [block.order for block in ordered] != expected_orders:
        raise PageReadingError("Page blocks must have consecutive orders.")
    return "\n".join(block.text for block in ordered)


class GoldReferencePageReader:
    kind = PageReaderKind.gold_reference
    version = "gold-reference-v1"
    preprocessing_version = "manual-cleaning-v1"

    def read(self, context: PageReadContext) -> PageContent:
        started_at = time.perf_counter()
        text = context.reference.gold_text
        return _content_from_blocks(
            context,
            reader=self.kind,
            reader_version=self.version,
            preprocessing_version=self.preprocessing_version,
            blocks=[PageContentBlock(order=0, text=text)],
            source_fingerprint=_text_fingerprint(text),
            latency_seconds=time.perf_counter() - started_at,
        )


class NativeSourcePageReader:
    kind = PageReaderKind.native_source
    preprocessing_version = "source-content-cleaner-v1"

    def __init__(self, source_units: Sequence[SourceUnitLike]) -> None:
        self.version = (
            "source-asset-parser-v1:pypdf-"
            f"{_package_version('pypdf')}"
        )
        self._units_by_page: dict[int, SourceUnitLike] = {}
        for unit in source_units:
            page_number = unit.locator.get("page_number")
            if not isinstance(page_number, int):
                continue
            if page_number in self._units_by_page:
                raise PageReadingError(
                    f"Multiple source units claim page {page_number}."
                )
            self._units_by_page[page_number] = unit

    def read(self, context: PageReadContext) -> PageContent:
        started_at = time.perf_counter()
        page_number = context.reference.page_number
        if page_number is None:
            return _abstained_content(
                context,
                reader=self.kind,
                reader_version=self.version,
                preprocessing_version=self.preprocessing_version,
                reason="Reference has no source page number.",
                latency_seconds=time.perf_counter() - started_at,
            )

        unit = self._units_by_page.get(page_number)
        if unit is None:
            return _abstained_content(
                context,
                reader=self.kind,
                reader_version=self.version,
                preprocessing_version=self.preprocessing_version,
                reason=f"No source unit matches page {page_number}.",
                latency_seconds=time.perf_counter() - started_at,
            )

        text = clean_native_source_text(unit.text)
        if not text:
            return _abstained_content(
                context,
                reader=self.kind,
                reader_version=self.version,
                preprocessing_version=self.preprocessing_version,
                reason=f"Source unit {unit.id} contains no text.",
                source_asset_id=unit.asset_id,
                source_unit_id=unit.id,
                latency_seconds=time.perf_counter() - started_at,
            )

        return _content_from_blocks(
            context,
            reader=self.kind,
            reader_version=self.version,
            preprocessing_version=self.preprocessing_version,
            blocks=[PageContentBlock(order=0, text=text)],
            source_asset_id=unit.asset_id,
            source_unit_id=unit.id,
            source_fingerprint=_text_fingerprint(f"{unit.id}\0{text}"),
            latency_seconds=time.perf_counter() - started_at,
        )


class RapidOcrPageReader:
    kind = PageReaderKind.rapidocr
    preprocessing_version = (
        "content-roi-v1:min-height-0.015:footer-0.90:watermark"
    )

    def __init__(self, engine: Any | None = None) -> None:
        self.version = f"rapidocr-{_package_version('rapidocr')}:ppocrv6-small"
        if engine is None:
            from rapidocr import RapidOCR

            engine = RapidOCR()
        self._engine = engine

    def read(self, context: PageReadContext) -> PageContent:
        started_at = time.perf_counter()
        try:
            result = self._engine(str(context.image_path))
        except Exception as exc:
            raise PageReadingError(
                f"RapidOCR failed for {context.image_path}: {exc}"
            ) from exc

        texts = tuple(result.txts) if result.txts is not None else ()
        boxes = tuple(result.boxes) if result.boxes is not None else ()
        scores = tuple(result.scores) if result.scores is not None else ()
        if texts and len(boxes) != len(texts):
            raise PageReadingError("RapidOCR returned mismatched text and boxes.")
        if scores and len(scores) != len(texts):
            raise PageReadingError("RapidOCR returned mismatched text and scores.")

        image_shape = getattr(getattr(result, "img", None), "shape", None)
        image_height = int(image_shape[0]) if image_shape is not None else None
        image_width = int(image_shape[1]) if image_shape is not None else None

        blocks: list[PageContentBlock] = []
        for index, text in enumerate(texts):
            cleaned_text = str(text).strip()
            if not cleaned_text:
                continue
            polygon = [
                (float(point[0]), float(point[1]))
                for point in boxes[index]
            ]
            if not _is_semantic_content_block(
                cleaned_text,
                polygon,
                image_width=image_width,
                image_height=image_height,
            ):
                continue
            confidence = float(scores[index]) if scores else None
            blocks.append(
                PageContentBlock(
                    order=len(blocks),
                    text=cleaned_text,
                    polygon=polygon,
                    confidence=confidence,
                )
            )

        elapsed = time.perf_counter() - started_at
        if not blocks:
            return _abstained_content(
                context,
                reader=self.kind,
                reader_version=self.version,
                preprocessing_version=self.preprocessing_version,
                reason="RapidOCR detected no readable text.",
                latency_seconds=elapsed,
            )

        confidences = [
            block.confidence
            for block in blocks
            if block.confidence is not None
        ]
        return _content_from_blocks(
            context,
            reader=self.kind,
            reader_version=self.version,
            preprocessing_version=self.preprocessing_version,
            blocks=blocks,
            confidence=(
                sum(confidences) / len(confidences)
                if confidences
                else None
            ),
            latency_seconds=elapsed,
        )


def _content_from_blocks(
    context: PageReadContext,
    *,
    reader: PageReaderKind,
    reader_version: str,
    preprocessing_version: str,
    blocks: list[PageContentBlock],
    latency_seconds: float,
    source_asset_id: str | None = None,
    source_unit_id: str | None = None,
    source_fingerprint: str = "",
    confidence: float | None = None,
) -> PageContent:
    raw_text = assemble_page_text(blocks)
    return PageContent(
        page_event_id=context.reference.page_event_id,
        lecture_id=context.reference.lecture_id,
        page_number=context.reference.page_number,
        stable_frame_timestamp=context.reference.stable_frame_timestamp,
        image_path=context.reference.image_path,
        image_sha256=context.image_sha256,
        reader=reader,
        reader_version=reader_version,
        preprocessing_version=preprocessing_version,
        cache_key=compute_page_reader_cache_key(
            image_sha256=context.image_sha256,
            reader=reader,
            reader_version=reader_version,
            preprocessing_version=preprocessing_version,
            source_fingerprint=source_fingerprint,
        ),
        raw_text=raw_text,
        normalized_text=normalize_ocr_text(raw_text, case_sensitive=True),
        ordered_blocks=blocks,
        source_asset_id=source_asset_id,
        source_unit_id=source_unit_id,
        confidence=confidence,
        latency_seconds=latency_seconds,
    )


def _abstained_content(
    context: PageReadContext,
    *,
    reader: PageReaderKind,
    reader_version: str,
    preprocessing_version: str,
    reason: str,
    latency_seconds: float,
    source_asset_id: str | None = None,
    source_unit_id: str | None = None,
) -> PageContent:
    return PageContent(
        page_event_id=context.reference.page_event_id,
        lecture_id=context.reference.lecture_id,
        page_number=context.reference.page_number,
        stable_frame_timestamp=context.reference.stable_frame_timestamp,
        image_path=context.reference.image_path,
        image_sha256=context.image_sha256,
        reader=reader,
        reader_version=reader_version,
        preprocessing_version=preprocessing_version,
        cache_key=compute_page_reader_cache_key(
            image_sha256=context.image_sha256,
            reader=reader,
            reader_version=reader_version,
            preprocessing_version=preprocessing_version,
        ),
        latency_seconds=latency_seconds,
        abstained=True,
        abstention_reason=reason,
        source_asset_id=source_asset_id,
        source_unit_id=source_unit_id,
    )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SOURCE_HEADER_PATTERN = re.compile(
    r"^.*\bLecture\s+\d+\s*-\s*[A-Za-z]+\s+\d{1,2},\s+\d{4}\d*$",
    re.IGNORECASE,
)
_ATTRIBUTION_MARKERS = (
    "this image",
    "image source:",
    "licensed under",
    "is licensed",
    "under cc-",
    "public domain",
    "creativecommons",
    "http://",
    "https://",
    "utm_",
)


def clean_native_source_text(
    text: str,
    *,
    discard_numeric_lines: bool = True,
) -> str:
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        if not line or _SOURCE_HEADER_PATTERN.match(line):
            continue
        if discard_numeric_lines and line.isdigit():
            continue
        if any(marker in lowered for marker in _ATTRIBUTION_MARKERS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _is_semantic_content_block(
    text: str,
    polygon: Sequence[tuple[float, float]],
    *,
    image_width: int | None,
    image_height: int | None,
) -> bool:
    lowered = text.casefold()
    if any(marker in lowered for marker in _ATTRIBUTION_MARKERS):
        return False
    if image_width is None or image_height is None:
        return True

    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    block_height = max(ys) - min(ys)
    top = min(ys)
    left = min(xs)
    if block_height < image_height * 0.015:
        return False
    if top >= image_height * 0.90:
        return False
    if top >= image_height * 0.80 and left >= image_width * 0.65:
        return False
    return True
