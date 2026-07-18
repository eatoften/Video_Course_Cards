from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Sequence
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .ctc_text import normalize_line_text
from .line_crop_dataset import (
    LineCropDatasetError,
    line_crop_sample_id,
    polygon_to_pixel_crop,
    portable_dataset_path,
    resolve_dataset_path,
)
from .page_reading import sha256_file
from .schemas import (
    GoldTextScope,
    LineCropReviewRecord,
    LineCropSample,
    LineLabelSource,
    LineReviewDecision,
    PageContent,
    PageReaderKind,
    PixelCrop,
    StablePageReference,
)


SOURCE_ALIGNED_REVIEW_METHOD = "official-deck-and-stable-frame-review-v1"


def make_line_review_template(
    contents: Sequence[PageContent],
) -> list[LineCropReviewRecord]:
    """Enumerate every detector polygon without using recognition correctness."""

    records: list[LineCropReviewRecord] = []
    seen: set[tuple[str, int]] = set()
    for content in contents:
        for block in content.ordered_blocks:
            if block.polygon is None:
                continue
            key = (content.page_event_id, block.order)
            if key in seen:
                raise LineCropDatasetError(f"Duplicate review candidate: {key}")
            seen.add(key)
            records.append(
                LineCropReviewRecord(
                    page_event_id=content.page_event_id,
                    page_content_cache_key=content.cache_key,
                    source_image_sha256=content.image_sha256,
                    source_block_order=block.order,
                    detected_text=block.text,
                )
            )
    if not records:
        raise LineCropDatasetError("No detector polygons are available for review.")
    return records


def build_source_aligned_line_crops(
    references: Sequence[StablePageReference],
    contents: Sequence[PageContent],
    reviews: Sequence[LineCropReviewRecord],
    *,
    image_root: str | Path,
    output_dir: str | Path,
    padding_pixels: int = 2,
) -> list[LineCropSample]:
    """Crop all reviewed blocks and attach labels independent of OCR text."""

    if padding_pixels < 0:
        raise LineCropDatasetError("padding_pixels cannot be negative.")
    references_by_id = _index_unique(
        references,
        key=lambda item: item.page_event_id,
        label="reference page_event_id",
    )
    contents_by_id = _index_unique(
        contents,
        key=lambda item: item.page_event_id,
        label="page content page_event_id",
    )
    if references_by_id.keys() != contents_by_id.keys():
        raise LineCropDatasetError(
            "Page contents must match the reviewed reference pages exactly."
        )

    review_by_key = _index_unique(
        reviews,
        key=lambda item: (item.page_event_id, item.source_block_order),
        label="review page/block key",
    )
    candidate_keys = {
        (content.page_event_id, block.order)
        for content in contents
        for block in content.ordered_blocks
        if block.polygon is not None
    }
    if set(review_by_key) != candidate_keys:
        missing = sorted(candidate_keys - set(review_by_key))
        extra = sorted(set(review_by_key) - candidate_keys)
        raise LineCropDatasetError(
            "Reviews must cover every polygon exactly; "
            f"missing={missing}, extra={extra}."
        )
    pending = [
        key
        for key, review in review_by_key.items()
        if review.decision is LineReviewDecision.pending
    ]
    if pending:
        raise LineCropDatasetError(f"Line reviews are still pending: {pending}")

    root = Path(image_root).resolve()
    crop_dir = Path(output_dir).resolve()
    crop_dir.mkdir(parents=True, exist_ok=True)
    cropper_version = source_aligned_cropper_version(padding_pixels)
    samples: list[LineCropSample] = []

    for content in contents:
        reference = references_by_id[content.page_event_id]
        _validate_page_provenance(reference, content)
        source_path = resolve_dataset_path(content.image_path, root=root)
        if not source_path.is_file():
            raise LineCropDatasetError(
                f"Stable source image does not exist: {source_path}"
            )
        source_hash = sha256_file(source_path)
        if source_hash != content.image_sha256:
            raise LineCropDatasetError(
                f"Stable source image hash changed: {source_path}"
            )

        with Image.open(source_path) as source_image:
            rgb_image = source_image.convert("RGB")
            for block in content.ordered_blocks:
                if block.polygon is None:
                    continue
                review = review_by_key[(content.page_event_id, block.order)]
                _validate_review_provenance(review, content, block.text)
                if review.decision is LineReviewDecision.exclude:
                    continue
                assert review.corrected_text is not None
                corrected_text = review.corrected_text.strip()
                normalized_text = normalize_line_text(corrected_text)
                bounding_box = polygon_to_pixel_crop(
                    block.polygon,
                    image_width=rgb_image.width,
                    image_height=rgb_image.height,
                    padding_pixels=padding_pixels,
                )
                sample_id = line_crop_sample_id(
                    source_image_sha256=source_hash,
                    page_content_cache_key=content.cache_key,
                    page_event_id=content.page_event_id,
                    block_order=block.order,
                    text=corrected_text,
                    cropper_version=cropper_version,
                    bounding_box=bounding_box,
                )
                crop_path = crop_dir / f"{sample_id}.png"
                rgb_image.crop(
                    (
                        bounding_box.x,
                        bounding_box.y,
                        bounding_box.x + bounding_box.width,
                        bounding_box.y + bounding_box.height,
                    )
                ).save(crop_path, format="PNG")
                samples.append(
                    LineCropSample(
                        sample_id=sample_id,
                        lecture_id=reference.lecture_id,
                        page_event_id=content.page_event_id,
                        page_number=reference.page_number,
                        stable_frame_timestamp=reference.stable_frame_timestamp,
                        source_image_path=portable_dataset_path(source_path, root=root),
                        source_image_sha256=source_hash,
                        crop_path=portable_dataset_path(crop_path, root=root),
                        crop_sha256=sha256_file(crop_path),
                        bounding_box=bounding_box,
                        text=corrected_text,
                        normalized_text=normalized_text,
                        label_source=LineLabelSource.source_aligned,
                        detector_reader=content.reader,
                        detector_version=content.reader_version,
                        detector_preprocessing_version=content.preprocessing_version,
                        detector_cache_key=content.cache_key,
                        source_block_order=block.order,
                    )
                )

    if not samples:
        raise LineCropDatasetError("No reviewed line crops were included.")
    return _validate_unique_samples(samples)


def build_synthetic_line_crops(
    references: Sequence[StablePageReference],
    *,
    image_root: str | Path,
    output_dir: str | Path,
    font_path: str | Path | None,
    seed: int = 17,
    variants_per_line: int = 4,
    min_characters: int = 2,
    max_characters: int = 48,
) -> list[LineCropSample]:
    """Render train-only gold lines with deterministic video-like degradation."""

    if variants_per_line <= 0:
        raise LineCropDatasetError("variants_per_line must be positive.")
    if min_characters <= 0 or max_characters < min_characters:
        raise LineCropDatasetError("Invalid synthetic line length bounds.")
    font = Path(font_path).resolve() if font_path is not None else None
    if font is not None and not font.is_file():
        raise FileNotFoundError(f"Synthetic line font does not exist: {font}")

    root = Path(image_root).resolve()
    crop_dir = Path(output_dir).resolve()
    crop_dir.mkdir(parents=True, exist_ok=True)
    font_sha256 = (
        sha256_file(font)
        if font is not None
        else _sha256_text(f"pillow-default-font:{PIL.__version__}")
    )
    builder_version = synthetic_line_builder_version(
        font_sha256=font_sha256,
        min_characters=min_characters,
        max_characters=max_characters,
    )
    samples: list[LineCropSample] = []

    for reference in references:
        if reference.gold_text_scope is not GoldTextScope.verbatim_content:
            raise LineCropDatasetError(
                "Synthetic labels require verbatim_content gold references."
            )
        line_index = -1
        for raw_line in reference.gold_text.splitlines():
            normalized = normalize_line_text(raw_line)
            if not (
                min_characters <= len(normalized) <= max_characters
                and any(character.isalpha() for character in normalized)
            ):
                continue
            line_index += 1
            for variant in range(variants_per_line):
                cache_key = _sha256_text(
                    "\0".join(
                        (
                            builder_version,
                            str(seed),
                            reference.page_event_id,
                            str(line_index),
                            str(variant),
                            normalized,
                        )
                    )
                )
                rendered = _render_synthetic_line(
                    normalized,
                    font_path=font,
                    seed_material=cache_key,
                )
                image_bytes = io.BytesIO()
                rendered.save(image_bytes, format="PNG")
                payload = image_bytes.getvalue()
                image_sha256 = hashlib.sha256(payload).hexdigest()
                bounding_box = PixelCrop(
                    x=0,
                    y=0,
                    width=rendered.width,
                    height=rendered.height,
                )
                sample_id = line_crop_sample_id(
                    source_image_sha256=image_sha256,
                    page_content_cache_key=cache_key,
                    page_event_id=reference.page_event_id,
                    block_order=line_index * variants_per_line + variant,
                    text=normalized,
                    cropper_version=builder_version,
                    bounding_box=bounding_box,
                )
                crop_path = crop_dir / f"{sample_id}.png"
                crop_path.write_bytes(payload)
                portable_path = portable_dataset_path(crop_path, root=root)
                samples.append(
                    LineCropSample(
                        sample_id=sample_id,
                        lecture_id=reference.lecture_id,
                        page_event_id=reference.page_event_id,
                        page_number=reference.page_number,
                        stable_frame_timestamp=reference.stable_frame_timestamp,
                        source_image_path=portable_path,
                        source_image_sha256=image_sha256,
                        crop_path=portable_path,
                        crop_sha256=image_sha256,
                        bounding_box=bounding_box,
                        text=normalized,
                        normalized_text=normalized,
                        label_source=LineLabelSource.synthetic_render,
                        detector_reader=PageReaderKind.gold_reference,
                        detector_version=builder_version,
                        detector_preprocessing_version=builder_version,
                        detector_cache_key=cache_key,
                        source_block_order=line_index * variants_per_line + variant,
                    )
                )

    if not samples:
        raise LineCropDatasetError("No eligible lines were available to render.")
    return _validate_unique_samples(samples)


def combine_line_crop_components(
    components: Sequence[Sequence[LineCropSample]],
) -> list[LineCropSample]:
    if not components or any(not component for component in components):
        raise LineCropDatasetError("Every dataset component must contain samples.")
    combined = [sample for component in components for sample in component]
    combined.sort(
        key=lambda sample: (
            sample.lecture_id,
            sample.page_event_id,
            sample.source_block_order,
            sample.sample_id,
        )
    )
    return _validate_unique_samples(combined)


def source_aligned_cropper_version(padding_pixels: int) -> str:
    return f"source-aligned-polygon-v1:padding-{padding_pixels}"


def synthetic_line_builder_version(
    *,
    font_sha256: str,
    min_characters: int,
    max_characters: int,
) -> str:
    return (
        "synthetic-line-v1:"
        f"font-{font_sha256[:12]}:chars-{min_characters}-{max_characters}"
    )


def _render_synthetic_line(
    text: str,
    *,
    font_path: Path | None,
    seed_material: str,
) -> Image.Image:
    rng = random.Random(int(seed_material[:16], 16))
    font_size = rng.randint(25, 34)
    font = (
        ImageFont.truetype(str(font_path), font_size)
        if font_path is not None
        else ImageFont.load_default(size=font_size)
    )
    scratch = Image.new("L", (1, 1), color=255)
    bounds = ImageDraw.Draw(scratch).textbbox((0, 0), text, font=font)
    padding_x = rng.randint(6, 12)
    padding_y = rng.randint(4, 8)
    width = bounds[2] - bounds[0] + 2 * padding_x
    height = bounds[3] - bounds[1] + 2 * padding_y
    background = rng.randint(238, 255)
    foreground = rng.randint(0, 35)
    image = Image.new("L", (width, height), color=background)
    ImageDraw.Draw(image).text(
        (padding_x - bounds[0], padding_y - bounds[1]),
        text,
        fill=foreground,
        font=font,
    )
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.65)))
    jpeg_buffer = io.BytesIO()
    image.save(jpeg_buffer, format="JPEG", quality=rng.randint(45, 90))
    jpeg_buffer.seek(0)
    with Image.open(jpeg_buffer) as compressed:
        return compressed.convert("RGB")


def _validate_page_provenance(
    reference: StablePageReference,
    content: PageContent,
) -> None:
    if content.lecture_id != reference.lecture_id:
        raise LineCropDatasetError("Page content belongs to the wrong lecture.")
    if content.page_number != reference.page_number:
        raise LineCropDatasetError("Page content has the wrong page number.")
    if content.stable_frame_timestamp != reference.stable_frame_timestamp:
        raise LineCropDatasetError("Page content has the wrong timestamp.")


def _validate_review_provenance(
    review: LineCropReviewRecord,
    content: PageContent,
    detected_text: str,
) -> None:
    if review.page_content_cache_key != content.cache_key:
        raise LineCropDatasetError("A review targets a different page-content run.")
    if review.source_image_sha256 != content.image_sha256:
        raise LineCropDatasetError("A review targets a different source image.")
    if review.detected_text != detected_text:
        raise LineCropDatasetError("Detected text changed after review.")


def _validate_unique_samples(
    samples: list[LineCropSample],
) -> list[LineCropSample]:
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise LineCropDatasetError("Generated line-crop IDs are not unique.")
    return samples


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _index_unique(items, *, key, label: str):
    indexed = {}
    for item in items:
        value = key(item)
        if value in indexed:
            raise LineCropDatasetError(f"Duplicate {label}: {value}")
        indexed[value] = item
    return indexed
