from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .ctc_text import normalize_line_text
from .page_reading import sha256_file
from .schemas import (
    DatasetSplit,
    GoldTextScope,
    LectureSplitManifest,
    LineCropSample,
    LineLabelSource,
    PageContent,
    PixelCrop,
    StablePageReference,
)


class LineCropDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class LineImageTransform:
    target_height: int = 32
    max_width: int = 512
    verify_hashes: bool = True

    def __post_init__(self) -> None:
        if self.target_height <= 0:
            raise ValueError("target_height must be positive.")
        if self.max_width < self.target_height:
            raise ValueError("max_width cannot be smaller than target_height.")


@dataclass(frozen=True)
class LineCropItem:
    sample: LineCropSample
    image: Tensor
    width: int


@dataclass(frozen=True)
class LineCropBatch:
    images: Tensor
    widths: Tensor
    texts: tuple[str, ...]
    sample_ids: tuple[str, ...]


class LineCropDataset(Dataset[LineCropItem]):
    def __init__(
        self,
        samples: Sequence[LineCropSample],
        *,
        image_root: str | Path,
        transform: LineImageTransform | None = None,
    ) -> None:
        if not samples:
            raise LineCropDatasetError("A line-crop dataset cannot be empty.")
        sample_ids = [sample.sample_id for sample in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise LineCropDatasetError("Line-crop sample IDs must be unique.")
        self._samples = list(samples)
        self._image_root = Path(image_root).resolve()
        self._transform = transform or LineImageTransform()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> LineCropItem:
        sample = self._samples[index]
        crop_path = _resolve_path(sample.crop_path, root=self._image_root)
        if not crop_path.is_file():
            raise LineCropDatasetError(f"Line crop does not exist: {crop_path}")
        if (
            self._transform.verify_hashes
            and sha256_file(crop_path) != sample.crop_sha256
        ):
            raise LineCropDatasetError(
                f"Line crop hash changed for sample {sample.sample_id}."
            )

        with Image.open(crop_path) as image:
            grayscale = image.convert("L")
            target_width = max(
                1,
                round(
                    grayscale.width
                    * self._transform.target_height
                    / grayscale.height
                ),
            )
            if target_width > self._transform.max_width:
                raise LineCropDatasetError(
                    f"Sample {sample.sample_id} needs width {target_width}, "
                    f"above configured max_width={self._transform.max_width}."
                )
            resized = grayscale.resize(
                (target_width, self._transform.target_height),
                Image.Resampling.BILINEAR,
            )
            pixels = np.asarray(resized, dtype=np.float32) / 255.0

        # Ink is positive and white padding is zero, which makes padding inert.
        tensor = 1.0 - torch.from_numpy(pixels.copy()).unsqueeze(0)
        return LineCropItem(sample=sample, image=tensor, width=target_width)


def collate_line_crops(
    items: Sequence[LineCropItem],
    *,
    width_multiple: int = 4,
) -> LineCropBatch:
    if not items:
        raise LineCropDatasetError("Cannot collate an empty line-crop batch.")
    if width_multiple <= 0:
        raise LineCropDatasetError("width_multiple must be positive.")

    heights = {item.image.shape[-2] for item in items}
    if len(heights) != 1:
        raise LineCropDatasetError("All line crops must have the same height.")
    max_width = max(item.width for item in items)
    padded_width = math.ceil(max_width / width_multiple) * width_multiple
    batch = torch.zeros(
        len(items),
        1,
        next(iter(heights)),
        padded_width,
        dtype=torch.float32,
    )
    for index, item in enumerate(items):
        batch[index, :, :, : item.width] = item.image

    return LineCropBatch(
        images=batch,
        widths=torch.tensor([item.width for item in items], dtype=torch.long),
        texts=tuple(item.sample.normalized_text for item in items),
        sample_ids=tuple(item.sample.sample_id for item in items),
    )


def build_exact_match_line_crops(
    references: Sequence[StablePageReference],
    contents: Sequence[PageContent],
    *,
    image_root: str | Path,
    output_dir: str | Path,
    padding_pixels: int = 2,
) -> list[LineCropSample]:
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
            "Page contents must match the reference page IDs exactly."
        )

    root = Path(image_root).resolve()
    crop_dir = Path(output_dir).resolve()
    crop_dir.mkdir(parents=True, exist_ok=True)
    cropper_version = line_cropper_version(padding_pixels)
    samples: list[LineCropSample] = []

    for reference in references:
        if reference.gold_text_scope is not GoldTextScope.verbatim_content:
            raise LineCropDatasetError(
                "Line labels require verbatim_content gold references."
            )
        content = contents_by_id[reference.page_event_id]
        _validate_page_provenance(reference, content)
        source_path = _resolve_path(content.image_path, root=root)
        if not source_path.is_file():
            raise LineCropDatasetError(
                f"Stable source image does not exist: {source_path}"
            )
        source_hash = sha256_file(source_path)
        if source_hash != content.image_sha256:
            raise LineCropDatasetError(
                f"Stable source image hash changed: {source_path}"
            )

        labels_by_normalized_text: dict[str, deque[str]] = defaultdict(deque)
        for line in reference.gold_text.splitlines():
            normalized = normalize_line_text(line)
            if normalized:
                labels_by_normalized_text[normalized].append(line.strip())

        with Image.open(source_path) as source_image:
            rgb_image = source_image.convert("RGB")
            for block in content.ordered_blocks:
                normalized_block = normalize_line_text(block.text)
                available_labels = labels_by_normalized_text[normalized_block]
                if not available_labels or block.polygon is None:
                    continue
                label = available_labels.popleft()
                bounding_box = _polygon_to_crop(
                    block.polygon,
                    image_width=rgb_image.width,
                    image_height=rgb_image.height,
                    padding_pixels=padding_pixels,
                )
                sample_id = _line_crop_sample_id(
                    source_image_sha256=source_hash,
                    page_content_cache_key=content.cache_key,
                    page_event_id=reference.page_event_id,
                    block_order=block.order,
                    text=label,
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
                        page_event_id=reference.page_event_id,
                        page_number=reference.page_number,
                        stable_frame_timestamp=reference.stable_frame_timestamp,
                        source_image_path=_portable_path(source_path, root=root),
                        source_image_sha256=source_hash,
                        crop_path=_portable_path(crop_path, root=root),
                        crop_sha256=sha256_file(crop_path),
                        bounding_box=bounding_box,
                        text=label,
                        normalized_text=normalize_line_text(label),
                        label_source=LineLabelSource.manual_exact_match,
                        detector_reader=content.reader,
                        detector_version=content.reader_version,
                        detector_preprocessing_version=(
                            content.preprocessing_version
                        ),
                        detector_cache_key=content.cache_key,
                        source_block_order=block.order,
                    )
                )

    if not samples:
        raise LineCropDatasetError("No OCR blocks exactly matched manual gold lines.")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise LineCropDatasetError("Generated line-crop IDs are not unique.")
    return samples


def line_cropper_version(padding_pixels: int) -> str:
    return f"gold-exact-polygon-v1:padding-{padding_pixels}"


def make_lecture_level_split(
    samples: Sequence[LineCropSample],
    *,
    dataset_sha256: str,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> LectureSplitManifest:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise LineCropDatasetError("Split fractions must be between zero and one.")
    lectures = sorted({sample.lecture_id for sample in samples})
    if len(lectures) < 3:
        raise LineCropDatasetError(
            "A formal lecture-level split requires at least three lectures."
        )

    shuffled = lectures.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    test_count = max(1, round(len(shuffled) * test_fraction))
    while validation_count + test_count >= len(shuffled):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise LineCropDatasetError(
                "Split fractions leave no lecture for training."
            )

    test_lectures = sorted(shuffled[:test_count])
    validation_lectures = sorted(
        shuffled[test_count : test_count + validation_count]
    )
    train_lectures = sorted(shuffled[test_count + validation_count :])
    return LectureSplitManifest(
        dataset_sha256=dataset_sha256,
        seed=seed,
        train_lecture_ids=train_lectures,
        validation_lecture_ids=validation_lectures,
        test_lecture_ids=test_lectures,
    )


def partition_by_lecture_split(
    samples: Sequence[LineCropSample],
    split: LectureSplitManifest,
) -> dict[DatasetSplit, list[LineCropSample]]:
    split_by_lecture = {
        **{
            lecture_id: DatasetSplit.train
            for lecture_id in split.train_lecture_ids
        },
        **{
            lecture_id: DatasetSplit.validation
            for lecture_id in split.validation_lecture_ids
        },
        **{
            lecture_id: DatasetSplit.test
            for lecture_id in split.test_lecture_ids
        },
    }
    sample_lectures = {sample.lecture_id for sample in samples}
    if sample_lectures != set(split_by_lecture):
        raise LineCropDatasetError(
            "Split lectures must exactly match the dataset lectures."
        )

    partitions = {
        DatasetSplit.train: [],
        DatasetSplit.validation: [],
        DatasetSplit.test: [],
    }
    for sample in samples:
        partitions[split_by_lecture[sample.lecture_id]].append(sample)
    return partitions


def _polygon_to_crop(
    polygon: Sequence[tuple[float, float]],
    *,
    image_width: int,
    image_height: int,
    padding_pixels: int,
) -> PixelCrop:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left = max(0, math.floor(min(xs)) - padding_pixels)
    top = max(0, math.floor(min(ys)) - padding_pixels)
    right = min(image_width, math.ceil(max(xs)) + padding_pixels)
    bottom = min(image_height, math.ceil(max(ys)) + padding_pixels)
    if right <= left or bottom <= top:
        raise LineCropDatasetError("A text polygon produced an empty crop.")
    return PixelCrop(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
    )


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


def _line_crop_sample_id(
    *,
    source_image_sha256: str,
    page_content_cache_key: str,
    page_event_id: str,
    block_order: int,
    text: str,
    cropper_version: str,
    bounding_box: PixelCrop,
) -> str:
    payload = "\0".join(
        (
            source_image_sha256,
            page_content_cache_key,
            page_event_id,
            str(block_order),
            text,
            cropper_version,
            str(bounding_box.x),
            str(bounding_box.y),
            str(bounding_box.width),
            str(bounding_box.height),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_path(path: str | Path, *, root: Path) -> Path:
    configured = Path(path)
    return (configured if configured.is_absolute() else root / configured).resolve()


def _portable_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _index_unique(items, *, key, label: str):
    indexed = {}
    for item in items:
        value = key(item)
        if value in indexed:
            raise LineCropDatasetError(f"Duplicate {label}: {value}")
        indexed[value] = item
    return indexed
