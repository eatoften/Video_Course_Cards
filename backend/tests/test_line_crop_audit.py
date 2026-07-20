from __future__ import annotations

from pathlib import Path

from PIL import Image

from multimodal_lab.line_crop_audit import render_line_crop_audit_sheets
from multimodal_lab.page_reading import sha256_file
from multimodal_lab.schemas import (
    LineCropSample,
    LineLabelSource,
    PageReaderKind,
    PixelCrop,
)


def test_render_line_crop_audit_sheets_hashes_every_crop(tmp_path: Path) -> None:
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (80, 24), color="white").save(crop_path)
    crop_hash = sha256_file(crop_path)
    sample = LineCropSample(
        sample_id="a" * 64,
        lecture_id="lecture",
        page_event_id="page",
        page_number=3,
        stable_frame_timestamp=1,
        source_image_path="crop.png",
        source_image_sha256=crop_hash,
        crop_path="crop.png",
        crop_sha256=crop_hash,
        bounding_box=PixelCrop(x=0, y=0, width=80, height=24),
        text="Array of 32x32x3 numbers",
        normalized_text="Array of 32x32x3 numbers",
        label_source=LineLabelSource.source_aligned,
        detector_reader=PageReaderKind.rapidocr,
        detector_version="test",
        detector_preprocessing_version="test",
        detector_cache_key="b" * 64,
        source_block_order=2,
    )

    report = render_line_crop_audit_sheets(
        [sample],
        image_root=tmp_path,
        output_dir=tmp_path / "audit",
        dataset_sha256="c" * 64,
        columns=1,
        rows=1,
    )

    assert report["sample_count"] == 1
    sheet_path = Path(report["sheets"][0]["path"])
    assert sheet_path.is_file()
    assert report["sheets"][0]["sha256"] == sha256_file(sheet_path)
