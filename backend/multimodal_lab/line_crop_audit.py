from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .line_crop_dataset import resolve_dataset_path
from .page_reading import sha256_file
from .schemas import LineCropSample


class LineCropAuditError(ValueError):
    pass


def render_line_crop_audit_sheets(
    samples: Sequence[LineCropSample],
    *,
    image_root: str | Path,
    output_dir: str | Path,
    dataset_sha256: str,
    columns: int = 2,
    rows: int = 15,
) -> dict[str, object]:
    if not samples:
        raise LineCropAuditError("Cannot audit an empty line-crop dataset.")
    if columns <= 0 or rows <= 0:
        raise LineCropAuditError("Contact-sheet dimensions must be positive.")
    root = Path(image_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    font = _load_font(16)
    samples_per_sheet = columns * rows
    sheet_records: list[dict[str, object]] = []

    for sheet_index, start in enumerate(
        range(0, len(samples), samples_per_sheet),
        start=1,
    ):
        page_samples = samples[start : start + samples_per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * 620, rows * 108),
            color=(242, 244, 247),
        )
        draw = ImageDraw.Draw(sheet)
        for offset, sample in enumerate(page_samples):
            row, column = divmod(offset, columns)
            left = column * 620
            top = row * 108
            crop_path = resolve_dataset_path(sample.crop_path, root=root)
            if not crop_path.is_file():
                raise LineCropAuditError(f"Missing crop: {crop_path}")
            if sha256_file(crop_path) != sample.crop_sha256:
                raise LineCropAuditError(
                    f"Crop hash changed for sample {sample.sample_id}."
                )
            with Image.open(crop_path) as crop:
                rendered = ImageOps.contain(
                    crop.convert("RGB"),
                    (596, 66),
                    Image.Resampling.LANCZOS,
                )
            image_left = left + 12
            image_top = top + 6 + (66 - rendered.height) // 2
            sheet.paste(rendered, (image_left, image_top))
            label = (
                f"{start + offset:03d} | p{sample.page_number or 0} "
                f"b{sample.source_block_order} | {sample.normalized_text}"
            )
            draw.text(
                (left + 12, top + 78),
                _truncate_label(label, maximum_characters=78),
                fill=(20, 24, 31),
                font=font,
            )
            draw.rectangle(
                (left + 4, top + 2, left + 615, top + 104),
                outline=(170, 176, 187),
                width=1,
            )

        output_path = destination / f"line_crop_audit_{sheet_index:02d}.png"
        sheet.save(output_path, format="PNG")
        sheet_records.append(
            {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "first_sample_index": start,
                "sample_count": len(page_samples),
            }
        )

    report = {
        "schema_version": "1.0",
        "dataset_sha256": dataset_sha256,
        "sample_count": len(samples),
        "sample_sequence_sha256": _sample_sequence_hash(samples),
        "columns": columns,
        "rows": rows,
        "sheets": sheet_records,
    }
    report_path = destination / "line_crop_audit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _truncate_label(text: str, *, maximum_characters: int) -> str:
    if len(text) <= maximum_characters:
        return text
    return text[: maximum_characters - 3] + "..."


def _sample_sequence_hash(samples: Sequence[LineCropSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(sample.normalized_text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
