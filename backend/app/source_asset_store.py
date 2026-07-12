import json
from datetime import datetime
from sqlite3 import Row

from .db import connect, ensure_db
from .source_asset import SourceAsset, SourceAssetDetail, SourceUnit


def _to_text(value: datetime) -> str:
    return value.isoformat()


def _from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_asset(row: Row) -> SourceAssetDetail:
    keys = set(row.keys())
    return SourceAssetDetail(
        id=row["id"],
        course_id=row["course_id"],
        job_id=row["job_id"],
        asset_type=row["asset_type"],
        original_filename=row["original_filename"],
        stored_path=row["stored_path"],
        mime_type=row["mime_type"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        extraction_status=row["extraction_status"],
        metadata=json.loads(row["metadata_json"]),
        error_message=row["error_message"],
        created_at=_from_text(row["created_at"]),
        updated_at=_from_text(row["updated_at"]),
        unit_count=row["unit_count"] if "unit_count" in keys else 0,
    )


def _row_to_unit(row: Row) -> SourceUnit:
    return SourceUnit(
        id=row["id"],
        asset_id=row["asset_id"],
        unit_type=row["unit_type"],
        ordinal=row["ordinal"],
        text=row["text"],
        locator=json.loads(row["locator_json"]),
        created_at=_from_text(row["created_at"]),
    )


def create_source_asset(asset: SourceAsset) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO source_assets (
                id, course_id, job_id, asset_type, original_filename,
                stored_path, mime_type, size_bytes, sha256,
                extraction_status, metadata_json, error_message,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset.id, asset.course_id, asset.job_id, asset.asset_type,
                asset.original_filename, asset.stored_path, asset.mime_type,
                asset.size_bytes, asset.sha256, asset.extraction_status,
                json.dumps(asset.metadata, ensure_ascii=False),
                asset.error_message, _to_text(asset.created_at),
                _to_text(asset.updated_at),
            ),
        )


def update_source_asset(asset: SourceAsset) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE source_assets
            SET extraction_status = ?, metadata_json = ?, error_message = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                asset.extraction_status,
                json.dumps(asset.metadata, ensure_ascii=False),
                asset.error_message,
                _to_text(asset.updated_at),
                asset.id,
            ),
        )


def get_source_asset(asset_id: str) -> SourceAssetDetail | None:
    ensure_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT source_assets.*, COUNT(source_units.id) AS unit_count
            FROM source_assets
            LEFT JOIN source_units ON source_units.asset_id = source_assets.id
            WHERE source_assets.id = ?
            GROUP BY source_assets.id
            """,
            (asset_id,),
        ).fetchone()
    return _row_to_asset(row) if row is not None else None


def list_source_assets_for_course(course_id: str) -> list[SourceAssetDetail]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT source_assets.*, COUNT(source_units.id) AS unit_count
            FROM source_assets
            LEFT JOIN source_units ON source_units.asset_id = source_assets.id
            WHERE source_assets.course_id = ?
            GROUP BY source_assets.id
            ORDER BY source_assets.updated_at DESC
            """,
            (course_id,),
        ).fetchall()
    return [_row_to_asset(row) for row in rows]


def replace_source_units(asset_id: str, units: list[SourceUnit]) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM source_units WHERE asset_id = ?", (asset_id,))
        conn.executemany(
            """
            INSERT INTO source_units (
                id, asset_id, unit_type, ordinal, text, locator_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    unit.id, unit.asset_id, unit.unit_type, unit.ordinal,
                    unit.text, json.dumps(unit.locator, ensure_ascii=False),
                    _to_text(unit.created_at),
                )
                for unit in units
            ],
        )


def list_source_units_for_asset(asset_id: str) -> list[SourceUnit]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM source_units WHERE asset_id = ? ORDER BY ordinal",
            (asset_id,),
        ).fetchall()
    return [_row_to_unit(row) for row in rows]


def list_source_units_for_assets(asset_ids: list[str]) -> list[SourceUnit]:
    if not asset_ids:
        return []
    ensure_db()
    placeholders = ",".join("?" for _ in asset_ids)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM source_units
            WHERE asset_id IN ({placeholders})
            ORDER BY asset_id, ordinal
            """,
            asset_ids,
        ).fetchall()
    return [_row_to_unit(row) for row in rows]


def delete_source_asset(asset_id: str) -> bool:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM source_units WHERE asset_id = ?", (asset_id,))
        cursor = conn.execute("DELETE FROM source_assets WHERE id = ?", (asset_id,))
    return cursor.rowcount > 0


def move_source_assets_to_course(
    source_course_id: str,
    target_course_id: str,
) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            "UPDATE source_assets SET course_id = ? WHERE course_id = ?",
            (target_course_id, source_course_id),
        )


def clear_source_assets() -> None:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM source_units")
        conn.execute("DELETE FROM source_assets")
