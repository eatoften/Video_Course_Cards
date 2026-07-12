import json
import re
import unicodedata
from dataclasses import dataclass

from .knowledge_card import KnowledgeCard
from .review_item import ReviewItem


INVALID_FILENAME_CHARS = r'<>:"/\|?*'
INVALID_FILENAME_PATTERN = re.compile(
    f"[{re.escape(INVALID_FILENAME_CHARS)}\\x00-\\x1f]"
)
WHITESPACE_PATTERN = re.compile(r"\s+")
MAX_COMPONENT_LENGTH = 80


@dataclass(frozen=True)
class MarkdownCardSource:
    course_title: str | None
    video_title: str | None
    job_id: str


def render_card_markdown(
    card: KnowledgeCard,
    source: MarkdownCardSource,
    review_items: list[ReviewItem] | None = None,
) -> str:
    active_review_items = review_items or []
    lines = [
        "---",
        "type: knowledge-card",
        f"course: {_yaml_string(source.course_title or 'Uncategorized')}",
        f"video: {_yaml_string(source.video_title or source.job_id)}",
        f"job_id: {_yaml_string(source.job_id)}",
        f"card_id: {_yaml_string(card.id)}",
        f"source_start: {_yaml_string(format_timestamp(card.source_start_seconds))}",
        f"source_end: {_yaml_string(format_timestamp(card.source_end_seconds))}",
        f"card_kind: {_yaml_string(card.card_kind)}",
        f"content_status: {_yaml_string(card.content_status)}",
        *_tag_frontmatter_lines(card.tags),
        "---",
        "",
        f"# {card.title}",
        "",
        card.summary,
        "",
        "## Source",
        "",
        f"- Course: {source.course_title or 'Uncategorized'}",
        f"- Video: {source.video_title or source.job_id}",
        f"- Time: {format_time_range(card.source_start_seconds, card.source_end_seconds)}",
        f"- Job: `{source.job_id}`",
        f"- Card: `{card.id}`",
        "",
    ]

    if card.key_points:
        lines.extend([
            "## Key Points",
            "",
            *[
                f"- {point}"
                for point in card.key_points
            ],
            "",
        ])

    lines.extend([
        "## Claims",
        "",
    ])

    for claim in card.claims:
        lines.append(f"- Claim: {claim.text}")
        for evidence in claim.evidence:
            lines.extend([
                f"  - Evidence: {_quote(evidence.quote)}",
                (
                    "    Source: "
                    f"{format_time_range(evidence.segment_start_seconds, evidence.segment_end_seconds)}"
                ),
            ])

    lines.extend(["", "## Active Recall", ""])
    if active_review_items:
        for index, item in enumerate(active_review_items, start=1):
            lines.extend([
                f"### {index}. {item.item_type.replace('_', ' ').title()}",
                "",
                f"Q: {item.prompt}",
                "",
                f"A: {item.expected_answer}",
                "",
            ])
    else:
        lines.extend(["_No review items saved._", ""])

    if card.unsupported_terms:
        lines.extend([
            "## Review Terms",
            "",
            *[
                f"- {term}"
                for term in card.unsupported_terms
            ],
            "",
        ])

    lines.extend([
        "## Metadata",
        "",
        f"- Card kind: {card.card_kind}",
        f"- Content status: {card.content_status}",
        f"- Provider: {card.provider or 'unknown'}",
        f"- Model: {card.model or 'unknown'}",
        f"- Created: {card.created_at.isoformat()}",
        f"- Updated: {card.updated_at.isoformat()}",
        "",
    ])

    return "\n".join(lines)


def safe_path_component(value: str | None, fallback: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").strip()
    text = INVALID_FILENAME_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip(" .")

    if not text:
        text = fallback

    return text[:MAX_COMPONENT_LENGTH].strip(" .") or fallback


def safe_markdown_filename(value: str | None, fallback: str) -> str:
    component = safe_path_component(value, fallback=fallback).lower()
    component = WHITESPACE_PATTERN.sub("-", component)

    return f"{component}.md"


def format_time_range(start_seconds: float, end_seconds: float) -> str:
    return f"{format_timestamp(start_seconds)} - {format_timestamp(end_seconds)}"


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

    return f"{minutes}:{remaining_seconds:02d}"


def _quote(value: str) -> str:
    return json.dumps(
        WHITESPACE_PATTERN.sub(" ", value).strip(),
        ensure_ascii=False,
    )


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _tag_frontmatter_lines(tags: list[str]) -> list[str]:
    if not tags:
        return ["tags: []"]

    return ["tags:"] + [
        f"  - {_yaml_string(tag)}"
        for tag in tags
    ]
