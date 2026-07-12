export type StudyCourse = {
  id: string
  title: string
  card_count: number
}

export type StudyCard = {
  id: string
  job_id: string
  title: string
  summary: string
  card_kind: string
  tags: string[]
  content_status: 'draft' | 'reviewed' | 'needs_fix'
  review_item_count: number
  learning_document_count: number
  source_video: string | null
  source_start_seconds: number
  source_end_seconds: number
  note_count: number
}

export type SourceAsset = {
  id: string
  course_id: string
  job_id: string | null
  asset_type: 'video' | 'audio' | 'pptx' | 'pdf' | 'docx' | 'text'
  original_filename: string
  stored_path: string
  mime_type: string | null
  size_bytes: number
  sha256: string
  extraction_status: 'pending' | 'ready' | 'failed'
  metadata: Record<string, unknown>
  error_message: string | null
  unit_count: number
  created_at: string
  updated_at: string
}

export type LearningDocument = {
  id: string
  course_id: string
  title: string
  summary: string
  body_markdown: string
  status: 'draft' | 'reviewed' | 'needs_fix'
  generation_mode: 'manual' | 'local_llm' | 'imported'
  provider: string | null
  model: string | null
  created_at: string
  updated_at: string
}

export type LearningDocumentCardLink = {
  id: string
  document_id: string
  card_id: string
  role: 'primary_anchor' | 'supporting' | 'example' | 'contrast' | 'prerequisite'
  position: number
  created_at: string
}

export type LearningDocumentSource = {
  id: string
  document_id: string
  source_type: 'card_claim' | 'source_unit'
  source_id: string
  card_id: string | null
  label: string
  quote: string
  locator: Record<string, unknown>
  position: number
  created_at: string
}

export type LearningDocumentVersion = {
  id: string
  document_id: string
  version_number: number
  title: string
  summary: string
  body_markdown: string
  change_source: 'manual' | 'local_llm' | 'imported'
  provider: string | null
  model: string | null
  created_at: string
}

export type LearningDocumentDetail = LearningDocument & {
  card_links: LearningDocumentCardLink[]
  sources: LearningDocumentSource[]
  versions: LearningDocumentVersion[]
}

export type LearningDocumentGenerationResult = {
  document: LearningDocumentDetail
  selected_source_units: number
  selected_cards: number
  warning: string | null
}
