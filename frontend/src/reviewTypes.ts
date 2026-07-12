import type { CourseMapCourse, CourseMapPayload } from './courseMapTypes'

export type ReviewCourse = CourseMapCourse
export type ReviewCourseMap = CourseMapPayload
export type ReviewRating = 'again' | 'hard' | 'good' | 'easy'
export type ReviewPhase = 'new' | 'learning' | 'review' | 'relearning'

export type ReviewEvidence = {
  id: string
  quote: string
  segment_start_seconds: number
  segment_end_seconds: number
}

export type ReviewClaim = {
  id: string
  text: string
  evidence: ReviewEvidence[]
}

export type ReviewQueueItem = {
  review_item: {
    id: string
    card_id: string
    item_type: string
    prompt: string
    expected_answer: string
    source_claim_ids: string[]
    source: string
    status: string
  }
  progress: {
    due_at: string
    review_count: number
    lapse_count: number
  }
  phase: ReviewPhase
  card_id: string
  card_title: string
  card_summary: string
  card_kind: string
  claims: ReviewClaim[]
  topic_id: string | null
  topic_title: string | null
  source_start_seconds: number
  source_end_seconds: number
}

export type ReviewQueue = {
  course_id: string
  topic_id: string | null
  due_count: number
  new_count: number
  learning_count: number
  review_count: number
  relearning_count: number
  items: ReviewQueueItem[]
}
