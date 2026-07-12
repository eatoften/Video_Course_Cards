export type GraphCourse = {
  id: string
  title: string
  card_count: number
}

export type CardRelationType =
  | 'semantic_similarity'
  | 'prerequisite'
  | 'related'
  | 'example_of'
  | 'contrast_with'
  | 'part_of'

export type CardSemanticRelationType = Exclude<
  CardRelationType,
  'semantic_similarity'
>

export type CardRelationStatus =
  | 'suggested'
  | 'accepted'
  | 'rejected'
  | 'hidden'

export type CardRelationMethod =
  | 'cosine_similarity'
  | 'local_llm'
  | 'manual'

export type CardGraphNode = {
  id: string
  job_id: string
  title: string
  summary: string
  tags: string[]
  review_state: 'draft' | 'reviewed' | 'needs_fix'
  source_start_seconds: number
  source_end_seconds: number
  x?: number
  y?: number
  vx?: number
  vy?: number
}

export type CardGraphEdge = {
  id: string
  source: string | CardGraphNode
  target: string | CardGraphNode
  relation_type: CardRelationType
  score: number
  method: CardRelationMethod
  status: CardRelationStatus
  explanation: string | null
}

export type CourseCardRelationsGraph = {
  course_id: string
  nodes: CardGraphNode[]
  edges: CardGraphEdge[]
}

export type CardRelationRecomputeResult = {
  course_id: string
  total_cards: number
  embedded_cards: number
  skipped_cards: number
  relations_written: number
  threshold: number
  top_k: number
}

export type CardRelationClassificationResult = {
  source_relation_id: string
  classification: CardSemanticRelationType | 'unclear'
  explanation: string
  model: string
  relation: CardGraphEdge | null
}
