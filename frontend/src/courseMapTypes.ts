export type TopicMethod = 'manual' | 'embedding_cluster' | 'local_llm' | 'system'
export type TopicStatus = 'suggested' | 'accepted' | 'hidden'
export type TopicRelationType = 'prerequisite' | 'related' | 'contrast_with'

export type CourseMapTopic = {
  id: string
  course_id: string
  parent_topic_id: string | null
  title: string
  summary: string | null
  position: number
  depth: number
  method: TopicMethod
  status: TopicStatus
  is_system: boolean
  created_at: string
  updated_at: string
}

export type CourseMapMembership = {
  id: string
  topic_id: string
  card_id: string
  role: 'primary' | 'supporting' | 'example'
  position: number
  method: TopicMethod
  confidence: number | null
  status: TopicStatus
}

export type CourseMapCard = {
  id: string
  job_id: string
  title: string
  summary: string
  card_kind: string
  tags: string[]
  content_status: 'draft' | 'reviewed' | 'needs_fix'
  review_item_count: number
  source_video: string | null
  source_start_seconds: number
  source_end_seconds: number
  note_count: number
  learning_document_count: number
}

export type TopicLearningCoverage = {
  topic_id: string
  card_count: number
  cards_with_review_items: number
  review_item_count: number
  due_review_item_count: number
  cards_with_learning_documents: number
  learning_document_count: number
}

export type CourseLearningCoverage = {
  total_cards: number
  cards_with_review_items: number
  review_item_count: number
  due_review_item_count: number
  cards_with_learning_documents: number
  learning_document_count: number
  source_asset_count: number
  unsorted_card_count: number
  topic_coverage: TopicLearningCoverage[]
}

export type CourseMapTopicRelation = {
  id: string
  course_id: string
  source_topic_id: string
  target_topic_id: string
  relation_type: TopicRelationType
  explanation: string | null
  method: TopicMethod
  status: TopicStatus
}

export type CourseMapPayload = {
  course_id: string
  topics: CourseMapTopic[]
  memberships: CourseMapMembership[]
  topic_relations: CourseMapTopicRelation[]
  cards: CourseMapCard[]
  coverage: CourseLearningCoverage
}

export type CourseMapCourse = {
  id: string
  title: string
  card_count: number
}
