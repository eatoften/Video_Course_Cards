import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowRight,
  BookOpenCheck,
  Clock3,
  Eye,
  RefreshCw,
  RotateCcw,
} from 'lucide-react'
import type { CourseMapPayload } from './courseMapTypes'
import type {
  ReviewCourse,
  ReviewQueue,
  ReviewQueueItem,
  ReviewRating,
} from './reviewTypes'


type ReviewViewProps = {
  apiBaseUrl: string
  courses: ReviewCourse[]
  selectedCourseId: string | null
  onSelectCourse: (courseId: string) => void
  onOpenWorkspaceCard: (cardId: string) => void
}


async function fetchJson<T>(apiBaseUrl: string, path: string, options?: RequestInit) {
  const response = await fetch(`${apiBaseUrl}${path}`, options)
  if (!response.ok) {
    let message = `HTTP ${response.status}`
    try {
      const payload = await response.json()
      if (typeof payload.detail === 'string') message = payload.detail
    } catch {
      // Keep status fallback.
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}


function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  return `${Math.floor(total / 60)}:${(total % 60).toString().padStart(2, '0')}`
}


function formatDue(value: string): string {
  return new Date(value).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}


function readClockMilliseconds(): number {
  return Date.now()
}


export function ReviewView({
  apiBaseUrl,
  courses,
  selectedCourseId,
  onSelectCourse,
  onOpenWorkspaceCard,
}: ReviewViewProps) {
  const [queue, setQueue] = useState<ReviewQueue | null>(null)
  const [courseMap, setCourseMap] = useState<CourseMapPayload | null>(null)
  const [topicId, setTopicId] = useState('')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [revealed, setRevealed] = useState(false)
  const [selfAnswer, setSelfAnswer] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isRating, setIsRating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const startedAtRef = useRef(0)

  const loadQueue = useCallback(async () => {
    if (!selectedCourseId) return
    setIsLoading(true)
    setError(null)
    const params = new URLSearchParams({ limit: '100' })
    if (topicId) params.set('topic_id', topicId)
    try {
      const [nextQueue, nextMap] = await Promise.all([
        fetchJson<ReviewQueue>(
          apiBaseUrl,
          `/courses/${selectedCourseId}/review/queue?${params}`,
        ),
        fetchJson<CourseMapPayload>(
          apiBaseUrl,
          `/courses/${selectedCourseId}/map`,
        ),
      ])
      setQueue(nextQueue)
      setCourseMap(nextMap)
      setCurrentIndex(0)
      setRevealed(false)
      setSelfAnswer('')
      startedAtRef.current = readClockMilliseconds()
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Review queue failed.')
    } finally {
      setIsLoading(false)
    }
  }, [apiBaseUrl, selectedCourseId, topicId])

  useEffect(() => {
    void loadQueue()
  }, [loadQueue])

  const currentItem = queue?.items[currentIndex] ?? null

  const topicDueCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const item of queue?.items ?? []) {
      if (item.topic_id) counts.set(item.topic_id, (counts.get(item.topic_id) ?? 0) + 1)
    }
    return [...counts.entries()]
      .map(([id, count]) => ({
        id,
        title: courseMap?.topics.find((topic) => topic.id === id)?.title ?? 'Unknown',
        count,
      }))
      .sort((left, right) => right.count - left.count)
  }, [courseMap, queue])

  async function rateCurrent(rating: ReviewRating) {
    if (!currentItem) return
    setIsRating(true)
    setError(null)
    try {
      await fetchJson(
        apiBaseUrl,
        `/review-items/${currentItem.review_item.id}/rate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rating,
            response_time_ms: Math.max(
              0,
              readClockMilliseconds() - startedAtRef.current,
            ),
          }),
        },
      )
      const remainingItems = queue?.items.filter(
        (item) => item.review_item.id !== currentItem.review_item.id,
      ) ?? []
      setQueue((current) => current ? {
        ...current,
        due_count: Math.max(0, current.due_count - 1),
        items: remainingItems,
      } : current)
      setCurrentIndex(0)
      setRevealed(false)
      setSelfAnswer('')
      setMessage(`Rated ${rating}.`)
      startedAtRef.current = readClockMilliseconds()
    } catch (ratingError) {
      setError(ratingError instanceof Error ? ratingError.message : 'Rating failed.')
    } finally {
      setIsRating(false)
    }
  }

  function renderEvidence(item: ReviewQueueItem) {
    const selectedClaimIds = new Set(item.review_item.source_claim_ids)
    const claims = selectedClaimIds.size > 0
      ? item.claims.filter((claim) => selectedClaimIds.has(claim.id))
      : item.claims
    return (
      <div className="review-evidence">
        <div className="review-section-title">Grounded evidence</div>
        {claims.map((claim) => (
          <div key={claim.id}>
            <strong>{claim.text}</strong>
            {claim.evidence.map((evidence) => (
              <blockquote key={evidence.id}>
                <span>
                  {formatTime(evidence.segment_start_seconds)} -{' '}
                  {formatTime(evidence.segment_end_seconds)}
                </span>
                {evidence.quote}
              </blockquote>
            ))}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="review-view">
      <header className="review-toolbar">
        <div>
          <div className="panel-title">Spaced repetition</div>
          <h1>Review</h1>
          <p>Recall first, reveal evidence, then rate honestly.</p>
        </div>
        <label>
          <span>Course</span>
          <select value={selectedCourseId ?? ''} onChange={(event) => onSelectCourse(event.target.value)}>
            {courses.map((course) => (
              <option key={course.id} value={course.id}>{course.title}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Topic</span>
          <select value={topicId} onChange={(event) => setTopicId(event.target.value)}>
            <option value="">All topics</option>
            {(courseMap?.topics ?? []).filter((topic) => topic.status !== 'hidden').map((topic) => (
              <option key={topic.id} value={topic.id}>{topic.title}</option>
            ))}
          </select>
        </label>
        <button type="button" disabled={isLoading} onClick={() => void loadQueue()}>
          <RefreshCw size={16} /> Refresh
        </button>
      </header>

      {(error || message) && (
        <div className={error ? 'review-notice error' : 'review-notice success'}>
          {error ?? message}
        </div>
      )}

      <div className="review-layout">
        <aside className="review-overview">
          <div className="review-summary-grid">
            <div><strong>{queue?.due_count ?? 0}</strong><span>Due now</span></div>
            <div><strong>{queue?.new_count ?? 0}</strong><span>New</span></div>
            <div><strong>{queue?.learning_count ?? 0}</strong><span>Learning</span></div>
            <div><strong>{queue?.review_count ?? 0}</strong><span>Review</span></div>
          </div>
          <section>
            <div className="review-section-title">Due by topic</div>
            <div className="review-topic-list">
              {topicDueCounts.length ? topicDueCounts.map((topic) => (
                <button key={topic.id} type="button" onClick={() => setTopicId(topic.id)}>
                  <span>{topic.title}</span><strong>{topic.count}</strong>
                </button>
              )) : <div className="review-empty-small">No due topics</div>}
            </div>
          </section>
          <section>
            <div className="review-section-title">Session queue</div>
            <div className="review-session-list">
              {(queue?.items ?? []).map((item, index) => (
                <button
                  key={item.review_item.id}
                  type="button"
                  className={index === currentIndex ? 'selected' : ''}
                  onClick={() => {
                    setCurrentIndex(index)
                    setRevealed(false)
                    setSelfAnswer('')
                    startedAtRef.current = Date.now()
                  }}
                >
                  <strong>{item.card_title}</strong>
                  <span>{item.topic_title ?? 'Unsorted'} · {item.phase}</span>
                </button>
              ))}
            </div>
          </section>
        </aside>

        <main className="review-session">
          {currentItem ? (
            <div className="review-card-session">
              <div className="review-card-context">
                <div>
                  <span>{currentItem.topic_title ?? 'Unsorted'}</span>
                  <span>{currentItem.card_kind}</span>
                  <span>{currentItem.phase}</span>
                </div>
                <button type="button" onClick={() => onOpenWorkspaceCard(currentItem.card_id)}>
                  Open card <ArrowRight size={15} />
                </button>
              </div>
              <div className="review-prompt">
                <BookOpenCheck size={24} />
                <h2>{currentItem.review_item.prompt}</h2>
                <p>{currentItem.card_title}</p>
              </div>
              <textarea
                className="review-self-answer"
                value={selfAnswer}
                onChange={(event) => setSelfAnswer(event.target.value)}
                placeholder="Optional: type what you remember before revealing"
                disabled={revealed}
              />
              {!revealed ? (
                <button className="review-reveal-button" type="button" onClick={() => setRevealed(true)}>
                  <Eye size={17} /> Reveal answer
                </button>
              ) : (
                <>
                  <section className="review-answer">
                    <div className="review-section-title">Expected answer</div>
                    <p>{currentItem.review_item.expected_answer}</p>
                  </section>
                  {renderEvidence(currentItem)}
                  <div className="review-rating-grid">
                    {(['again', 'hard', 'good', 'easy'] as ReviewRating[]).map((rating) => (
                      <button
                        key={rating}
                        type="button"
                        className={`rating-${rating}`}
                        disabled={isRating}
                        onClick={() => void rateCurrent(rating)}
                      >
                        {rating === 'again' && <RotateCcw size={15} />}
                        {rating}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="review-complete">
              <BookOpenCheck size={32} />
              <h2>{isLoading ? 'Loading review queue' : 'Review complete'}</h2>
              <p>No review items are due under this filter.</p>
            </div>
          )}
        </main>

        <aside className="review-details">
          {currentItem ? (
            <>
              <div className="review-section-title">Card context</div>
              <h3>{currentItem.card_title}</h3>
              <p>{currentItem.card_summary}</p>
              <dl>
                <div><dt>Due</dt><dd>{formatDue(currentItem.progress.due_at)}</dd></div>
                <div><dt>Reviews</dt><dd>{currentItem.progress.review_count}</dd></div>
                <div><dt>Lapses</dt><dd>{currentItem.progress.lapse_count}</dd></div>
                <div><dt>Source</dt><dd>{formatTime(currentItem.source_start_seconds)} - {formatTime(currentItem.source_end_seconds)}</dd></div>
              </dl>
              <div className="review-tip">
                <Clock3 size={16} />
                Rate the quality of recall, not how familiar the answer looks.
              </div>
            </>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
