import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type MouseEvent,
} from 'react'
import './App.css'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

type JobStatus =
  | 'uploaded'
  | 'probing'
  | 'extracting_audio'
  | 'transcribing'
  | 'completed'
  | 'failed'

type VideoMetadata = {
  duration_seconds: number
  width: number
  height: number
  video_codec: string
  has_audio: boolean
}

type VideoJob = {
  id: string
  video_path: string
  status: JobStatus
  original_filename: string | null
  stored_name: string | null
  size_bytes: number | null
  metadata: VideoMetadata | null
  transcript_path: string | null
  error_message: string | null
  created_at: string
  updated_at: string
  started_at: string | null
  completed_at: string | null
}

type TranscriptSegment = {
  start_seconds: number
  end_seconds: number
  text: string
}

type TranscriptionResult = {
  language: string
  language_probability: number
  duration_seconds: number
  segments: TranscriptSegment[]
}

type TranscriptContext = {
  job_id: string
  source_video: string
  start_seconds: number
  end_seconds: number
  segments: TranscriptSegment[]
  text: string
}

type LlmStatus = {
  provider: string
  base_url: string
  model: string
  available: boolean
  error_message: string | null
}

type LlmModelList = {
  provider: string
  base_url: string
  default_model: string
  models: string[]
  available: boolean
  error_message: string | null
}

type KnowledgeCardDraft = {
  title: string
  summary: string
  key_points: string[]
  question: string
  answer: string
  difficulty: 'easy' | 'medium' | 'hard'
  source_start_seconds: number
  source_end_seconds: number
}

type CardDraftResponse = {
  job_id: string
  source_video: string
  start_seconds: number
  end_seconds: number
  provider: string
  model: string
  cards: KnowledgeCardDraft[]
}

type KnowledgeCard = Omit<KnowledgeCardDraft, 'question' | 'answer'> & {
  id: string
  job_id: string
  question: string | null
  answer: string | null
  provider: string | null
  model: string | null
  created_at: string
  updated_at: string
}

type CardEditForm = {
  title: string
  summary: string
  key_points: string
  question: string
  answer: string
  difficulty: KnowledgeCardDraft['difficulty']
}

type UploadResponse = {
  id: string
  filename: string
  stored_name: string
  size_bytes: number
  status: JobStatus
}

type SegmentRange = {
  anchorIndex: number
  startIndex: number
  endIndex: number
}

const runningStatuses: JobStatus[] = [
  'probing',
  'extracting_audio',
  'transcribing',
]

async function fetchJson<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options)

  if (!response.ok) {
    let message = `HTTP ${response.status}`

    try {
      const payload = await response.json()
      if (typeof payload.detail === 'string') {
        message = payload.detail
      }
    } catch {
      // Keep the HTTP status message.
    }

    throw new Error(message)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}

function formatTime(seconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(totalSeconds / 60)
  const remainingSeconds = totalSeconds % 60

  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

function formatSize(bytes: number | null): string {
  if (!bytes) {
    return '-'
  }

  const megabytes = bytes / 1024 / 1024
  return `${megabytes.toFixed(1)} MB`
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [jobs, setJobs] = useState<VideoJob[]>([])
  const [job, setJob] = useState<VideoJob | null>(null)
  const [transcript, setTranscript] = useState<TranscriptionResult | null>(null)
  const [selectedRange, setSelectedRange] = useState<SegmentRange | null>(null)
  const [transcriptContext, setTranscriptContext] =
    useState<TranscriptContext | null>(null)
  const [llmStatus, setLlmStatus] = useState<LlmStatus | null>(null)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [cardDraft, setCardDraft] = useState<CardDraftResponse | null>(null)
  const [savedCards, setSavedCards] = useState<KnowledgeCard[]>([])
  const [editingCardId, setEditingCardId] = useState<string | null>(null)
  const [cardEditForm, setCardEditForm] = useState<CardEditForm | null>(null)
  const [cardFocus, setCardFocus] = useState('')
  const [currentTime, setCurrentTime] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [isLoadingContext, setIsLoadingContext] = useState(false)
  const [isLoadingJobs, setIsLoadingJobs] = useState(false)
  const [isSavingCard, setIsSavingCard] = useState(false)
  const [isDeletingJob, setIsDeletingJob] = useState(false)
  const [isCheckingLlm, setIsCheckingLlm] = useState(false)
  const [isDraftingCards, setIsDraftingCards] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const activeSegmentIndex = useMemo(() => {
    if (!transcript) {
      return -1
    }

    return transcript.segments.findIndex((segment) => {
      return (
        currentTime >= segment.start_seconds &&
        currentTime < segment.end_seconds
      )
    })
  }, [currentTime, transcript])

  const selectedSegments = useMemo(() => {
    if (!transcript || !selectedRange) {
      return []
    }

    return transcript.segments.slice(
      selectedRange.startIndex,
      selectedRange.endIndex + 1,
    )
  }, [selectedRange, transcript])

  function clearTranscriptSelection() {
    setSelectedRange(null)
    setTranscriptContext(null)
    setCardDraft(null)
    setEditingCardId(null)
    setCardEditForm(null)
  }

  function handleVideoChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]

    if (!file) {
      return
    }

    const objectUrl = URL.createObjectURL(file)

    setSelectedFile(file)
    setVideoUrl((previousUrl) => {
      if (previousUrl) {
        URL.revokeObjectURL(previousUrl)
      }

      return objectUrl
    })
    setJob(null)
    setTranscript(null)
    clearTranscriptSelection()
    setErrorMessage(null)
  }

  async function refreshJob(jobId: string): Promise<VideoJob> {
    const nextJob = await fetchJson<VideoJob>(`/jobs/${jobId}`)
    setJob(nextJob)

    return nextJob
  }

  async function loadJobs() {
    setIsLoadingJobs(true)

    try {
      const nextJobs = await fetchJson<VideoJob[]>('/jobs')
      setJobs(nextJobs)
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Job list failed.',
      )
    } finally {
      setIsLoadingJobs(false)
    }
  }

  async function loadTranscript(jobId: string) {
    const nextTranscript = await fetchJson<TranscriptionResult>(
      `/jobs/${jobId}/transcript`,
    )
    setTranscript(nextTranscript)
    setCardDraft(null)
  }

  async function loadSavedCards(jobId: string) {
    const cards = await fetchJson<KnowledgeCard[]>(
      `/jobs/${jobId}/cards`,
    )
    setSavedCards(cards)
  }

  async function openJob(nextJob: VideoJob) {
    setJob(nextJob)
    setSelectedFile(null)
    setVideoUrl((previousUrl) => {
      if (previousUrl) {
        URL.revokeObjectURL(previousUrl)
      }

      return null
    })
    setTranscript(null)
    setSavedCards([])
    clearTranscriptSelection()
    setErrorMessage(null)

    try {
      await loadSavedCards(nextJob.id)

      if (nextJob.transcript_path) {
        await loadTranscript(nextJob.id)
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Job loading failed.',
      )
    }
  }

  async function deleteVideoJob(jobToDelete: VideoJob) {
    setIsDeletingJob(true)
    setErrorMessage(null)

    try {
      await fetchJson<void>(
        `/jobs/${jobToDelete.id}`,
        {
          method: 'DELETE',
        },
      )

      setJobs((previousJobs) =>
        previousJobs.filter((item) => item.id !== jobToDelete.id),
      )

      if (job?.id === jobToDelete.id) {
        setJob(null)
        setSelectedFile(null)
        setVideoUrl((previousUrl) => {
          if (previousUrl) {
            URL.revokeObjectURL(previousUrl)
          }

          return null
        })
        setTranscript(null)
        setSavedCards([])
        clearTranscriptSelection()
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Video delete failed.',
      )
    } finally {
      setIsDeletingJob(false)
    }
  }

  async function checkLlmStatus() {
    setIsCheckingLlm(true)

    try {
      const [status, modelList] = await Promise.all([
        fetchJson<LlmStatus>('/llm/status'),
        fetchJson<LlmModelList>('/llm/models'),
      ])
      setLlmStatus(status)
      setAvailableModels(modelList.models)
      setSelectedModel((previousModel) => {
        if (previousModel) {
          return previousModel
        }

        if (modelList.models.includes(status.model)) {
          return status.model
        }

        return modelList.models[0] ?? status.model
      })
    } catch (error) {
      setLlmStatus(null)
      setAvailableModels([])
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Local model status check failed.',
      )
    } finally {
      setIsCheckingLlm(false)
    }
  }

  async function uploadSelectedVideo() {
    if (!selectedFile) {
      return
    }

    setIsUploading(true)
    setErrorMessage(null)
    setTranscript(null)
    clearTranscriptSelection()

    try {
      const formData = new FormData()
      formData.append('video', selectedFile)

      const upload = await fetchJson<UploadResponse>('/videos', {
        method: 'POST',
        body: formData,
      })

      await refreshJob(upload.id)
      await loadJobs()
      await loadSavedCards(upload.id)
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Upload failed.',
      )
    } finally {
      setIsUploading(false)
    }
  }

  async function startProcessing(path: 'run' | 'retry') {
    if (!job) {
      return
    }

    setIsStarting(true)
    setErrorMessage(null)
    setTranscript(null)
    clearTranscriptSelection()

    try {
      const nextJob = await fetchJson<VideoJob>(
        `/jobs/${job.id}/${path}`,
        { method: 'POST' },
      )
      setJob(nextJob)
      await loadJobs()
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Processing failed.',
      )
    } finally {
      setIsStarting(false)
    }
  }

  function selectSegment(
    segment: TranscriptSegment,
    index: number,
    event: MouseEvent<HTMLButtonElement>,
  ) {
    setCardDraft(null)

    setSelectedRange((previousRange) => {
      if (event.shiftKey && previousRange) {
        return {
          anchorIndex: previousRange.anchorIndex,
          startIndex: Math.min(previousRange.anchorIndex, index),
          endIndex: Math.max(previousRange.anchorIndex, index),
        }
      }

      return {
        anchorIndex: index,
        startIndex: index,
        endIndex: index,
      }
    })

    if (videoRef.current) {
      videoRef.current.currentTime = segment.start_seconds
      void videoRef.current.play()
    }
  }

  async function generateCards() {
    if (!job || selectedSegments.length === 0) {
      return
    }

    const firstSegment = selectedSegments[0]
    const lastSegment = selectedSegments[selectedSegments.length - 1]

    setIsDraftingCards(true)
    setErrorMessage(null)
    setCardDraft(null)

    try {
      const draft = await fetchJson<CardDraftResponse>('/cards/draft', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: job.id,
          start_seconds: firstSegment.start_seconds,
          end_seconds: lastSegment.end_seconds,
          card_count: 3,
          focus: cardFocus.trim() || null,
          model: selectedModel || null,
        }),
      })

      setCardDraft(draft)
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Card generation failed.',
      )
    } finally {
      setIsDraftingCards(false)
    }
  }

  async function saveDraftCard(card: KnowledgeCardDraft) {
    if (!job || !cardDraft) {
      return
    }

    setIsSavingCard(true)
    setErrorMessage(null)

    try {
      const savedCard = await fetchJson<KnowledgeCard>(
        `/jobs/${job.id}/cards`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ...card,
            provider: cardDraft.provider,
            model: cardDraft.model,
          }),
        },
      )

      setSavedCards((previousCards) => [...previousCards, savedCard])
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Card save failed.',
      )
    } finally {
      setIsSavingCard(false)
    }
  }

  function startEditingCard(card: KnowledgeCard) {
    setEditingCardId(card.id)
    setCardEditForm({
      title: card.title,
      summary: card.summary,
      key_points: card.key_points.join('\n'),
      question: card.question ?? '',
      answer: card.answer ?? '',
      difficulty: card.difficulty,
    })
  }

  async function saveEditedCard(cardId: string) {
    if (!cardEditForm) {
      return
    }

    setIsSavingCard(true)
    setErrorMessage(null)

    try {
      const updatedCard = await fetchJson<KnowledgeCard>(
        `/cards/${cardId}`,
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            title: cardEditForm.title,
            summary: cardEditForm.summary,
            key_points: cardEditForm.key_points
              .split('\n')
              .map((point) => point.trim())
              .filter(Boolean),
            question: cardEditForm.question || null,
            answer: cardEditForm.answer || null,
            difficulty: cardEditForm.difficulty,
          }),
        },
      )

      setSavedCards((previousCards) =>
        previousCards.map((card) =>
          card.id === updatedCard.id ? updatedCard : card,
        ),
      )
      setEditingCardId(null)
      setCardEditForm(null)
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Card update failed.',
      )
    } finally {
      setIsSavingCard(false)
    }
  }

  async function deleteSavedCard(cardId: string) {
    setIsSavingCard(true)
    setErrorMessage(null)

    try {
      await fetchJson<void>(
        `/cards/${cardId}`,
        {
          method: 'DELETE',
        },
      )
      setSavedCards((previousCards) =>
        previousCards.filter((card) => card.id !== cardId),
      )
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Card delete failed.',
      )
    } finally {
      setIsSavingCard(false)
    }
  }

  useEffect(() => {
    void checkLlmStatus()
    void loadJobs()
  }, [])

  useEffect(() => {
    if (!job || !transcript || !selectedRange) {
      setTranscriptContext(null)
      return
    }

    const firstSegment = transcript.segments[selectedRange.startIndex]
    const lastSegment = transcript.segments[selectedRange.endIndex]

    if (!firstSegment || !lastSegment) {
      setTranscriptContext(null)
      return
    }

    const controller = new AbortController()
    const params = new URLSearchParams({
      start_seconds: String(firstSegment.start_seconds),
      end_seconds: String(lastSegment.end_seconds),
    })

    setIsLoadingContext(true)

    void fetchJson<TranscriptContext>(
      `/jobs/${job.id}/context?${params.toString()}`,
      { signal: controller.signal },
    )
      .then((context) => {
        setTranscriptContext(context)
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }

        setErrorMessage(
          error instanceof Error
            ? error.message
            : 'Context loading failed.',
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoadingContext(false)
        }
      })

    return () => {
      controller.abort()
    }
  }, [job, selectedRange, transcript])

  useEffect(() => {
    if (!job || !runningStatuses.includes(job.status)) {
      return
    }

    const intervalId = window.setInterval(() => {
      void refreshJob(job.id)
        .then((nextJob) => {
          if (nextJob.status === 'completed') {
            void loadJobs()
            void loadSavedCards(nextJob.id)
            return loadTranscript(nextJob.id)
          }

          return undefined
        })
        .catch((error) => {
          setErrorMessage(
            error instanceof Error ? error.message : 'Polling failed.',
          )
        })
    }, 1500)

    return () => window.clearInterval(intervalId)
  }, [job])

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl)
      }
    }
  }, [videoUrl])

  const canUpload = selectedFile && !isUploading
  const canRun = job?.status === 'uploaded' && !isStarting
  const canRetry = job?.status === 'failed' && !isStarting
  const canGenerateCards =
    selectedSegments.length > 0 && !isDraftingCards && !isLoadingContext

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <h1>Video Course Cards</h1>
          <p className="subtle">Local video transcription workspace</p>
        </div>
        <div className="top-statuses">
          <button
            type="button"
            className={`llm-pill ${
              llmStatus?.available ? 'llm-ready' : 'llm-offline'
            }`}
            onClick={() => void checkLlmStatus()}
          >
            {isCheckingLlm
                ? 'checking model'
                : llmStatus
                ? `${selectedModel || llmStatus.model} ${
                    llmStatus.available ? 'ready' : 'offline'
                  }`
                : 'model unknown'}
          </button>
          <select
            className="model-select"
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
          >
            {availableModels.length > 0 ? (
              availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))
            ) : (
              <option value={selectedModel || llmStatus?.model || ''}>
                {selectedModel || llmStatus?.model || 'No model loaded'}
              </option>
            )}
          </select>
          <div className={`status-pill status-${job?.status ?? 'idle'}`}>
            {job?.status ?? 'idle'}
          </div>
        </div>
      </header>

      <section className="workspace">
        <div className="media-pane">
          <div className="toolbar">
            <label className="file-picker">
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoChange}
              />
              Choose video
            </label>
            <button
              type="button"
              disabled={!canUpload}
              onClick={uploadSelectedVideo}
            >
              {isUploading ? 'Uploading' : 'Upload'}
            </button>
            <button
              type="button"
              disabled={!canRun}
              onClick={() => void startProcessing('run')}
            >
              {isStarting ? 'Starting' : 'Run'}
            </button>
            <button
              type="button"
              disabled={!canRetry}
              onClick={() => void startProcessing('retry')}
            >
              Retry
            </button>
          </div>

          <div className="video-stage">
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                preload="metadata"
                onTimeUpdate={(event) => {
                  setCurrentTime(event.currentTarget.currentTime)
                }}
              />
            ) : (
              <div className="empty-state">No video selected</div>
            )}
          </div>

          {errorMessage && (
            <div className="error-banner">{errorMessage}</div>
          )}

          {selectedRange && selectedSegments.length > 0 && (
            <div className="selection-panel">
              <div className="panel-title">Context window</div>
              <p>
                {formatTime(selectedSegments[0].start_seconds)} -{' '}
                {formatTime(
                  selectedSegments[selectedSegments.length - 1].end_seconds,
                )}
              </p>
              <p className="context-text">
                {isLoadingContext
                  ? 'Loading context'
                  : transcriptContext?.text ||
                    selectedSegments
                      .map((segment) => segment.text)
                      .join('\n')}
              </p>
              <div className="card-controls">
                <input
                  className="focus-input"
                  value={cardFocus}
                  onChange={(event) => setCardFocus(event.target.value)}
                  placeholder="Optional focus, e.g. exam review or core concept"
                />
                <button
                  type="button"
                  disabled={!canGenerateCards}
                  onClick={() => void generateCards()}
                >
                  {isDraftingCards ? 'Generating' : 'Generate cards'}
                </button>
              </div>
            </div>
          )}

          {cardDraft && (
            <section className="cards-panel">
              <div className="panel-title">
                Draft cards · {cardDraft.model}
              </div>
              <div className="card-list">
                {cardDraft.cards.map((card, index) => (
                  <article
                    className="knowledge-card"
                    key={`${card.title}-${index}`}
                  >
                    <div className="card-heading">
                      <h3>{card.title}</h3>
                      <span>{card.difficulty}</span>
                    </div>
                    <p>{card.summary}</p>
                    <ul>
                      {card.key_points.map((point) => (
                        <li key={point}>{point}</li>
                      ))}
                    </ul>
                    <div className="qa-block">
                      <strong>{card.question}</strong>
                      <p>{card.answer}</p>
                    </div>
                    <div className="source-range">
                      {formatTime(card.source_start_seconds)} -{' '}
                      {formatTime(card.source_end_seconds)}
                    </div>
                    <div className="card-actions">
                      <button
                        type="button"
                        disabled={isSavingCard}
                        onClick={() => void saveDraftCard(card)}
                      >
                        Save
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          )}

          <section className="cards-panel">
            <div className="panel-title">Saved cards</div>
            <div className="card-list">
              {savedCards.length ? (
                savedCards.map((card) => {
                  const isEditing = editingCardId === card.id

                  return (
                    <article className="knowledge-card" key={card.id}>
                      {isEditing && cardEditForm ? (
                        <div className="edit-card-form">
                          <input
                            value={cardEditForm.title}
                            onChange={(event) =>
                              setCardEditForm({
                                ...cardEditForm,
                                title: event.target.value,
                              })
                            }
                          />
                          <textarea
                            value={cardEditForm.summary}
                            onChange={(event) =>
                              setCardEditForm({
                                ...cardEditForm,
                                summary: event.target.value,
                              })
                            }
                          />
                          <textarea
                            value={cardEditForm.key_points}
                            onChange={(event) =>
                              setCardEditForm({
                                ...cardEditForm,
                                key_points: event.target.value,
                              })
                            }
                          />
                          <input
                            value={cardEditForm.question}
                            onChange={(event) =>
                              setCardEditForm({
                                ...cardEditForm,
                                question: event.target.value,
                              })
                            }
                          />
                          <input
                            value={cardEditForm.answer}
                            onChange={(event) =>
                              setCardEditForm({
                                ...cardEditForm,
                                answer: event.target.value,
                              })
                            }
                          />
                          <select
                            value={cardEditForm.difficulty}
                            onChange={(event) =>
                              setCardEditForm({
                                ...cardEditForm,
                                difficulty: event.target.value as
                                  KnowledgeCardDraft['difficulty'],
                              })
                            }
                          >
                            <option value="easy">easy</option>
                            <option value="medium">medium</option>
                            <option value="hard">hard</option>
                          </select>
                          <div className="card-actions">
                            <button
                              type="button"
                              disabled={isSavingCard}
                              onClick={() => void saveEditedCard(card.id)}
                            >
                              Save edit
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setEditingCardId(null)
                                setCardEditForm(null)
                              }}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="card-heading">
                            <h3>{card.title}</h3>
                            <span>{card.difficulty}</span>
                          </div>
                          <p>{card.summary}</p>
                          <ul>
                            {card.key_points.map((point) => (
                              <li key={point}>{point}</li>
                            ))}
                          </ul>
                          {(card.question || card.answer) && (
                            <div className="qa-block">
                              {card.question && <strong>{card.question}</strong>}
                              {card.answer && <p>{card.answer}</p>}
                            </div>
                          )}
                          <div className="source-range">
                            {formatTime(card.source_start_seconds)} -{' '}
                            {formatTime(card.source_end_seconds)}
                          </div>
                          <div className="card-actions">
                            <button
                              type="button"
                              onClick={() => startEditingCard(card)}
                            >
                              Edit
                            </button>
                            <button
                              type="button"
                              disabled={isSavingCard}
                              onClick={() => void deleteSavedCard(card.id)}
                            >
                              Delete
                            </button>
                          </div>
                        </>
                      )}
                    </article>
                  )
                })
              ) : (
                <div className="empty-list">No saved cards</div>
              )}
            </div>
          </section>
        </div>

        <aside className="side-pane">
          <section className="jobs-panel">
            <div className="panel-heading-row">
              <h2>Videos</h2>
              <button type="button" onClick={() => void loadJobs()}>
                {isLoadingJobs ? 'Loading' : 'Refresh'}
              </button>
            </div>
            <div className="job-list">
              {jobs.length ? (
                jobs.map((item) => (
                  <div
                    key={item.id}
                    className={[
                      'job-list-row',
                      item.id === job?.id ? 'selected' : '',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                  >
                    <button
                      type="button"
                      className="job-list-item"
                      onClick={() => void openJob(item)}
                    >
                      <span>{item.original_filename ?? item.stored_name ?? item.id}</span>
                      <small>{item.status}</small>
                    </button>
                    <button
                      type="button"
                      className="danger-button"
                      disabled={isDeletingJob}
                      onClick={() => void deleteVideoJob(item)}
                    >
                      Delete
                    </button>
                  </div>
                ))
              ) : (
                <div className="empty-list">No uploaded videos</div>
              )}
            </div>
          </section>
          <section className="job-panel">
            <h2>Job</h2>
            <dl>
              <div>
                <dt>ID</dt>
                <dd>{job?.id ?? '-'}</dd>
              </div>
              <div>
                <dt>File</dt>
                <dd>{job?.original_filename ?? selectedFile?.name ?? '-'}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd>{formatSize(job?.size_bytes ?? selectedFile?.size ?? null)}</dd>
              </div>
              <div>
                <dt>Updated</dt>
                <dd>{job?.updated_at ? new Date(job.updated_at).toLocaleTimeString() : '-'}</dd>
              </div>
            </dl>
            {job?.metadata && (
              <div className="metadata-row">
                {job.metadata.width}x{job.metadata.height} |{' '}
                {formatTime(job.metadata.duration_seconds)} |{' '}
                {job.metadata.video_codec}
              </div>
            )}
            {job?.error_message && (
              <div className="error-text">{job.error_message}</div>
            )}
          </section>

          <section className="transcript-panel">
            <h2>Transcript</h2>
            <div className="segment-list">
              {transcript?.segments.length ? (
                transcript.segments.map((segment, index) => {
                  const isInSelectedRange =
                    selectedRange !== null &&
                    index >= selectedRange.startIndex &&
                    index <= selectedRange.endIndex
                  const className = [
                    'segment',
                    index === activeSegmentIndex ? 'active' : '',
                    isInSelectedRange ? 'selected' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')

                  return (
                    <button
                      type="button"
                      key={`${segment.start_seconds}-${index}`}
                      className={className}
                      onClick={(event) => selectSegment(segment, index, event)}
                    >
                      <span>{formatTime(segment.start_seconds)}</span>
                      <p>{segment.text}</p>
                    </button>
                  )
                })
              ) : (
                <div className="empty-list">Transcript unavailable</div>
              )}
            </div>
          </section>
        </aside>
      </section>
    </main>
  )
}

export default App
