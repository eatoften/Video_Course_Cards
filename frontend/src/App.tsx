import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react'
import './App.css'

const API_BASE_URL = 'http://127.0.0.1:8000'

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

type UploadResponse = {
  id: string
  filename: string
  stored_name: string
  size_bytes: number
  status: JobStatus
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
  const [job, setJob] = useState<VideoJob | null>(null)
  const [transcript, setTranscript] = useState<TranscriptionResult | null>(null)
  const [selectedSegment, setSelectedSegment] =
    useState<TranscriptSegment | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
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
    setSelectedSegment(null)
    setErrorMessage(null)
  }

  async function refreshJob(jobId: string): Promise<VideoJob> {
    const nextJob = await fetchJson<VideoJob>(`/jobs/${jobId}`)
    setJob(nextJob)

    return nextJob
  }

  async function loadTranscript(jobId: string) {
    const nextTranscript = await fetchJson<TranscriptionResult>(
      `/jobs/${jobId}/transcript`,
    )
    setTranscript(nextTranscript)
  }

  async function uploadSelectedVideo() {
    if (!selectedFile) {
      return
    }

    setIsUploading(true)
    setErrorMessage(null)
    setTranscript(null)
    setSelectedSegment(null)

    try {
      const formData = new FormData()
      formData.append('video', selectedFile)

      const upload = await fetchJson<UploadResponse>('/videos', {
        method: 'POST',
        body: formData,
      })

      await refreshJob(upload.id)
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
    setSelectedSegment(null)

    try {
      const nextJob = await fetchJson<VideoJob>(
        `/jobs/${job.id}/${path}`,
        { method: 'POST' },
      )
      setJob(nextJob)
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Processing failed.',
      )
    } finally {
      setIsStarting(false)
    }
  }

  function jumpToSegment(segment: TranscriptSegment) {
    setSelectedSegment(segment)

    if (videoRef.current) {
      videoRef.current.currentTime = segment.start_seconds
      void videoRef.current.play()
    }
  }

  useEffect(() => {
    if (!job || !runningStatuses.includes(job.status)) {
      return
    }

    const intervalId = window.setInterval(() => {
      void refreshJob(job.id)
        .then((nextJob) => {
          if (nextJob.status === 'completed') {
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

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <h1>Video Course Cards</h1>
          <p className="subtle">Local video transcription workspace</p>
        </div>
        <div className={`status-pill status-${job?.status ?? 'idle'}`}>
          {job?.status ?? 'idle'}
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

          {selectedSegment && (
            <div className="selection-panel">
              <div className="panel-title">Selected segment</div>
              <p>
                {formatTime(selectedSegment.start_seconds)} -{' '}
                {formatTime(selectedSegment.end_seconds)}
              </p>
              <p>{selectedSegment.text}</p>
            </div>
          )}
        </div>

        <aside className="side-pane">
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
                {job.metadata.width}x{job.metadata.height} ·{' '}
                {formatTime(job.metadata.duration_seconds)} ·{' '}
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
                transcript.segments.map((segment, index) => (
                  <button
                    type="button"
                    key={`${segment.start_seconds}-${index}`}
                    className={
                      index === activeSegmentIndex
                        ? 'segment active'
                        : 'segment'
                    }
                    onClick={() => jumpToSegment(segment)}
                  >
                    <span>{formatTime(segment.start_seconds)}</span>
                    <p>{segment.text}</p>
                  </button>
                ))
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
