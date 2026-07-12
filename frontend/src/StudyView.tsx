import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import ReactMarkdown from 'react-markdown'
import {
  BookOpenText,
  Eye,
  FilePlus2,
  History,
  Link2,
  Pencil,
  Save,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react'
import type {
  LearningDocument,
  LearningDocumentDetail,
  LearningDocumentGenerationResult,
  LearningDocumentSource,
  SourceAsset,
  StudyCard,
  StudyCourse,
} from './studyTypes'


type StudyViewProps = {
  apiBaseUrl: string
  courses: StudyCourse[]
  selectedCourseId: string | null
  selectedModel: string
  initialCardId: string | null
  initialDocumentId: string | null
  onSelectCourse: (courseId: string) => void
}


async function fetchJson<T>(
  apiBaseUrl: string,
  path: string,
  options?: RequestInit,
): Promise<T> {
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
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}


function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}


function sourceLocation(source: LearningDocumentSource): string {
  const locator = source.locator
  if (typeof locator.slide_number === 'number') return `Slide ${locator.slide_number}`
  if (typeof locator.page_number === 'number') return `Page ${locator.page_number}`
  if (typeof locator.paragraph_number === 'number') return `Paragraph ${locator.paragraph_number}`
  if (typeof locator.start_seconds === 'number') {
    const minutes = Math.floor(locator.start_seconds / 60)
    const seconds = Math.floor(locator.start_seconds % 60).toString().padStart(2, '0')
    return `${minutes}:${seconds}`
  }
  return source.source_type === 'card_claim' ? 'Card claim' : 'Source excerpt'
}


export function StudyView({
  apiBaseUrl,
  courses,
  selectedCourseId,
  selectedModel,
  initialCardId,
  initialDocumentId,
  onSelectCourse,
}: StudyViewProps) {
  const [cards, setCards] = useState<StudyCard[]>([])
  const [documents, setDocuments] = useState<LearningDocument[]>([])
  const [assets, setAssets] = useState<SourceAsset[]>([])
  const [selectedCardId, setSelectedCardId] = useState(initialCardId ?? '')
  const [selectedDocumentId, setSelectedDocumentId] = useState(initialDocumentId ?? '')
  const [document, setDocument] = useState<LearningDocumentDetail | null>(null)
  const [title, setTitle] = useState('')
  const [summary, setSummary] = useState('')
  const [body, setBody] = useState('')
  const [status, setStatus] = useState<LearningDocument['status']>('draft')
  const [mode, setMode] = useState<'edit' | 'preview'>('preview')
  const [selectedAssetIds, setSelectedAssetIds] = useState<Set<string>>(new Set())
  const [supportingCardIds, setSupportingCardIds] = useState<Set<string>>(new Set())
  const [focus, setFocus] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadLibrary = useCallback(async () => {
    if (!selectedCourseId) return
    setIsLoading(true)
    setError(null)
    try {
      const [nextCards, nextDocuments, nextAssets] = await Promise.all([
        fetchJson<StudyCard[]>(apiBaseUrl, `/courses/${selectedCourseId}/card-index`),
        fetchJson<LearningDocument[]>(
          apiBaseUrl,
          `/courses/${selectedCourseId}/learning-documents`,
        ),
        fetchJson<SourceAsset[]>(apiBaseUrl, `/courses/${selectedCourseId}/source-assets`),
      ])
      setCards(nextCards)
      setDocuments(nextDocuments)
      setAssets(nextAssets)
      setSelectedCardId((current) => {
        if (initialCardId && nextCards.some((card) => card.id === initialCardId)) {
          return initialCardId
        }
        if (current && nextCards.some((card) => card.id === current)) return current
        return nextCards[0]?.id ?? ''
      })
      setSelectedDocumentId((current) => {
        if (
          initialDocumentId &&
          nextDocuments.some((item) => item.id === initialDocumentId)
        ) return initialDocumentId
        if (current && nextDocuments.some((item) => item.id === current)) return current
        return nextDocuments[0]?.id ?? ''
      })
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Study library failed.')
    } finally {
      setIsLoading(false)
    }
  }, [apiBaseUrl, initialCardId, initialDocumentId, selectedCourseId])

  useEffect(() => {
    void loadLibrary()
  }, [loadLibrary])

  useEffect(() => {
    if (!selectedDocumentId) {
      setDocument(null)
      return
    }
    let active = true
    void fetchJson<LearningDocumentDetail>(
      apiBaseUrl,
      `/learning-documents/${selectedDocumentId}`,
    ).then((detail) => {
      if (!active) return
      setDocument(detail)
      setTitle(detail.title)
      setSummary(detail.summary)
      setBody(detail.body_markdown)
      setStatus(detail.status)
      const primary = detail.card_links.find((link) => link.role === 'primary_anchor')
      if (primary) setSelectedCardId(primary.card_id)
      setSupportingCardIds(
        new Set(
          detail.card_links
            .filter((link) => link.role !== 'primary_anchor')
            .map((link) => link.card_id),
        ),
      )
      const url = new URL(window.location.href)
      url.searchParams.set('document', detail.id)
      if (primary) url.searchParams.set('card', primary.card_id)
      window.history.replaceState({}, '', url)
    }).catch((loadError) => {
      if (active) setError(loadError instanceof Error ? loadError.message : 'Document failed.')
    })
    return () => { active = false }
  }, [apiBaseUrl, selectedDocumentId])

  const primaryCard = cards.find((card) => card.id === selectedCardId) ?? null
  const readyAssets = assets.filter((asset) => asset.extraction_status === 'ready')
  const supportCandidates = useMemo(
    () => cards.filter((card) => card.id !== selectedCardId),
    [cards, selectedCardId],
  )

  async function createDocument() {
    if (!selectedCardId) return
    setIsSaving(true)
    setError(null)
    try {
      const created = await fetchJson<LearningDocumentDetail>(
        apiBaseUrl,
        `/cards/${selectedCardId}/learning-documents`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        },
      )
      setSelectedDocumentId(created.id)
      setMode('edit')
      setMessage('Study document created.')
      await loadLibrary()
      setSelectedDocumentId(created.id)
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Create failed.')
    } finally {
      setIsSaving(false)
    }
  }

  async function saveDocument() {
    if (!document) return
    setIsSaving(true)
    setError(null)
    try {
      const updated = await fetchJson<LearningDocumentDetail>(
        apiBaseUrl,
        `/learning-documents/${document.id}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title, summary, body_markdown: body, status }),
        },
      )
      setDocument(updated)
      setMessage(`Saved version ${updated.versions[0]?.version_number ?? ''}.`)
      await loadLibrary()
      setSelectedDocumentId(updated.id)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Save failed.')
    } finally {
      setIsSaving(false)
    }
  }

  async function generateDocument() {
    if (!document) return
    setIsGenerating(true)
    setError(null)
    setMessage(null)
    try {
      const result = await fetchJson<LearningDocumentGenerationResult>(
        apiBaseUrl,
        `/learning-documents/${document.id}/generate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_asset_ids: [...selectedAssetIds],
            supporting_card_ids: [...supportingCardIds],
            focus: focus.trim() || null,
            model: selectedModel || null,
          }),
        },
      )
      setDocument(result.document)
      setTitle(result.document.title)
      setSummary(result.document.summary)
      setBody(result.document.body_markdown)
      setStatus(result.document.status)
      setMode('preview')
      setMessage(
        `${result.selected_cards} cards and ${result.selected_source_units} source units used.${result.warning ? ` ${result.warning}` : ''}`,
      )
      await loadLibrary()
      setSelectedDocumentId(result.document.id)
    } catch (generationError) {
      setError(
        generationError instanceof Error ? generationError.message : 'Generation failed.',
      )
    } finally {
      setIsGenerating(false)
    }
  }

  async function uploadAsset(file: File) {
    if (!selectedCourseId) return
    setIsUploading(true)
    setError(null)
    const form = new FormData()
    form.append('file', file)
    try {
      await fetchJson(apiBaseUrl, `/courses/${selectedCourseId}/source-assets`, {
        method: 'POST',
        body: form,
      })
      setMessage(`${file.name} imported.`)
      await loadLibrary()
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Import failed.')
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function deleteAsset(asset: SourceAsset) {
    if (!window.confirm(`Delete source "${asset.original_filename}"?`)) return
    try {
      await fetchJson<void>(apiBaseUrl, `/source-assets/${asset.id}`, { method: 'DELETE' })
      setSelectedAssetIds((current) => {
        const next = new Set(current)
        next.delete(asset.id)
        return next
      })
      await loadLibrary()
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Delete failed.')
    }
  }

  async function deleteDocument() {
    if (!document || !window.confirm(`Delete "${document.title}"?`)) return
    try {
      await fetchJson<void>(apiBaseUrl, `/learning-documents/${document.id}`, {
        method: 'DELETE',
      })
      setSelectedDocumentId('')
      setDocument(null)
      setMessage('Study document deleted.')
      await loadLibrary()
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Delete failed.')
    }
  }

  async function restoreVersion(versionNumber: number) {
    if (!document || !window.confirm(`Restore version ${versionNumber}?`)) return
    try {
      const restored = await fetchJson<LearningDocumentDetail>(
        apiBaseUrl,
        `/learning-documents/${document.id}/restore`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ version_number: versionNumber }),
        },
      )
      setDocument(restored)
      setTitle(restored.title)
      setSummary(restored.summary)
      setBody(restored.body_markdown)
      setMessage(`Version ${versionNumber} restored as a new version.`)
    } catch (restoreError) {
      setError(restoreError instanceof Error ? restoreError.message : 'Restore failed.')
    }
  }

  return (
    <div className="study-view">
      <header className="study-toolbar">
        <div>
          <div className="panel-title">Concept learning</div>
          <h1>Study documents</h1>
          <p>Grow grounded cards into editable, source-backed explanations.</p>
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
          <span>Anchor card</span>
          <select value={selectedCardId} onChange={(event) => setSelectedCardId(event.target.value)}>
            {cards.map((card) => (
              <option key={card.id} value={card.id}>{card.title}</option>
            ))}
          </select>
        </label>
        <button type="button" disabled={!selectedCardId || isSaving} onClick={() => void createDocument()}>
          <FilePlus2 size={16} /> New document
        </button>
      </header>

      {(error || message) && (
        <div className={error ? 'study-notice error' : 'study-notice success'}>
          {error ?? message}
        </div>
      )}

      <div className="study-layout">
        <aside className="study-library">
          <section>
            <div className="study-section-heading">
              <div><strong>Documents</strong><span>{documents.length}</span></div>
              <BookOpenText size={16} />
            </div>
            <div className="study-document-list">
              {documents.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  className={item.id === selectedDocumentId ? 'selected' : ''}
                  onClick={() => setSelectedDocumentId(item.id)}
                >
                  <strong>{item.title}</strong>
                  <span>{item.summary || 'No summary'}</span>
                  <small>{item.status} · {item.generation_mode}</small>
                </button>
              ))}
              {!documents.length && !isLoading && <div className="study-empty-small">No study documents yet.</div>}
            </div>
          </section>

          <section>
            <div className="study-section-heading">
              <div><strong>Local sources</strong><span>{assets.length}</span></div>
              <Upload size={16} />
            </div>
            <input
              ref={fileInputRef}
              className="study-file-input"
              type="file"
              accept=".pptx,.pdf,.docx,.txt,.md,.markdown"
              disabled={isUploading}
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void uploadAsset(file)
              }}
            />
            <div className="study-asset-list">
              {assets.map((asset) => (
                <div key={asset.id}>
                  <div>
                    <strong>{asset.original_filename}</strong>
                    <span>{asset.asset_type} · {asset.unit_count} units · {formatBytes(asset.size_bytes)}</span>
                  </div>
                  <button type="button" title="Delete source" aria-label={`Delete ${asset.original_filename}`} onClick={() => void deleteAsset(asset)}>
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          </section>
        </aside>

        <main className="study-document-workspace">
          {document ? (
            <>
              <div className="study-document-toolbar">
                <div className="study-mode-toggle">
                  <button type="button" className={mode === 'preview' ? 'active' : ''} onClick={() => setMode('preview')}>
                    <Eye size={15} /> Preview
                  </button>
                  <button type="button" className={mode === 'edit' ? 'active' : ''} onClick={() => setMode('edit')}>
                    <Pencil size={15} /> Edit
                  </button>
                </div>
                <select value={status} onChange={(event) => setStatus(event.target.value as LearningDocument['status'])}>
                  <option value="draft">draft</option>
                  <option value="reviewed">reviewed</option>
                  <option value="needs_fix">needs fix</option>
                </select>
                <button type="button" disabled={isSaving} onClick={() => void saveDocument()}>
                  <Save size={15} /> Save
                </button>
                <button className="danger-button" type="button" onClick={() => void deleteDocument()}>
                  <Trash2 size={15} />
                </button>
              </div>
              {mode === 'edit' ? (
                <div className="study-editor">
                  <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Document title" />
                  <textarea className="study-summary-input" value={summary} onChange={(event) => setSummary(event.target.value)} placeholder="Short summary" />
                  <textarea className="study-markdown-input" value={body} onChange={(event) => setBody(event.target.value)} placeholder="Write Markdown" />
                </div>
              ) : (
                <article className="study-markdown">
                  <ReactMarkdown>{body}</ReactMarkdown>
                </article>
              )}
            </>
          ) : (
            <div className="study-empty">
              <BookOpenText size={34} />
              <h2>Select or create a study document</h2>
              <p>{primaryCard ? `Use ${primaryCard.title} as the anchor.` : 'Choose an anchor card first.'}</p>
            </div>
          )}
        </main>

        <aside className="study-inspector">
          {document ? (
            <>
              <section>
                <div className="study-section-heading">
                  <div><strong>Generate draft</strong><span>{selectedModel || 'default model'}</span></div>
                  <Sparkles size={16} />
                </div>
                <textarea value={focus} onChange={(event) => setFocus(event.target.value)} placeholder="Optional focus, e.g. mathematical intuition" />
                <div className="study-check-list">
                  <div className="study-list-label">Source files</div>
                  {readyAssets.map((asset) => (
                    <label key={asset.id}>
                      <input
                        type="checkbox"
                        checked={selectedAssetIds.has(asset.id)}
                        onChange={(event) => setSelectedAssetIds((current) => {
                          const next = new Set(current)
                          if (event.target.checked) next.add(asset.id)
                          else next.delete(asset.id)
                          return next
                        })}
                      />
                      <span>{asset.original_filename}</span>
                    </label>
                  ))}
                </div>
                <div className="study-check-list supporting">
                  <div className="study-list-label">Supporting cards</div>
                  {supportCandidates.map((card) => (
                    <label key={card.id}>
                      <input
                        type="checkbox"
                        checked={supportingCardIds.has(card.id)}
                        onChange={(event) => setSupportingCardIds((current) => {
                          const next = new Set(current)
                          if (event.target.checked) next.add(card.id)
                          else next.delete(card.id)
                          return next
                        })}
                      />
                      <span>{card.title}</span>
                    </label>
                  ))}
                </div>
                <button className="study-generate-button" type="button" disabled={isGenerating} onClick={() => void generateDocument()}>
                  <Sparkles size={16} /> {isGenerating ? 'Generating locally' : 'Generate grounded draft'}
                </button>
              </section>

              <section>
                <div className="study-section-heading">
                  <div><strong>References</strong><span>{document.sources.length}</span></div>
                  <Link2 size={16} />
                </div>
                <div className="study-reference-list">
                  {document.sources.map((source) => (
                    <div key={source.id}>
                      <strong>[{source.label}] {sourceLocation(source)}</strong>
                      <span>{source.source_type === 'card_claim' ? 'course evidence' : 'supplementary source'}</span>
                      <p>{source.quote}</p>
                    </div>
                  ))}
                  {!document.sources.length && <div className="study-empty-small">No generated references yet.</div>}
                </div>
              </section>

              <section>
                <div className="study-section-heading">
                  <div><strong>Versions</strong><span>{document.versions.length}</span></div>
                  <History size={16} />
                </div>
                <div className="study-version-list">
                  {document.versions.map((version) => (
                    <button key={version.id} type="button" onClick={() => void restoreVersion(version.version_number)}>
                      <strong>v{version.version_number}</strong>
                      <span>{version.change_source} · {new Date(version.created_at).toLocaleString()}</span>
                    </button>
                  ))}
                </div>
              </section>
            </>
          ) : <div className="study-empty-small">Document tools appear after selection.</div>}
        </aside>
      </div>
    </div>
  )
}
