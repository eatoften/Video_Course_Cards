import { useEffect, useState, type ChangeEvent } from 'react'
import './App.css'

type HealthResponse = {
  status: string
}

function App() {
  const [message, setMessage] = useState('尚未检查')
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [videoName, setVideoName] = useState<string | null>(null)

  async function checkBackend() {
    setMessage('检查中...')

    try {
      const response = await fetch('http://127.0.0.1:8000/health')

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data: HealthResponse = await response.json()

      if (data.status === 'ok') {
        setMessage('后端连接成功')
      } else {
        setMessage('后端返回异常')
      }
    } catch (error) {
      console.error(error)
      setMessage('无法连接后端')
    }
  }

  function handleVideoChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]

    if (!file) {
      return
    }

    const objectUrl = URL.createObjectURL(file)

    setVideoUrl(objectUrl)
    setVideoName(file.name)
  }

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl)
      }
    }
  }, [videoUrl])

  return (
    <main>
      <h1>Video Course Cards</h1>

      <section>
        <h2>Backend</h2>
        <p>{message}</p>
        <button onClick={checkBackend}>检查后端连接</button>
      </section>

      <section>
        <h2>Video</h2>

        <input
          type="file"
          accept="video/*"
          onChange={handleVideoChange}
        />

        {videoName && <p>当前视频：{videoName}</p>}

        {videoUrl && (
          <video
            src={videoUrl}
            controls
            preload="metadata"
            width="800"
          />
        )}
      </section>
    </main>
  )
}

export default App