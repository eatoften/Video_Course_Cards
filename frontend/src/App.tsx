import { useState } from 'react'
import './App.css'

type HealthResponse = {
  status: string
}

function App() {
  const [message, setMessage] = useState('尚未检查')

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

  return (
    <main>
      <h1>Video Course Cards</h1>
      <p>{message}</p>
      <button onClick={checkBackend}>检查后端连接</button>
    </main>
  )
}

export default App