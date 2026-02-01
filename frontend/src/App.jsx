import React, { useState, useCallback } from 'react'
import PromptScreen from './components/PromptScreen'
import LoadingScreen from './components/LoadingScreen'
import WorldView from './components/WorldView'

// API base URL - use Modal backend in production
const API_URL = import.meta.env.PROD 
  ? 'https://revelium-studio--lingbot-world-fastapi-app.modal.run'
  : ''

// Application states
const STATES = {
  PROMPT: 'prompt',
  LOADING: 'loading',
  WORLD: 'world',
}

function App() {
  const [appState, setAppState] = useState(STATES.PROMPT)
  const [sessionId, setSessionId] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [loadingStatus, setLoadingStatus] = useState('Initializing...')
  const [error, setError] = useState(null)

  const handleGenerate = useCallback(async (promptText) => {
    setPrompt(promptText)
    setAppState(STATES.LOADING)
    setLoadingStatus('Creating world...')
    setError(null)

    try {
      const response = await fetch(`${API_URL}/api/world/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to create world')
      }

      const data = await response.json()
      setSessionId(data.session_id)
      setLoadingStatus('Generating initial frames...')

      // Wait a moment for initial frames to generate
      await new Promise(resolve => setTimeout(resolve, 1500))

      setAppState(STATES.WORLD)
    } catch (err) {
      console.error('Generation error:', err)
      setError(err.message)
      setAppState(STATES.PROMPT)
    }
  }, [])

  const handleBackToPrompt = useCallback(async () => {
    // Clean up session
    if (sessionId) {
      try {
        await fetch(`${API_URL}/api/world/${sessionId}`, { method: 'DELETE' })
      } catch (err) {
        console.error('Failed to delete session:', err)
      }
    }

    setSessionId(null)
    setAppState(STATES.PROMPT)
    setError(null)
  }, [sessionId])

  return (
    <div className="app">
      {appState === STATES.PROMPT && (
        <PromptScreen
          onGenerate={handleGenerate}
          initialPrompt={prompt}
          error={error}
        />
      )}

      {appState === STATES.LOADING && (
        <LoadingScreen
          prompt={prompt}
          status={loadingStatus}
        />
      )}

      {appState === STATES.WORLD && (
        <WorldView
          sessionId={sessionId}
          prompt={prompt}
          onBack={handleBackToPrompt}
          apiUrl={API_URL}
        />
      )}
    </div>
  )
}

export default App
