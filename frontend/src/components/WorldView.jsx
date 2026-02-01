import React, { useEffect, useRef, useState, useCallback } from 'react'

// Control key mappings
const KEY_ACTIONS = {
  KeyW: 'move_forward',
  KeyS: 'move_backward',
  KeyA: 'move_left',
  KeyD: 'move_right',
  KeyQ: 'turn_left',
  KeyE: 'turn_right',
  ArrowUp: 'look_up',
  ArrowDown: 'look_down',
  ArrowLeft: 'turn_left',
  ArrowRight: 'turn_right',
  Space: 'move_up',
  ShiftLeft: 'move_down',
  ShiftRight: 'move_down',
}

function WorldView({ sessionId, prompt, onBack, apiUrl = '' }) {
  const [frameData, setFrameData] = useState(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [isConnected, setIsConnected] = useState(false)
  const [isGenerating, setIsGenerating] = useState(true)
  const [activeKeys, setActiveKeys] = useState(new Set())
  const [pointerLocked, setPointerLocked] = useState(false)
  const [statusMessage, setStatusMessage] = useState('Connecting...')
  
  const wsRef = useRef(null)
  const viewportRef = useRef(null)
  const frameRef = useRef(null)
  const keysRef = useRef(new Set())
  const controlIntervalRef = useRef(null)

  // WebSocket connection
  useEffect(() => {
    if (!sessionId) return

    // Determine WebSocket URL based on apiUrl
    let wsUrl
    if (apiUrl) {
      // Use Modal backend WebSocket
      wsUrl = apiUrl.replace('https://', 'wss://').replace('http://', 'ws://') + `/ws/${sessionId}`
    } else {
      // Local development
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`
    }
    
    console.log('Connecting to WebSocket:', wsUrl)
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        
        if (message.type === 'frame') {
          setFrameData(message.data)
          setFrameIndex(message.frame_index)
          setIsGenerating(message.is_generating)
        } else if (message.type === 'status') {
          console.log('Status:', message.status, message.message)
        }
      } catch (err) {
        console.error('Failed to parse message:', err)
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setIsConnected(false)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    // Fallback: poll for frames if WebSocket fails
    const pollInterval = setInterval(async () => {
      if (ws.readyState !== WebSocket.OPEN) {
        try {
          const response = await fetch(`${apiUrl}/api/world/${sessionId}/frame`)
          if (response.ok) {
            const blob = await response.blob()
            const reader = new FileReader()
            reader.onloadend = () => {
              const base64 = reader.result.split(',')[1]
              setFrameData(base64)
            }
            reader.readAsDataURL(blob)
          }
        } catch (err) {
          // Ignore polling errors if WebSocket is working
        }
      }
    }, 500)

    return () => {
      clearInterval(pollInterval)
      ws.close()
    }
  }, [sessionId])

  // Send control action via WebSocket
  const sendControl = useCallback((action, mouseDx = 0, mouseDy = 0) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'control',
        action,
        mouse_dx: mouseDx,
        mouse_dy: mouseDy,
      }))
    }
  }, [])

  // Continuous control sending for held keys
  useEffect(() => {
    controlIntervalRef.current = setInterval(() => {
      keysRef.current.forEach((action) => {
        sendControl(action)
      })
    }, 50) // Send controls at 20Hz

    return () => {
      if (controlIntervalRef.current) {
        clearInterval(controlIntervalRef.current)
      }
    }
  }, [sendControl])

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      const action = KEY_ACTIONS[e.code]
      if (action && !keysRef.current.has(action)) {
        keysRef.current.add(action)
        setActiveKeys(new Set(keysRef.current))
        sendControl(action)
      }
    }

    const handleKeyUp = (e) => {
      const action = KEY_ACTIONS[e.code]
      if (action) {
        keysRef.current.delete(action)
        setActiveKeys(new Set(keysRef.current))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [sendControl])

  // Mouse controls (pointer lock for smooth camera)
  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return

    const handleClick = () => {
      viewport.requestPointerLock?.()
    }

    const handlePointerLockChange = () => {
      setPointerLocked(document.pointerLockElement === viewport)
    }

    const handleMouseMove = (e) => {
      if (document.pointerLockElement === viewport) {
        sendControl(null, e.movementX, e.movementY)
      }
    }

    viewport.addEventListener('click', handleClick)
    document.addEventListener('pointerlockchange', handlePointerLockChange)
    document.addEventListener('mousemove', handleMouseMove)

    return () => {
      viewport.removeEventListener('click', handleClick)
      document.removeEventListener('pointerlockchange', handlePointerLockChange)
      document.removeEventListener('mousemove', handleMouseMove)
    }
  }, [sendControl])

  // Exit pointer lock with Escape
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Escape' && pointerLocked) {
        document.exitPointerLock?.()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [pointerLocked])

  const isKeyActive = (keys) => {
    return keys.some(key => {
      const action = KEY_ACTIONS[key]
      return action && activeKeys.has(action)
    })
  }

  return (
    <div className="world-view">
      <div className="world-viewport" ref={viewportRef}>
        {frameData ? (
          <img
            ref={frameRef}
            className={`world-frame ${isGenerating ? 'world-frame--loading' : ''}`}
            src={`data:image/jpeg;base64,${frameData}`}
            alt="Generated world"
          />
        ) : (
          <div className="loading-spinner" />
        )}

        {/* HUD */}
        <div className="hud">
          <div className="hud-item">
            <span className="hud-label">Frame</span>
            <span className="hud-value">{frameIndex}</span>
          </div>
          <div className="hud-item">
            <span className="hud-label">Status</span>
            <span className="hud-value">
              {isConnected ? (isGenerating ? 'Generating' : 'Ready') : 'Connecting...'}
            </span>
          </div>
        </div>

        {/* Top bar */}
        <div className="top-bar">
          <button className="top-bar-btn" onClick={onBack}>
            ← New World
          </button>
        </div>

        {/* Pointer lock hint */}
        {!pointerLocked && (
          <div className="pointer-hint">
            Click to enable mouse look
          </div>
        )}

        {/* On-screen controls */}
        <div className="controls-overlay">
          <div className="control-group">
            <button
              className={`control-btn ${isKeyActive(['KeyQ']) ? 'control-btn--active' : ''}`}
              onMouseDown={() => sendControl('turn_left')}
            >
              Q
            </button>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div className="control-group" style={{ justifyContent: 'center' }}>
              <button
                className={`control-btn ${isKeyActive(['KeyW']) ? 'control-btn--active' : ''}`}
                onMouseDown={() => sendControl('move_forward')}
              >
                W
              </button>
            </div>
            <div className="control-group">
              <button
                className={`control-btn ${isKeyActive(['KeyA']) ? 'control-btn--active' : ''}`}
                onMouseDown={() => sendControl('move_left')}
              >
                A
              </button>
              <button
                className={`control-btn ${isKeyActive(['KeyS']) ? 'control-btn--active' : ''}`}
                onMouseDown={() => sendControl('move_backward')}
              >
                S
              </button>
              <button
                className={`control-btn ${isKeyActive(['KeyD']) ? 'control-btn--active' : ''}`}
                onMouseDown={() => sendControl('move_right')}
              >
                D
              </button>
            </div>
          </div>

          <div className="control-group">
            <button
              className={`control-btn ${isKeyActive(['KeyE']) ? 'control-btn--active' : ''}`}
              onMouseDown={() => sendControl('turn_right')}
            >
              E
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WorldView
