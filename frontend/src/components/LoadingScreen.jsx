import React from 'react'

function LoadingScreen({ prompt, status }) {
  return (
    <div className="loading-screen">
      <div className="loading-spinner" />
      <p className="loading-text">Generating your world</p>
      <p className="loading-status">{status}</p>
      <p 
        style={{ 
          marginTop: '2rem', 
          color: 'var(--text-muted)',
          maxWidth: '400px',
          textAlign: 'center',
          fontSize: '0.875rem'
        }}
      >
        "{prompt}"
      </p>
    </div>
  )
}

export default LoadingScreen
