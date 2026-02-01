import React, { useState } from 'react'

const EXAMPLE_PROMPTS = [
  'A neon cyberpunk alley at night with rain',
  'Ancient forest with glowing mushrooms',
  'Futuristic space station interior',
  'Underwater coral reef city',
  'Steampunk Victorian streets',
]

function PromptScreen({ onGenerate, initialPrompt = '', error }) {
  const [prompt, setPrompt] = useState(initialPrompt)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!prompt.trim() || isSubmitting) return

    setIsSubmitting(true)
    try {
      await onGenerate(prompt.trim())
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleExampleClick = (example) => {
    setPrompt(example)
  }

  return (
    <div className="prompt-screen">
      <header className="prompt-screen__header">
        <h1 className="prompt-screen__title">LingBot-World</h1>
        <p className="prompt-screen__subtitle">
          Describe a world and explore it in real-time. 
          Powered by open-source AI world generation.
        </p>
      </header>

      <form className="prompt-form" onSubmit={handleSubmit}>
        <div className="prompt-input-wrapper">
          <input
            type="text"
            className="prompt-input"
            placeholder="Describe your world..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            autoFocus
            disabled={isSubmitting}
          />
        </div>

        <button
          type="submit"
          className="generate-btn"
          disabled={!prompt.trim() || isSubmitting}
        >
          <span>✦</span>
          {isSubmitting ? 'Generating...' : 'Generate World'}
        </button>

        {error && (
          <p style={{ color: 'var(--accent-secondary)', textAlign: 'center' }}>
            {error}
          </p>
        )}
      </form>

      <div className="examples">
        <p className="examples__title">Try an example</p>
        <div className="examples__list">
          {EXAMPLE_PROMPTS.map((example, index) => (
            <button
              key={index}
              className="example-chip"
              onClick={() => handleExampleClick(example)}
              disabled={isSubmitting}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default PromptScreen
