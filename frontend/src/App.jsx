import { useState } from 'react'

const VIEWS = [
  { id: 'upload', label: 'Upload' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'constellation', label: 'Constellation' },
  { id: 'satellite', label: 'Satellite' },
]

export default function App() {
  const [view, setView] = useState('upload')

  return (
    <>
      {/* Must stay visible on every view. */}
      <p role="note" className="banner">
        Synthetic telemetry — generated from published iRAFS specifications. Not live ISRO data.
      </p>

      <header className="masthead">
        <p className="wordmark">NavIC Pulse</p>
        <nav aria-label="Views" className="views">
          {VIEWS.map(({ id, label }) => {
            const active = view === id
            return (
              <button
                key={id}
                type="button"
                aria-current={active ? 'page' : undefined}
                onClick={() => setView(id)}
              >
                {label}
              </button>
            )
          })}
        </nav>
      </header>

      {/* All views stay mounted so they keep internal state; inactive ones are hidden. */}
      {VIEWS.map(({ id, label }) => (
        <main key={id} hidden={view !== id} className="view">
          <h1>{label}</h1>
        </main>
      ))}
    </>
  )
}
