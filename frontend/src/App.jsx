import { useState } from 'react'
import Constellation from './Constellation.jsx'
import Pipeline from './Pipeline.jsx'
import Satellite from './Satellite.jsx'
import Upload from './Upload.jsx'
import { useRun } from './data.js'

const VIEWS = [
  { id: 'upload', label: 'Upload' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'constellation', label: 'Constellation' },
  { id: 'satellite', label: 'Satellite' },
]

export default function App() {
  const { run, failed } = useRun()
  const [view, setView] = useState('upload')
  const [selectedId, setSelectedId] = useState(null)

  function selectSatellite(id) {
    setSelectedId(id)
    setView('satellite')
  }

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
      {VIEWS.map(({ id }) => {
        if (id === 'upload') {
          return <Upload key={id} hidden={view !== id} onComplete={() => setView('pipeline')} />
        }
        if (id === 'constellation') {
          return (
            <Constellation
              key={id}
              run={run}
              failed={failed}
              hidden={view !== id}
              onSelect={selectSatellite}
            />
          )
        }
        if (id === 'satellite') {
          return (
            <Satellite
              key={id}
              run={run}
              failed={failed}
              satelliteId={selectedId}
              hidden={view !== id}
              onBack={() => setView('constellation')}
            />
          )
        }
        if (id === 'pipeline') {
          return <Pipeline key={id} run={run} hidden={view !== id} />
        }
        return null
      })}
    </>
  )
}
