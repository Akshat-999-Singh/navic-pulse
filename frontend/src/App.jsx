import { useEffect, useState } from 'react'
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

const IDLE = { status: 'idle', fileName: null, startedAt: null }

// "2026-09-16T18:20:11+00:00" -> "2026-09-16 18:20 UTC"; anything else as given.
function formatGenerated(iso) {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(iso ?? '')
  return match ? `${match[1]} ${match[2]} UTC` : (iso ?? '—')
}

// Everything above the views: banner, masthead, run status. Their summed
// height is published as --chrome-height, which the scene column's sticky
// offset reads, so the scene holds its position whatever wraps.
function useChromeHeight() {
  useEffect(() => {
    const root = document.documentElement
    const parts = () => [...document.querySelectorAll('.banner, .masthead, .run-status')]
    const publish = () => {
      const height = parts().reduce((sum, el) => sum + el.getBoundingClientRect().height, 0)
      // Exact, not rounded: a subpixel over the in-flow position lets the
      // column's end nudge the scene at the bottom of the page.
      root.style.setProperty('--chrome-height', `${height.toFixed(2)}px`)
    }
    const observer = new ResizeObserver(publish)
    parts().forEach((el) => observer.observe(el))
    publish()
    return () => {
      observer.disconnect()
      root.style.removeProperty('--chrome-height')
    }
  }) // no dependency list: the run-status bar comes and goes, and must be re-observed
}

function RunStatus({ run, fileName, onReset }) {
  if (!run) return null
  const computed = run.source === 'computed'
  return (
    <div className="run-status" data-source={run.source}>
      <p className="run-status-text">
        <span className="run-status-label">{computed ? 'Computed run' : 'Reference run'}</span>
        <span className="quiet">
          {computed
            ? `${fileName ?? 'Uploaded batch'} · analysed ${formatGenerated(run.generated_at)}`
            : 'Precomputed offline'}
        </span>
      </p>
      {computed && (
        <button type="button" className="run-status-reset" onClick={onReset}>
          Return to reference run
        </button>
      )}
    </div>
  )
}

export default function App() {
  const { run: referenceRun, failed } = useRun()
  // The run every view shows: a computed run once an upload succeeds, otherwise
  // the reference run useRun() loaded on mount.
  const [computed, setComputed] = useState(null) // { run, fileName }
  const [analysis, setAnalysis] = useState(IDLE)
  const [view, setView] = useState('upload')
  const [selectedId, setSelectedId] = useState(null)

  const run = computed?.run ?? referenceRun
  useChromeHeight()

  function selectSatellite(id) {
    setSelectedId(id)
    setView('satellite')
  }

  // A different run may not contain the selected satellite, and a satellite
  // present in both has different windows: start from the grid again.
  function showRun(next) {
    setComputed(next)
    setSelectedId(null)
  }

  function startAnalysis(fileName) {
    setAnalysis({ status: 'running', fileName, startedAt: Date.now() })
    setView('pipeline')
  }

  function analysisSucceeded(result, fileName) {
    showRun({ run: result, fileName })
    setAnalysis(IDLE)
    setView('constellation')
  }

  // 'rejected': the service answered with row errors or a size refusal, shown
  // on Upload. 'unavailable': no analysis happened; the reference run stays.
  function analysisFailed(reason) {
    if (reason === 'unavailable') {
      showRun(null)
      setAnalysis({ ...IDLE, status: 'unavailable' })
    } else {
      setAnalysis(IDLE)
    }
    setView('upload')
  }

  function returnToReference() {
    showRun(null)
    setAnalysis(IDLE)
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

      <RunStatus run={run} fileName={computed?.fileName} onReset={returnToReference} />

      {/* All views stay mounted so they keep internal state; inactive ones are hidden. */}
      {VIEWS.map(({ id }) => {
        if (id === 'upload') {
          return (
            <Upload
              key={id}
              run={run}
              hidden={view !== id}
              analysing={analysis.status === 'running'}
              unavailable={analysis.status === 'unavailable'}
              onAnalysisStart={startAnalysis}
              onAnalysisSuccess={analysisSucceeded}
              onAnalysisFailure={analysisFailed}
            />
          )
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
          return (
            <Pipeline
              key={id}
              run={run}
              fileName={computed?.fileName}
              analysis={analysis}
              hidden={view !== id}
            />
          )
        }
        return null
      })}
    </>
  )
}
