import { useEffect, useState } from 'react'
import { leadTimeSummary } from './Constellation.jsx'
import OrbitScene from './OrbitScene.jsx'
import { FALLBACK_NOTE } from './data.js'

// Reference run: architecture, not a live run. The fixture carries no stage
// timings, so this screen shows none. Every stage is "complete" because the
// committed results were produced offline before this build.
const STAGES = [
  { name: 'Ingest', detail: 'Parse telemetry CSV, check required channels' },
  { name: 'Window', detail: '30-day sliding windows per satellite' },
  { name: 'Encode', detail: 'Autoencoder (150-64-12-64-150), trained on healthy data only' },
  { name: 'Score', detail: 'Reconstruction error per window' },
  { name: 'Detect', detail: 'CUSUM on the score, plus a static threshold at 0.498' },
  { name: 'Persistence', detail: '60 consecutive flagged windows required to confirm' },
  { name: 'Baseline', detail: 'Fixed-limit check on raw channels, mean ±3 sigma from healthy satellites' },
  { name: 'Prognosis', detail: 'Signal-level decline fitted to remaining useful life' },
]

// Computed run: one entry per module the function timed, keyed by run.stages[].name.
const COMPUTED_STAGES = {
  ingest: { name: 'Ingest', detail: 'Parse the CSV; check columns, days, cadence and values' },
  window: { name: 'Window', detail: '30-day sliding windows per satellite, scaled per channel' },
  infer: { name: 'Infer', detail: 'Autoencoder (150-64-12-64-150) forward pass, reconstruction error per window' },
  detect: { name: 'Detect', detail: 'Static threshold, CUSUM and 60-window persistence, parameters fixed offline' },
  baseline: {
    name: 'Baseline',
    detail: 'Fixed-limit check on raw channels, mean ±3 sigma from the satellites in the batch the model never flagged',
  },
  explain: { name: 'Explain', detail: "Each channel's share of the reconstruction error" },
  prognosis: { name: 'Prognosis', detail: 'Signal-level decline fitted to remaining useful life' },
}

const WHAT_IS_REAL = {
  demo: [
    'Detection, prognosis and the baseline comparison run offline in Python; the results shown in this interface are those outputs.',
    'Telemetry is synthetic, generated from published iRAFS specifications.',
    "Threshold limits are derived from the healthy satellites' own observed distributions, not from a spec sheet.",
  ],
  computed: [
    'Detection, prognosis and the baseline comparison ran on the uploaded batch in a Python function; the model and thresholds are the ones trained offline.',
    'Telemetry is synthetic, generated from published iRAFS specifications.',
    'Stage timings are measured inside the function for this batch; they exclude network time.',
  ],
}

const formatMs = (ms) => (typeof ms === 'number' ? `${ms < 10 ? ms.toFixed(2) : ms.toFixed(1)} ms` : '—')

function Elapsed({ since }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])
  return <>{Math.max(0, Math.round((now - since) / 1000))} s</>
}

function Running({ analysis }) {
  return (
    <>
      <header className="intake-header">
        <h1>Analysis pipeline</h1>
        <p className="lede" aria-live="polite">
          Analysing {analysis.fileName ?? 'the uploaded batch'}…
        </p>
        <p className="analysis-progress" role="status">
          <span className="analysis-spinner" aria-hidden="true" />
          Waiting for the analysis service · <Elapsed since={analysis.startedAt ?? Date.now()} />
        </p>
      </header>
      <ol className="stages" aria-busy="true">
        {Object.values(COMPUTED_STAGES).map((stage) => (
          <li key={stage.name} className="stage" data-state="pending">
            <span className="stage-name">{stage.name}</span>
            <span className="stage-detail">{stage.detail}</span>
          </li>
        ))}
      </ol>
      <p className="quiet">Stage timings appear when the function returns; it reports them all at once.</p>
    </>
  )
}

export default function Pipeline({ run, fileName, analysis, hidden }) {
  const summary = run?.summary
  const detail = summary?.lead_time_detail
  const computed = run?.source === 'computed'
  const stages = computed && Array.isArray(run.stages) ? run.stages : []
  const totalMs = stages.reduce((sum, s) => sum + (typeof s.ms === 'number' ? s.ms : 0), 0)

  if (analysis?.status === 'running') {
    return (
      <main hidden={hidden} className="view pipeline split">
        <div className="split-primary">
          <Running analysis={analysis} />
        </div>
        <aside className="split-aside">
          <div className="split-sticky">
            <OrbitScene />
          </div>
        </aside>
      </main>
    )
  }

  return (
    <main hidden={hidden} className="view pipeline split">
      <div className="split-primary">
        <header className="intake-header">
          <h1>Analysis pipeline</h1>
          {computed ? (
            <p className="lede">
              These stages ran on {fileName ?? 'the uploaded batch'} in the analysis function, in{' '}
              {formatMs(totalMs)} of compute.
            </p>
          ) : (
            <p className="lede">
              The stages below describe how the committed results were produced offline. This view is
              illustrative, not a live run.
            </p>
          )}
          {analysis?.status === 'unavailable' && (
            <p className="fallback-note" role="alert">
              {FALLBACK_NOTE}
            </p>
          )}
        </header>

        {computed ? (
          <ol className="stages">
            {stages.map((stage) => {
              const known = COMPUTED_STAGES[stage.name]
              return (
                <li key={stage.name} className="stage" data-state="complete">
                  <span className="stage-head">
                    <span className="stage-name">{known?.name ?? stage.name}</span>
                    <span className="stage-ms">{formatMs(stage.ms)}</span>
                  </span>
                  {known && <span className="stage-detail">{known.detail}</span>}
                </li>
              )
            })}
          </ol>
        ) : (
          <ol className="stages">
            {STAGES.map((stage) => (
              <li key={stage.name} className="stage" data-state="complete">
                <span className="stage-name">{stage.name}</span>
                <span className="stage-detail">{stage.detail}</span>
              </li>
            ))}
          </ol>
        )}

        {summary && (
          <>
            <section className="intake-section" aria-labelledby="results-heading">
              <h2 id="results-heading">Results</h2>
              <p className="result-line">
                {summary.satellites} satellite{summary.satellites === 1 ? '' : 's'} analysed,{' '}
                {summary.flagged} flagged.
              </p>
              {detail && (
                <>
                  <p className="result-line">{leadTimeSummary(detail)}</p>
                  <p className="note">{detail.note}</p>
                </>
              )}
            </section>

            <section className="intake-section" aria-labelledby="real-heading">
              <h2 id="real-heading">What is real</h2>
              <ul className="real-list">
                {WHAT_IS_REAL[computed ? 'computed' : 'demo'].map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>
          </>
        )}
      </div>

      <aside className="split-aside">
        <div className="split-sticky">
          <OrbitScene />
        </div>
      </aside>
    </main>
  )
}
