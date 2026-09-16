import { leadTimeSummary } from './Constellation.jsx'
import ConstellationRail from './ConstellationRail.jsx'

// Architecture, not a live run: the fixture carries no stage timings, so this
// screen shows none. Every stage is "complete" because the committed results
// were produced offline before this build.
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

const WHAT_IS_REAL = [
  'Detection, prognosis and the baseline comparison run offline in Python; the results shown in this interface are those outputs.',
  'Telemetry is synthetic, generated from published iRAFS specifications.',
  "Threshold limits are derived from the healthy satellites' own observed distributions, not from a spec sheet.",
]

export default function Pipeline({ run, hidden }) {
  const summary = run?.summary
  const detail = summary?.lead_time_detail

  return (
    <main hidden={hidden} className="view pipeline split">
      <div className="split-primary">
        <header className="intake-header">
          <h1>Analysis pipeline</h1>
          <p className="lede">
            The stages below describe how the committed results were produced offline. This view is
            illustrative, not a live run.
          </p>
        </header>

        <ol className="stages">
          {STAGES.map((stage) => (
            <li key={stage.name} className="stage" data-state="complete">
              <span className="stage-name">{stage.name}</span>
              <span className="stage-detail">{stage.detail}</span>
            </li>
          ))}
        </ol>

        {summary && (
          <>
            <section className="intake-section" aria-labelledby="results-heading">
              <h2 id="results-heading">Results</h2>
              <p className="result-line">
                {summary.satellites} satellites analysed, {summary.flagged} flagged.
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
                {WHAT_IS_REAL.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>
          </>
        )}
      </div>

      <aside className="split-aside">
        <div className="split-sticky">
          <ConstellationRail run={run} />
        </div>
      </aside>
    </main>
  )
}
