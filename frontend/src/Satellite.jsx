import { useMemo, useState } from 'react'
import AnomalyChart from './AnomalyChart.jsx'
import { ClockPill, OrbitText } from './Constellation.jsx'
import { healthColor, healthTextColor, satelliteHealth } from './health.js'

const TRACK_STOPS = 64 // gradient stops sampled across the scrubber track

const PROGNOSIS_CAVEAT =
  'Estimate is ±1 standard error and assumes the observed decline continues at its current rate.'

// The windows array is thinned, so every position here is an ARRAY position,
// never a window.index.
function defaultPosition(satellite) {
  const { windows, persistence_confirmed_index: confirmed } = satellite
  if (windows.length === 0) return 0

  const atConfirmed = windows.findIndex((w) => w.index === confirmed)
  if (atConfirmed !== -1) return atConfirmed

  const healthy = confirmed == null && !windows.some((w) => w.flagged)
  if (healthy) return windows.length - 1

  let highest = 0
  windows.forEach((w, i) => {
    if (w.error > windows[highest].error) highest = i
  })
  return highest
}

function timelineAlt(satellite) {
  const parts = ['reconstruction error per window across the mission', 'the detection threshold']
  if (satellite.windows.some((w) => w.flagged)) parts.push('flagged windows shaded')
  if (satellite.persistence_confirmed_index != null) {
    parts.push('the persistence-confirmed detection marked')
  }
  if (satellite.baseline_flagged_index != null) {
    parts.push('the fixed-limit baseline detection marked')
  }
  const last = parts.pop()
  return `Anomaly timeline for ${satellite.id}: ${parts.join(', ')} and ${last}.`
}

function Header({ satellite, onBack }) {
  return (
    <header className="detail-header">
      <button type="button" className="back" onClick={onBack}>
        Back to Constellation
      </button>
      <div className="sat-head">
        <h1 className="detail-id">{satellite.id}</h1>
        <ClockPill clockType={satellite.clock_type} />
      </div>
      <p className="sat-meta detail-line">
        {satellite.name} · {satellite.launch_date} · <OrbitText orbit={satellite.orbit} />
      </p>
      <p className="sat-severity detail-line">
        <span className="severity-dot" data-severity={satellite.severity} aria-hidden="true" />
        {satellite.severity}
      </p>
      {satellite.peak_severity !== satellite.severity && (
        <p className="sat-muted detail-line">Peak: {satellite.peak_severity}</p>
      )}
    </header>
  )
}

function Timeline({ satellite, position, onPosition }) {
  const { windows } = satellite
  const selected = windows[position]
  const firstFlag = windows.findIndex((w) => w.flagged)
  const confirmed = windows.findIndex((w) => w.index === satellite.persistence_confirmed_index)
  const describe = (w) => `Window ${w.index} · days ${w.start}–${w.end} · error ${w.error.toFixed(4)}`

  // The track is the satellite's life on the health spectrum, laid out by array
  // position so it lines up with the thumb (the windows array is thinned).
  const health = useMemo(() => satelliteHealth(satellite), [satellite])
  const track = useMemo(() => {
    const last = windows.length - 1
    const count = Math.min(TRACK_STOPS, windows.length)
    const stops = []
    for (let k = 0; k < count; k++) {
      const i = count === 1 ? 0 : Math.round((k * last) / (count - 1))
      const at = last === 0 ? 0 : (i / last) * 100
      stops.push(`${healthColor(health.t(windows[i].error))} ${at.toFixed(2)}%`)
    }
    return `linear-gradient(90deg, ${stops.join(', ')})`
  }, [windows, health])
  const selectedT = selected ? health.t(selected.error) : 0
  // Data-driven gradient and the colours sampled from it, handed to CSS.
  const spectrum = {
    '--track-gradient': track,
    '--thumb-color': healthColor(selectedT),
    '--readout-color': healthTextColor(selectedT),
  }

  return (
    <section className="detail-section" aria-labelledby="timeline-heading">
      <h2 id="timeline-heading">Anomaly timeline</h2>
      {/* satellite.plot (the pre-rendered PNG) may still be in the fixture; it is not rendered. */}
      <AnomalyChart
        windows={windows}
        threshold={satellite.threshold}
        confirmedIndex={satellite.persistence_confirmed_index}
        baselineIndex={satellite.baseline_flagged_index}
        selectedIndex={position}
        onSelect={onPosition}
        health={health}
        label={timelineAlt(satellite)}
      />

      {selected && (
        <div className="scrubber" style={spectrum}>
          <input
            type="range"
            min="0"
            max={windows.length - 1}
            step="1"
            value={position}
            onChange={(event) => onPosition(Number(event.target.value))}
            aria-label="Window"
            aria-valuetext={`${describe(selected)}, ${selected.flagged ? 'flagged' : 'not flagged'}`}
          />
          <p className="window-readout">
            <span className="window-value">{describe(selected)}</span>
            <span className="flag-state" data-flagged={selected.flagged}>
              {selected.flagged ? 'Flagged' : 'Not flagged'}
            </span>
          </p>
          <div className="jump-buttons">
            <button type="button" disabled={firstFlag === -1} onClick={() => onPosition(firstFlag)}>
              First flag
            </button>
            <button type="button" disabled={confirmed === -1} onClick={() => onPosition(confirmed)}>
              Confirmed
            </button>
          </div>
        </div>
      )}
    </section>
  )
}

function Contributions({ window }) {
  const channels = [...window.channels].sort((a, b) => b.contribution - a.contribution)

  return (
    <section className="detail-section" aria-labelledby="contribution-heading">
      <h2 id="contribution-heading">Why this window was flagged</h2>
      {!window.flagged && (
        <p className="quiet">
          This window is within normal limits; contributions show the ordinary reconstruction
          error.
        </p>
      )}
      <ul className="channel-bars">
        {channels.map((channel, i) => {
          const pct = channel.contribution * 100
          return (
            <li key={channel.name} className="channel-bar">
              <span className="channel-name">{channel.name}</span>
              <svg className="bar" aria-hidden="true">
                <rect className="bar-track" width="100%" height="14" rx="7" />
                <rect
                  className={i === 0 ? 'channel-fill channel-fill-top' : 'channel-fill'}
                  width={`${Math.min(Math.max(pct, 0), 100)}%`}
                  height="14"
                  rx="7"
                />
              </svg>
              <span className="channel-pct">{pct.toFixed(1)}%</span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function Prognosis({ life }) {
  if (life == null) {
    return (
      <section className="detail-section" aria-labelledby="prognosis-heading">
        <h2 id="prognosis-heading">Prognosis</h2>
        <p className="quiet">No decline trend detected</p>
      </section>
    )
  }

  // Scale from zero so the band's width honestly reflects its size relative to
  // the estimate, rather than filling the bar whatever the uncertainty.
  const max = Math.max(life.high, life.estimate) * 1.25
  const x = (days) => (Math.max(days, 0) / max) * 100

  return (
    <section className="detail-section" aria-labelledby="prognosis-heading">
      <h2 id="prognosis-heading">Prognosis</h2>
      <div className="prognosis">
        <p className="prognosis-estimate">
          <span className="prognosis-value">{life.estimate}</span>
          <span className="prognosis-unit">
            d remaining ({life.low}–{life.high})
          </span>
        </p>
        <p className="prognosis-caveat">{PROGNOSIS_CAVEAT}</p>
        <svg className="bar range-bar" aria-hidden="true">
          <rect className="bar-track" width="100%" height="14" rx="7" />
          <rect className="range-band" x={`${x(life.low)}%`} width={`${x(life.high) - x(life.low)}%`} height="14" rx="7" />
          <rect className="range-mark" x={`${x(life.estimate)}%`} width="2" height="14" transform="translate(-1 0)" />
        </svg>
        <p className="range-scale">
          <span>0 d</span>
          <span>{Math.round(max)} d</span>
        </p>
      </div>
    </section>
  )
}

function Detail({ satellite, onBack }) {
  const [position, setPosition] = useState(() => defaultPosition(satellite))
  const selected = satellite.windows[position]

  return (
    <>
      <Header satellite={satellite} onBack={onBack} />
      <Timeline satellite={satellite} position={position} onPosition={setPosition} />
      {selected && <Contributions window={selected} />}
      <Prognosis life={satellite.remaining_life_days} />
    </>
  )
}

export default function Satellite({ run, failed, satelliteId, hidden, onBack }) {
  let body
  if (!run) {
    body = <p className="quiet">{failed ? 'Run data unavailable.' : 'Loading run…'}</p>
  } else if (satelliteId == null) {
    body = <p className="quiet">Select a satellite from the Constellation view.</p>
  } else {
    const satellite = run.satellites.find((s) => s.id === satelliteId)
    body = satellite ? (
      // Keyed so the scrubber resets to the new satellite's default window.
      <Detail key={satellite.id} satellite={satellite} onBack={onBack} />
    ) : (
      <p className="quiet">{satelliteId} is not in this run.</p>
    )
  }

  const showsDetail = run && satelliteId != null && run.satellites.some((s) => s.id === satelliteId)

  return (
    <main hidden={hidden} className="view satellite">
      {!showsDetail && <h1>Satellite</h1>}
      {body}
    </main>
  )
}
