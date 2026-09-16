import { useId, useMemo, useState } from 'react'
import { healthColor, satelliteHealth } from './health.js'

const SEVERITY_RANK = { critical: 0, warning: 1, normal: 2 }

const CLOCK_LABEL = { imported: 'Imported', irafs: 'Indigenous iRAFS' }

const SORTS = [
  { id: 'severity', label: 'Severity' },
  { id: 'life', label: 'Remaining life' },
  { id: 'id', label: 'ID' },
]

const CLOCK_FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'imported', label: 'Imported' },
  { id: 'irafs', label: 'Indigenous iRAFS' },
]

const ORBIT_FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'GEO', label: 'GEO' },
  { id: 'IGSO', label: 'IGSO' },
  { id: 'not-on-station', label: 'Not on station' },
]

const byId = (a, b) => a.id.localeCompare(b.id)

const COMPARE = {
  severity: (a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || byId(a, b),
  // Satellites with no remaining-life figure (the healthy units) sort last.
  life: (a, b) =>
    (a.remaining_life_days?.estimate ?? Infinity) - (b.remaining_life_days?.estimate ?? Infinity) ||
    byId(a, b),
  id: byId,
}

// GEO and IGSO match exactly. An "(intended)" orbit was never reached -- NVS-02
// is not on station -- so it matches neither and has a filter of its own.
const NOT_ON_STATION = '(intended)'

function matchesOrbit(orbit, filter) {
  if (filter === 'all') return true
  if (filter === 'not-on-station') return orbit.includes(NOT_ON_STATION)
  return orbit === filter
}

export function OrbitText({ orbit }) {
  const match = orbit.match(/^(.*?)\s*(\(.*\))$/)
  if (!match) return orbit
  return (
    <>
      {match[1]} <span className="orbit-qualifier">{match[2]}</span>
    </>
  )
}

export function ClockPill({ clockType }) {
  return (
    <span className="pill" data-clock={clockType}>
      {CLOCK_LABEL[clockType] ?? clockType}
    </span>
  )
}

const signed = (n) => (n > 0 ? `+${n}` : n < 0 ? `−${Math.abs(n)}` : '0')

export function leadTimeSummary(detail) {
  const direction = detail.mean_days >= 0 ? 'earlier' : 'later'
  return (
    `Mean ${Math.abs(detail.mean_days)} d ${direction} than fixed-limit monitoring ` +
    `(range ${signed(detail.min_days)} to ${signed(detail.max_days)} d, ` +
    `earlier on ${detail.faults_where_cusum_earlier} of ${detail.faults_total} faults)`
  )
}

function leadTimeText(days) {
  if (days == null) return '—'
  if (days > 0) return `Confirmed ${days} d before fixed limits`
  if (days < 0) return `Confirmed ${Math.abs(days)} d after fixed limits`
  return 'Confirmed the same day as fixed limits'
}

function remainingLifeText(life) {
  if (life == null) return '—'
  return `${life.estimate} d remaining (${life.low}–${life.high})`
}

function ButtonGroup({ label, options, value, onChange }) {
  return (
    <div className="control">
      <span className="control-label" id={`control-${label}`}>
        {label}
      </span>
      <div className="segmented" role="group" aria-labelledby={`control-${label}`}>
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            aria-pressed={value === option.id}
            onClick={() => onChange(option.id)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function SatelliteCard({ satellite, onSelect }) {
  const ratio = satellite.threshold > 0 ? satellite.anomaly_score / satellite.threshold : 0
  const fill = Math.min(Math.max(ratio, 0), 1) * 100
  // useId output contains characters that are awkward inside url(#...).
  const gradientId = `score-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`

  return (
    <button
      type="button"
      className="sat-card"
      data-severity={satellite.severity}
      onClick={() => onSelect(satellite.id)}
    >
      <span className="sat-head">
        <span className="sat-id">{satellite.id}</span>
        <ClockPill clockType={satellite.clock_type} />
      </span>
      <span className="sat-meta">
        {satellite.name} · {satellite.launch_date} · <OrbitText orbit={satellite.orbit} />
      </span>

      <span className="sat-severity">
        <span className="severity-dot" data-severity={satellite.severity} aria-hidden="true" />
        {satellite.severity}
      </span>
      {satellite.peak_severity !== satellite.severity && (
        <span className="sat-muted">Peak: {satellite.peak_severity}</span>
      )}

      <span className="sat-score">
        <span className="sat-row">
          <span className="sat-label">Score</span>
          <span>
            {satellite.anomaly_score.toFixed(2)}
            <span className="sat-muted"> / threshold {satellite.threshold.toFixed(2)}</span>
          </span>
        </span>
        <svg className="bar" aria-hidden="true">
          <defs>
            <linearGradient id={gradientId}>
              <stop offset="0" className="stop-h0" />
              <stop offset="1" stopColor={healthColor(satelliteHealth(satellite).latest)} />
            </linearGradient>
          </defs>
          <rect className="bar-track" width="100%" height="14" rx="7" />
          <rect width={`${fill}%`} height="14" rx="7" fill={`url(#${gradientId})`} />
        </svg>
      </span>

      <span className="sat-row">
        <span className="sat-label">Lead time</span>
        <span>{leadTimeText(satellite.lead_time_days)}</span>
      </span>
      <span className="sat-row">
        <span className="sat-label">Remaining life</span>
        <span>{remainingLifeText(satellite.remaining_life_days)}</span>
      </span>
    </button>
  )
}

export default function Constellation({ run, failed, hidden, onSelect }) {
  const [sort, setSort] = useState('severity')
  const [clock, setClock] = useState('all')
  const [orbit, setOrbit] = useState('all')

  const satellites = useMemo(() => {
    if (!run) return []
    return run.satellites
      .filter((s) => clock === 'all' || s.clock_type === clock)
      .filter((s) => matchesOrbit(s.orbit, orbit))
      .sort(COMPARE[sort])
  }, [run, sort, clock, orbit])

  if (!run) {
    return (
      <main hidden={hidden} className="view constellation">
        <h1>Constellation</h1>
        <p className="quiet">{failed ? 'Run data unavailable.' : 'Loading run…'}</p>
      </main>
    )
  }

  const { summary } = run
  const detail = summary.lead_time_detail
  const clocks = summary.clock_comparison

  return (
    <main hidden={hidden} className="view constellation">
      <h1>Constellation</h1>

      <section className="summary-strip" aria-label="Run summary">
        <div className="stats">
          <p className="stat">
            <span className="stat-value">{summary.satellites}</span>
            <span className="stat-label">satellites</span>
          </p>
          <p className="stat">
            <span className="stat-value">{summary.flagged}</span>
            <span className="stat-label">flagged</span>
          </p>
        </div>
        {detail && (
          <div className="lead-time">
            <p className="lead-time-headline">{leadTimeSummary(detail)}</p>
            <p className="note">{detail.note}</p>
          </div>
        )}
      </section>

      {clocks && (
        <section className="clock-comparison" aria-label="Clock comparison">
          <p>
            Healthy imported and indigenous clocks:{' '}
            {clocks.healthy_imported_mean_error.toFixed(5)} vs{' '}
            {clocks.healthy_irafs_mean_error.toFixed(5)} mean error (
            {clocks.difference_pct.toFixed(2)}% apart
            {clocks.conclusive ? '' : ' — not conclusive'})
          </p>
          <p className="note">{clocks.note}</p>
        </section>
      )}

      <section className="controls" aria-label="Sort and filter">
        <ButtonGroup label="Sort" options={SORTS} value={sort} onChange={setSort} />
        <ButtonGroup label="Clock" options={CLOCK_FILTERS} value={clock} onChange={setClock} />
        <ButtonGroup label="Orbit" options={ORBIT_FILTERS} value={orbit} onChange={setOrbit} />
      </section>

      <p className="quiet" aria-live="polite">
        {satellites.length} of {run.satellites.length} satellites
      </p>

      <div className="sat-grid">
        {satellites.map((satellite) => (
          <SatelliteCard key={satellite.id} satellite={satellite} onSelect={onSelect} />
        ))}
      </div>
    </main>
  )
}
