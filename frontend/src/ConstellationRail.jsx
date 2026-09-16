import { healthColor, satelliteHealth } from './health.js'

const SEVERITY_RANK = { critical: 0, warning: 1, normal: 2 }

// Worst health first; severity then id break ties so the order is stable.
function byHealth(a, b) {
  return (
    b.t - a.t ||
    SEVERITY_RANK[a.satellite.severity] - SEVERITY_RANK[b.satellite.severity] ||
    a.satellite.id.localeCompare(b.satellite.id)
  )
}

function HealthKey() {
  return (
    <figure className="health-key">
      <svg className="bar" aria-hidden="true">
        <defs>
          <linearGradient id="health-key-gradient">
            <stop offset="0" className="stop-h0" />
            <stop offset="0.25" className="stop-h1" />
            <stop offset="0.5" className="stop-h2" />
            <stop offset="0.75" className="stop-h3" />
            <stop offset="1" className="stop-h4" />
          </linearGradient>
        </defs>
        <rect width="100%" height="14" rx="7" fill="url(#health-key-gradient)" />
      </svg>
      <figcaption className="health-key-labels">
        <span>healthy</span>
        <span>critical</span>
      </figcaption>
    </figure>
  )
}

export default function ConstellationRail({ run }) {
  const rows = run
    ? run.satellites.map((satellite) => ({ satellite, t: satelliteHealth(satellite).latest })).sort(byHealth)
    : null

  return (
    <section className="rail" aria-labelledby="rail-heading">
      <h2 id="rail-heading" className="rail-heading">
        Constellation health
      </h2>
      {rows && (
        <ol className="rail-list">
          {rows.map(({ satellite, t }) => (
            <li key={satellite.id} className="rail-row">
              <svg className="rail-bar" aria-hidden="true">
                <rect width="6" height="100%" rx="3" fill={healthColor(t)} />
              </svg>
              <span className="rail-id">{satellite.id}</span>
              <span className="rail-severity">{satellite.severity}</span>
              <span className="rail-life">
                {satellite.remaining_life_days ? `${satellite.remaining_life_days.estimate} d` : '—'}
              </span>
            </li>
          ))}
        </ol>
      )}
      <HealthKey />
    </section>
  )
}
