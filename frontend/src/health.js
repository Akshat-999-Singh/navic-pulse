// The health spectrum. Clock health is a continuous decline, so every health
// colour on screen comes from one interpolation across --h0 (healthy) to --h4
// (critical). The stops are read from :root, so the palette lives in CSS only.

const STOP_TOKENS = ['--h0', '--h1', '--h2', '--h3', '--h4']

// Where each severity sits on the spectrum; matches the status mapping in CSS
// (normal --h0 .. --h1 band, warning --h2, critical --h4).
const SEVERITY_T = { normal: 0.25, warning: 0.5, critical: 1 }
const THRESHOLD_T = 0.25 // the detection threshold sits between healthy and early drift

let tokens = null

function hexToRgb(value) {
  const hex = value.trim().replace('#', '')
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16))
}

function readTokens() {
  if (tokens) return tokens
  const style = getComputedStyle(document.documentElement)
  tokens = {
    stops: STOP_TOKENS.map((name) => hexToRgb(style.getPropertyValue(name))),
    ink: style.getPropertyValue('--ink').trim().split(/\s+/).map(Number),
    page: hexToRgb(style.getPropertyValue('--page')),
  }
  return tokens
}

const clamp01 = (t) => Math.min(Math.max(t, 0), 1)
const toCss = (rgb) => `rgb(${rgb.map((c) => Math.round(c)).join(', ')})`

function healthRgb(t) {
  const { stops } = readTokens()
  const position = clamp01(t) * (stops.length - 1)
  const i = Math.min(Math.floor(position), stops.length - 2)
  const f = position - i
  return stops[i].map((c, k) => c + (stops[i + 1][k] - c) * f)
}

/** Colour for health t in [0, 1], linearly interpolated in RGB across the five stops. */
export function healthColor(t) {
  return toCss(healthRgb(t))
}

function luminance(rgb) {
  const [r, g, b] = rgb.map((c) => {
    const v = c / 255
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

// Body text on the dark ground reads at 7:1, not the 4.5:1 minimum: small
// light-on-dark type needs the margin.
const TEXT_CONTRAST = 7
// A status colour is never text at full saturation: at least this much ink is
// mixed in, so it cannot vibrate against the dark ground.
const TEXT_MIN_MIX = 0.3

/**
 * The same hue as healthColor(t), mixed toward ink: at least TEXT_MIN_MIX, and
 * further if that is what it takes to read at TEXT_CONTRAST on --page.
 */
export function healthTextColor(t) {
  const { ink, page } = readTokens()
  const base = healthRgb(t)
  for (let k = TEXT_MIN_MIX; k <= 1; k += 0.05) {
    const mixed = base.map((c, i) => c + (ink[i] - c) * k)
    if (contrast(mixed, page) >= TEXT_CONTRAST) return toCss(mixed)
  }
  return toCss(ink)
}

/**
 * Health mapping for one satellite.
 *
 * Below the detection threshold, t rises linearly to THRESHOLD_T, so a healthy
 * clock stays green-to-chartreuse however its score wanders. Above it, t is the
 * log of score/threshold normalised against the satellite's own maximum, and
 * tops out at the spectrum position of the satellite's final severity -- so
 * the colour a satellite ends on agrees with the severity word beside it.
 *
 * Normalising raw score against the satellite's maximum alone would paint a
 * healthy satellite's noisiest window deep red, which is exactly the false
 * alarm the design must not show.
 */
export function satelliteHealth(satellite) {
  const { threshold, windows, anomaly_score: latest } = satellite
  const maxError = Math.max(latest, ...windows.map((w) => w.error))
  const ceiling = Math.max(SEVERITY_T[satellite.severity] ?? THRESHOLD_T, THRESHOLD_T)
  const span = Math.log(maxError / threshold)

  const t = (error) => {
    if (!(threshold > 0) || error <= threshold) return THRESHOLD_T * clamp01(error / threshold)
    const f = span > 0 ? clamp01(Math.log(error / threshold) / span) : 1
    return THRESHOLD_T + (ceiling - THRESHOLD_T) * f
  }

  return { t, latest: t(latest) }
}
