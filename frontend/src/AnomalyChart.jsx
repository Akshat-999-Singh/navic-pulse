import { useEffect, useRef, useState } from 'react'
import { healthColor } from './health.js'

// Hand-drawn SVG anomaly timeline: reconstruction error per window, the
// detection threshold, flagged spans, the persistence-confirmed window, the
// fixed-limit baseline window, and the selected window.
//
// The viewBox is sized to the container's pixel width (via ResizeObserver), so
// the chart fills its container while labels stay a constant, legible size
// instead of shrinking at 360px and ballooning on wide screens.

const HEIGHT = 280
const HEIGHT_NARROW = 220
const NARROW = 520 // px of chart width below which the shorter height is used
const MARGIN = { top: 44, right: 16, bottom: 30, left: 56 }
const TICKS = 5
const FALLBACK_WIDTH = 720
const LOG_FLOOR = 0.8 // domain starts at this fraction of the smallest non-zero error
const LOG_CEILING = 1.2 // and ends at this multiple of the largest value shown

// Log-axis ticks: every power of ten in [min, max]; with fewer than three of
// those, half-decade ticks (3 x 10^n) are added so the axis is not sparse.
function logTicks(min, max) {
  const decades = []
  for (let k = Math.ceil(Math.log10(min)); k <= Math.floor(Math.log10(max)); k++) decades.push(10 ** k)
  if (decades.length >= 3) return decades
  const withHalves = [...decades]
  for (let k = Math.floor(Math.log10(min)); k <= Math.floor(Math.log10(max)); k++) {
    const half = 3 * 10 ** k
    if (half >= min && half <= max) withHalves.push(half)
  }
  return withHalves.sort((a, b) => a - b)
}

// 0.1, 0.3, 1, 3, 10, 100: one significant figure, no trailing zeros.
const formatLogTick = (value) => String(Number(value.toPrecision(1)))

// Round tick steps (1, 2, 2.5, 5 x 10^n) covering [min, max], choosing the step
// whose tick count is closest to `count`.
function niceTicks(min, max, count) {
  const span = max - min
  if (!(span > 0)) return { ticks: [min], step: 1 }
  const magnitude = 10 ** Math.floor(Math.log10(span / count))
  const candidates = [0.5, 1, 2, 2.5, 5, 10, 20].map((m) => m * magnitude)
  const step = candidates.reduce((best, s) =>
    Math.abs(span / s - count) < Math.abs(span / best - count) ? s : best,
  )
  const ticks = []
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(v)
  return { ticks, step }
}

function formatTick(value, step) {
  const decimals = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)))
  return value.toFixed(decimals)
}

// Nearest window by index value (the array is sorted by index, and thinned).
function nearestPosition(windows, index) {
  let lo = 0
  let hi = windows.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (windows[mid].index < index) lo = mid
    else hi = mid
  }
  return Math.abs(windows[lo].index - index) <= Math.abs(windows[hi].index - index) ? lo : hi
}

// Contiguous runs of flagged windows (adjacent in the array) as [firstIndex, lastIndex].
function flaggedSpans(windows) {
  const spans = []
  let start = null
  windows.forEach((w, i) => {
    if (w.flagged && start === null) start = w.index
    const endsRun = w.flagged && (i === windows.length - 1 || !windows[i + 1].flagged)
    if (endsRun) {
      spans.push([start, w.index])
      start = null
    }
  })
  return spans
}

export default function AnomalyChart({
  windows,
  threshold,
  confirmedIndex,
  baselineIndex,
  selectedIndex,
  onSelect,
  health,
  label,
}) {
  const frameRef = useRef(null)
  const [width, setWidth] = useState(FALLBACK_WIDTH)

  useEffect(() => {
    const frame = frameRef.current
    // Content width, not clientWidth: padding would make the SVG draw slightly
    // smaller than its viewBox and shrink the labels with it.
    const measure = () => {
      const style = getComputedStyle(frame)
      const content = frame.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
      if (content > 0) setWidth(Math.round(content))
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(frame)
    return () => observer.disconnect()
  }, [])

  if (!windows?.length) return <div className="anomaly-chart" ref={frameRef} />

  const height = width < NARROW ? HEIGHT_NARROW : HEIGHT
  const plot = {
    left: MARGIN.left,
    right: width - MARGIN.right,
    top: MARGIN.top,
    bottom: height - MARGIN.bottom,
  }

  // X is the window index value, never array position: the array is thinned.
  const xMin = windows[0].index
  const xMax = windows[windows.length - 1].index
  // Y is log10, so the pre-fault noise floor gets real vertical space next to a
  // fault that is three decades larger.
  const positive = windows.map((w) => w.error).filter((e) => e > 0)
  const maxError = positive.length ? Math.max(...positive) : 1
  const minError = positive.length ? Math.min(...positive) : maxError / 10
  const yMin = LOG_FLOOR * Math.min(minError, threshold > 0 ? threshold : minError)
  // Include the threshold, so it stays on the chart for a clock that never reaches it.
  const yMax = LOG_CEILING * Math.max(maxError, threshold > 0 ? threshold : 0)
  const logMin = Math.log10(yMin)
  const logSpan = Math.log10(yMax) - logMin || 1

  const x = (index) =>
    plot.left + (xMax === xMin ? 0 : ((index - xMin) / (xMax - xMin)) * (plot.right - plot.left))
  // Zero or negative errors have no logarithm: clamp them to the domain minimum.
  const y = (error) =>
    plot.bottom - ((Math.log10(Math.max(error, yMin)) - logMin) / logSpan) * (plot.bottom - plot.top)

  const trace = windows.map((w, i) => `${i === 0 ? 'M' : 'L'}${x(w.index).toFixed(1)},${y(w.error).toFixed(1)}`).join('')
  // Fewer x ticks on a narrow chart, where four-digit labels would crowd.
  const xTicks = niceTicks(xMin, xMax, width < NARROW ? 3 : TICKS)
  const yTicks = logTicks(yMin, yMax)
  const selected = windows[selectedIndex]
  const selectedColor = selected && health ? healthColor(health.t(selected.error)) : undefined

  // Marker labels sit in two rows above the plot, pointing away from each other,
  // so "confirmed" and "fixed limits" stay readable when the windows are close.
  const markerLabelX = (index, side) => {
    const px = x(index)
    return side === 'left' ? Math.max(px - 6, plot.left) : Math.min(px + 6, plot.right)
  }
  const confirmedAnchor = x(confirmedIndex ?? 0) - plot.left < 90 ? 'start' : 'end'
  const baselineAnchor = plot.right - x(baselineIndex ?? 0) < 90 ? 'end' : 'start'

  function handlePointer(event) {
    const svg = event.currentTarget.ownerSVGElement
    const rect = svg.getBoundingClientRect()
    const px = ((event.clientX - rect.left) / rect.width) * width
    const ratio = (Math.min(Math.max(px, plot.left), plot.right) - plot.left) / (plot.right - plot.left)
    onSelect?.(nearestPosition(windows, xMin + ratio * (xMax - xMin)))
  }

  return (
    <div className="anomaly-chart" ref={frameRef}>
      <svg className="anomaly-svg" viewBox={`0 0 ${width} ${height}`} height={height} role="img" aria-label={label}>
        {flaggedSpans(windows).map(([from, to]) => (
          <rect
            key={from}
            className="chart-flagged"
            x={x(from)}
            y={plot.top}
            width={Math.max(x(to) - x(from), 1)}
            height={plot.bottom - plot.top}
          />
        ))}

        {/* Axes: tick marks and labels only, no grid. */}
        <line className="chart-axis" x1={plot.left} x2={plot.right} y1={plot.bottom} y2={plot.bottom} />
        <line className="chart-axis" x1={plot.left} x2={plot.left} y1={plot.top} y2={plot.bottom} />
        {xTicks.ticks.map((t) => (
          <g key={`x${t}`}>
            <line className="chart-tick" x1={x(t)} x2={x(t)} y1={plot.bottom} y2={plot.bottom + 5} />
            <text className="chart-label" x={x(t)} y={plot.bottom + 18} textAnchor="middle">
              {formatTick(t, xTicks.step)}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="chart-tick" x1={plot.left - 5} x2={plot.left} y1={y(t)} y2={y(t)} />
            <text className="chart-label" x={plot.left - 8} y={y(t) + 4} textAnchor="end">
              {formatLogTick(t)}
            </text>
          </g>
        ))}
        {/* So nobody reads the trace as linear. */}
        <text
          className="chart-label chart-scale-label"
          transform={`translate(12 ${(plot.top + plot.bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          log scale
        </text>

        {threshold > 0 && (
          <g>
            <line className="chart-threshold" x1={plot.left} x2={plot.right} y1={y(threshold)} y2={y(threshold)} />
            <text className="chart-label chart-threshold-label" x={plot.right} y={y(threshold) - 5} textAnchor="end">
              threshold {threshold.toFixed(2)}
            </text>
          </g>
        )}

        <path className="chart-trace" d={trace} />

        {baselineIndex != null && (
          <g>
            <line className="chart-baseline" x1={x(baselineIndex)} x2={x(baselineIndex)} y1={plot.top - 14} y2={plot.bottom} />
            <text
              className="chart-label chart-marker-label"
              x={markerLabelX(baselineIndex, baselineAnchor === 'start' ? 'right' : 'left')}
              y={plot.top - 18}
              textAnchor={baselineAnchor}
            >
              fixed limits
            </text>
          </g>
        )}
        {confirmedIndex != null && (
          <g>
            <line className="chart-confirmed" x1={x(confirmedIndex)} x2={x(confirmedIndex)} y1={plot.top - 30} y2={plot.bottom} />
            <text
              className="chart-label chart-marker-label"
              x={markerLabelX(confirmedIndex, confirmedAnchor === 'end' ? 'left' : 'right')}
              y={plot.top - 34}
              textAnchor={confirmedAnchor}
            >
              confirmed
            </text>
          </g>
        )}

        {selected && (
          <g>
            <line className="chart-selected" x1={x(selected.index)} x2={x(selected.index)} y1={plot.top} y2={plot.bottom} stroke={selectedColor} />
            <circle className="chart-selected-dot" cx={x(selected.index)} cy={y(selected.error)} r="5" fill={selectedColor} />
          </g>
        )}

        {/* Catches clicks and drags anywhere over the plot area. */}
        <rect
          className="chart-hit"
          x={plot.left}
          y={0}
          width={Math.max(plot.right - plot.left, 0)}
          height={height}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture?.(event.pointerId)
            handlePointer(event)
          }}
          onPointerMove={(event) => {
            if (event.buttons & 1) handlePointer(event)
          }}
        />
      </svg>
    </div>
  )
}
