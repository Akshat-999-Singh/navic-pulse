import { useEffect, useState } from 'react'

// The only data layer in the frontend. The reference run is public/demo-run.json,
// written by export_demo_run.py; an uploaded batch is analysed by the Python
// function at /api/analyze, which returns a run of the same shape.

export const LIVE_NOTE =
  'Analysis runs on your uploaded batch. Telemetry is synthetic; the model and thresholds are the ones trained offline.'

export const FALLBACK_NOTE =
  'Showing the precomputed reference run. The analysis service is unavailable, so your upload was validated but not analysed.'

const RUN_URL = '/demo-run.json'
const ANALYZE_URL = '/api/analyze'
const ANALYZE_TIMEOUT_MS = 300_000 // the function's maxDuration in vercel.json

const isRun = (value) => Boolean(value) && Array.isArray(value.satellites)

// POSTs one batch for analysis. Resolves with the computed run; rejects with
// exactly one of {validation: [{row, column, message}]}, {tooLarge: true} or
// {unavailable: true}.
export async function analyze(file, cadence, metadata = {}) {
  const form = new FormData()
  form.append('file', file, file.name ?? 'batch.csv')
  form.append('cadence', cadence)
  for (const key of ['name', 'launch_date', 'clock_type', 'orbit']) {
    const value = metadata[key]?.trim()
    if (value) form.append(key, value)
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS)
  let response
  let body = null
  try {
    response = await fetch(ANALYZE_URL, { method: 'POST', body: form, signal: controller.signal })
    // A missing function can come back as the SPA's index.html, so an
    // unparseable body is treated like no service at all.
    body = await response.json().catch(() => null)
  } catch {
    throw { unavailable: true }
  } finally {
    clearTimeout(timer)
  }

  if (response.status === 200 && isRun(body)) return body
  if (response.status === 422 && Array.isArray(body?.errors)) throw { validation: body.errors }
  if (response.status === 413) throw { tooLarge: true }
  throw { unavailable: true }
}

export function useRun() {
  const [state, setState] = useState({ run: null, loading: true, failed: false })

  useEffect(() => {
    const controller = new AbortController()

    async function load() {
      try {
        const response = await fetch(RUN_URL, { cache: 'no-store', signal: controller.signal })
        if (!response.ok) throw new Error(`${RUN_URL} responded ${response.status}`)
        // A missing file can come back as the SPA's index.html with a 200, so
        // a parse failure or the wrong shape counts as a failed load too.
        const run = await response.json()
        if (!isRun(run)) throw new Error(`${RUN_URL} is not a run`)
        setState({ run, loading: false, failed: false })
      } catch (error) {
        if (error?.name === 'AbortError') return
        setState({ run: null, loading: false, failed: true })
      }
    }

    load()
    return () => controller.abort()
  }, [])

  return state
}

const REQUIRED_COLUMNS = ['day', 'satellite_id', 'clock_type']

const CHANNEL_COLUMNS = [
  'freq_offset_y',
  'clock_bias_ns',
  'trb_core_temp_c',
  'rafs_signal_level',
  'bus_voltage_v',
  'steering_correction_ns',
]

const INTEGER = /^[+-]?\d+$/

// Hand-written on purpose: the telemetry export is plain comma-separated values
// with no quoting, so a CSV library would add weight without adding safety.
// lines[i] is the 1-based file line that rows[i] came from.
export function parseTelemetryCsv(text) {
  const rows = []
  const lines = []
  const errors = []
  const source = String(text ?? '').replace(/^\uFEFF/, '').split(/\r?\n/)

  const headerIndex = source.findIndex((line) => line.trim() !== '')
  if (headerIndex === -1) {
    errors.push({ row: 1, column: null, message: 'File is empty.' })
    return { rows, lines, columns: [], satellites: [], errors }
  }

  const headerRow = headerIndex + 1
  const columns = source[headerIndex].split(',').map((name) => name.trim())

  for (const name of REQUIRED_COLUMNS) {
    if (!columns.includes(name)) {
      errors.push({ row: headerRow, column: name, message: `Missing required column "${name}".` })
    }
  }
  if (!CHANNEL_COLUMNS.some((name) => columns.includes(name))) {
    errors.push({
      row: headerRow,
      column: null,
      message: `Needs at least one telemetry channel: ${CHANNEL_COLUMNS.join(', ')}.`,
    })
  }
  if (errors.length > 0) return { rows, lines, columns, satellites: [], errors }

  const lastDay = new Map()
  const seen = new Set()

  for (let i = headerIndex + 1; i < source.length; i++) {
    if (source[i].trim() === '') continue
    const row = i + 1
    const cells = source[i].split(',').map((cell) => cell.trim())

    if (cells.length !== columns.length) {
      errors.push({
        row,
        column: null,
        message: `Expected ${columns.length} fields, found ${cells.length}.`,
      })
      continue
    }

    const record = {}
    columns.forEach((name, j) => {
      record[name] = cells[j]
    })
    for (const name of CHANNEL_COLUMNS) {
      if (name in record) record[name] = record[name] === '' ? null : Number(record[name])
    }

    const satellite = record.satellite_id
    if (!INTEGER.test(record.day)) {
      errors.push({ row, column: 'day', message: `Day "${record.day}" is not an integer.` })
      record.day = null
    } else {
      const day = Number(record.day)
      record.day = day

      const key = JSON.stringify([satellite, day])
      if (seen.has(key)) {
        errors.push({ row, column: 'day', message: `Duplicate day ${day} for ${satellite}.` })
      }
      seen.add(key)

      const previous = lastDay.get(satellite)
      if (previous !== undefined && day < previous) {
        errors.push({
          row,
          column: 'day',
          message: `Day ${day} comes after day ${previous} for ${satellite}; days must not decrease.`,
        })
      } else {
        lastDay.set(satellite, day)
      }
    }

    rows.push(record)
    lines.push(row)
  }

  const satellites = [...new Set(rows.map((record) => record.satellite_id))].sort()
  return { rows, lines, columns, satellites, errors }
}
