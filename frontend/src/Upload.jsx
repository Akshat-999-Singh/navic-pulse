import { useMemo, useRef, useState } from 'react'
import { PRECOMPUTED_NOTE, parseTelemetryCsv } from './data.js'
import ConstellationRail from './ConstellationRail.jsx'

const MAX_BYTES = 4 * 1024 * 1024
const SAMPLE_URL = '/sample-batch.csv'
const SAMPLE_NAME = 'sample-batch.csv'

// Reference metadata for the sample batch's source satellite, matching the
// REFERENCE table in export_demo_run.py. Operator-supplied, not read from the CSV.
const SAMPLE_METADATA = {
  name: 'IRNSS-1I',
  launchDate: '2018-04-11',
  clockType: 'imported',
  orbit: 'IGSO',
}

const EMPTY_METADATA = { name: '', launchDate: '', clockType: '', orbit: '' }

// Allowed day step between consecutive rows of one satellite. Calendar months
// run 28-31 days, so monthly accepts that range rather than one fixed step.
const CADENCES = [
  { id: 'daily', label: 'Daily', min: 1, max: 1, describe: '1 day' },
  { id: 'weekly', label: 'Weekly', min: 7, max: 7, describe: '7 days' },
  { id: 'monthly', label: 'Monthly', min: 28, max: 31, describe: '28–31 days' },
]

const MAX_LISTED_ROWS = 20

function cadenceMismatches(parsed, cadence) {
  const { rows, lines } = parsed
  const previous = new Map()
  const mismatched = []
  rows.forEach((row, i) => {
    if (row.day == null) return
    const last = previous.get(row.satellite_id)
    previous.set(row.satellite_id, row.day)
    if (last === undefined) return
    const step = row.day - last
    if (step < cadence.min || step > cadence.max) mismatched.push(lines[i])
  })
  return mismatched
}

function listRows(rows) {
  const shown = rows.slice(0, MAX_LISTED_ROWS).join(', ')
  const rest = rows.length - MAX_LISTED_ROWS
  return rest > 0 ? `${shown} and ${rest} more` : shown
}

function Field({ id, label, children }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
    </div>
  )
}

export default function Upload({ run, hidden, onComplete }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null) // { name, bytes, parsed }
  const [fileError, setFileError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loadingSample, setLoadingSample] = useState(false)
  const [metadata, setMetadata] = useState(EMPTY_METADATA)
  const [cadenceId, setCadenceId] = useState('daily')

  const cadence = CADENCES.find((c) => c.id === cadenceId)

  const issues = useMemo(() => {
    if (!file?.parsed) return []
    const { parsed } = file
    const list = parsed.errors.map((e) => ({
      label: `Row ${e.row}`,
      text: e.column ? `${e.column}: ${e.message}` : e.message,
    }))
    if (parsed.errors.length === 0 && parsed.rows.length === 0) {
      list.push({ label: 'File', text: 'The file has a header but no data rows.' })
    }
    const mismatched = cadenceMismatches(parsed, cadence)
    if (mismatched.length > 0) {
      list.push({
        label: 'Cadence',
        text:
          `${mismatched.length} row${mismatched.length === 1 ? '' : 's'} not ${cadence.describe} ` +
          `after the previous row for the same satellite (${cadence.label.toLowerCase()} cadence): ` +
          `rows ${listRows(mismatched)}.`,
      })
    }
    return list
  }, [file, cadence])

  const metadataComplete = Object.values(metadata).every((value) => value.trim() !== '')
  const clean = Boolean(file?.parsed) && !fileError && issues.length === 0
  const canSubmit = clean && metadataComplete

  function accept(name, bytes, text, presetMetadata) {
    const parsed = parseTelemetryCsv(text)
    setFile({ name, bytes, parsed })
    setFileError(null)
    if (presetMetadata) {
      setMetadata(presetMetadata)
    } else if (parsed.satellites.length === 1) {
      setMetadata((m) => ({ ...m, name: parsed.satellites[0] }))
    }
  }

  async function handleFiles(fileList) {
    if (!fileList || fileList.length === 0) return
    if (fileList.length > 1) {
      setFile(null)
      setFileError('Drop one file at a time.')
      return
    }
    const picked = fileList[0]
    if (!picked.name.toLowerCase().endsWith('.csv')) {
      setFile(null)
      setFileError(`"${picked.name}" is not a .csv file. Choose a comma-separated telemetry export.`)
      return
    }
    if (picked.size >= MAX_BYTES) {
      setFile({ name: picked.name, bytes: picked.size, parsed: null })
      setFileError(
        `"${picked.name}" is ${(picked.size / 1024 / 1024).toFixed(1)} MB. Batches must be under 4 MB; split the file and upload it in parts.`,
      )
      return
    }
    try {
      accept(picked.name, picked.size, await picked.text())
    } catch {
      setFile(null)
      setFileError(`"${picked.name}" could not be read.`)
    }
  }

  async function loadSample() {
    setLoadingSample(true)
    try {
      const response = await fetch(SAMPLE_URL, { cache: 'no-store' })
      if (!response.ok) throw new Error(String(response.status))
      const text = await response.text()
      // Same guard as useRun: a missing file can come back as index.html.
      if (text.trimStart().startsWith('<')) throw new Error('not a CSV')
      accept(SAMPLE_NAME, new Blob([text]).size, text, SAMPLE_METADATA)
    } catch {
      setFile(null)
      setFileError('The sample batch could not be loaded.')
    } finally {
      setLoadingSample(false)
    }
  }

  const setField = (key) => (event) => setMetadata((m) => ({ ...m, [key]: event.target.value }))

  return (
    <main hidden={hidden} className="view upload split">
      <div className="split-primary">
        <header className="intake-header">
          <h1>Telemetry intake</h1>
          <p className="lede">
            A batch of satellite clock telemetry is validated and inspected here: required columns,
            day ordering, duplicate days, file size and delivery cadence.
          </p>
        </header>

        <section className="intake-section" aria-labelledby="file-heading">
          <h2 id="file-heading">Batch file</h2>
          <button
            type="button"
            className="drop-zone"
            data-dragging={dragging}
            onClick={() => inputRef.current?.click()}
            onDragEnter={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              handleFiles(event.dataTransfer.files)
            }}
          >
            <span className="drop-title">Drop a .csv file here</span>
            <span className="quiet">or click to browse · one file, under 4 MB</span>
          </button>
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept=".csv"
            tabIndex={-1}
            aria-hidden="true"
            onChange={(event) => {
              handleFiles(event.target.files)
              // Allow the same file to be chosen again after edits on disk.
              event.target.value = ''
            }}
          />

          {file && (
            <p className="file-summary">
              <span>{file.name}</span>
              <span className="quiet">{(file.bytes / 1024).toFixed(1)} KB</span>
              {file.parsed && <span className="quiet">{file.parsed.rows.length} rows</span>}
            </p>
          )}
          {fileError && (
            <p className="inline-error" role="alert">
              {fileError}
            </p>
          )}
        </section>

        <section className="intake-section" aria-labelledby="metadata-heading">
          <h2 id="metadata-heading">Source satellite</h2>
          <p className="quiet">Operator-supplied reference data; not derived from the file.</p>
          <div className="fields">
            <Field id="meta-name" label="Satellite name">
              <input
                id="meta-name"
                type="text"
                required
                value={metadata.name}
                onChange={setField('name')}
                autoComplete="off"
              />
            </Field>
            <Field id="meta-launch" label="Launch date">
              <input
                id="meta-launch"
                type="date"
                required
                value={metadata.launchDate}
                onChange={setField('launchDate')}
              />
            </Field>
            <Field id="meta-clock" label="Clock type">
              <select id="meta-clock" required value={metadata.clockType} onChange={setField('clockType')}>
                <option value="">Select…</option>
                <option value="imported">Imported</option>
                <option value="irafs">Indigenous iRAFS</option>
              </select>
            </Field>
            <Field id="meta-orbit" label="Orbit">
              <select id="meta-orbit" required value={metadata.orbit} onChange={setField('orbit')}>
                <option value="">Select…</option>
                <option value="GEO">GEO</option>
                <option value="IGSO">IGSO</option>
                <option value="not-on-station">Not on station</option>
              </select>
            </Field>
          </div>
        </section>

        <section className="intake-section" aria-labelledby="cadence-heading">
          <h2 id="cadence-heading">Cadence</h2>
          <div className="segmented" role="group" aria-labelledby="cadence-heading">
            {CADENCES.map((option) => (
              <button
                key={option.id}
                type="button"
                aria-pressed={cadenceId === option.id}
                onClick={() => setCadenceId(option.id)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <p className="quiet">
            The cadence your telemetry feed delivers at. Uploaded rows are validated against it.
          </p>
        </section>

        <section className="intake-section" aria-labelledby="validation-heading" aria-live="polite">
          <h2 id="validation-heading">Validation</h2>
          {!file && !fileError && <p className="quiet">No batch selected.</p>}
          {file?.parsed && issues.length === 0 && (
            <p className="validation-ok">
              <span className="severity-dot" data-severity="normal" aria-hidden="true" />
              Valid: {file.parsed.rows.length} rows · {file.parsed.satellites.length} satellite
              {file.parsed.satellites.length === 1 ? '' : 's'} ({file.parsed.satellites.join(', ')})
            </p>
          )}
          {issues.length > 0 && (
            <>
              <p className="validation-bad">
                <span className="severity-dot" data-severity="critical" aria-hidden="true" />
                {issues.length} problem{issues.length === 1 ? '' : 's'} found
              </p>
              <ul className="issues">
                {issues.map((issue, i) => (
                  <li key={i}>
                    <span className="issue-row">{issue.label}</span>
                    <span>{issue.text}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {clean && !metadataComplete && (
            <p className="quiet">Complete every source satellite field to continue.</p>
          )}
        </section>

        <section className="intake-actions" aria-label="Actions">
          <div className="action-buttons">
            <button type="button" className="primary" disabled={!canSubmit} onClick={() => onComplete()}>
              Validate and continue
            </button>
            <button type="button" className="secondary" disabled={loadingSample} onClick={loadSample}>
              {loadingSample ? 'Loading sample…' : 'Load sample batch'}
            </button>
          </div>
          <p className="precomputed-note">{PRECOMPUTED_NOTE}</p>
        </section>
      </div>

      <aside className="split-aside">
        <div className="split-sticky">
          <ConstellationRail run={run} />
        </div>
      </aside>
    </main>
  )
}
