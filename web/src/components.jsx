import React, { useEffect, useRef, useState } from 'react'
import { fmtTime } from './api.js'
import { WavRecorder } from './recorder.js'

export function Card({ step, title, children, hidden }) {
  if (hidden) return null
  return (
    <section className="card">
      <h2>{step != null && <span className="step">{step}</span>}{title}</h2>
      {children}
    </section>
  )
}

export function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone || ''}`}>
      <div className="k">{label}</div>
      <div className="v">{value}</div>
    </div>
  )
}

export function Progress({ job }) {
  if (!job) return null
  const pct = Math.round((job.progress || 0) * 100)
  return (
    <div className="progress-wrap">
      <div className="progress"><div style={{ width: `${Math.max(pct, 2)}%` }} /></div>
      <div className="muted small">{job.status === 'failed' ? `Failed: ${job.error}` : `${job.message || job.status} · ${pct}%`}</div>
    </div>
  )
}

export function RecordButton({ label = 'Record', onDone, disabled }) {
  const rec = useRef(null)
  const [secs, setSecs] = useState(null)
  useEffect(() => {
    if (secs == null) return undefined
    const t = setInterval(() => rec.current && setSecs(rec.current.seconds), 500)
    return () => clearInterval(t)
  }, [secs != null])

  const toggle = async () => {
    if (!rec.current) {
      const r = new WavRecorder()
      try { await r.start() } catch (e) { alert(`Microphone access failed: ${e.message}`); return }
      rec.current = r
      setSecs(0)
    } else {
      const blob = await rec.current.stop()
      rec.current = null
      setSecs(null)
      onDone(blob)
    }
  }
  return (
    <button className={`btn ${secs != null ? 'rec' : ''}`} onClick={toggle} disabled={disabled}>
      {secs != null ? `■ Stop  ${fmtTime(secs)}` : `● ${label}`}
    </button>
  )
}

export const tone = (v, ok, warn) => (v >= ok ? 'ok' : v >= warn ? 'warn' : 'bad')
