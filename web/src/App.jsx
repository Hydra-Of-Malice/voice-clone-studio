import React, { useCallback, useEffect, useState } from 'react'
import { api, fmtTime, followJob, postForm, postJson } from './api.js'
import { Card, Progress, RecordButton, Stat, tone } from './components.jsx'

const STYLES = ['natural', 'conversational', 'professional', 'energetic']
const LANGS = [
  ['auto', 'Auto Detect'],
  ['hi', 'Hindi / Hinglish (Devanagari)'],
  ['hi-Latn', 'Hinglish (Roman script)'],
  ['en', 'English'],
]

export default function App() {
  const [status, setStatus] = useState(null)
  const [profiles, setProfiles] = useState([])
  const [profileId, setProfileId] = useState('')
  const [tab, setTab] = useState(() => {
    const t = new URLSearchParams(window.location.search).get('tab')
    return ['studio', 'voices', 'verify', 'history'].includes(t) ? t : 'studio'
  })

  const refreshProfiles = useCallback(async () => {
    const rows = await api('/api/profiles')
    setProfiles(rows)
    setProfileId((cur) => (rows.some((r) => r.id === cur) ? cur : (rows.find((r) => r.consent_status === 'verified') || rows[0] || {}).id || ''))
    return rows
  }, [])

  useEffect(() => {
    const load = () => api('/api/status').then(setStatus).catch(() => setStatus(null))
    load()
    refreshProfiles().catch(() => {})
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [refreshProfiles])

  return (
    <>
      <header>
        <h1>Voice Clone Studio</h1>
        <StatusLine status={status} />
      </header>
      <nav className="tabs">
        {[['studio', 'Studio'], ['voices', `Voices (${profiles.length})`], ['verify', 'Verify audio'], ['history', 'History']].map(([k, label]) => (
          <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{label}</button>
        ))}
      </nav>
      <main>
        {tab === 'studio' && <Studio profiles={profiles} profileId={profileId} setProfileId={setProfileId} refreshProfiles={refreshProfiles} />}
        {tab === 'voices' && <Voices profiles={profiles} refreshProfiles={refreshProfiles} />}
        {tab === 'verify' && <Verify />}
        {tab === 'history' && <History profiles={profiles} />}
      </main>
    </>
  )
}

function StatusLine({ status }) {
  if (!status) return <div className="muted small">server unreachable</div>
  const w = status.worker || {}
  const g = status.gpu || {}
  return (
    <div className="muted small status">
      <span className={`dot ${w.alive ? 'ok' : 'bad'}`} />
      {w.alive ? (w.current_job ? 'Working' : 'Ready') : 'Worker starting…'}
      {g.name && ` · ${g.name.replace('NVIDIA GeForce ', '')} · ${g.free_gb}/${g.total_gb} GB free`}
      {` · TTS ${status.engines.tts} · ASR ${status.engines.asr}`}
    </div>
  )
}

/* ------------------------------------------------------------------ studio ---------------- */
function Studio({ profiles, profileId, setProfileId, refreshProfiles }) {
  const [sample, setSample] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [job1, setJob1] = useState(null)
  const [err1, setErr1] = useState('')
  const [name, setName] = useState('My voice')
  const [pendingConsent, setPendingConsent] = useState(null)   // profile id awaiting consent
  const profile = profiles.find((p) => p.id === profileId)

  const upload = async (blob, filename) => {
    setErr1(''); setAnalysis(null); setJob1({ progress: 0.01, message: 'Uploading' })
    try {
      const up = await postForm('/api/upload', { file: [blob, filename] })
      setSample(up.sample_id)
      setAnalysis(await followJob(up.job_id, setJob1))
    } catch (e) { setErr1(e.message) }
  }

  const createProfile = async () => {
    setErr1('')
    try {
      const r = await postJson('/api/create-profile', { sample_id: sample, name: name || 'My voice' })
      const res = await followJob(r.job_id, setJob1)
      await refreshProfiles()
      setProfileId(res.profile_id)
      setPendingConsent(res.profile_id)
    } catch (e) { setErr1(e.message) }
  }

  const needsConsent = pendingConsent || (profile && profile.consent_status !== 'verified' ? profile.id : null)

  return (
    <>
      <Card step="1" title="Create Voice">
        <p className="muted">Upload 5–10 minutes of clear, natural speech from a single speaker: no music, no other voices, the same microphone throughout. Or record it here.</p>
        <div className="row">
          <label className="btn primary">
            Upload Audio
            <input type="file" hidden accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.webm"
              onChange={(e) => e.target.files[0] && upload(e.target.files[0], e.target.files[0].name)} />
          </label>
          <RecordButton label="Record sample" onDone={(b) => upload(b, 'recording.wav')} />
        </div>
        {job1 && !analysis && <Progress job={job1} />}
        {err1 && <p className="error">{err1}</p>}
        {analysis && <Analysis a={analysis} />}
        {analysis && (
          <div className="row">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Voice name" />
            <button className="btn primary" disabled={!analysis.quality.accepted} onClick={createProfile}>Create Voice Profile</button>
            {job1 && job1.kind === 'create_profile' && job1.status !== 'completed' && <span className="muted small">{job1.message}</span>}
          </div>
        )}
      </Card>

      {needsConsent && (
        <Consent profileId={needsConsent} onVerified={async () => { setPendingConsent(null); await refreshProfiles() }} />
      )}

      <Script profiles={profiles} profileId={profileId} setProfileId={setProfileId} />
    </>
  )
}

function Analysis({ a }) {
  const q = a.quality, t = a.transcript, p = a.prosody
  return (
    <>
      <div className="stats">
        <Stat label="Voice Quality" value={`${q.score}%`} tone={tone(q.score, 75, 55)} />
        <Stat label="Duration" value={fmtTime(q.duration_s)} />
        <Stat label="Language" value={`${t.language} (${Math.round(t.language_confidence * 100)}%)`} />
        <Stat label="Speaker Detected" value={q.speakers_detected} tone={q.speakers_detected === 1 ? 'ok' : 'bad'} />
        <Stat label="Noise Level" value={`${q.noise_level} · ${q.snr_db} dB`} tone={q.noise_level === 'Low' ? 'ok' : q.noise_level === 'Medium' ? 'warn' : 'bad'} />
        <Stat label="Transcript confidence" value={`${Math.round(t.confidence * 100)}%`} tone={tone(t.confidence, 0.8, 0.6)} />
        <Stat label="Pitch" value={`${p.f0_median_hz} Hz`} />
        <Stat label="Pace" value={`${p.articulation_rate_sps} syl/s · ${p.style_label}`} />
      </div>
      {a.enhancement && a.enhancement.applied && (
        <p className="note">Background noise was reduced automatically: {a.enhancement.snr_before_db} → {a.enhancement.snr_after_db} dB SNR
          (voice match {a.enhancement.identity_cosine}). Quality before cleaning: {a.quality_before_enhancement.score}%.</p>
      )}
      {(q.issues.length > 0 || q.suggestions.length > 0) && (
        <ul className="issues">
          {q.issues.map((i) => <li key={i}>{i}</li>)}
          {q.suggestions.map((s) => <li key={s} className="muted">{s}</li>)}
        </ul>
      )}
      <p className={q.accepted ? 'ok-text' : 'error'}>
        {q.accepted ? 'Sample accepted. You can create the voice profile.' : 'Sample rejected. Please improve the recording and upload again.'}
      </p>
      <details>
        <summary>Transcript ({t.engine})</summary>
        <div className="transcript">{(a.segments || []).map((s) => `[${fmtTime(s.start)}] ${s.text}`).join('\n')}</div>
      </details>
    </>
  )
}

function Consent({ profileId, onVerified }) {
  const [st, setSt] = useState(null)
  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [lang, setLang] = useState('')

  const load = useCallback(async (language) => {
    setResult(null); setError('')
    try { setSt(await postJson('/api/consent/statement', { profile_id: profileId, language: language || null })) }
    catch (e) { setError(e.message) }
  }, [profileId])
  useEffect(() => { load(lang) }, [load, lang])

  const submit = async (blob) => {
    setError(''); setResult(null)
    try {
      const r = await postForm('/api/consent/verify', { profile_id: profileId, consent_id: st.consent_id, file: [blob, 'consent.wav'] })
      const res = await followJob(r.job_id, setJob)
      setResult(res)
      if (res.verified) await onVerified()
    } catch (e) { setError(e.message) } finally { setJob(null) }
  }

  return (
    <Card step="✓" title="Confirm it's your voice">
      <p className="muted">Read this statement aloud in your normal voice. The words must match, the voice must match your sample, and the recording must be live. Generation is unlocked only after this passes.</p>
      <blockquote>{st ? st.statement : '…'}</blockquote>
      <div className="row">
        <RecordButton label="Record statement" onDone={submit} disabled={!st} />
        <button className="btn" onClick={() => load(lang)}>New statement</button>
        <label>Statement language
          <select value={lang} onChange={(e) => setLang(e.target.value)}>
            <option value="">Same as sample</option><option value="en">English</option><option value="hi">Hindi</option>
          </select>
        </label>
      </div>
      <Progress job={job} />
      {error && <p className="error">{error}</p>}
      {result && (result.verified
        ? <p className="ok-text">Consent verified (text match {Math.round(result.text_match * 100)}%, voice match ✓).</p>
        : <p className="error">Not verified. {result.reasons.join(' ')}<br /><span className="small muted">Heard: “{result.heard}”</span></p>)}
    </Card>
  )
}

function Script({ profiles, profileId, setProfileId }) {
  const [text, setText] = useState('')
  const [language, setLanguage] = useState('auto')
  const [style, setStyle] = useState('natural')
  const [speed, setSpeed] = useState(1)
  const [useFt, setUseFt] = useState(true)
  const [preview, setPreview] = useState(null)
  const [job, setJob] = useState(null)
  const [out, setOut] = useState(null)
  const [error, setError] = useState('')
  const profile = profiles.find((p) => p.id === profileId)
  const adapter = profile && (profile.adapters || []).find((a) => a.active)
  const ready = profile && profile.consent_status === 'verified'

  useEffect(() => {
    if (!text.trim()) { setPreview(null); return undefined }
    const t = setTimeout(() => postJson('/api/text/preview', { text, language }).then(setPreview).catch(() => {}), 500)
    return () => clearTimeout(t)
  }, [text, language])

  const run = async (spoken) => {
    setError(''); setJob({ progress: 0.01, message: 'Queued' })
    try {
      const body = { profile_id: profileId, text: spoken || text, language: spoken ? preview.engine_language : language, style, speed: Number(speed), use_fine_tuned: useFt }
      const r = await postJson('/api/generate', body)
      setOut(await followJob(r.job_id, setJob))
    } catch (e) { setError(e.message) } finally { setJob(null) }
  }

  return (
    <>
      <Card step="2" title="Enter Script">
        <div className="row">
          <label>Voice
            <select value={profileId} onChange={(e) => setProfileId(e.target.value)}>
              {profiles.length === 0 && <option value="">No voice profiles yet</option>}
              {profiles.map((p) => <option key={p.id} value={p.id}>{p.name} · {p.language} · {p.consent_status}</option>)}
            </select>
          </label>
          <label>Language
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {LANGS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          <label>Style
            <select value={style} onChange={(e) => setStyle(e.target.value)}>
              {STYLES.map((s) => <option key={s} value={s}>{s[0].toUpperCase() + s.slice(1)}</option>)}
            </select>
          </label>
          <label>Speed {Number(speed).toFixed(2)}x
            <input type="range" min="0.8" max="1.2" step="0.05" value={speed} onChange={(e) => setSpeed(e.target.value)} />
          </label>
        </div>
        <textarea rows="6" placeholder="Paste your script here…" value={text} onChange={(e) => setText(e.target.value)} />
        {preview && preview.text !== text.trim() && (
          <div className="preview">
            <div className="small muted">Will be spoken as ({preview.language}{preview.transliteration ? ` · ${preview.transliteration}` : ''}):</div>
            <div className="spoken">{preview.text}</div>
            <button className="btn small-btn" onClick={() => { setText(preview.text); setLanguage(preview.engine_language) }}>Edit this text</button>
          </div>
        )}
        {adapter && (
          <label className="check"><input type="checkbox" checked={useFt} onChange={(e) => setUseFt(e.target.checked)} /> Use fine-tuned voice</label>
        )}
        <div className="row">
          <button className="btn primary" disabled={!ready || !text.trim() || !!job} onClick={() => run()}>Generate Voice</button>
          {profile && !ready && <span className="error small">Consent not verified for this voice.</span>}
        </div>
        <Progress job={job} />
        {error && <p className="error">{error}</p>}
      </Card>

      <Card step="3" title="Preview" hidden={!out}>
        {out && <Output out={out} onRegenerate={() => run()} busy={!!job} />}
      </Card>
    </>
  )
}

function Output({ out, onRegenerate, busy }) {
  const m = out.metrics
  const prov = m.provenance || {}
  return (
    <>
      <audio controls autoPlay src={`${out.wav_url}&t=${out.audio_id}`} />
      <div className="row">
        <button className="btn" onClick={onRegenerate} disabled={busy}>Regenerate</button>
        <a className="btn" href={`${out.wav_url}&download=1`}>Download WAV</a>
        <a className="btn" href={`${out.mp3_url}&download=1`}>Download MP3</a>
      </div>
      <div className="stats">
        <Stat label="Duration" value={fmtTime(m.duration_s)} />
        <Stat label="Speaker similarity" value={m.speaker_similarity.toFixed(3)} tone={m.similarity_ok ? 'ok' : 'warn'} />
        <Stat label="Loudness" value={`${m.loudness_lufs} LUFS`} />
        <Stat label="Language" value={m.language} />
        <Stat label="Voice" value={m.fine_tuned ? 'Fine-tuned' : 'Zero-shot'} />
        <Stat label="Provenance" value={[prov.audioseal && prov.audioseal.embedded && 'Watermark', prov.c2pa && prov.c2pa.signed && 'C2PA'].filter(Boolean).join(' + ') || 'None'} />
      </div>
      {!m.similarity_ok && <p className="note">Similarity is below this speaker's own range ({m.similarity_band_low}). Try Regenerate, another style, or a script in the sample's language.</p>}
      <p className="muted small">48 kHz 24-bit WAV and 320 kbps MP3, watermarked and labelled as AI-generated speech made with the speaker's consent.</p>
    </>
  )
}

/* ------------------------------------------------------------------ voices ---------------- */
function Voices({ profiles, refreshProfiles }) {
  const [jobs, setJobs] = useState({})
  const [errors, setErrors] = useState({})

  const fineTune = async (p) => {
    setErrors((e) => ({ ...e, [p.id]: '' }))
    try {
      const r = await postJson('/api/fine-tune', { profile_id: p.id })
      await followJob(r.job_id, (j) => setJobs((s) => ({ ...s, [p.id]: j })))
    } catch (e) { setErrors((s) => ({ ...s, [p.id]: e.message })) }
    setJobs((s) => ({ ...s, [p.id]: null }))
    await refreshProfiles()
  }
  const toggle = async (a, active) => { await postJson(`/api/adapters/${a.id}`, { active }); await refreshProfiles() }
  const remove = async (p) => {
    if (!window.confirm(`Delete the voice "${p.name}"? Its clips, embeddings and fine-tunes are erased and consent is revoked.`)) return
    await api(`/api/profiles/${p.id}`, { method: 'DELETE' })
    await refreshProfiles()
  }

  if (!profiles.length) return <Card title="Voices"><p className="muted">No voice profiles yet. Create one in the Studio tab.</p></Card>
  return profiles.map((p) => (
    <Card key={p.id} title={p.name}>
      <div className="stats">
        <Stat label="Consent" value={p.consent_status} tone={p.consent_status === 'verified' ? 'ok' : 'warn'} />
        <Stat label="Language" value={p.language} />
        <Stat label="Quality" value={`${p.quality_score}%`} tone={tone(p.quality_score, 75, 55)} />
        <Stat label="Reference clips" value={p.clips} />
        <Stat label="Pitch" value={`${(p.pitch || {}).median_hz} Hz`} />
        <Stat label="Style" value={(p.style || {}).label} />
      </div>
      <h3>Fine-tuning</h3>
      <p className="muted small">Trains a small adapter on this speaker's recording (a few minutes on the GPU). It is kept only if it sounds more like the speaker than the zero-shot voice without losing intelligibility.</p>
      {(p.adapters || []).map((a) => (
        <div key={a.id} className={`adapter ${a.status}`}>
          <div>
            <b>{a.status}</b>{a.active && ' · in use'} <span className="muted small">{new Date(a.created_at * 1000).toLocaleString()}</span>
            {a.metrics.verdict && <div className="small">{a.metrics.verdict}</div>}
            {a.metrics.zero_shot && (
              <div className="small muted">
                similarity {a.metrics.zero_shot.similarity} → {a.metrics.fine_tuned.similarity} · CER {a.metrics.zero_shot.cer} → {a.metrics.fine_tuned.cer} · {a.metrics.clips} clips · {a.metrics.steps} steps · {Math.round(a.metrics.train_seconds)} s
              </div>
            )}
            {a.metrics.error && <div className="small error">{a.metrics.error}</div>}
          </div>
          {a.status === 'accepted' && <button className="btn small-btn" onClick={() => toggle(a, !a.active)}>{a.active ? 'Stop using' : 'Use'}</button>}
        </div>
      ))}
      <Progress job={jobs[p.id]} />
      {errors[p.id] && <p className="error">{errors[p.id]}</p>}
      <div className="row">
        <button className="btn" disabled={p.consent_status !== 'verified' || !!jobs[p.id]} onClick={() => fineTune(p)}>Fine-tune this voice</button>
        <button className="btn danger" onClick={() => remove(p)}>Delete voice</button>
      </div>
    </Card>
  ))
}

/* ------------------------------------------------------------------ verify ---------------- */
function Verify() {
  const [job, setJob] = useState(null)
  const [res, setRes] = useState(null)
  const [error, setError] = useState('')
  const check = async (file) => {
    setRes(null); setError(''); setJob({ progress: 0.01, message: 'Uploading' })
    try {
      const r = await postForm('/api/verify-audio', { file: [file, file.name] })
      setRes(await followJob(r.job_id, setJob))
    } catch (e) { setError(e.message) } finally { setJob(null) }
  }
  return (
    <Card title="Verify audio">
      <p className="muted">Check whether a recording was generated here or carries an AI-speech watermark or Content Credentials.</p>
      <label className="btn primary">Choose file
        <input type="file" hidden accept="audio/*" onChange={(e) => e.target.files[0] && check(e.target.files[0])} />
      </label>
      <Progress job={job} />
      {error && <p className="error">{error}</p>}
      {res && (
        <>
          <h3 className={res.ai_generated ? 'warn-text' : 'ok-text'}>{res.verdict}</h3>
          <div className="stats">
            <Stat label="AudioSeal watermark" value={res.audioseal.detected ? `Found (${Math.round(res.audioseal.probability * 100)}%)` : 'Not found'} tone={res.audioseal.detected ? 'warn' : ''} />
            <Stat label="Perth watermark" value={res.perth.detected ? 'Found' : 'Not found'} tone={res.perth.detected ? 'warn' : ''} />
            <Stat label="Content Credentials" value={res.c2pa.present ? (res.c2pa.validation_state || 'Present') : 'None'} tone={res.c2pa.present ? 'warn' : ''} />
            {res.c2pa.present && <Stat label="Signed by" value={res.c2pa.issuer || '—'} />}
            {res.matched_output && <Stat label="Output id" value={res.matched_output.audio_id} />}
            {res.matched_output && <Stat label="Matched by" value={res.matched_output.matched_by} />}
          </div>
          <p className="muted small">{res.note}</p>
        </>
      )}
    </Card>
  )
}

/* ------------------------------------------------------------------ history --------------- */
function History({ profiles }) {
  const [rows, setRows] = useState(null)
  useEffect(() => { api('/api/history').then(setRows).catch(() => setRows([])) }, [])
  const nameOf = (id) => (profiles.find((p) => p.id === id) || {}).name || 'deleted voice'
  if (!rows) return <Card title="History"><p className="muted">Loading…</p></Card>
  if (!rows.length) return <Card title="History"><p className="muted">Nothing generated yet.</p></Card>
  return (
    <Card title="History">
      {rows.map((r) => (
        <div key={r.id} className="history-row">
          <div className="small muted">{new Date(r.created_at * 1000).toLocaleString()} · {nameOf(r.profile_id)} · {fmtTime(r.duration_s)} · similarity {r.metrics.speaker_similarity}</div>
          <div className="script">{r.script}</div>
          <audio controls preload="none" src={`/api/audio/${r.id}?fmt=mp3`} />
        </div>
      ))}
    </Card>
  )
}
