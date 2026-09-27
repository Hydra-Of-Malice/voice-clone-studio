export async function api(path, opts = {}) {
  const res = await fetch(path, opts)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || res.statusText)
  return body
}

export const postJson = (path, data) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })

export function postForm(path, fields) {
  const fd = new FormData()
  for (const [k, v] of Object.entries(fields)) {
    if (Array.isArray(v)) fd.append(k, v[0], v[1])
    else fd.append(k, v)
  }
  return api(path, { method: 'POST', body: fd })
}

// Follows a job over server-sent events, falling back to polling if the stream breaks.
export function followJob(jobId, onProgress) {
  return new Promise((resolve, reject) => {
    let done = false
    const finish = (job) => {
      if (done) return
      if (job.status === 'completed') { done = true; resolve(job.result) }
      else if (job.status === 'failed') { done = true; reject(new Error(job.error || 'Job failed')) }
    }
    const poll = async () => {
      while (!done) {
        try {
          const job = await api(`/api/jobs/${jobId}`)
          onProgress && onProgress(job)
          finish(job)
        } catch (e) { done = true; reject(e) }
        if (!done) await new Promise((r) => setTimeout(r, 800))
      }
    }
    if (typeof EventSource === 'undefined') { poll(); return }
    const es = new EventSource(`/api/jobs/${jobId}/events`)
    const handle = (ev) => {
      const job = JSON.parse(ev.data)
      onProgress && onProgress(job)
      if (job.status === 'completed' || job.status === 'failed') { es.close(); finish(job) }
    }
    for (const name of ['queued', 'processing', 'completed', 'failed']) es.addEventListener(name, handle)
    es.onerror = () => { es.close(); if (!done) poll() }
  })
}

export const fmtTime = (s) =>
  `${String(Math.floor((s || 0) / 60)).padStart(2, '0')}:${String(Math.floor((s || 0) % 60)).padStart(2, '0')}`
