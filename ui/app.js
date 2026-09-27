const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
};
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
async function waitJob(jobId, onProgress) {
  for (;;) {
    const j = await api(`/api/jobs/${jobId}`);
    onProgress && onProgress(j);
    if (j.status === 'completed') return j.result;
    if (j.status === 'failed') throw new Error(j.error || 'Job failed');
    await sleep(700);
  }
}
const fmtTime = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const stat = (k, v, cls = '') => `<div class="stat ${cls}"><div class="k">${k}</div><div class="v">${v}</div></div>`;

let currentSample = null, currentProfile = null, currentConsent = null, lastRequest = null;

// ---------------------------------------------------------------- status ----------------
async function refreshStatus() {
  try {
    const s = await api('/api/status');
    const g = s.gpu.name ? `${s.gpu.name} · ${s.gpu.free_gb} / ${s.gpu.total_gb} GB free` : 'CPU mode';
    $('gpu').textContent = `${g} · TTS: ${s.engines.tts} · ASR: ${s.engines.asr_model}`;
  } catch { $('gpu').textContent = 'server unreachable'; }
}
async function refreshProfiles() {
  const rows = await api('/api/profiles');
  const sel = $('profileSelect'); sel.innerHTML = '';
  for (const r of rows) {
    const o = document.createElement('option');
    o.value = r.id; o.textContent = `${r.name} (${r.language || '?'}, ${r.consent_status})`;
    if (r.id === currentProfile) o.selected = true;
    sel.appendChild(o);
  }
  if (!rows.length) { const o = document.createElement('option'); o.textContent = 'No voice profiles yet'; o.value = ''; sel.appendChild(o); }
}

// ---------------------------------------------------------------- step 1 ----------------
async function uploadBlob(blob, name) {
  $('analysis').classList.remove('hidden'); $('createProfile').disabled = true;
  $('stats').innerHTML = ''; $('issues').innerHTML = ''; $('p1').style.width = '2%'; $('p1msg').textContent = 'Uploading…';
  const fd = new FormData(); fd.append('file', blob, name);
  const up = await api('/api/upload', { method: 'POST', body: fd });
  currentSample = up.sample_id;
  const res = await waitJob(up.job_id, j => { $('p1').style.width = `${Math.round(j.progress * 100)}%`; $('p1msg').textContent = j.message; });
  showAnalysis(res);
}
function showAnalysis(res) {
  const q = res.quality, t = res.transcript, p = res.prosody;
  const cls = (v, ok, warn) => v >= ok ? 'ok' : v >= warn ? 'warn' : 'bad';
  $('stats').innerHTML =
    stat('Voice Quality', `${q.score}%`, cls(q.score, 75, 55)) +
    stat('Duration', fmtTime(q.duration_s)) +
    stat('Language', `${t.language} (${Math.round(t.language_confidence * 100)}%)`) +
    stat('Speaker Detected', q.speakers_detected, q.speakers_detected === 1 ? 'ok' : 'bad') +
    stat('Noise Level', `${q.noise_level} (${q.snr_db} dB SNR)`, q.noise_level === 'Low' ? 'ok' : q.noise_level === 'Medium' ? 'warn' : 'bad') +
    stat('Transcript confidence', `${Math.round(t.confidence * 100)}%`, cls(t.confidence, 0.8, 0.6)) +
    stat('Pitch', `${p.f0_median_hz} Hz median`) +
    stat('Pace', `${p.articulation_rate_sps} syl/s · ${p.style_label}`);
  $('issues').innerHTML = [...q.issues.map(i => `<li>${i}</li>`), ...q.suggestions.map(s => `<li class="muted">${s}</li>`)].join('');
  $('transcript').textContent = (res.segments || []).map(s => `[${fmtTime(s.start)}] ${s.text}`).join('\n');
  $('createProfile').disabled = !q.accepted;
  $('p1msg').textContent = q.accepted ? 'Sample accepted. You can create the voice profile.' : 'Sample rejected — please improve the recording and upload again.';
}
$('file').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { await uploadBlob(f, f.name); } catch (err) { $('p1msg').textContent = 'Error: ' + err.message; }
});

// recording helpers
function bindRecorder(btnId, timeId, onDone) {
  let rec = null, timer = null;
  $(btnId).addEventListener('click', async () => {
    if (!rec) {
      rec = new WavRecorder();
      try { await rec.start(); } catch (e) { alert('Microphone access failed: ' + e.message); rec = null; return; }
      $(btnId).classList.add('rec'); $(btnId).textContent = '■ Stop';
      timer = setInterval(() => { $(timeId).textContent = fmtTime(rec.seconds); }, 500);
    } else {
      clearInterval(timer); const blob = await rec.stop(); rec = null;
      $(btnId).classList.remove('rec'); $(btnId).textContent = '● Record'; $(timeId).textContent = '';
      onDone(blob);
    }
  });
}
bindRecorder('recSample', 'recSampleTime', async (blob) => { try { await uploadBlob(blob, 'recording.wav'); } catch (err) { $('p1msg').textContent = 'Error: ' + err.message; } });

$('createProfile').addEventListener('click', async () => {
  if (!currentSample) return;
  $('createProfile').disabled = true; $('p1msg').textContent = 'Building voice profile…';
  try {
    const r = await api('/api/create-profile', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sample_id: currentSample, name: $('profileName').value || 'My voice' }) });
    const res = await waitJob(r.job_id, j => { $('p1').style.width = `${Math.round(j.progress * 100)}%`; $('p1msg').textContent = j.message; });
    currentProfile = res.profile_id;
    $('p1msg').textContent = `Profile "${res.name}" created with ${res.clips} reference clips. Now confirm consent below.`;
    await refreshProfiles(); await startConsent();
  } catch (err) { $('p1msg').textContent = 'Error: ' + err.message; $('createProfile').disabled = false; }
});

// ---------------------------------------------------------------- consent ---------------
async function startConsent() {
  if (!currentProfile) return;
  $('stepConsent').classList.remove('hidden'); $('consentResult').textContent = '';
  const r = await api('/api/consent/statement', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ profile_id: currentProfile }) });
  currentConsent = r.consent_id; $('statement').textContent = r.statement;
  $('stepConsent').scrollIntoView({ behavior: 'smooth' });
}
$('newStatement').addEventListener('click', startConsent);
bindRecorder('recConsent', 'recConsentTime', async (blob) => {
  $('consentResult').textContent = 'Verifying…';
  try {
    const fd = new FormData(); fd.append('profile_id', currentProfile); fd.append('consent_id', currentConsent); fd.append('file', blob, 'consent.wav');
    const r = await api('/api/consent/verify', { method: 'POST', body: fd });
    const res = await waitJob(r.job_id, j => { $('consentResult').textContent = j.message; });
    if (res.verified) {
      $('consentResult').innerHTML = `<span style="color:var(--ok)">Consent verified</span> (text match ${Math.round(res.text_match * 100)}%, voice match ✓). You can generate speech now.`;
      await refreshProfiles(); $('step2').scrollIntoView({ behavior: 'smooth' });
    } else {
      $('consentResult').innerHTML = `<span style="color:var(--bad)">Not verified.</span> ${res.reasons.join(' ')}<br><span class="small">Heard: “${res.heard}”</span>`;
    }
  } catch (err) { $('consentResult').textContent = 'Error: ' + err.message; }
});

// ---------------------------------------------------------------- step 2/3 --------------
$('speed').addEventListener('input', () => { $('speedVal').textContent = `${Number($('speed').value).toFixed(2)}x`; });
$('profileSelect').addEventListener('change', () => { currentProfile = $('profileSelect').value; });

async function runGenerate(req) {
  lastRequest = req; $('genMsg').textContent = 'Queued…'; $('p2').style.width = '2%'; $('generate').disabled = true;
  try {
    const r = await api('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) });
    const res = await waitJob(r.job_id, j => { $('p2').style.width = `${Math.round(j.progress * 100)}%`; $('genMsg').textContent = j.message; });
    $('step3').classList.remove('hidden');
    $('player').src = res.wav_url + '&t=' + Date.now(); $('player').play().catch(() => {});
    $('dlWav').href = res.wav_url + '&download=1'; $('dlMp3').href = res.mp3_url + '&download=1';
    const m = res.metrics;
    $('metrics').innerHTML =
      stat('Duration', fmtTime(m.duration_s)) +
      stat('Speaker similarity', m.speaker_similarity.toFixed(3), m.similarity_ok ? 'ok' : 'warn') +
      stat('Loudness', `${m.loudness_lufs} LUFS`) +
      stat('Language', m.language) + stat('Style', m.style) + stat('Chunks', m.chunks);
    $('genMsg').textContent = m.similarity_ok ? 'Done.' : 'Done — similarity below the speaker\'s own band; try Regenerate or another style.';
    $('step3').scrollIntoView({ behavior: 'smooth' });
  } catch (err) { $('genMsg').textContent = 'Error: ' + err.message; }
  finally { $('generate').disabled = false; }
}
$('generate').addEventListener('click', () => {
  const pid = $('profileSelect').value; if (!pid) { alert('Create a voice profile first.'); return; }
  const text = $('script').value.trim(); if (!text) { alert('Enter a script.'); return; }
  runGenerate({ profile_id: pid, text, language: $('language').value, style: $('style').value, speed: Number($('speed').value) });
});
$('regen').addEventListener('click', () => lastRequest && runGenerate(lastRequest));

refreshStatus(); refreshProfiles(); setInterval(refreshStatus, 10000);
