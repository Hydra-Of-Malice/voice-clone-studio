// In-browser WAV recorder (48 kHz PCM16). Produces WAV directly so the server never needs ffmpeg
// to decode WebM/Opus from MediaRecorder.
class WavRecorder {
  constructor() { this.chunks = []; this.ctx = null; this.stream = null; this.node = null; this.startedAt = 0; }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    this.ctx = new AudioContext({ sampleRate: 48000 });
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    this.chunks = [];
    this.node.onaudioprocess = (e) => { this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
    src.connect(this.node); this.node.connect(this.ctx.destination);
    this.startedAt = Date.now();
  }
  get seconds() { return (Date.now() - this.startedAt) / 1000; }
  async stop() {
    const sr = this.ctx.sampleRate;
    this.node.disconnect(); this.stream.getTracks().forEach(t => t.stop()); await this.ctx.close();
    const len = this.chunks.reduce((a, c) => a + c.length, 0);
    const pcm = new Float32Array(len); let o = 0;
    for (const c of this.chunks) { pcm.set(c, o); o += c.length; }
    return this.encodeWav(pcm, sr);
  }
  encodeWav(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2); const v = new DataView(buf);
    const w = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
    v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data');
    v.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true); }
    return new Blob([buf], { type: 'audio/wav' });
  }
}
window.WavRecorder = WavRecorder;
