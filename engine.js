/* Reel Studio engine. */
(function (root) {
'use strict';

const FPS = 30, SR = 48000, SPF = SR / FPS, OUT_W = 1080, OUT_H = 1920;
const LIBS = {
  mp4box: 'https://cdn.jsdelivr.net/npm/mp4box@2.3.0/dist/mp4box.all.js',
  muxer: 'https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.2/build/mp4-muxer.js',
  ort: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.js',
  tjs: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js',
};
const WHISPER_ID = 'onnx-community/whisper-base.en_timestamped';

function defaults() {
  return {
    clap: { hp: 2000, level: 8.5, rise: 26, decay: 10, maxWidth: 0.07, iso: 10, erase: 1.5 },
    flubPause: 0.60, maxFlub: 9,
    maxGap: 0.40, pad: 0.12,
    nr: 12, lufs: -14,
    deadzone: 0.04, smooth: 0.35, faceY: 0.38,
    zoom: { minGap: 6.5, window: 2.5, cut: 1.12, push: 1.10, ramp: 3.0, maxHold: 4.5 },
    broll: { min: 1.8, target: 2.0, max: 2.4, maxFrac: 0.25, autoMinStart: 1.5, gap: 2.8 },
    cap: { size: 86, accent: '#D6B06C', baseline: 0.70, baselineUp: 0.468, maxWords: 4, side: 150 },
    pop: { on: true, x: 140, y: 985, w: 800, h: 520, radius: 28, min: 1.8, max: 2.4, gap: 2.8, inF: 13, outF: 11 },
    music: { mode: 'auto', mood: 'warm', level: -10 },
    videoBitrate: 12_000_000,
  };
}

// ------------------------------------------------------------------ small helpers
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const tick = () => new Promise(r => setTimeout(r, 0));
const sleep = ms => new Promise(r => setTimeout(r, ms));
function fmt(t) { t = Math.max(0, t); const m = Math.floor(t / 60); return `${m}:${(t - m * 60).toFixed(2).padStart(5, '0')}`; }
function percentile(arr, p) {
  const a = Float64Array.from(arr).sort(); if (!a.length) return 0;
  const i = (a.length - 1) * p / 100, lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}
function median(a) { return percentile(a, 50); }
class Cancelled extends Error { constructor() { super('Cancelled'); this.name = 'Cancelled'; } }
function check(signal) { if (signal && signal.aborted) throw new Cancelled(); }
function loadScript(src) {
  return new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) return res();
    const s = document.createElement('script'); s.src = src; s.onload = res;
    s.onerror = () => rej(new Error('Could not load ' + src.split('/').slice(4, 6).join(' ')));
    document.head.appendChild(s);
  });
}
const DIAG = {};
const DISSOLVE = 7; // frames (~0.23 s) of cross-dissolve into and out of every cutaway
const hex = (u, n = 12) => Array.from(u.slice(0, n)).map(x => x.toString(16).padStart(2, '0')).join(' ');
const rmsDb = a => { let e = 0, c = 0; for (let i = 0; i < a.length; i += 11) { e += a[i] * a[i]; c++; } return +(10 * Math.log10(e / Math.max(1, c) + 1e-12)).toFixed(1); };
const BASE = () => new URL('./', root.location ? root.location.href : 'http://x/').href;

// ------------------------------------------------------------------ DSP
function biquad(type, f0, Q, gainDb = 0, fs = SR) {
  const A = Math.pow(10, gainDb / 40), w = 2 * Math.PI * f0 / fs, cw = Math.cos(w), sw = Math.sin(w), al = sw / (2 * Q);
  let b0, b1, b2, a0, a1, a2;
  if (type === 'hp') { b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; }
  else if (type === 'lp') { b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; }
  else if (type === 'peak') { b0 = 1 + al * A; b1 = -2 * cw; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cw; a2 = 1 - al / A; }
  else if (type === 'hs') {
    const sq = 2 * Math.sqrt(A) * al;
    b0 = A * ((A + 1) + (A - 1) * cw + sq); b1 = -2 * A * ((A - 1) + (A + 1) * cw); b2 = A * ((A + 1) + (A - 1) * cw - sq);
    a0 = (A + 1) - (A - 1) * cw + sq; a1 = 2 * ((A - 1) - (A + 1) * cw); a2 = (A + 1) - (A - 1) * cw - sq;
  }
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}
function filt(x, c, out) {
  out = out || new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0; const [b0, b1, b2, a1, a2] = c;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i], y = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = xi; y2 = y1; y1 = y; out[i] = y;
  }
  return out;
}
function highpass4(x, f) { return filt(filt(x, biquad('hp', f, 0.5412)), biquad('hp', f, 1.3066)); }

function frameDb(x, win = 0.010, hop = 0.005, rate = SR) {
  const w = Math.round(win * rate), h = Math.round(hop * rate);
  const n = Math.max(0, Math.floor((x.length - w) / h) + 1);
  const cs = new Float64Array(x.length + 1);
  for (let i = 0; i < x.length; i++) cs[i + 1] = cs[i] + x[i] * x[i];
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) { const a = k * h; out[k] = 10 * Math.log10((cs[a + w] - cs[a]) / w + 1e-12); }
  return { db: out, dt: h / rate };
}

// radix-2 FFT with cached twiddles
const _tw = {};
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  let tw = _tw[n];
  if (!tw) { tw = _tw[n] = { c: new Float64Array(n / 2), s: new Float64Array(n / 2) }; for (let k = 0; k < n / 2; k++) { tw.c[k] = Math.cos(-2 * Math.PI * k / n); tw.s[k] = Math.sin(-2 * Math.PI * k / n); } }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) for (let j = 0; j < half; j++) {
      const wr = tw.c[j * step], wi = tw.s[j * step], a = i + j, b = a + half;
      const vr = re[b] * wr - im[b] * wi, vi = re[b] * wi + im[b] * wr;
      re[b] = re[a] - vr; im[b] = im[a] - vi; re[a] += vr; im[a] += vi;
    }
  }
}
function ifft(re, im) { for (let i = 0; i < im.length; i++) im[i] = -im[i]; fft(re, im); const n = re.length; for (let i = 0; i < n; i++) { re[i] /= n; im[i] = -im[i] / n; } }

/** Spectral-gate noise reduction (stands in for ffmpeg afftdn). */
async function denoise(x, reduceDb = 12, signal) {
  const N = 1024, H = 256, nb = N / 2 + 1, win = new Float32Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  const frames = Math.max(1, Math.ceil((x.length - N) / H) + 1);
  const mags = new Float32Array(frames * nb), en = new Float64Array(frames);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f < frames; f++) {
    const o = f * H;
    for (let i = 0; i < N; i++) { re[i] = (x[o + i] || 0) * win[i]; im[i] = 0; }
    fft(re, im);
    let e = 0;
    for (let k = 0; k < nb; k++) { const m = Math.hypot(re[k], im[k]); mags[f * nb + k] = m; e += m * m; }
    en[f] = e;
    if (f % 2000 === 0) { check(signal); await tick(); }
  }
  const thr = percentile(en, 15), noise = new Float64Array(nb); let cnt = 0;
  for (let f = 0; f < frames; f++) if (en[f] <= thr) { cnt++; for (let k = 0; k < nb; k++) noise[k] += mags[f * nb + k]; }
  for (let k = 0; k < nb; k++) noise[k] /= Math.max(1, cnt);
  const floor = Math.pow(10, -reduceDb / 20), out = new Float32Array(x.length), prevG = new Float32Array(nb).fill(1);
  for (let f = 0; f < frames; f++) {
    const o = f * H;
    for (let i = 0; i < N; i++) { re[i] = (x[o + i] || 0) * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < nb; k++) {
      const m = mags[f * nb + k] + 1e-9;
      let g = Math.max(floor, 1 - 1.6 * noise[k] / m);
      g = Math.max(g, 0.55 * prevG[k] + 0.45 * g * (g < prevG[k] ? 1 : 0)); // gentle release, no musical noise
      g = Math.min(1, g); prevG[k] = g;
      re[k] *= g; im[k] *= g;
      if (k > 0 && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; }
    }
    ifft(re, im);
    for (let i = 0; i < N; i++) if (o + i < out.length) out[o + i] += re[i] * win[i] / 1.5;
    if (f % 2000 === 0) { check(signal); await tick(); }
  }
  return out;
}

function compress(x, { thr = -22, ratio = 2.5, knee = 4, att = 0.012, rel = 0.2, makeupDb = 3.5 } = {}) {
  const out = new Float32Array(x.length);
  const ca = Math.exp(-1 / (att * SR)), cr = Math.exp(-1 / (rel * SR)), mk = Math.pow(10, makeupDb / 20);
  let env = 0;
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i]);
    env = a > env ? ca * env + (1 - ca) * a : cr * env + (1 - cr) * a;
    const l = 20 * Math.log10(env + 1e-9), over = l - thr;
    let gr = 0;
    if (over > knee / 2) gr = over * (1 - 1 / ratio);
    else if (over > -knee / 2) gr = (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee);
    out[i] = x[i] * Math.pow(10, -gr / 20) * mk;
  }
  return out;
}

/** Integrated loudness (ITU-R BS.1770 / EBU R128) of 1–2 channels. */
function loudness(chs) {
  const kw = chs.map(c => filt(filt(c, biquad('hs', 1681.974, 0.7071752, 3.99984)), biquad('hp', 38.13547, 0.5003270)));
  const n = kw[0].length, blk = Math.round(0.4 * SR), hop = Math.round(0.1 * SR);
  const cs = kw.map(c => { const s = new Float64Array(n + 1); for (let i = 0; i < n; i++) s[i + 1] = s[i] + c[i] * c[i]; return s; });
  const z = [];
  for (let s = 0; s + blk <= n; s += hop) { let v = 0; for (const c of cs) v += (c[s + blk] - c[s]) / blk; z.push(v); }
  if (!z.length) return -70;
  const L = v => -0.691 + 10 * Math.log10(v + 1e-15);
  const g1 = z.filter(v => L(v) > -70); if (!g1.length) return -70;
  const rel = L(g1.reduce((a, b) => a + b, 0) / g1.length) - 10;
  const g2 = g1.filter(v => L(v) > rel);
  return L(g2.reduce((a, b) => a + b, 0) / Math.max(1, g2.length));
}

function limit(chs, ceilingDb = -1.2) {
  const ceil = Math.pow(10, ceilingDb / 20), n = chs[0].length, L = Math.round(0.005 * SR);
  const req = new Float32Array(n);
  for (let i = 0; i < n; i++) { let p = 0; for (const c of chs) p = Math.max(p, Math.abs(c[i])); req[i] = p > ceil ? ceil / p : 1; }
  // running minimum over the look-ahead window (monotonic deque)
  const g = new Float32Array(n), dq = new Int32Array(n); let h = 0, t = 0;
  for (let i = n - 1; i >= 0; i--) {
    while (t > h && req[dq[t - 1]] >= req[i]) t--;
    dq[t++] = i;
    while (dq[h] > i + L) h++;
    g[i] = req[dq[h]];
  }
  const rc = 1 - Math.exp(-1 / (0.08 * SR)); let e = 1;
  for (let i = 0; i < n; i++) { e = Math.min(g[i], e + (1 - e) * rc); for (const c of chs) c[i] *= e; }
  return chs;
}

function resample(x, from, to) {
  if (from === to) return x;
  const n = Math.floor(x.length * to / from), out = new Float32Array(n), r = from / to;
  if (to < from) { // anti-alias before decimating
    const lp = biquad('lp', to * 0.45, 0.707, 0, from);
    x = filt(filt(x, lp), lp);
  }
  for (let i = 0; i < n; i++) { const p = i * r, k = Math.floor(p), f = p - k; out[i] = (x[k] || 0) * (1 - f) + (x[k + 1] || 0) * f; }
  return out;
}

// ------------------------------------------------------------------ 1. claps
function detectClaps(audio, cfg) {
  const hp = highpass4(audio, cfg.clap.hp);
  const { db: h, dt } = frameDb(hp), { db: f } = frameDb(audio);
  if (h.length < 50) return { claps: [], near: [] };
  const floor = percentile(f, 10);
  let voiced = []; for (let i = 0; i < f.length; i++) if (f[i] > floor + 15) voiced.push(h[i]);
  if (voiced.length < 50) { const m = percentile(f, 50); voiced = []; for (let i = 0; i < f.length; i++) if (f[i] > m) voiced.push(h[i]); }
  const ref = percentile(voiced, 90), n = h.length, W = Math.round(0.05 / dt);
  const sp = []; for (let i = 0; i < f.length; i++) if (f[i] > floor + 15) sp.push(f[i]);
  const fmed = sp.length > 50 ? percentile(sp, 50) : percentile(f, 70);
  const meanDb = (a, b) => { a = clamp(Math.round(a), 0, n - 1); b = clamp(Math.round(b), a + 1, n); let e = 0; for (let j = a; j < b; j++) e += Math.pow(10, f[j] / 10); return 10 * Math.log10(e / (b - a) + 1e-12); };
  const found = [], near = [];
  for (let i = 4; i < n; i++) {
    if (h[i] <= ref + cfg.clap.level - 6) continue;
    let isMax = true;
    for (let j = Math.max(0, i - W); j <= Math.min(n - 1, i + W); j++) if (h[j] > h[i]) { isMax = false; break; }
    if (!isMax) continue;
    let pre = -Infinity; for (let j = Math.max(0, i - 12); j < i - 3; j++) pre = Math.max(pre, h[j]);
    const rise = h[i] - pre;
    let post = -Infinity; const pa = i + 20, pb = Math.min(n, i + 31);
    if (pa < n) for (let j = pa; j < pb; j++) post = Math.max(post, h[j]);
    const decay = isFinite(post) ? h[i] - post : 99;
    let j0 = i; while (j0 > 0 && h[j0 - 1] > h[i] - 6) j0--;
    let j1 = i; while (j1 < n - 1 && h[j1 + 1] > h[i] - 6) j1++;
    const width = (j1 - j0 + 1) * dt, level = h[i] - ref;
    let k = i; while (k < n - 1 && h[k] > Math.max(pre + 3, h[i] - 35) && (k - i) * dt < 0.4) k++;
    const c = { t: +(i * dt).toFixed(3), start: +Math.max(0, (j0 - 2) * dt).toFixed(3), end: +(k * dt + 0.01).toFixed(3),
      level: +level.toFixed(1), rise: +rise.toFixed(1), decay: +decay.toFixed(1), widthMs: Math.round(width * 1000) };
    // a real clap sits in or at the edge of a pause; a plosive or a knock mid-sentence has speech on both sides
    c.iso = +(fmed - Math.min(meanDb(j0 - 0.30 / dt, j0 - 0.05 / dt), meanDb(k + 0.04 / dt, k + 0.30 / dt))).toFixed(1);
    const ok = level >= cfg.clap.level && rise >= cfg.clap.rise && decay >= cfg.clap.decay && width <= cfg.clap.maxWidth && c.iso >= cfg.clap.iso;
    (ok ? found : near).push(c);
  }
  const claps = [];
  for (const c of found) {
    if (claps.length && c.start - claps[claps.length - 1].end < 0.5) claps[claps.length - 1].end = Math.max(claps[claps.length - 1].end, c.end);
    else claps.push(c);
  }
  return { claps, near: near.sort((a, b) => b.level - a.level).slice(0, 5) };
}

// ------------------------------------------------------------------ 2. speech & dead air
function vad(x) {
  const { db, dt } = frameDb(x, 0.02, 0.01);
  const floor = percentile(db, 10), hi = percentile(db, 95);
  const on = floor + Math.max(6, 0.3 * (hi - floor)), off = on - 3, hang = Math.round(0.12 / dt);
  const segs = []; let active = false, start = 0, quiet = 0;
  for (let i = 0; i < db.length; i++) {
    if (!active && db[i] > on) { active = true; start = i; quiet = 0; }
    else if (active) {
      if (db[i] < off) { if (++quiet > hang) { segs.push([start * dt, (i - quiet + 1) * dt + 0.02]); active = false; } }
      else quiet = 0;
    }
  }
  if (active) segs.push([start * dt, db.length * dt]);
  const merged = [];
  for (let [s, e] of segs) {
    s = Math.max(0, s - 0.03); e += 0.03;
    if (e - s < 0.08) continue;
    if (merged.length && s - merged[merged.length - 1][1] < 0.12) merged[merged.length - 1][1] = e;
    else merged.push([s, e]);
  }
  return merged;
}
function addMissedWords(speech, words) {
  const out = speech.map(s => s.slice());
  for (const w of words) {
    if (!out.some(([s, e]) => w.e > s && w.s < e)) out.push([w.s, w.e]);
  }
  out.sort((a, b) => a[0] - b[0]);
  const m = [];
  for (const s of out) { if (m.length && s[0] <= m[m.length - 1][1]) m[m.length - 1][1] = Math.max(m[m.length - 1][1], s[1]); else m.push(s); }
  return m;
}
function subtract(intervals, cuts) {
  const out = [];
  for (const [s, e] of intervals) {
    let pieces = [[s, e]];
    for (const [a, b] of cuts) {
      const nxt = [];
      for (const [x, y] of pieces) {
        if (b <= x || a >= y) nxt.push([x, y]);
        else { if (a > x) nxt.push([x, a]); if (b < y) nxt.push([b, y]); }
      }
      pieces = nxt;
    }
    out.push(...pieces);
  }
  return out;
}

const STOP = new Set(`a an the and or but so to of in on at for with from by as is are was were be been being am i im i'm
you your you're we us our they them their he she it its it's this that these those there here what which who whom whose when
where why how do does did doing done have has had having not no nope yes yeah just like really very can could would should
will shall may might must get got gonna wanna if then than too also all any some each every more most much many my me mine
his her hers him one ones up down out over into about off again still don't doesn't didn't can't won't isn't aren't wasn't
that's what's let's okay ok alright right well oh um uh`.split(/\s+/));
const norm = w => w.toLowerCase().replace(/’/g, "'").replace(/[^a-z0-9']/g, '');
function stem(w) {
  w = w.toLowerCase().replace(/[^a-z0-9]/g, '');
  for (const suf of ['ings', 'ing', 'es', 'ed', 's']) if (w.length > suf.length + 2 && w.endsWith(suf)) return w.slice(0, -suf.length);
  return w;
}
// similarity ratio like difflib (Levenshtein based)
function ratio(a, b) {
  if (a === b) return 1; if (!a.length || !b.length) return 0;
  let prev = new Array(b.length + 1); for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return 1 - prev[b.length] / Math.max(a.length, b.length);
}

/** What to cut for each clap: a clap is an intentional marker that erases the 1.5 seconds before it (and the clap itself). */
function flubRegions(claps, speech, words, cfg) {
  return claps.map(c => {
    const c0 = c.start, c1 = c.end; let from = Math.max(0, c0 - cfg.clap.erase);
    const split = words.find(w => w.s < from - 0.03 && w.e > from + 0.03);     // the 1.5 s mark lands mid-word: take the whole word
    if (split && from - split.s <= 0.4) from = Math.max(0, split.s - 0.02);
    const f = { clapT: c.t, flubStart: from, clapEnd: c1, redoStart: c1, cut: [from, c1 + 0.03], method: `marker clap: erased the ${(c0 - from).toFixed(1)} s before it` };
    const after = speech.filter(([, e]) => e > c1 + 0.05).map(([x]) => x); f.redoStart = after.length ? Math.max(after[0], c1) : c1;
    f.flubText = words.filter(w => w.e > from + 0.05 && w.s < c0).map(w => w.w).join(' ');
    f.redoText = words.filter(w => w.s >= c1 - 0.05).slice(0, 6).map(w => w.w).join(' ');
    f.similarity = 1; f.long = false;
    return f;
  });
}

function buildKeep(speech, removed, total, cfg, restored = []) {
  const sp = subtract(speech, removed).filter(([s, e]) => e - s >= 0.06);
  const keep = [], trimmed = [];
  for (const [s, e] of sp) {
    const a = Math.max(0, s - cfg.pad), b = Math.min(total, e + cfg.pad);
    const isRestored = keep.length && restored.some(([x, y]) => Math.abs(x - keep[keep.length - 1][2]) < 0.06 && Math.abs(y - s) < 0.06);
    if (keep.length && (s - keep[keep.length - 1][2] <= cfg.maxGap || isRestored)) { keep[keep.length - 1][1] = Math.max(keep[keep.length - 1][1], b); keep[keep.length - 1][2] = e; }
    else { if (keep.length) trimmed.push([keep[keep.length - 1][2], s]); keep.push([a, b, e]); }
  }
  const kept = subtract(keep.map(([a, b]) => [a, b]), removed);
  const tr = trimmed.map(([a, b]) => [a, b, (b - a) - removed.reduce((acc, [x, y]) => acc + Math.max(0, Math.min(b, y) - Math.max(a, x)), 0)])
    .filter(t => t[2] > cfg.maxGap);
  const frames = [];
  for (const [a, b] of kept) {
    const f0 = Math.ceil(a * FPS), f1 = Math.floor(b * FPS);
    if (f1 - f0 < 3) continue;
    if (frames.length && f0 <= frames[frames.length - 1][1]) frames[frames.length - 1][1] = Math.max(frames[frames.length - 1][1], f1);
    else frames.push([f0, f1]);
  }
  return { keep: frames, trimmed: tr };
}

class TimeMap {
  constructor(keep) {
    this.keep = keep; this.starts = []; let o = 0;
    for (const [f0, f1] of keep) { this.starts.push(o); o += f1 - f0; }
    this.nOut = o;
  }
  rawToOut(t) {
    const f = t * FPS;
    for (let i = 0; i < this.keep.length; i++) {
      const [f0, f1] = this.keep[i];
      if (f < f0) return this.starts[i] / FPS;
      if (f < f1) return (this.starts[i] + f - f0) / FPS;
    }
    return this.nOut / FPS;
  }
  outToRawFrame(oi) { // output frame index -> raw frame index
    let lo = 0, hi = this.keep.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (this.starts[m] <= oi) lo = m; else hi = m - 1; }
    return this.keep[lo][0] + (oi - this.starts[lo]);
  }
  outToRaw(t) { const f = t * FPS, k = this.segIndex(Math.floor(f)); return (this.keep[k][0] + (f - this.starts[k])) / FPS; }
  segIndex(oi) { let lo = 0, hi = this.keep.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (this.starts[m] <= oi) lo = m; else hi = m - 1; } return lo; }
  cutPoints() { return this.starts.slice(1).map(o => o / FPS); }
}

function cutAudio(a, keep) {
  const fade = Math.round(0.004 * SR); let n = 0;
  for (const [f0, f1] of keep) n += (f1 - f0) * SPF;
  const out = new Float32Array(n); let o = 0;
  for (const [f0, f1] of keep) {
    const len = (f1 - f0) * SPF;
    out.set(a.subarray(f0 * SPF, Math.min(a.length, f1 * SPF)), o);
    for (let i = 0; i < fade; i++) { const g = i / fade; out[o + i] *= g; out[o + len - 1 - i] *= g; }
    o += len;
  }
  return out;
}

function mapWords(words, tmap) {
  const out = [];
  for (const w of words) {
    // keep a word if enough of it survives the cut (Whisper often starts the first word a little early)
    let k = -1, best = 0;
    tmap.keep.forEach(([f0, f1], i) => { const ov = Math.min(w.e, f1 / FPS) - Math.max(w.s, f0 / FPS); if (ov > best) { best = ov; k = i; } });
    if (k < 0 || (best < 0.1 && best < 0.35 * (w.e - w.s))) continue;
    const [f0, f1] = tmap.keep[k];
    const s = Math.max(w.s, f0 / FPS), e = Math.min(w.e, f1 / FPS);
    out.push({ w: w.w, s: +((tmap.starts[k] + s * FPS - f0) / FPS).toFixed(3), e: +((tmap.starts[k] + e * FPS - f0) / FPS).toFixed(3) });
  }
  for (let i = 1; i < out.length; i++) { out[i].s = Math.max(out[i].s, out[i - 1].s); out[i].e = Math.max(out[i].e, out[i].s + 0.05); }
  return out;
}

/** Snap word edges to where the voice actually starts and stops. Whisper often starts a word early, inside the pause
 *  before it, so captions would show before he speaks. Conservative: each word only moves inside its own span,
 *  only out of clear silence, and never pushes the words after it. */
function refineTiming(words, voice) {
  if (!words.length) return { moved: 0, shifted: 0 };
  const hop = 0.01, { db } = frameDb(voice, 0.02, hop), floor = percentile(db, 8), hi = percentile(db, 95);
  const n = db.length, at = i => db[clamp(Math.round(i), 0, n - 1)];
  const loud = floor + Math.max(8, 0.22 * (hi - floor)), quiet = floor + Math.max(5, 0.12 * (hi - floor));
  const isQuiet = (a, b) => { for (let i = Math.round(a); i < Math.round(b); i++) if (at(i) > quiet) return false; return true; };
  const orig = words.map(w => [w.s, w.e]);
  let moved = 0, shifted = 0;
  words.forEach((w, k) => {
    const [s0, e0] = orig[k], nextS = k + 1 < words.length ? orig[k + 1][0] : Infinity;
    let fs = s0 / hop, fe = e0 / hop;
    // start: if the word opens on 80 ms+ of clear silence, move it to the first loud frame inside the word
    if (fe - fs > 10 && isQuiet(fs, fs + 8)) { let f = fs + 8; while (f < fe - 4 && at(f) <= loud) f++; if (f < fe - 4) fs = f - 3; }
    else if (fe - fs > 20) { // or if a clear pause sits inside the word's span, the word really starts after it
      let f = fs, gapEnd = -1;
      while (f < fe - 6) { if (at(f) <= quiet) { let g = f; while (g < fe && at(g) <= quiet) g++; if (g - f >= 12 && g < fe - 5) gapEnd = g; f = g + 1; } else f++; }
      if (gapEnd > 0) fs = gapEnd - 3;
    }
    if (fs === s0 / hop && at(fs) > quiet) {  // starts late: the voice was already going; back up to where it began, if a pause precedes it
      const prevE = k ? orig[k - 1][1] / hop : 0; let f = fs;
      while (f > prevE && fs - f < 35 && at(f - 1) > quiet) f--;
      if (f > prevE + 1 && at(f - 1) <= quiet) fs = f - 2;
    }
    // end: pull back out of clear trailing silence; run on through a voiced tail (not into the next word)
    if (isQuiet(fe - 6, fe)) { let f = fe - 6; while (f > fs + 6 && at(f) <= quiet) f--; fe = f + 4; }
    else { let f = fe; const lim = Math.min(fe + 25, nextS / hop - 2); while (f < lim && at(f + 1) > quiet) f++; fe = f + 2; }
    const ns = +(fs * hop).toFixed(3), ne = +Math.max(ns + 0.06, fe * hop).toFixed(3);
    if (ns < e0 && ne > ns) { w.s = ns; w.e = ne; }
    if (w.s - s0 > 0.25) shifted++;
    moved += Math.abs(w.s - s0) + Math.abs(w.e - e0);
  });
  for (let i = 1; i < words.length; i++) {
    if (words[i].s < words[i - 1].s + 0.02) words[i].s = Math.min(words[i].e - 0.04, words[i - 1].s + 0.02);
    if (words[i - 1].e > words[i].s) words[i - 1].e = Math.max(words[i - 1].s + 0.04, words[i].s);
  }
  return { moved: +(moved / words.length).toFixed(3), shifted };
}
/** Your corrections list. Text lines like:  patients will => patience will */
function parseCorrections(text) {
  return String(text || '').split(/\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#') && l.includes('=>'))
    .map(l => { const [a, b] = l.split('=>'); return [a.trim().toLowerCase().split(/\s+/).filter(Boolean), b.trim().split(/\s+/).filter(Boolean)]; })
    .filter(([a]) => a.length);
}
function applyCorrections(words, rules) {
  const applied = [];
  for (const [pat, rep] of rules) {
    for (let i = 0; i + pat.length <= words.length; i++) {
      if (!pat.every((w, k) => norm(words[i + k].w) === norm(w))) continue;
      const s0 = words[i].s, e0 = words[i + pat.length - 1].e, tail = (words[i + pat.length - 1].w.match(/[.,;:!?…]+$/) || [''])[0];
      const nw = rep.map((t, k) => ({ ...words[i], w: t + (k === rep.length - 1 && !/[.,;:!?]$/.test(t) ? tail : ''),
        s: +(s0 + (e0 - s0) * k / rep.length).toFixed(3), e: +(s0 + (e0 - s0) * (k + 1) / rep.length).toFixed(3) }));
      words.splice(i, pat.length, ...nw); applied.push(`"${pat.join(' ')}" → "${rep.join(' ')}"`); i += Math.max(0, rep.length - 1);
    }
  }
  return applied;
}
const DEFAULT_CORRECTIONS = `patients will => patience will
are deals => RDLs
our pe => RPE
progressive over load => progressive overload`;
function addEmphasis(words, audio) {
  if (!words.length) return;
  const dbs = words.map(w => {
    const a = Math.floor(w.s * SR), b = Math.max(a + 1, Math.floor(w.e * SR)); let e = 0;
    for (let i = a; i < b && i < audio.length; i++) e += audio[i] * audio[i];
    return 10 * Math.log10(e / (b - a) + 1e-10);
  });
  words.forEach((w, i) => {
    const near = dbs.filter((_, j) => Math.abs(words[j].s - w.s) < 3);
    w.emph = +(dbs[i] - median(near)).toFixed(2);
    w.sent = i === 0 || /[.?!]["')]*$/.test(words[i - 1].w);
  });
}

// ------------------------------------------------------------------ 7. B-roll from the library
const GENERIC = new Set(['homegym', 'train', 'mood', 'contrast', 'still', 'tight', 'wide', 'broll', 'clip', 'shot', 'home', 'gym', 'training']);
function itemTerms(it) {
  const words = new Set(), phrases = [];
  for (const t of it.tags || []) {
    const toks = t.toLowerCase().split(/[\s\-]+/).filter(Boolean);
    if (toks.length >= 2) phrases.push(t.toLowerCase());
    for (const x of toks) if (x.length >= 3 && !STOP.has(x) && !GENERIC.has(x)) words.add(stem(x));
  }
  for (const x of String(it.id).toLowerCase().split(/[_\-\s]+/)) if (x.length >= 3 && !STOP.has(x) && !GENERIC.has(x)) words.add(stem(x));
  return { words: [...words], phrases };
}
function parseNotes(text) {
  const out = [];
  for (let line of String(text || '').split(/\n/)) {
    line = line.trim().replace(/^[-•*]\s*/, '');
    if (!line || line.startsWith('#')) continue;
    let parts = line.split(/\s*(?:→|->|=>|—>|–>)\s*/);
    if (parts.length < 2) { const i = line.indexOf(':'); if (i > 0) parts = [line.slice(0, i), line.slice(i + 1)]; }
    if (parts.length < 2) parts = line.split(/\s+over\s+/i);
    if (parts.length < 2) { out.push({ raw: line, error: "couldn't split into clip and line" }); continue; }
    let [clip, ...rest] = parts; let target = rest.join(' ').trim();
    target = target.replace(/^(over|on|at|during|when i say)\s+/i, '').replace(/^["'“”‘’\s]+|["'“”‘’\s]+$/g, '');
    out.push({ raw: line, clip: clip.trim(), target });
  }
  return out;
}
function parseTime(s) {
  s = s.trim().replace(/^@\s*/, '');
  let m = s.match(/^(\d+):(\d{1,2}(?:\.\d+)?)$/); if (m) return +m[1] * 60 + +m[2];
  m = s.match(/^(\d+(?:\.\d+)?)\s*s(ec)?$/); return m ? +m[1] : null;
}
function bestItem(name, items) {
  const key = name.toLowerCase().replace(/[_\-]+/g, ' ').trim(); let best = null, bs = 0;
  for (const it of items) {
    const cands = [String(it.id).replace(/[_\-]+/g, ' '), String(it.name || '').toLowerCase(), ...(it.tags || []).map(t => t.toLowerCase())];
    let s = Math.max(...cands.map(c => ratio(key, c)));
    const kt = new Set(key.split(' ')), it2 = new Set(cands.join(' ').split(/\s+/));
    if ([...kt].some(t => it2.has(t))) s += 0.15;
    if (s > bs) { bs = s; best = it; }
  }
  return bs >= 0.55 ? best : null;
}
function findPhrase(words, phrase) {
  const p = phrase.split(/\s+/).map(norm).filter(Boolean); if (!p.length || !words.length) return null;
  const wt = words.map(w => norm(w.w)), target = p.join(' '); let best = [0, null];
  for (let L = Math.max(1, p.length - 2); L <= p.length + 2; L++)
    for (let i = 0; i + L <= wt.length; i++) {
      const r = ratio(target, wt.slice(i, i + L).join(' '));
      if (r > best[0]) best = [r, [i, i + L]];
    }
  if (best[0] < 0.6) return null;
  const [i, j] = best[1];
  return { i, s: words[i].s, e: words[j - 1].e, score: +best[0].toFixed(2), text: words.slice(i, j).map(w => w.w).join(' ') };
}
function snapEnd(words, t, lo, hi) {
  const gaps = [];
  for (let k = 0; k + 1 < words.length; k++) { const g = (words[k].e + words[k + 1].s) / 2; if (words[k + 1].s - words[k].e > 0.06 && g >= lo && g <= hi) gaps.push(g); }
  const ends = words.map(w => w.e + 0.05).filter(x => x >= lo && x <= hi);
  const pool = gaps.length ? gaps : ends;
  return pool.length ? pool.reduce((a, b) => Math.abs(b - t) < Math.abs(a - t) ? b : a) : clamp(t, lo, hi);
}
/** Group words into short phrases (for the review screen). */
function phrases(words, maxWords = 12) {
  const out = []; let cur = [];
  words.forEach((w, i) => {
    cur.push(i);
    const end = /[.?!]["')]*$/.test(w.w) || (/[,;:—]$/.test(w.w) && cur.length >= 5) || cur.length >= maxWords ||
      (i + 1 < words.length && words[i + 1].s - w.e > 0.45);
    if (end) { out.push(cur); cur = []; }
  });
  if (cur.length) out.push(cur);
  return out.map(ix => ({ a: ix[0], b: ix[ix.length - 1], s: words[ix[0]].s, e: words[ix[ix.length - 1]].e, text: ix.map(i => words[i].w).join(' ') }));
}
function catalogLine(it) {
  const src = it.source === 'stock' ? 'stock' : 'yours';
  const flags = [it.shirtless && 'shirtless', it.others_visible && 'strangers visible'].filter(Boolean).join(', ');
  return `${it.id} | ${src} | ${it.category || ''} | ${it.energy || ''} | ${(it.subject || '').slice(0, 90)} | ${(it.tags || []).slice(0, 10).join(', ')}${flags ? ' | ' + flags : ''}`;
}
// ------------------------------------------------------------------ delivery: how he says each line
/** Per line: loudness against your average, pace, pause before it, and whether you're punching it. */
function deliveryRows(words, voice, ph) {
  const rows = ph.map((p, k) => {
    const a = Math.floor(p.s * SR), b = Math.max(a + 1, Math.floor(p.e * SR)); let e = 0;
    for (let i = a; i < b && i < voice.length; i++) e += voice[i] * voice[i];
    return { db: 10 * Math.log10(e / (b - a) + 1e-10), wps: (p.b - p.a + 1) / Math.max(0.3, p.e - p.s), pause: k ? p.s - ph[k - 1].e : p.s, text: p.text };
  });
  const mdb = median(rows.map(r => r.db)), mw = median(rows.map(r => r.wps));
  const out = new Map();
  ph.forEach((p, k) => { const r = rows[k]; out.set(`${p.a}-${p.b}`, { loud: r.db - mdb, slow: r.wps < 0.8 * mw, pause: r.pause, emphatic: r.db - mdb > 2.5 || /!/.test(r.text) }); });
  return out;
}
let _embedder = null;
/** A small open-source sentence model (MiniLM, ~23 MB) that runs in the browser: matches what you mean, not just words. */
async function loadEmbedder() {
  if (_embedder) return _embedder;
  const T = await import(LIBS.tjs);
  T.env.allowLocalModels = false; T.env.allowRemoteModels = true; T.env.useBrowserCache = true;
  T.env.backends.onnx.wasm.numThreads = 1; T.env.backends.onnx.wasm.proxy = false;
  _embedder = await T.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8', device: 'wasm' });
  return _embedder;
}
async function embed(texts) {
  const m = await loadEmbedder(), out = [];
  for (let i = 0; i < texts.length; i += 16) {
    const t = await m(texts.slice(i, i + 16), { pooling: 'mean', normalize: true });
    const d = t.dims[1]; for (let k = 0; k < t.dims[0]; k++) out.push(t.data.slice(k * d, (k + 1) * d));
  }
  return out;
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function itemText(it) { return `${it.subject || ''}. ${(it.tags || []).join(', ')}. ${String(it.category || '').replace('_', ' ')}`; }
/** Meaning similarity of every line (with a little context) to every clip and photo: Map "a-b" -> Map(id -> similarity). */
async function meaningSims(words, items) {
  const ph = phrases(words, 10);
  const lineV = await embed(ph.map(p => p.text));
  const ctxV = await embed(ph.map(p => words.slice(Math.max(0, p.a - 6), Math.min(words.length, p.b + 7)).map(w => w.w).join(' ')));
  const itemV = await embed(items.map(itemText));
  const out = new Map();
  ph.forEach((p, k) => { const m = new Map(); items.forEach((it, j) => m.set(it.id, 0.7 * dot(lineV[k], itemV[j]) + 0.3 * dot(ctxV[k], itemV[j]))); out.set(`${p.a}-${p.b}`, m); });
  return out;
}
function targetVisuals(total) { return total < 9 ? 0 : clamp(Math.round(total / 7), 1, 9); }
function popAlpha(oi, p, cfg) {
  const f0 = Math.round(p.s * FPS), f1 = Math.round(p.e * FPS);
  if (oi < f0 || oi >= f1) return null;
  const C = cfg.pop, kIn = oi - f0, kOut = f1 - 1 - oi;
  const ein = clamp((kIn + 1) / C.inF, 0, 1), eout = clamp((kOut + 1) / C.outF, 0, 1);
  const easeOut = u => 1 - Math.pow(1 - u, 4), easeIO = u => u * u * (3 - 2 * u);
  const i = easeOut(ein), o = easeIO(eout);
  return { a: Math.min(i, o), dy: (1 - i) * 42 + (1 - o) * 18, sc: (0.94 + 0.06 * i) * (0.985 + 0.015 * o), u: (oi - f0) / Math.max(1, f1 - f0) };
}
/** 0..1: how far the captions have moved up to make room for a photo card (they move first, then the card arrives). */
function captionLift(t, pops) {
  let m = 0; const R = 0.3, sm = u => { u = clamp(u, 0, 1); return u * u * (3 - 2 * u); };
  for (const p of pops) m = Math.max(m, Math.min(sm((t - (p.s - R)) / R), sm((p.e + R - t) / R)));
  return m;
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function drawPop(ctx, img, st, cfg) {
  const C = cfg.pop, cx = C.x + C.w / 2, cy = C.y + C.h / 2;
  ctx.save();
  ctx.globalAlpha = st.a;
  ctx.translate(cx, cy + st.dy); ctx.scale(st.sc, st.sc); ctx.translate(-C.w / 2, -C.h / 2);
  ctx.shadowColor = 'rgba(0,0,0,0.38)'; ctx.shadowBlur = 44; ctx.shadowOffsetY = 14;
  roundRect(ctx, 0, 0, C.w, C.h, C.radius); ctx.fillStyle = '#111'; ctx.fill();
  ctx.shadowColor = 'transparent';
  ctx.save(); roundRect(ctx, 0, 0, C.w, C.h, C.radius); ctx.clip();
  const drift = 1 + 0.04 * st.u, k = Math.max(C.w / img.width, C.h / img.height) * drift;
  ctx.drawImage(img, (C.w - img.width * k) / 2, (C.h - img.height * k) / 2, img.width * k, img.height * k);
  ctx.restore();
  roundRect(ctx, 0, 0, C.w, C.h, C.radius); ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.stroke();
  ctx.restore();
}

/** Move a cut-in to the breath before the phrase, and slide it so a jump cut lands under the cutaway, not in its dissolve. */
function rhythmStart(words, s, cuts) {
  let best = null;
  for (let k = 0; k + 1 < words.length; k++) {
    const g0 = words[k].e, g1 = words[k + 1].s;
    if (g1 - g0 >= 0.12 && g1 <= s + 0.06 && g1 >= s - 0.45) best = Math.max(g0, g1 - 0.12);
  }
  if (best != null) s = best;
  for (const c of cuts || []) if (c > s - 0.08 && c < s + (DISSOLVE + 3) / FPS) s = Math.max(0, c - (DISSOLVE + 2) / FPS);
  return s;
}
/** Place every visual, full-screen clips and half-screen photo cards, by one set of rules:
 *  about 2 s each, cut in on a breath, at least ~3 s apart, never in the opening or closing line,
 *  and only when it relates to the line. Your notes first, then a fill to a steady rhythm. */
function planVisuals(words, items, photos, notes, tmap, total, cfg, aiPicks, ctx = {}) {
  const B = cfg.broll, PC = cfg.pop, budget = B.maxFrac * total, broll = [], pops = [], log = [], cuts = tmap.cutPoints();
  let shirtless = false;
  const firstEnd = Math.min(4, (() => { for (const w of words) if (/[.?!]["')]*$/.test(w.w)) return w.e; return 3; })());
  const lastSentence = Math.max(total - 4, (() => { for (let i = words.length - 2; i >= 0; i--) if (/[.?!]["')]*$/.test(words[i].w)) return words[i + 1].s; return total; })());
  const all = () => [...broll, ...pops].filter(p => !p.off);
  function when(s, maxDur, auto) {
    s = rhythmStart(words, Math.max(0, s - 0.1), cuts);                                    // cut in on the breath before the thought
    if (auto && s < Math.max(B.autoMinStart, firstEnd - 0.15)) return 'in the opening line';
    if (auto && s > lastSentence - 0.3) return 'on the closing line';
    const hi = Math.min(s + Math.min(B.max, maxDur), total - 0.4), lo = s + Math.min(B.min, maxDur);
    if (hi - s < (auto ? B.min - 0.05 : 0.9)) return 'no room before the end';
    let e = snapEnd(words, s + Math.min(B.target, maxDur), Math.min(lo, hi), hi);             // ~2 s, ending between words
    s = Math.round(s * FPS) / FPS; e = Math.round(e * FPS) / FPS;
    if (e - s < (auto ? B.min - 0.07 : 0.9)) return 'no room before the end';
    const near = all().find(p => !(e + B.gap <= p.s || s >= p.e + B.gap));
    if (near) return `too close to ${near.id}`;
    return { s, e };
  }
  function addClip(it, s, why, text, auto) {
    if (all().some(p => p.id === it.id)) return 'already used';
    if (it.shirtless && shirtless) return 'already used one shirtless shot';
    const dur0 = it.kind === 'still' ? 2.6 : it.duration;
    if (auto && dur0 < B.min - 0.05) return 'clip is shorter than 2 s';
    const r = when(s, dur0, auto); if (typeof r === 'string') return r;
    if (broll.filter(p => !p.off).reduce((a, p) => a + p.e - p.s, 0) + (r.e - r.s) > budget) return 'would push full-screen B-roll past 25% of the reel';
    const clipIn = it.kind === 'still' ? 0 : Math.max(0, (it.duration - (r.e - r.s)) / 2);
    broll.push({ id: it.id, item: it, s: r.s, e: r.e, clipIn: +clipIn.toFixed(3), why, text });
    if (it.shirtless) shirtless = true;
    return null;
  }
  function addPhoto(it, s, why, text, auto) {
    if (!PC.on) return 'photo cards are off';
    if (all().some(p => p.id === it.id)) return 'already used';
    const r = when(s, 99, auto); if (typeof r === 'string') return r;
    pops.push({ id: it.id, item: it, s: r.s, e: r.e, why, text });
    return null;
  }
  const add = (type, it, s, why, text, auto) => type === 'photo' ? addPhoto(it, s, why, text, auto) : addClip(it, s, why, text, auto);
  let mode = 'auto';
  if (notes && notes.trim()) {
    mode = 'notes';
    for (const n of parseNotes(notes)) {
      if (n.error) { log.push(`Skipped "${n.raw}": ${n.error}`); continue; }
      const it = bestItem(n.clip, [...items, ...photos]);
      if (!it) { log.push(`Skipped "${n.raw}": no clip or photo matches "${n.clip}"`); continue; }
      const t = parseTime(n.target); let hit;
      if (t != null) { const s = tmap.rawToOut(t); hit = { s, text: `raw ${fmt(t)}` }; } else hit = findPhrase(words, n.target);
      if (!hit) { log.push(`Skipped "${n.raw}": couldn't find that line in what you said`); continue; }
      const err = add(it.kind === 'photo' ? 'photo' : 'clip', it, hit.s, 'your note', hit.text, false);
      if (err) log.push(`Skipped "${n.raw}": ${err}`);
    }
  }
  if (aiPicks) {
    mode = mode === 'notes' ? 'notes+ai' : 'ai';
    for (const p of aiPicks.slice().sort((x, y) => y.score - x.score || x.a - y.a)) {     // best picks claim their spot first
      const text = words.slice(p.a, p.b + 1).map(w => w.w).join(' ');
      let err = add(p.type, p.it, words[p.a].s, p.why || 'fits the line', text, true);
      if (err && /too close|opening/.test(err)) {                                       // try later on the same line
        for (let k = p.a + 1; k <= p.b && err; k++) if (words[p.b].e - words[k].s > 1.2) err = add(p.type, p.it, words[k].s, p.why || 'fits the line', text, true);
      }
      if (err) log.push(`${p.it.id}: skipped, ${err}`);
    }
  }
  const before = broll.length + pops.length;
  fillVisuals(words, items, photos, add, all, cuts, total, cfg, log, firstEnd, lastSentence, ctx);
  if (broll.length + pops.length > before) mode += '+fill';
  broll.sort((a, b) => a.s - b.s); pops.sort((a, b) => a.s - b.s);
  DIAG.visuals = { mode, clips: broll.length, photos: pops.length, filled: broll.length + pops.length - before, clipSecs: +broll.reduce((a, p) => a + p.e - p.s, 0).toFixed(1), of: +total.toFixed(1) };
  return { broll, pops, log, mode };
}
const CUES = {
  nature: ['patient', 'patience', 'slow', 'years', 'decades', 'lifetime', 'life', 'calm', 'stress', 'peace', 'breathe', 'perspective', 'season', 'simple', 'enough', 'quiet', 'rest', 'sleep', 'recover', 'recovery', 'mind', 'grateful', 'trust', 'process', 'journey', 'grow', 'growth', 'nature', 'outside', 'walk', 'walking', 'sun', 'morning', 'tomorrow', 'forever', 'decade', 'overwhelmed', 'tired', 'burnout', 'anxious', 'worry'],
  training: ['lift', 'lifting', 'rep', 'reps', 'set', 'sets', 'weight', 'weights', 'squat', 'bench', 'deadlift', 'press', 'train', 'training', 'workout', 'gym', 'muscle', 'strong', 'strength', 'overload', 'heavy', 'barbell', 'dumbbell', 'dumbbells', 'pullup', 'pullups', 'pushup', 'pushups', 'row', 'curl', 'curls', 'effort', 'work', 'progress', 'progressive', 'form', 'exercise', 'cardio', 'run', 'running', 'body', 'sweat', 'grind', 'boring'],
  home_gym: ['home', 'gym', 'garage', 'basement', 'bench', 'equipment', 'budget', 'cheap', 'setup', 'rack', 'plates', 'plate', 'membership', 'fancy', 'expensive', 'wobbly'],
  habits: ['habit', 'habits', 'consistent', 'consistency', 'routine', 'daily', 'track', 'write', 'notebook', 'journal', 'plan', 'discipline', 'goal', 'goals', 'water', 'steps', 'alarm', 'early'],
  food: ['protein', 'eat', 'eating', 'food', 'meal', 'meals', 'calorie', 'calories', 'diet', 'nutrition', 'carbs', 'fat', 'chicken', 'eggs', 'rice', 'hungry', 'snack', 'cook', 'gram', 'grams', 'pizza', 'dinner', 'breakfast', 'lunch'],
  family: ['kid', 'kids', 'son', 'daughter', 'wife', 'family', 'dad', 'father', 'children', 'child', 'parent', 'love'],
  contrast: ['influencer', 'influencers', 'fancy', 'hype', 'bro', 'bros', 'supplements', 'supplement', 'shortcut', 'hack', 'hacks', 'instagram', 'program', 'expensive', 'magic', 'ripped', 'shredded'],
  mood: ['honestly', 'truth', 'real', 'hard', 'quit', 'doubt', 'feel', 'feeling', 'think', 'remember', 'proud', 'alone', 'start', 'again'],
};
/** Fill the reel to a natural rhythm with visuals that relate to the line, clip or photo by the same scoring:
 *  what he names, the feeling of the line (cue words), his own footage first. Slots over a jump cut come first. */
function fillVisuals(words, items, photos, add, all, cuts, total, cfg, log, firstEnd, lastSentence, ctx = {}) {
  const want = targetVisuals(total), B = cfg.broll;
  if (all().length >= want || !words.length) return;
  const pool = [
    ...items.filter(it => !it.others_visible && it.kind !== 'photo' && (it.kind === 'still' || (it.duration || 0) >= B.min - 0.05)).map(it => ({ it, type: 'clip' })),
    ...(cfg.pop.on ? photos.filter(it => !it.others_visible).map(it => ({ it, type: 'photo' })) : []),
  ];
  if (!pool.length) return;
  const ph = phrases(words, 10);
  const terms = pool.map(({ it }) => { const t = itemTerms(it); String(it.subject || '').toLowerCase().split(/[^a-z']+/).forEach(x => { if (x.length >= 4 && !STOP.has(x)) t.words.push(stem(x)); }); return [...new Set(t.words)]; });
  const df = {}; for (const t of terms) for (const w of t) df[w] = (df[w] || 0) + 1;
  const idf = w => Math.log((pool.length + 1) / ((df[w] || 0) + 0.5));
  const cueSets = Object.fromEntries(Object.entries(CUES).map(([k, v]) => [k, new Set(v.map(stem))]));
  const slots = ph.filter(p => p.s >= Math.max(B.autoMinStart, firstEnd - 0.1) && p.s < lastSentence - 0.4 && p.e - p.s >= 0.6).map(p => {
    let sc = 0; const d = p.e - p.s;
    if (cuts.some(c => c >= p.s - 0.15 && c <= p.s + Math.min(2.4, d))) sc += 3;
    if (d >= 1.2 && d <= 3.6) sc += 1;
    return { ...p, base: sc };
  });
  let guard = 0;
  while (all().length < want && slots.length && guard++ < 40) {
    const centers = all().map(p => (p.s + p.e) / 2);
    slots.forEach(sl => { const c = (sl.s + sl.e) / 2, d = centers.length ? Math.min(...centers.map(x => Math.abs(x - c))) : total / 2; sl.score = sl.base + Math.min(4, d / Math.max(2, total / (want + 1)) * 3) - (d < 3.5 ? 6 : 0); });
    slots.sort((a, b) => b.score - a.score);
    const sl = slots.shift(); if (sl.score < -2) break;
    const inSlot = new Set(words.slice(sl.a, sl.b + 1).map(w => stem(w.w))), near = new Set(words.slice(Math.max(0, sl.a - 6), Math.min(words.length, sl.b + 7)).map(w => stem(w.w)));
    const cueHit = k => !cueSets[k] ? 0 : [...cueSets[k]].filter(c => inSlot.has(c)).length + 0.4 * [...cueSets[k]].filter(c => near.has(c) && !inSlot.has(c)).length;
    const lineMood = Object.keys(CUES).map(k => [k, cueHit(k)]).sort((a, b) => b[1] - a[1])[0];
    const prev = all().filter(p => p.e <= sl.s).sort((a, b) => b.e - a.e)[0];
    const ranked = pool.map(({ it, type }, k) => {
      if (all().some(p => p.id === it.id)) return null;
      const hit = terms[k].filter(t => inSlot.has(t)), hitNear = terms[k].filter(t => near.has(t) && !inSlot.has(t));
      const lw = it.source === 'stock' ? (hit.length >= 2 ? 0.75 : 0.4) : 1.2;
      let sc = hit.reduce((a, t) => a + idf(t), 0) * (type === 'photo' ? lw + 0.3 : lw) + hitNear.reduce((a, t) => a + idf(t), 0) * 0.35;
      const cat = it.category;
      if (cueSets[cat]) sc += cueHit(cat) * 1.6;
      if (lineMood && lineMood[1] >= 1 && cat === lineMood[0]) sc += 0.6;                  // matches the feeling of the line overall
      if (cat === 'contrast' || it.energy === 'high') sc += cueHit('contrast') > 0.9 ? 1 : -4;
      if (it.source !== 'stock') sc += 0.8; else if (cat === 'nature') sc += 0.5;
      const key = `${sl.a}-${sl.b}`, sim = ctx.sims && ctx.sims.get(key) ? ctx.sims.get(key).get(it.id) : null;
      if (sim != null) sc += Math.max(0, sim - 0.25) * 5;                           // a 0.5 meaning match adds ~1.25
      const dv = ctx.delivery && ctx.delivery.get(key);
      if (dv && dv.emphatic) sc += type === 'photo' ? 0.6 : -0.6;                   // you're punching the line: stay on camera
      if (dv && (dv.slow || dv.pause > 0.6) && dv.loud < 0 && cat === 'nature') sc += 0.6; // a quiet, slowed line can hold on a calm shot
      if (prev && prev.item.category === cat) sc -= 0.6;
      if (prev && (prev.item.kind === 'photo') === (type === 'photo')) sc -= 0.2;      // gently alternate clips and photos
      sc += ((k * 2654435761 + Math.round(sl.s * 10)) % 97) / 400;
      return { it, type, sc, why: hit.length ? `fits "${hit.slice(0, 2).join('", "')}"` : cueSets[cat] && cueHit(cat) >= 1 ? `${String(cat).replace('_', ' ')} feel for this line` : sim != null && sim > 0.4 ? 'matches what you mean' : `${String(cat).replace('_', ' ')} shot` };
    }).filter(c => c && c.sc >= 2.1).sort((a, b) => b.sc - a.sc);
    for (const c of ranked.slice(0, 6)) {
      const err = add(c.type, c.it, sl.s, c.why, sl.text, true);
      if (!err) break;
      if (/past 25%/.test(err)) continue;
      if (/too close|closing|opening|no room/.test(err)) break;
    }
  }
}

// ------------------------------------------------------------------ 6. zooms
function planZooms(words, cutPoints, broll, total, cfg) {
  const Z = cfg.zoom, cuts = cutPoints.length ? cutPoints : [1e9];
  const atCut = t => Math.min(...cuts.map(c => Math.abs(c - t))) < 0.2;
  const cands = [];
  words.forEach((w, i) => {
    const ac = atCut(w.s);
    if (w.sent) cands.push({ t: w.s, score: 1 + 0.5 * ac + 0.1 * Math.max(w.emph || 0, 0), i });
    else if (i > 0 && /[,;:—]$/.test(words[i - 1].w)) cands.push({ t: w.s, score: 0.7 + 0.3 * ac, i });
    else if (ac) cands.push({ t: w.s, score: 0.65, i });
    else if (!STOP.has(norm(w.w)) && norm(w.w).length >= 4 && (w.emph || 0) > 2.5) cands.push({ t: Math.max(0, w.s - 0.05), score: 0.6 + 0.1 * w.emph, i });
  });
  const inBroll = (a, b) => broll.some(p => a < p.e + 0.15 && b > p.s - 0.15);
  const nextBroll = t => Math.min(total, ...broll.filter(p => p.s > t).map(p => p.s));
  const gapsBetween = (a, b) => {
    const g = []; for (let k = 0; k + 1 < words.length; k++) if (words[k + 1].s - words[k].e >= 0.12 && words[k].e >= a && words[k].e <= b) g.push((words[k].e + words[k + 1].s) / 2);
    return g;
  };
  const ev = []; let kind = 'push', tOk = 0, guard = 0;
  while (guard++ < 500) {
    const win = cands.filter(c => c.t >= tOk && c.t <= tOk + Z.window && !inBroll(c.t, c.t + 1.2));
    if (!win.length) { const later = cands.filter(c => c.t > tOk + Z.window); if (!later.length) break; tOk = later[0].t; continue; }
    const c = win.reduce((a, b) => b.score > a.score ? b : a);
    const t0 = Math.round(c.t * FPS) / FPS;
    let lo, hi;
    if (kind === 'push') { lo = t0 + Z.ramp + 0.3; hi = t0 + Z.ramp + 2.5; } else { lo = t0 + 1.8; hi = t0 + Z.maxHold; }
    hi = Math.min(hi, nextBroll(t0) - 0.05, total - 0.2);
    const ends = cutPoints.filter(x => x >= lo && x <= hi); const pool = ends.length ? ends : gapsBetween(lo, hi);
    let end = pool.length ? pool[0] : Math.min(hi, lo + 1.0); end = Math.round(end * FPS) / FPS;
    if (end - t0 < (kind === 'push' ? Z.ramp : 1.2)) {
      if (kind === 'push') { kind = 'cut'; continue; }
      tOk = c.t + 0.5; continue;
    }
    const w = words[c.i];
    ev.push({ kind, t0, t1: end, scale: kind === 'push' ? Z.push : Z.cut, word: w.w,
      reason: w.sent ? 'sentence start' : (c.i && /[,;:—]$/.test(words[c.i - 1].w) ? 'clause start' : 'emphasis / jump cut') });
    tOk = Math.max(t0 + Z.minGap, end + 1.0);
    kind = kind === 'push' ? 'cut' : 'push';
  }
  return ev;
}
function zoomCurve(events, nOut, cfg) {
  const z = new Float32Array(nOut).fill(1);
  for (const e of events) {
    const f0 = Math.round(e.t0 * FPS), f1 = Math.min(nOut, Math.round(e.t1 * FPS));
    for (let f = f0; f < f1; f++) {
      if (e.kind === 'cut') z[f] = e.scale;
      else { const u = Math.min(1, (f - f0) / (cfg.zoom.ramp * FPS)); z[f] = 1 + (e.scale - 1) * u * u * (3 - 2 * u); }
    }
  }
  return z;
}

// ------------------------------------------------------------------ 8. captions
function display(w) {
  return w.replace(/’/g, "'").trim().replace(/^["'“”(\[]+/, '').replace(/["'“”)\]]+$/, '').replace(/[.,;:!…]+$/, '');
}
function pickHighlight(ws) {
  const kk = ws.findIndex(w => w.key);
  if (kk >= 0) return kk;
  let best = null, bs = 0;
  ws.forEach((w, k) => {
    const t = norm(w.w); if (!t || STOP.has(t)) return;
    let s = 0.15 * t.length + 0.25 * Math.max(0, w.emph || 0) + (/^\d+$/.test(t) ? 1 : 0);
    if (t.length >= 4 || /^\d+$/.test(t)) s += 0.5;
    if (s > bs) { bs = s; best = k; }
  });
  return bs >= 1 ? best : null;
}
const WEAK_END = new Set('a an the to of and but or nor so in on at for with from by into my your his her their our its i you we they he she it is are was were be been am that this than as if then when while because like just very really not'.split(' '));
/** Split words into caption lines with dynamic programming: 2–4 words, never across a pause, never ending on "the"/"to"/"and". */
function chunkWords(words, measure, maxw, cfg) {
  const n = words.length; if (!n) return [];
  const hardEnd = w => /[.?!]["')]*$/.test(w), softEnd = w => /[,;:—–]$/.test(w);
  const width = (i, j) => measure(words.slice(i, j + 1).map(x => display(x.w)).join(' '));
  const lineCost = (i, j) => {
    const k = j - i + 1; if (k > cfg.cap.maxWords) return Infinity;
    if (k > 1 && width(i, j) > maxw) return Infinity;
    let c = 0.7 * (k - 3) * (k - 3);
    for (let m = i; m < j; m++) {
      const gap = words[m + 1].s - words[m].e;
      if (gap > 0.6) c += 25; else if (gap > 0.32) c += 5;
      if (hardEnd(words[m].w)) c += 14; else if (softEnd(words[m].w)) c += 4.5;
    }
    const last = words[j].w;
    if (hardEnd(last)) c -= 1.6; else if (softEnd(last)) c -= 0.8;
    else if (WEAK_END.has(norm(last)) && j < n - 1) c += 3.2;
    if (j + 1 < n && words[j + 1].s - words[j].e > 0.32) c -= 1.2;          // line ends on a pause: good
    if (k === 1) { const alone = (i === 0 || hardEnd(words[i - 1].w)) && (hardEnd(last) || j === n - 1); c += alone ? 0.5 : 3.5; }
    const dur = words[j].e - words[i].s; if (dur < 0.3) c += 2.5; if (dur > 2.8) c += 2;
    return c;
  };
  const best = new Array(n + 1).fill(Infinity), back = new Array(n + 1).fill(-1); best[0] = 0;
  for (let j = 1; j <= n; j++) for (let i = Math.max(0, j - cfg.cap.maxWords); i < j; i++) {
    const c = best[i] + lineCost(i, j - 1); if (c < best[j]) { best[j] = c; back[j] = i; }
  }
  const cuts = []; for (let j = n; j > 0; j = back[j]) cuts.unshift([back[j], j - 1]);
  const lines = cuts.map(([i, j]) => words.slice(i, j + 1));
  // each line appears just before its first word is heard; a slow previous line never pushes it late
  const starts = []; lines.forEach((c, k) => starts.push(Math.max(0, c[0].s - 0.08, k ? starts[k - 1] + 0.2 : 0)));
  const out = [];
  lines.forEach((c, k) => {
    const last = c[c.length - 1], nx = lines[k + 1], s = starts[k];
    let e = nx && nx[0].s - last.e < 0.7 ? starts[k + 1] : last.e + 0.35;  // hold through short pauses, clear on long ones
    e = Math.max(e, Math.min(s + 0.4, nx ? starts[k + 1] : Infinity), last.e);
    out.push({ s: +s.toFixed(3), e: +e.toFixed(3), words: c.map(x => display(x.w)), hl: pickHighlight(c), w0: c[0].s });
  });
  for (let k = 0; k + 1 < out.length; k++) if (out[k].e > out[k + 1].s) out[k].e = out[k + 1].s;
  return out;
}
function srtTime(t) { const ms = Math.round(Math.max(0, t) * 1000); const p = (n, l = 2) => String(n).padStart(l, '0'); return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`; }
function toSrt(chunks) { return chunks.map((c, k) => `${k + 1}\n${srtTime(c.s)} --> ${srtTime(c.e)}\n${c.words.join(' ')}\n`).join('\n'); }
const CAP_FONT = '"Inter", "Helvetica Neue", Arial, sans-serif';
function capFont(cfg) { return `800 ${Math.round(cfg.cap.size / 1.21)}px ${CAP_FONT}`; }
function drawCaption(ctx, chunk, cfg, t, next, lift = null) {
  if (!chunk) return;
  let a = 1, rise = 0;
  if (t != null) {
    const u = clamp((t - chunk.s) / 0.1, 0, 1); a = u * u * (3 - 2 * u); rise = (1 - a) * 5;     // fade up over ~3 frames
    const followed = next && next.s - chunk.e < 0.04;
    if (!followed) { const v = clamp((chunk.e - t) / 0.12, 0, 1); a = Math.min(a, v * v * (3 - 2 * v)); } // fade out only before a pause
  }
  if (a <= 0.01) return;
  ctx.save(); ctx.globalAlpha = a; ctx.translate(0, rise);
  ctx.font = capFont(cfg); ctx.textBaseline = 'alphabetic'; ctx.lineJoin = 'round'; ctx.miterLimit = 2;
  const sp = ctx.measureText(' ').width + 0.5;
  const ws = chunk.words.map(w => ({ w, wd: ctx.measureText(w).width }));
  const total = ws.reduce((a, b) => a + b.wd, 0) + sp * (ws.length - 1);
  const lf = lift == null ? (chunk.up ? 1 : 0) : lift;
  const y = Math.round(OUT_H * (cfg.cap.baseline + (cfg.cap.baselineUp - cfg.cap.baseline) * lf)); let x = OUT_W / 2 - total / 2;
  const pos = ws.map(w => { const p = x; x += w.wd + sp; return p; });
  // soft drop shadow, then dark outline, then fill
  ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = 6.4;
  ws.forEach((w, k) => { ctx.strokeText(w.w, pos[k] + 2.2, y + 2.2); ctx.fillText(w.w, pos[k] + 2.2, y + 2.2); });
  ctx.strokeStyle = 'rgba(20,20,20,0.70)'; ctx.lineWidth = 6.4;
  ws.forEach((w, k) => ctx.strokeText(w.w, pos[k], y));
  ws.forEach((w, k) => { ctx.fillStyle = k === chunk.hl ? cfg.cap.accent : '#FFFFFF'; ctx.fillText(w.w, pos[k], y); });
  ctx.restore();
}

// ------------------------------------------------------------------ MP4 demux / decode
let MP4Box = null;
async function mp4box() { if (!MP4Box) MP4Box = await import(LIBS.mp4box); return MP4Box; }

class FileWindow {
  constructor(file) { this.file = file; this.start = 0; this.buf = null; }
  async read(off, size) {
    if (!this.buf || off < this.start || off + size > this.start + this.buf.byteLength) {
      const len = Math.max(size, 8 << 20);
      this.start = off; this.buf = new Uint8Array(await this.file.slice(off, Math.min(this.file.size, off + len)).arrayBuffer());
    }
    return this.buf.subarray(off - this.start, off - this.start + size);
  }
}

async function openMovie(file, opts = {}) {
  const M = await mp4box();
  const boxes = []; let off = 0;
  while (off + 8 <= file.size) {
    const dv = new DataView(await file.slice(off, off + 16).arrayBuffer());
    let size = dv.getUint32(0); const type = String.fromCharCode(dv.getUint8(4), dv.getUint8(5), dv.getUint8(6), dv.getUint8(7));
    if (size === 1) size = Number(dv.getBigUint64(8)); else if (size === 0) size = file.size - off;
    if (size < 8) break;
    boxes.push({ type, off, size }); off += size;
  }
  const mp4 = M.createFile(); let info = null, err = null;
  mp4.onReady = i => { info = i; }; mp4.onError = e => { err = e; };
  for (const b of boxes) {
    const len = b.type === 'mdat' ? Math.min(b.size, 16) : b.size;
    if (b.type !== 'mdat' && b.size > 300e6) continue;
    const ab = await file.slice(b.off, b.off + len).arrayBuffer();
    const buf = M.MP4BoxBuffer ? M.MP4BoxBuffer.fromArrayBuffer(ab, b.off) : Object.assign(ab, { fileStart: b.off });
    mp4.appendBuffer(buf);
    if (info) break;
  }
  if (!info) mp4.flush();
  if (!info) throw new Error(err ? String(err) : "This file isn't a video Reel Studio can read (MP4 or MOV).");
  const v = info.videoTracks[0];
  const a = info.audioTracks[0] || null;
  if (!v && opts.audioOnly && a) return { file, mp4, M, info, a, trakA: mp4.getTrackById(a.id), duration: a.duration / a.timescale };
  if (!v) throw new Error('No video track found in this file.');
  const trakV = mp4.getTrackById(v.id), trakA = a ? mp4.getTrackById(a.id) : null;
  const m = v.matrix || [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824];
  let rot = Math.round(Math.atan2(m[1], m[0]) * 180 / Math.PI); rot = ((rot % 360) + 360) % 360;
  const vs = trakV.samples, tsv = vs[0].timescale, ctsMin = Math.min(...vs.slice(0, 64).map(s => s.cts));
  // Edit lists say which part of each track is shown and when. iPhone trims, encoder delay and audio priming all live here.
  const eV = editList(trakV, info.timescale), eA = trakA ? editList(trakA, info.timescale) : null;
  const cts0 = eV ? Math.round((eV.start - eV.delay) * tsv) : ctsMin;
  let dur = v.duration / v.timescale || (vs.length ? (vs[vs.length - 1].cts - ctsMin + vs[vs.length - 1].duration) / tsv : 0);
  if (eV && eV.dur > 0.1) dur = Math.min(dur + 0.5, eV.delay + eV.dur);
  let audioOffset = 0, aInfo = null;
  if (trakA && trakA.samples.length) {
    const tsa = trakA.samples[0].timescale, ac0 = Math.min(...trakA.samples.slice(0, 16).map(s => s.cts));
    const pa = eA ? eA.delay + ac0 / tsa - eA.start : ac0 / tsa;    // when the first decoded audio sample is heard
    const pv = eV ? 0 : ctsMin / tsv;                                // when the first video frame is shown
    audioOffset = pa - pv;
    let sum = 0; for (const s of trakA.samples) sum += s.duration;
    aInfo = { mediaSec: sum / tsa, editV: eV, editA: eA };
  }
  DIAG.takeTiming = { videoSec: +dur.toFixed(3), audioOffset: +audioOffset.toFixed(4), editV: eV, editA: eA, vFrames: vs.length };
  return { file, mp4, M, info, v, a, trakV, trakA, rot, cts0, duration: dur, aInfo,
    codedW: v.video.width, codedH: v.video.height,
    W: (rot === 90 || rot === 270) ? v.video.height : v.video.width,
    H: (rot === 90 || rot === 270) ? v.video.width : v.video.height,
    fps: vs.length / Math.max(0.1, dur), audioOffset };
}

/** First edit of a track: initial empty delay (s), where the shown media starts (s, media time) and how long it runs (s). */
function editList(trak, movieTs) {
  try {
    const es = trak.edts && trak.edts.elst && trak.edts.elst.entries; if (!es || !es.length) return null;
    const ts = trak.mdia.mdhd.timescale; let delay = 0;
    for (const e of es) {
      const mt = Number(e.media_time), sd = Number(e.segment_duration) / (movieTs || ts);
      if (mt === -1 || mt === 0xFFFFFFFF) { delay += sd; continue; }
      return { delay: +delay.toFixed(4), start: +(mt / ts).toFixed(4), dur: +sd.toFixed(4) };
    }
  } catch (e) { /* no usable edit list */ }
  return null;
}
function describe(movie, trak) {
  const M = movie.M;
  for (const entry of trak.mdia.minf.stbl.stsd.entries) {
    const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
    if (box) {
      const s = new M.DataStream(undefined, 0, M.Endianness ? M.Endianness.BIG_ENDIAN : 1);
      box.write(s); return new Uint8Array(s.buffer, 8);
    }
  }
  return undefined;
}

/** Sequential frame access by presentation time; frames are borrowed until the next call. */
class VideoSource {
  constructor(movie) {
    this.mv = movie; this.samples = movie.trakV.samples; this.ts = this.samples[0].timescale;
    this.fw = new FileWindow(movie.file);
    this.config = { codec: movie.v.codec, codedWidth: movie.codedW, codedHeight: movie.codedH, description: describe(movie, movie.trakV) };
  }
  async open() {
    if (!('VideoDecoder' in root)) throw new Error('This browser cannot decode video frames (WebCodecs missing).');
    const sup = await VideoDecoder.isConfigSupported(this.config);
    if (!sup.supported) throw new Error(`This device can't decode ${this.config.codec} video.`);
    this._make(); this.i = 0;
  }
  _make() {
    this.out = []; this.cur = null; this.err = null; this.flushed = false;
    this.dec = new VideoDecoder({ output: f => this.out.push(f), error: e => { this.err = e; } });
    this.dec.configure(this.config);
  }
  seek(t) {
    const target = t * this.ts + this.mv.cts0; let k = 0;
    for (let i = 0; i < this.samples.length; i++) { if (this.samples[i].is_sync && this.samples[i].cts <= target) k = i; if (this.samples[i].cts > target + this.ts * 2) break; }
    this.close(); this._make(); this.i = k;
  }
  async _feed(n) {
    for (let c = 0; c < n && this.i < this.samples.length; c++) {
      const s = this.samples[this.i++];
      const data = await this.fw.read(s.offset, s.size);
      this.dec.decode(new EncodedVideoChunk({ type: s.is_sync ? 'key' : 'delta', timestamp: Math.round((s.cts - this.mv.cts0) / this.ts * 1e6), duration: Math.round(s.duration / this.ts * 1e6), data }));
    }
  }
  async frameAt(t) {
    const lim = (t + 0.25 / FPS) * 1e6;
    for (let guard = 0; guard < 100000; guard++) {
      if (this.err) throw this.err;
      this.out.sort((a, b) => a.timestamp - b.timestamp);
      while (this.out.length && this.out[0].timestamp <= lim) { if (this.cur) this.cur.close(); this.cur = this.out.shift(); }
      if (this.out.length) break;
      if (this.i >= this.samples.length) {
        if (!this.flushed) { this.flushed = true; await this.dec.flush(); continue; }
        break;
      }
      if (this.dec.decodeQueueSize < 8) await this._feed(4); else await sleep(1);
      if (!this.out.length) await sleep(0);
    }
    return this.cur || this.out[0] || null;
  }
  close() { try { this.out.forEach(f => f.close()); if (this.cur) this.cur.close(); this.out = []; this.cur = null; if (this.dec && this.dec.state !== 'closed') this.dec.close(); } catch (e) { /* already closed */ } }
}

const AAC_RATES = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];
/** A plain AAC-LC AudioSpecificConfig for this rate and channel count. */
function ascFor(sr, ch) { const fi = Math.max(0, AAC_RATES.indexOf(sr)); const c = clamp(ch || 2, 1, 7); return new Uint8Array([(2 << 3) | (fi >> 1), ((fi & 1) << 7) | (c << 3)]); }
/** Core sample rate written in an AudioSpecificConfig. */
function ascRate(u) { if (!u || u.length < 2) return 0; const fi = ((u[0] & 7) << 1) | (u[1] >> 7); return fi === 15 ? 0 : AAC_RATES[fi] || 0; }
/** The AAC AudioSpecificConfig, unwrapped if a browser handed over a full MPEG-4 ES descriptor instead. */
function sanitizeAsc(d) {
  if (!d) return AAC_ASC;
  const u = d instanceof Uint8Array ? d : ArrayBuffer.isView(d) ? new Uint8Array(d.buffer, d.byteOffset, d.byteLength) : new Uint8Array(d);
  const ot = u[0] >> 3;
  if (u.length >= 2 && u.length <= 16 && u[0] !== 0x03 && u[0] !== 0x04 && u[0] !== 0x05 && ot >= 1) return u;
  let i = 0;
  const readLen = () => { let n = 0, b; do { b = u[i++]; n = (n << 7) | (b & 0x7f); } while ((b & 0x80) && i < u.length); return n; };
  for (let guard = 0; i < u.length && guard < 32; guard++) {
    const tag = u[i++], len = readLen();
    if (tag === 0x05) return u.slice(i, i + len);
    if (tag === 0x03) { i += 2; const fl = u[i++]; if (fl & 0x80) i += 2; if (fl & 0x40) i += 1 + u[i]; if (fl & 0x20) i += 2; continue; }
    if (tag === 0x04) { i += 13; continue; }
    i += len;
  }
  return AAC_ASC;
}
async function decodeAudioTrack(movie, signal, opts = {}) {
  if (!movie.trakA) return null;
  const trak = movie.trakA, a = movie.a, entry = trak.mdia.minf.stbl.stsd.entries[0];
  const sr = a.audio.sample_rate, ch = a.audio.channel_count;
  let pcm = null, sr2 = sr;
  if ('AudioDecoder' in root) {
    try {
      let desc;
      const esds = entry.esds || (entry.wave && entry.wave.esds) || (entry.boxes || []).map(b => b.esds || b).find(b => b && b.type === 'esds');
      try { desc = esds.esd.descs[0].descs[0].data; } catch (e) { desc = undefined; }
      let codec = a.codec;
      if (a.codec.startsWith('mp4a')) {
        desc = desc ? sanitizeAsc(desc) : null;
        if (!desc || desc === AAC_ASC || ascRate(desc) !== sr) { DIAG.ascRebuilt = desc ? hex(desc) : 'missing'; desc = ascFor(sr, ch); } // MOV files often hide it; build it from the track header
        codec = 'mp4a.40.' + Math.max(1, desc[0] >> 3);
      }
      const cfg = { codec, sampleRate: sr, numberOfChannels: ch, description: desc };
      // Place every decoded block at its own timestamp, at the rate the decoder actually reports.
      // Packets the decoder drops or merges then can't slide the rest of the voice out of sync.
      const ts = trak.samples[0].timescale, t0 = Math.round(Math.min(...trak.samples.slice(0, 16).map(x => x.cts)) / ts * 1e6);
      const parts = []; let err = null, outSr = 0;
      const dec = new AudioDecoder({ output: d => {
        const n = d.numberOfFrames, nc = d.numberOfChannels, L = new Float32Array(n), R = new Float32Array(n), tmp = new Float32Array(n);
        if (!outSr) outSr = d.sampleRate || sr;
        if (opts.stereo) { d.copyTo(L, { planeIndex: 0, format: 'f32-planar' }); if (nc > 1) d.copyTo(R, { planeIndex: 1, format: 'f32-planar' }); else R.set(L); }
        else { for (let c = 0; c < nc; c++) { d.copyTo(tmp, { planeIndex: c, format: 'f32-planar' }); for (let i = 0; i < n; i++) L[i] += tmp[i] / nc; } }
        parts.push({ ts: d.timestamp - t0, L, R: opts.stereo ? R : null }); d.close();
      }, error: e => { err = e; } });
      dec.configure(cfg);
      const fw = new FileWindow(movie.file);
      for (let i = 0; i < trak.samples.length; i++) {
        const s = trak.samples[i];
        dec.decode(new EncodedAudioChunk({ type: 'key', timestamp: Math.round(s.cts / ts * 1e6), duration: Math.round(s.duration / ts * 1e6), data: await fw.read(s.offset, s.size) }));
        if (dec.decodeQueueSize > 64) await sleep(1);
        if (i % 500 === 0) { check(signal); await tick(); }
      }
      await dec.flush(); dec.close();
      if (err) throw err;
      if (!parts.length) throw new Error('the decoder returned no audio');
      outSr = outSr || sr;
      // timestamps are usable if they increase and roughly agree with the running sample count
      let run = parts[0].ts, tsOk = parts[0].ts >= 0 && parts[0].ts < 0.25e6;
      for (let k = 1; k < parts.length && tsOk; k++) { run += parts[k - 1].L.length / outSr * 1e6; if (parts[k].ts < parts[k - 1].ts || Math.abs(parts[k].ts - run) > 0.25e6) { tsOk = false; break; } }
      const lastEnd = tsOk ? parts[parts.length - 1].ts / 1e6 * outSr + parts[parts.length - 1].L.length : parts.reduce((a, p) => a + p.L.length, 0);
      const total = Math.ceil(lastEnd) + 8, outL = new Float32Array(total), outR = opts.stereo ? new Float32Array(total) : null;
      let o = 0;
      for (const p of parts) { if (tsOk) o = Math.max(0, Math.round(p.ts / 1e6 * outSr)); if (o + p.L.length > total) break; outL.set(p.L, o); if (outR) outR.set(p.R, o); o += p.L.length; }
      // the track's own sample table is the truth about how long the audio is: if the decoded length disagrees, the rate is mislabelled
      const media = movie.aInfo ? movie.aInfo.mediaSec : trak.samples.reduce((a, s) => a + s.duration, 0) / ts;
      let rate = outSr; const got = lastEnd / outSr;
      if (media > 1 && Math.abs(got / media - 1) > 0.015 && Math.abs(got - media) > 0.15) { // 44.1 vs 48 kHz is a 9% error
        rate = lastEnd / media; const std = AAC_RATES.find(r => Math.abs(rate / r - 1) < 0.01); if (std) rate = std;
      }
      DIAG.takeDecode = { declaredSr: sr, decoderSr: outSr, usedSr: +rate.toFixed(1), decodedSec: +got.toFixed(3), mediaSec: +media.toFixed(3), tsPlacement: tsOk, blocks: parts.length };
      const trimTo = x => x.subarray(0, Math.ceil(lastEnd));
      if (opts.stereo) return [resample(trimTo(outL), rate, SR), resample(trimTo(outR), rate, SR)];
      pcm = trimTo(outL); sr2 = rate;
    } catch (e) { if (e instanceof Cancelled) throw e; DIAG.takeDecoderError = String(e && e.message || e); pcm = null; if (opts.stereo) throw e; }
  }
  DIAG.takeAudio = { codec: a.codec, sr, usedSr: +sr2.toFixed(1), ch, path: pcm ? 'AudioDecoder' : 'decodeAudioData', samples: pcm ? pcm.length : 0, rmsDb: pcm ? rmsDb(pcm) : null };
  if (!pcm) { // fallback: let the browser decode the whole file
    const ac = new OfflineAudioContext(1, 1, sr);
    const buf = await ac.decodeAudioData(await movie.file.arrayBuffer());
    pcm = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) pcm[i] += d[i] / buf.numberOfChannels; }
    return resample(pcm, buf.sampleRate, SR);
  }
  return resample(pcm, sr2, SR);
}

// ------------------------------------------------------------------ models
let _yunet = null;
const MODELS = {
  small: { id: 'onnx-community/whisper-small.en_timestamped', label: 'Accurate', mb: 250 },
  base: { id: 'onnx-community/whisper-base.en_timestamped', label: 'Fast', mb: 80 },
};
const _whisperBy = {};
/** Whisper (open source) via transformers.js. Downloaded once from Hugging Face, then kept in the browser's cache. */
async function loadWhisper(onProgress, which = 'small') {
  if (_whisperBy[which]) return _whisperBy[which];
  const M = MODELS[which] || MODELS.base, T = await import(LIBS.tjs);
  T.env.allowLocalModels = false; T.env.allowRemoteModels = true; T.env.useBrowserCache = true;
  T.env.backends.onnx.wasm.numThreads = 1; T.env.backends.onnx.wasm.proxy = false;
  const files = {};
  _whisperBy[which] = await T.pipeline('automatic-speech-recognition', M.id, {
    dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' }, device: 'wasm',
    progress_callback: p => {
      if (p.status === 'progress' && p.total) { files[p.file] = [p.loaded, p.total];
        const v = Object.values(files), got = v.reduce((a, x) => a + x[0], 0), tot = v.reduce((a, x) => a + x[1], 0);
        onProgress && onProgress(Math.min(1, got / Math.max(tot, M.mb * 1e6 * 0.9))); }
    } });
  onProgress && onProgress(1);
  return _whisperBy[which];
}
async function transcribe(audio48, onProgress, signal, which = 'small') {
  let asr, used = which;
  try { asr = await loadWhisper(p => onProgress && onProgress(0.55 * p, p < 1 ? 'Loading the speech model' : null), which); }
  catch (e) {
    if (which === 'base' || e instanceof Cancelled) throw e;
    DIAG.modelFallback = String(e.message || e); used = 'base';
    asr = await loadWhisper(p => onProgress && onProgress(0.55 * p, 'Loading the fast speech model'), 'base');
  }
  DIAG.model = used;
  check(signal);
  onProgress && onProgress(0.58, 'Transcribing what you said');
  const a16 = resample(audio48, SR, 16000), R16 = 16000;
  // Split the take at its quietest pauses into pieces under 28 s and transcribe each on its own.
  // The library's own 30 s windowing can drop or misplace words where windows overlap.
  const { db } = frameDb(a16, 0.03, 0.01, R16), segs = []; let pos = 0;
  const total = a16.length / R16;
  while (total - pos > 28) {
    let best = pos + 24, bv = Infinity;
    for (let t = pos + 16; t <= pos + 27.5; t += 0.01) { const i = Math.round(t * 100); let m = -Infinity; for (let k = -15; k <= 15; k++) m = Math.max(m, db[clamp(i + k, 0, db.length - 1)]); if (m < bv) { bv = m; best = t; } }
    segs.push([pos, best]); pos = best;
  }
  segs.push([pos, total]);
  const out = [];
  const asrRange = async (t0, t1) => {
    const res = await asr(a16.subarray(Math.round(t0 * R16), Math.round(t1 * R16)), { return_timestamps: 'word' });
    const ws = [];
    for (const c of res.chunks || []) {
      const raw = c.text || '', txt = raw.trim();
      if (!txt || /^[\[(*].*[\])*]$/.test(txt) || /clap/i.test(txt) && !/[a-z]{6,}/i.test(txt.replace(/clap\w*/i, ''))) continue;
      let [s, e] = c.timestamp || [0, 0]; if (s == null) s = ws.length ? ws[ws.length - 1].e - t0 : 0; if (e == null) e = s + 0.3;
      s += t0; e += t0;
      if (ws.length && /^[-'’]/.test(txt) && !raw.startsWith(' ')) { ws[ws.length - 1].w += txt; ws[ws.length - 1].e = +e.toFixed(3); continue; }
      ws.push({ w: txt, s: +s.toFixed(3), e: +Math.min(e, t1).toFixed(3) });
    }
    return ws;
  };
  for (let k = 0; k < segs.length; k++) {
    check(signal);
    onProgress && onProgress(0.58 + 0.36 * k / segs.length, segs.length > 1 ? `Transcribing what you said (part ${k + 1} of ${segs.length})` : 'Transcribing what you said');
    out.push(...await asrRange(segs[k][0], segs[k][1]));
  }
  // Safety net: any stretch of clear speech that came back with no words gets transcribed again on its own.
  const speech = vad(audio48), redo = [];
  for (const [s, e] of speech) if (e - s > 1.2 && !out.some(w => w.e > s - 0.2 && w.s < e + 0.2)) redo.push([Math.max(0, s - 0.3), Math.min(total, e + 0.3)]);
  for (const [s, e] of redo.slice(0, 6)) {
    check(signal); const ws = (await asrRange(s, e)).filter(w => !out.some(o => o.e > w.s + 0.05 && o.s < w.e - 0.05));
    if (ws.length >= 2 && ws.length <= (e - s) * 4.5) out.push(...ws);                 // ignore one-word or run-on replies (likely invented)
  }
  out.sort((x, y) => x.s - y.s);
  DIAG.asr = { parts: segs.length, cuts: segs.map(x => +x[1].toFixed(2)), words: out.length, recovered: redo.length, redo };
  for (let i = 1; i < out.length; i++) { out[i].s = Math.max(out[i].s, out[i - 1].s); out[i].e = Math.max(out[i].e, out[i].s + 0.05); }
  out.model = used;
  onProgress && onProgress(1);
  return out;
}
async function loadYunet() {
  if (_yunet) return _yunet;
  await loadScript(LIBS.ort);
  const ort = root.ort;
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  const model = new Uint8Array(await (await fetch(BASE() + 'models/yunet.onnx')).arrayBuffer());
  _yunet = { ort, sess: await ort.InferenceSession.create(model, { executionProviders: ['wasm'] }) };
  return _yunet;
}

// ------------------------------------------------------------------ 5. face tracking
function drawRotated(ctx, frame, movie, k, dx = 0, dy = 0) {
  // draws the decoded frame in display orientation, scaled by k, offset by (dx, dy) in display pixels
  const w0 = movie.codedW, h0 = movie.codedH;
  ctx.save(); ctx.translate(dx, dy); ctx.scale(k, k);
  if (movie.rot === 90) { ctx.translate(h0, 0); ctx.rotate(Math.PI / 2); }
  else if (movie.rot === 180) { ctx.translate(w0, h0); ctx.rotate(Math.PI); }
  else if (movie.rot === 270) { ctx.translate(0, w0); ctx.rotate(-Math.PI / 2); }
  ctx.drawImage(frame, 0, 0, w0, h0);
  ctx.restore();
}
function gauss1d(x, sigma) {
  const r = Math.ceil(sigma * 3), k = []; let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) ** 2); k.push(v); s += v; }
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) { let a = 0; for (let j = -r; j <= r; j++) a += x[clamp(i + j, 0, x.length - 1)] * k[j + r]; out[i] = a / s; }
  return out;
}
async function faceTrack(movie, tmap, broll, cfg, onProgress, signal, cache = new Map()) {
  const { W, H } = movie, n = tmap.nOut;
  const fx = new Float32Array(n).fill(NaN), fy = new Float32Array(n).fill(NaN);
  const skip = new Uint8Array(n);
  for (const p of broll) for (let f = Math.floor(p.s * FPS); f < Math.min(n, Math.ceil(p.e * FPS)); f++) skip[f] = 1;
  let detector = 'none';
  try {
    const { ort, sess } = await loadYunet(); detector = 'YuNet';
    const cv = new OffscreenCanvas(640, 640), ctx = cv.getContext('2d', { willReadFrequently: true });
    const k = 640 / Math.max(W, H), input = new Float32Array(3 * 640 * 640);
    const src = new VideoSource(movie); await src.open();
    let prev = null, done = 0; const total = Math.ceil(n / 3);
    for (let oi = 0; oi < n; oi += 3) {
      if (skip[oi]) continue;
      const rf = tmap.outToRawFrame(oi);
      let hitC = null; for (let d = 0; d <= 2 && !hitC; d++) hitC = cache.get(rf - d) || cache.get(rf + d);
      if (hitC) { if (hitC.length) { fx[oi] = hitC[0]; fy[oi] = hitC[1]; prev = { cx: hitC[0], cy: hitC[1] }; } continue; }   // already looked at this moment
      const frame = await src.frameAt(rf / FPS); if (!frame) { cache.set(rf, []); continue; }
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 640, 640); drawRotated(ctx, frame, movie, k);
      const px = ctx.getImageData(0, 0, 640, 640).data, P = 640 * 640;
      for (let i = 0; i < P; i++) { input[i] = px[i * 4 + 2]; input[P + i] = px[i * 4 + 1]; input[2 * P + i] = px[i * 4]; }
      const out = await sess.run({ input: new ort.Tensor('float32', input, [1, 3, 640, 640]) });
      const dets = [];
      for (const st of [8, 16, 32]) {
        const g = 640 / st, cls = out['cls_' + st].data, obj = out['obj_' + st].data, bb = out['bbox_' + st].data;
        for (let i = 0; i < cls.length; i++) {
          const sc = Math.sqrt(clamp(cls[i], 0, 1) * clamp(obj[i], 0, 1)); if (sc < 0.6) continue;
          const r = Math.floor(i / g), c = i % g;
          dets.push({ sc, cx: (c + bb[i * 4]) * st / k, cy: (r + bb[i * 4 + 1]) * st / k, w: Math.exp(bb[i * 4 + 2]) * st / k });
        }
      }
      if (dets.length) {
        const best = prev ? dets.reduce((a, d) => (d.w * d.sc * Math.exp(-Math.hypot(d.cx - prev.cx, d.cy - prev.cy) / (0.25 * W)) > a.w * a.sc * Math.exp(-Math.hypot(a.cx - prev.cx, a.cy - prev.cy) / (0.25 * W)) ? d : a))
                          : dets.reduce((a, d) => d.w * d.sc > a.w * a.sc ? d : a);
        prev = best; fx[oi] = best.cx; fy[oi] = best.cy; cache.set(rf, [best.cx, best.cy]);
      } else cache.set(rf, []);
      if (++done % 10 === 0) { onProgress && onProgress(done / total); check(signal); await tick(); }
    }
    src.close();
  } catch (e) { if (e instanceof Cancelled) throw e; console.warn('face tracking failed', e); detector = 'none (' + e.message + ')'; }
  const valid = []; for (let i = 0; i < n; i++) if (!isNaN(fx[i])) valid.push(i);
  const sampled = Math.max(1, Math.ceil((n - skip.reduce((a, b) => a + b, 0)) / 3));
  let cx, cy;
  if (!valid.length) { cx = new Float32Array(n).fill(W / 2); cy = new Float32Array(n).fill(H * 0.4); }
  else {
    const med = (arr) => { const v = valid.map(i => arr[i]); const o = v.map((_, j) => median(v.slice(Math.max(0, j - 2), j + 3))); valid.forEach((i, j) => { arr[i] = o[j]; }); };
    if (valid.length >= 5) { med(fx); med(fy); }
    const interp = arr => { const out = new Float32Array(n); let j = 0; for (let i = 0; i < n; i++) { while (j + 1 < valid.length && valid[j + 1] <= i) j++; const a = valid[j], b = valid[Math.min(j + 1, valid.length - 1)]; out[i] = (i <= a || a === b) ? arr[a] : (i >= b ? arr[b] : arr[a] + (arr[b] - arr[a]) * (i - a) / (b - a)); } return out; };
    cx = interp(fx); cy = interp(fy);
  }
  const w0 = Math.min(W, H * 9 / 16), h0 = w0 * 16 / 9, starts = new Set(tmap.starts);
  const follow = (target, dz) => {
    const c = new Float32Array(n); let cur = target[0];
    for (let i = 0; i < n; i++) {
      const t = target[i];
      if (starts.has(i) && Math.abs(t - cur) > dz * 0.5) cur = t;
      else if (t - cur > dz) cur = t - dz; else if (cur - t > dz) cur = t + dz;
      c[i] = cur;
    }
    const b = [...starts, n].sort((x, y) => x - y);
    for (let s = 0; s + 1 < b.length; s++) if (b[s + 1] - b[s] > 2) c.set(gauss1d(c.subarray(b[s], b[s + 1]), cfg.smooth * FPS), b[s]);
    return c;
  };
  return { cx: follow(cx, cfg.deadzone * w0), cy: follow(cy, cfg.deadzone * h0), w0, h0, detector, coverage: Math.min(1, valid.length / sampled), cache };
}
function cropRect(i, tr, z, W, H, cfg) {
  const { w0, h0 } = tr, cx = tr.cx[i], fy = tr.cy[i];
  const x0 = clamp(cx - w0 / 2, 0, W - w0), y0 = clamp(fy + (0.5 - cfg.faceY) * h0 - h0 / 2, 0, H - h0);
  const px = clamp(cx, x0, x0 + w0), py = clamp(fy, y0, y0 + h0), s = z[i];
  return [px + (x0 - px) / s, py + (y0 - py) / s, w0 / s, h0 / s];
}

// ------------------------------------------------------------------ music
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
const MOODS = {
  warm:       { bpm: 76, prog: [[0, 4, 7, 11], [9, 12, 16, 19], [5, 9, 12, 16], [7, 11, 14, 17]], label: 'Warm' },
  steady:     { bpm: 84, prog: [[9, 12, 16, 19], [5, 9, 12, 16], [0, 4, 7, 11], [7, 11, 14, 17]], label: 'Steady', pulse: true },
  reflective: { bpm: 66, prog: [[0, 4, 7, 11, 14], [5, 9, 12, 16], [9, 12, 16, 19], [5, 9, 12, 14]], label: 'Reflective' },
};
async function composeMusic(seconds, mood = 'warm', seed = Date.now()) {
  const m = MOODS[mood] || MOODS.warm, r = rng(seed);
  const key = [0, 2, 3, 5, 7, 9][Math.floor(r() * 6)] + 48; // C3..A3 roots
  const beat = 60 / m.bpm, bar = beat * 4, chordLen = bar * 2, len = Math.ceil((seconds + 1) * SR);
  const ac = new OfflineAudioContext(2, len, SR);
  const master = ac.createGain(); master.gain.value = 0.9;
  // room: synthetic stereo reverb
  const ir = ac.createBuffer(2, Math.round(2.6 * SR), SR);
  for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); for (let i = 0; i < d.length; i++) d[i] = (r() * 2 - 1) * Math.pow(1 - i / d.length, 3.2) * 0.6; }
  const verb = ac.createConvolver(); verb.buffer = ir; const wet = ac.createGain(); wet.gain.value = 0.32;
  verb.connect(wet).connect(master); master.connect(ac.destination);
  const dry = ac.createGain(); dry.gain.value = 0.85; dry.connect(master); dry.connect(verb);
  const hz = n => 440 * Math.pow(2, (n - 69) / 12);
  const nChords = Math.ceil(seconds / chordLen) + 1;
  for (let ci = 0; ci < nChords; ci++) {
    const t0 = ci * chordLen, chord = m.prog[ci % m.prog.length];
    // pad: detuned saws through a soft low-pass
    const lp = ac.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 900; lp.Q.value = 0.6;
    lp.frequency.setValueAtTime(700, t0); lp.frequency.linearRampToValueAtTime(1150, t0 + chordLen * 0.6); lp.frequency.linearRampToValueAtTime(800, t0 + chordLen + 1.2);
    const pg = ac.createGain(); pg.gain.setValueAtTime(0, t0); pg.gain.linearRampToValueAtTime(0.055, t0 + 1.4);
    pg.gain.setValueAtTime(0.055, t0 + chordLen - 0.2); pg.gain.linearRampToValueAtTime(0, t0 + chordLen + 1.4);
    lp.connect(pg).connect(dry);
    chord.slice(0, 4).forEach((iv, vi) => {
      for (const det of [-6, 7]) {
        const o = ac.createOscillator(); o.type = 'sawtooth'; o.frequency.value = hz(key + 12 + iv); o.detune.value = det * 0.6 + (r() - 0.5) * 1.5;
        const pan = ac.createStereoPanner(); pan.pan.value = (vi - 1.5) * 0.25 + det / 40;
        o.connect(pan).connect(lp); o.start(t0); o.stop(t0 + chordLen + 1.5);
      }
    });
    // bass: soft sine on the root, each bar
    for (let b = 0; b < 2; b++) {
      const tb = t0 + b * bar, o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine'; o.frequency.value = hz(key - 12 + chord[0] % 12);
      g.gain.setValueAtTime(0, tb); g.gain.linearRampToValueAtTime(0.16, tb + 0.08); g.gain.exponentialRampToValueAtTime(0.03, tb + bar * 0.95); g.gain.linearRampToValueAtTime(0, tb + bar);
      o.connect(g).connect(dry); o.start(tb); o.stop(tb + bar + 0.05);
    }
    // felt-piano arpeggio: sparse eighths
    const tones = [...chord, chord[0] + 12, chord[1] + 12];
    for (let e8 = 0; e8 < 16; e8++) {
      if (r() < (mood === 'reflective' ? 0.55 : 0.35)) continue;
      const te = Math.max(0, t0 + e8 * beat / 2 + (r() - 0.5) * 0.012), note = key + 24 + tones[Math.floor(r() * tones.length)];
      const vel = 0.05 + r() * 0.035, o1 = ac.createOscillator(), o2 = ac.createOscillator(), g = ac.createGain(), f = ac.createBiquadFilter();
      o1.type = 'triangle'; o2.type = 'sine'; o1.frequency.value = hz(note); o2.frequency.value = hz(note) * 2.001;
      f.type = 'lowpass'; f.frequency.value = 2400;
      const g2 = ac.createGain(); g2.gain.value = 0.25; o2.connect(g2).connect(f); o1.connect(f); f.connect(g);
      g.gain.setValueAtTime(0, te); g.gain.linearRampToValueAtTime(vel, te + 0.006); g.gain.exponentialRampToValueAtTime(0.0008, te + 1.6);
      const pan = ac.createStereoPanner(); pan.pan.value = (r() - 0.5) * 0.6; g.connect(pan).connect(dry);
      o1.start(te); o2.start(te); o1.stop(te + 1.7); o2.stop(te + 1.7);
    }
    if (m.pulse) for (let b = 0; b < 8; b += 2) { // a soft heartbeat on 1 and 3
      const tb = t0 + b * beat, o = ac.createOscillator(), g = ac.createGain();
      o.frequency.setValueAtTime(70, tb); o.frequency.exponentialRampToValueAtTime(42, tb + 0.18);
      g.gain.setValueAtTime(0, tb); g.gain.linearRampToValueAtTime(0.12, tb + 0.01); g.gain.exponentialRampToValueAtTime(0.001, tb + 0.35);
      o.connect(g).connect(dry); o.start(tb); o.stop(tb + 0.4);
    }
  }
  // gentle air
  const nb = ac.createBuffer(2, len, SR); for (let c = 0; c < 2; c++) { const d = nb.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (r() * 2 - 1) * 0.006; }
  const ns = ac.createBufferSource(); ns.buffer = nb; const nf = ac.createBiquadFilter(); nf.type = 'bandpass'; nf.frequency.value = 3000; nf.Q.value = 0.4;
  ns.connect(nf).connect(dry); ns.start(0);
  const buf = await ac.startRendering();
  return { chs: [buf.getChannelData(0), buf.getChannelData(1)], name: `Original · ${m.label} · ${['C', 'D', 'E♭', 'F', 'G', 'A'][[0, 2, 3, 5, 7, 9].indexOf(key - 48)]} · ${m.bpm} bpm` };
}
async function decodeTrack(url) {
  const blob = await (await fetch(url)).blob();
  let b;
  try { const ac = new OfflineAudioContext(2, 1, SR); b = await ac.decodeAudioData(await blob.arrayBuffer()); }
  catch (e) { // tracks saved by an earlier version have a mangled AAC header: decode them sample by sample instead
    const mv = await openMovie(blob, { audioOnly: true });
    return await decodeAudioTrack(mv, null, { stereo: true });
  }
  const L = b.getChannelData(0), R = b.numberOfChannels > 1 ? b.getChannelData(1) : L;
  return [resample(L, b.sampleRate, SR), resample(R, b.sampleRate, SR)];
}
function fitTrack(chs, seconds) {
  const n = Math.ceil(seconds * SR), xf = Math.round(1.0 * SR), out = [new Float32Array(n), new Float32Array(n)];
  for (let c = 0; c < 2; c++) {
    const src = chs[c]; let o = 0;
    while (o < n) { // loop with a 1 s crossfade if the track is shorter than the reel
      for (let i = 0; i < src.length && o + i < n; i++) {
        let g = 1; if (o > 0 && i < xf) g = i / xf;
        out[c][o + i] = out[c][o + i] * (o > 0 && i < xf ? 1 - g : 1) + src[i] * g;
      }
      if (src.length >= n - o) break; o += src.length - xf;
    }
  }
  return out;
}
function mixMusic(voice, music, speech, levelLU) {
  const n = voice.length, mL = music[0], mR = music[1];
  const vL = loudness([voice, voice]), mLu = loudness([mL.subarray(0, n), mR.subarray(0, n)]);
  const base = Math.pow(10, ((vL + levelLU) - mLu) / 20);
  // duck: full bed level while you speak, +3 dB in the gaps, smooth both ways
  const env = new Float32Array(n), up = Math.pow(10, 4 / 20);
  const act = new Uint8Array(Math.ceil(n / 480));
  for (const [s, e] of speech) for (let k = Math.floor(s * 100); k < Math.min(act.length, Math.ceil(e * 100)); k++) act[k] = 1;
  let g = 1; const att = 1 - Math.exp(-1 / (0.15 * SR)), rel = 1 - Math.exp(-1 / (0.5 * SR));
  for (let i = 0; i < n; i++) { const tgt = act[Math.floor(i / 480)] ? 1 : up; g += (tgt - g) * (tgt < g ? att : rel); env[i] = g; }
  const fi = Math.round(1.2 * SR), fo = Math.round(1.8 * SR);
  const L = new Float32Array(n), R = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let f = 1; if (i < fi) f = i / fi; if (i > n - fo) f = Math.min(f, (n - i) / fo);
    const k = base * env[i] * f * f;
    L[i] = voice[i] + (mL[i] || 0) * k; R[i] = voice[i] + (mR[i] || 0) * k;
  }
  return [L, R];
}

// ------------------------------------------------------------------ analysis: everything before the render
async function analyze({ file, items, notes, cfg, onStage, signal, photos, model }) {
  cfg = cfg || defaults();
  const stage = (id, pct, msg) => onStage && onStage(id, pct, msg);
  const t00 = performance.now(), times = {};
  const mark = k => { times[k] = Math.round((performance.now() - t00) / 100) / 10; };

  stage('read', 0, 'Reading your take');
  const movie = await openMovie(file);
  const nFrames = Math.floor(movie.duration * FPS), total = nFrames / FPS;
  let raw = await decodeAudioTrack(movie, signal);
  if (!raw) throw new Error('This take has no audio track.');
  const off = Math.round(movie.audioOffset * SR), need = nFrames * SPF, aligned = new Float32Array(need);
  for (let i = 0; i < need; i++) { const j = i - off; aligned[i] = j >= 0 && j < raw.length ? raw[j] : 0; }
  raw = aligned; mark('read'); stage('read', 1);

  stage('claps', 0, 'Listening for claps');
  const { claps, near } = detectClaps(raw, cfg); await tick(); check(signal);
  stage('claps', 0.4, 'Cleaning up the audio');
  let proc = await denoise(highpass4(raw, 80), cfg.nr, signal);
  proc = filt(filt(filt(proc, biquad('peak', 220, 1.0, -2.5)), biquad('peak', 450, 1.4, -1.5)), biquad('peak', 3200, 1.0, 1.5));
  proc = compress(proc); mark('audio'); stage('claps', 1);

  stage('transcribe', 0, 'Loading the speech model');
  const words = await transcribe(proc, (p, msg) => stage('transcribe', p, msg || undefined), signal, model || 'small');
  const rawTiming = refineTiming(words, proc);                    // fix word times before they decide what counts as dead air
  mark('transcribe'); check(signal);

  stage('cut', 0, 'Cutting claps and dead air');
  const speech = addMissedWords(vad(proc), words);
  const flubs = flubRegions(claps, speech, words, cfg);
  DIAG.claps = { found: claps.map(c => `${c.t}s L${c.level} R${c.rise} I${c.iso}`), cuts: flubs.map(f => `${f.cut[0].toFixed(2)}-${f.cut[1].toFixed(2)} ${f.method}`), near: near.slice(0, 3).map(c => `${c.t}s L${c.level} R${c.rise} I${c.iso}`) };
  const { keep, trimmed } = buildKeep(speech, flubs.map(f => f.cut), total, cfg);
  if (!keep.length) throw new Error("Couldn't hear any speech in this take.");
  const tmap = new TimeMap(keep), outDur = tmap.nOut / FPS;
  flubs.forEach(f => { f.outJoin = tmap.rawToOut(f.cut[1]); });
  let voice = cutAudio(proc, keep);
  const inLufs = loudness([voice]);
  const g = Math.pow(10, ((cfg.lufs - 3) - inLufs) / 20); // -17 LUFS mono = -14 LUFS once doubled to stereo
  for (let i = 0; i < voice.length; i++) voice[i] *= g;
  let outWords = mapWords(words, tmap);
  const timing = Object.assign(refineTiming(outWords, voice), { rawShifted: rawTiming.shifted });
  addEmphasis(outWords, voice);
  const proof = { applied: applyCorrections(outWords, parseCorrections(cfg.corrections ?? DEFAULT_CORRECTIONS)), note: null };
  check(signal);
  mark('cut'); stage('cut', 1);

  stage('plan', 0, 'Choosing clips and photos');
  const ctx = { delivery: deliveryRows(outWords, voice, phrases(outWords, 10)) }; let aiNote = null;
  if (cfg.meaning !== false && ((items || []).length || (photos || []).length)) {
    try { stage('plan', 0.2, 'Matching visuals to what you mean'); ctx.sims = await meaningSims(outWords, [...(items || []), ...(photos || [])]); aiNote = 'Meaning matching: on (MiniLM, running on this device).'; }
    catch (e) { if (e instanceof Cancelled) throw e; aiNote = `Meaning matching unavailable (${e.message || e}); matched by words and feeling.`; }
  }
  const { broll, pops, log: brollLog, mode: brollMode } = planVisuals(outWords, items || [], photos || [], notes, tmap, outDur, cfg, null, ctx);
  if (aiNote) brollLog.unshift(aiNote);
  const popLog = [];
  const zooms = planZooms(outWords, tmap.cutPoints(), broll, outDur, cfg);
  const zc = zoomCurve(zooms, tmap.nOut, cfg);
  mark('plan'); stage('plan', 1);

  stage('face', 0, 'Finding your face');
  const track = await faceTrack(movie, tmap, broll, cfg, p => stage('face', p), signal);
  mark('face'); stage('face', 1);
  // everything below lets each edit be undone later without redoing the whole analysis
  const surv = outWords.map(w => {   // back to take time; a word never spans a cut (its end stays in the piece its start is in)
    const k = tmap.segIndex(Math.floor(w.s * FPS)), segEnd = tmap.keep[k][1] / FPS, s0 = tmap.outToRaw(w.s);
    return { ...w, s: +s0.toFixed(3), e: +Math.min(segEnd, Math.max(s0 + 0.05, tmap.outToRaw(Math.max(w.s, w.e - 0.001)))).toFixed(3) };
  });
  const cutWords = words.filter(w => !surv.some(o => o.e > w.s + 0.02 && o.s < w.e - 0.02)).map(w => ({ ...w, cut: true }));  // e.g. the flubbed take, for if it's put back
  const keepRaw = { proc, raw, speech, rawWords: [...surv, ...cutWords].sort((x, y) => x.s - y.s) };

  return { movie, cfg, total, outDur, claps, near, flubs, keep, trimmed, tmap, words, outWords, voice, inLufs,
    speechOut: vad(voice), broll, brollLog, brollMode, pops, popLog, zooms, timing, proof, model: words.model, zc, track, times, chunks: null, music: null, notes, ...keepRaw, cleanup: true, capsOff: false };
}


/** Redo the cut after edits were switched off or back on (a clap cut, a trimmed pause, voice cleanup):
 *  rebuild the timeline and move every other edit (B-roll, photos, zooms, caption fixes) to follow its words. */
async function retime(P, onStage, signal) {
  const cfg = P.cfg, old = P.tmap, stage = (id, p, m) => onStage && onStage(id, p, m);
  stage('cut', 0, 'Rebuilding the cut');
  const restored = P.trimmed.filter(t => t.off).map(t => [t[0], t[1]]);
  const removed = [...P.flubs.filter(f => !f.off).map(f => f.cut), ...(P.manual || []).filter(m => !m.off).map(m => m.range)].sort((a, b) => a[0] - b[0]);
  const { keep, trimmed } = buildKeep(P.speech, removed, P.total, cfg, restored);
  if (!keep.length) throw new Error('Nothing left to show with these edits removed.');
  const tmap = new TimeMap(keep), outDur = tmap.nOut / FPS;
  const move = t => tmap.rawToOut(old.outToRaw(t));
  // pauses: keep the ones the user put back listed (marked off) so they can be trimmed again
  const tr = trimmed.map(t => t); for (const r of P.trimmed.filter(t => t.off)) { const x = [r[0], r[1], r[2]]; x.off = true; tr.push(x); }
  tr.sort((a, b) => a[0] - b[0]);
  P.flubs.forEach(f => { f.outJoin = tmap.rawToOut(f.off ? f.cut[0] : f.cut[1]); });
  (P.manual || []).forEach(m => { m.outJoin = tmap.rawToOut(m.off ? m.range[0] : m.range[1]); });
  // voice
  let voice = cutAudio(P.cleanup ? P.proc : P.raw, keep);
  const g = Math.pow(10, ((cfg.lufs - 3) - loudness([voice])) / 20); for (let i = 0; i < voice.length; i++) voice[i] *= g;
  // words follow their raw times; caption fixes are kept by the raw time of each line's first word
  const outWords = [];
  for (const w of P.rawWords) {
    const tol = w.cut ? 0 : 2;                     // words that were cut only come back if their stretch is fully back
    const k = tmap.keep.findIndex(([f0, f1]) => w.s * FPS >= f0 - tol && w.s * FPS < f1);
    if (k < 0) continue;
    const s = tmap.rawToOut(w.s), e = Math.max(s + 0.05, tmap.rawToOut(w.e));
    outWords.push({ ...w, s: +s.toFixed(3), e: +e.toFixed(3), rawS: w.s });
  }
  // visuals and zooms keep their length and follow their words
  const shift = p => { const d = p.e - p.s; p.s = Math.round(move(p.s) * FPS) / FPS; p.e = Math.min(outDur - 0.1, Math.round((p.s + d) * FPS) / FPS); };
  P.broll.forEach(shift); (P.pops || []).forEach(shift);
  P.zooms.forEach(z => { const d = z.t1 - z.t0; z.t0 = Math.round(move(z.t0) * FPS) / FPS; z.t1 = Math.min(outDur, z.t0 + d); });
  Object.assign(P, { tmap, keep, trimmed: tr, outDur, voice, outWords, speechOut: vad(voice) });
  P.zc = zoomCurve(P.zooms.filter(z => !z.off), tmap.nOut, cfg);
  check(signal); stage('cut', 1);
  stage('face', 0, 'Following your face');
  P.track = await faceTrack(P.movie, tmap, P.broll.filter(b => !b.off), cfg, p => stage('face', p), signal, P.track && P.track.cache);
  stage('face', 1);
  await rebuildCaptions(P);
  return P;
}
/** Re-make the caption lines from the words (fresh line breaks and timing), keeping any line you fixed by hand. */
async function rebuildCaptions(P) {
  const fixes = new Map(); (P.chunks || []).forEach(c => { if (c.edited && c.rawS != null) fixes.set(Math.round(c.rawS * 20), c); });
  await buildCaptions(P);
  for (const c of P.chunks) { const f = fixes.get(Math.round((c.rawS ?? -9) * 20)); if (f && f.words.length) { c.words = f.words; c.hl = f.hl; c.edited = true; } }
  DIAG.captions && (DIAG.captions.kept = fixes.size);
  return P.chunks;
}
function setZooms(P) { P.zc = zoomCurve(P.zooms.filter(z => !z.off), P.tmap.nOut, P.cfg); }

async function buildCaptions(P) {
  try { await document.fonts.load(capFont(P.cfg)); } catch (e) { /* fallback fonts are fine */ }
  const c = new OffscreenCanvas(8, 8).getContext('2d'); c.font = capFont(P.cfg);
  P.chunks = chunkWords(P.outWords, s => c.measureText(s).width, OUT_W - 2 * P.cfg.cap.side - 20, P.cfg);
  if (P.tmap) for (const ch of P.chunks) ch.rawS = +P.tmap.outToRaw(ch.w0).toFixed(3);
  // how much of the speech has a caption on screen (should be ~100%)
  const sp = P.speechOut || [], tot = sp.reduce((a, [s, e]) => a + e - s, 0);
  const cov = sp.reduce((a, [s, e]) => a + P.chunks.reduce((b, c) => b + Math.max(0, Math.min(e, c.e) - Math.max(s, c.s)), 0), 0);
  DIAG.captions = { lines: P.chunks.length, words: P.outWords.length, speechSec: +tot.toFixed(1), covered: tot ? +(cov / tot).toFixed(2) : null, text: P.outWords.map(w => w.w).join(' ').slice(0, 1500) };
  return P.chunks;
}

async function buildAudio(P, musicOpts, onStage, signal) {
  const M = Object.assign({}, P.cfg.music, musicOpts || {});
  let music = null, chosen = { kind: 'none', name: 'No music' };
  const mine = (M.tracks || []).filter(t => !M.mood || !t.mood || t.mood === M.mood);
  try {
    if (M.mode === 'mine' || (M.mode === 'auto' && mine.length)) {
      const pool = M.mode === 'mine' ? (M.tracks || []) : mine;
      if (pool.length) {
        const pick = M.trackId ? pool.find(t => t.id === M.trackId) || pool[0] : pool[Math.floor(Math.random() * pool.length)];
        onStage && onStage('music', 0.3, `Fitting "${pick.name}"`);
        try { music = fitTrack(await decodeTrack(pick.url), P.outDur); chosen = { kind: 'track', name: pick.name, id: pick.id }; }
        catch (e) { DIAG.trackError = String(e.message || e); music = null; M.fellBack = `couldn't read "${pick.name}"`; }
      }
    }
    if (!music && M.mode !== 'off') {
      onStage && onStage('music', 0.3, 'Composing an original bed');
      const seed = M.seed || Math.floor(Math.random() * 1e9);
      const g = await composeMusic(P.outDur, M.mood, seed); music = g.chs; chosen = { kind: 'generated', name: g.name + (M.fellBack ? ` (${M.fellBack})` : ''), seed };
    }
  } catch (e) { if (e instanceof Cancelled) throw e; console.warn(e); music = null; chosen = { kind: 'none', name: 'No music (' + e.message + ')' }; }
  check(signal);
  let L, R;
  if (music) [L, R] = mixMusic(P.voice, music, P.speechOut, M.level);
  else { L = Float32Array.from(P.voice); R = Float32Array.from(P.voice); }
  const lu = loudness([L, R]), gg = Math.pow(10, (P.cfg.lufs - lu) / 20);
  for (let i = 0; i < L.length; i++) { L[i] *= gg; R[i] *= gg; }
  limit([L, R], -1.2);
  P.mix = [L, R]; P.music = Object.assign(chosen, { mood: M.mood, level: M.level });
  P.finalLufs = loudness([L, R]);
  onStage && onStage('music', 1);
  return P.mix;
}

// ------------------------------------------------------------------ 9. render
let _muxLoaded = false;
const AAC_ASC = new Uint8Array([0x11, 0x90]); // AudioSpecificConfig: AAC-LC, 48 kHz, stereo
/** Muxer output for AAC that works whether or not the browser supplies the config header or ADTS framing. */
function aacOutput(mux) {
  let first = true;
  return (chunk, meta) => {
    let data = new Uint8Array(chunk.byteLength); chunk.copyTo(data);
    if (data.length > 7 && data[0] === 0xFF && (data[1] & 0xF6) === 0xF0) data = data.slice((data[1] & 1) ? 7 : 9);
    let m;
    DIAG.aacChunks = (DIAG.aacChunks || 0) + 1; DIAG.aacBytes = (DIAG.aacBytes || 0) + data.length;
    if (first) {
      const d = meta && meta.decoderConfig && meta.decoderConfig.description;
      DIAG.aacFirst = { rawHead: hex(new Uint8Array(chunk.byteLength).map((_, i) => 0) && (() => { const u = new Uint8Array(chunk.byteLength); chunk.copyTo(u); return u; })()),
        type: chunk.type, ts: chunk.timestamp, dur: chunk.duration, metaKeys: meta ? Object.keys(meta) : null,
        decCfg: meta && meta.decoderConfig ? { codec: meta.decoderConfig.codec, sampleRate: meta.decoderConfig.sampleRate, ch: meta.decoderConfig.numberOfChannels, descLen: d ? (d.byteLength || 0) : 0, desc: d ? hex(new Uint8Array(d.buffer || d, d.byteOffset || 0, d.byteLength)) : null } : null };
      const ok = d && (d.byteLength || 0) >= 2;
      m = { decoderConfig: { codec: 'mp4a.40.2', sampleRate: SR, numberOfChannels: 2, description: ok ? sanitizeAsc(d) : AAC_ASC } };
      DIAG.aacAscUsed = hex(m.decoderConfig.description);
      first = false;
    }
    const dur = chunk.duration != null ? chunk.duration : Math.round(1024 / SR * 1e6);
    mux.addAudioChunkRaw(data, 'key', Math.max(0, chunk.timestamp), dur, m);
  };
}
/** Decode the finished file's audio to make sure it actually plays. */
async function verifyAudio(blob) {
  try {
    const ac = new OfflineAudioContext(2, 1, SR);
    const b = await ac.decodeAudioData(await blob.arrayBuffer());
    const d = b.getChannelData(0); let e = 0; for (let i = 0; i < d.length; i += 7) e += d[i] * d[i];
    const db = 10 * Math.log10(e / Math.max(1, d.length / 7) + 1e-12);
    return { ok: db > -55 && b.duration > 0.5, db: +db.toFixed(1), duration: +b.duration.toFixed(2) };
  } catch (e) { return { ok: false, error: e.message || String(e) }; }
}
async function ensureMuxer() { if (!_muxLoaded) { await loadScript(LIBS.muxer); _muxLoaded = true; } }
function toWav(chs) {
  const n = chs[0].length, nc = chs.length, buf = new ArrayBuffer(44 + n * nc * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * nc * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, nc, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * nc * 2, true); v.setUint16(32, nc * 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, n * nc * 2, true);
  let o = 44; for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) { v.setInt16(o, clamp(chs[c][i], -1, 1) * 32767, true); o += 2; }
  return new Blob([buf], { type: 'audio/wav' });
}
/** Convert a music file to a small AAC .m4a (audio-only MP4) so it can be stored on this device. */
async function audioToM4a(file) {
  await ensureMuxer();
  const ac = new OfflineAudioContext(2, 1, SR); const b = await ac.decodeAudioData(await file.arrayBuffer());
  const L = resample(b.getChannelData(0), b.sampleRate, SR), R = b.numberOfChannels > 1 ? resample(b.getChannelData(1), b.sampleRate, SR) : L;
  let codec = 'aac', cfg = { codec: 'mp4a.40.2', sampleRate: SR, numberOfChannels: 2, bitrate: 192000 };
  if (!('AudioEncoder' in root) || !(await AudioEncoder.isConfigSupported(cfg)).supported) { codec = 'opus'; cfg = { codec: 'opus', sampleRate: SR, numberOfChannels: 2, bitrate: 160000 }; }
  const Mx = root.Mp4Muxer, target = new Mx.ArrayBufferTarget();
  const mux = new Mx.Muxer({ target, audio: { codec, numberOfChannels: 2, sampleRate: SR }, fastStart: 'in-memory', firstTimestampBehavior: 'offset' });
  let err = null; const enc = new AudioEncoder({ output: codec === 'aac' ? aacOutput(mux) : (c, m) => mux.addAudioChunk(c, m), error: e => { err = e; } });
  enc.configure(cfg);
  for (let o = 0; o < L.length; o += 4800) {
    const m = Math.min(4800, L.length - o), d = new Float32Array(m * 2); d.set(L.subarray(o, o + m), 0); d.set(R.subarray(o, o + m), m);
    enc.encode(new AudioData({ format: 'f32-planar', sampleRate: SR, numberOfFrames: m, numberOfChannels: 2, timestamp: Math.round(o / SR * 1e6), data: d }));
    if (enc.encodeQueueSize > 50) await sleep(1);
  }
  await enc.flush(); enc.close(); if (err) throw err; mux.finalize();
  return { blob: new Blob([target.buffer], { type: 'video/mp4' }), duration: L.length / SR };
}
async function render(P, onProgress, signal) {
  await ensureMuxer();
  const Mx = root.Mp4Muxer, cfg = P.cfg, n = P.tmap.nOut;
  const vcfg = { codec: 'avc1.640028', width: OUT_W, height: OUT_H, bitrate: cfg.videoBitrate, framerate: FPS, avc: { format: 'avc' } };
  if (!(await VideoEncoder.isConfigSupported(vcfg)).supported) { delete vcfg.avc; }
  let acodec = 'aac', acfg = { codec: 'mp4a.40.2', sampleRate: SR, numberOfChannels: 2, bitrate: 192000 };
  if (!('AudioEncoder' in root) || !(await AudioEncoder.isConfigSupported(acfg)).supported) { acodec = 'opus'; acfg = { codec: 'opus', sampleRate: SR, numberOfChannels: 2, bitrate: 160000 }; }
  const target = new Mx.ArrayBufferTarget();
  const mux = new Mx.Muxer({ target, video: { codec: 'avc', width: OUT_W, height: OUT_H, frameRate: FPS },
    audio: { codec: acodec, numberOfChannels: 2, sampleRate: SR }, fastStart: 'in-memory', firstTimestampBehavior: 'offset' });
  let encErr = null;
  const venc = new VideoEncoder({ output: (c, m) => mux.addVideoChunk(c, m), error: e => { encErr = e; } });
  venc.configure(vcfg);
  DIAG.render = { acodec, acfg: { ...acfg }, mixRmsDb: rmsDb(P.mix[0]), mixLen: P.mix[0].length, voiceRmsDb: rmsDb(P.voice) }; DIAG.aacChunks = 0; DIAG.aacBytes = 0; DIAG.aacFirst = null;
  const aenc = new AudioEncoder({ output: acodec === 'aac' ? aacOutput(mux) : (c, m) => mux.addAudioChunk(c, m), error: e => { encErr = e; } });
  aenc.configure(acfg);
  // audio first (small)
  const [L, R] = P.mix, step = 4800;
  for (let o = 0; o < L.length; o += step) {
    const m = Math.min(step, L.length - o), buf = new Float32Array(m * 2);
    buf.set(L.subarray(o, o + m), 0); buf.set(R.subarray(o, o + m), m);
    aenc.encode(new AudioData({ format: 'f32-planar', sampleRate: SR, numberOfFrames: m, numberOfChannels: 2, timestamp: Math.round(o / SR * 1e6), data: buf }));
  }
  await aenc.flush(); aenc.close(); DIAG.render.audioEncState = 'flushed'; DIAG.render.encErr = encErr ? String(encErr.message || encErr) : null;

  const canvas = new OffscreenCanvas(OUT_W, OUT_H), ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  const src = new VideoSource(P.movie); await src.open();
  let capK = 0; const capAt = t => { while (capK < P.chunks.length && P.chunks[capK].e <= t) capK++; const c = P.chunks[capK]; return c && c.s <= t ? c : null; };
  const bAt = new Map(); P.broll.forEach(p => bAt.set(Math.round(p.s * FPS), p));
  const pops = (P.pops || []).filter(p => !p.off), popImg = new Map();
  for (const p of pops) { try { popImg.set(p, await createImageBitmap(await (await fetch(p.item.url)).blob())); } catch (e) { DIAG.popError = String(e.message || e); } }
  for (const c of P.chunks) c.up = false;
  let bsrc = null, bend = -1, bimg = null, bplace = null;
  const t0 = performance.now();
  try {
    for (let oi = 0; oi < n; oi++) {
      if (encErr) throw encErr;
      const t = oi / FPS;
      if (bAt.has(oi)) {
        const p = bAt.get(oi); bplace = p; bend = Math.round(p.e * FPS);
        if (bsrc) { bsrc.close(); bsrc = null; } bimg = null;
        const blob = await (await fetch(p.item.url)).blob();
        if (p.item.kind === 'still') bimg = await createImageBitmap(blob);
        else { const mv = await openMovie(blob); bsrc = new VideoSource(mv); bsrc.mv2 = mv; await bsrc.open(); if (p.clipIn > 0.05) bsrc.seek(p.clipIn); }
      }
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, OUT_W, OUT_H);
      const drawHead = async () => {
        const f = await src.frameAt(P.tmap.outToRawFrame(oi) / FPS);
        if (f) { const [x, y, w] = cropRect(oi, P.track, P.zc, P.movie.W, P.movie.H, cfg), k = OUT_W / w; drawRotated(ctx, f, P.movie, k, -x * k, -y * k); }
      };
      if (bplace && oi < bend) {
        const lt = t - bplace.s, D = Math.max(0.5, bplace.e - bplace.s), F = DISSOLVE;
        const kIn = oi - Math.round(bplace.s * FPS), kOut = bend - 1 - oi;
        let a = 1; if (kIn < F) a = Math.min(a, (kIn + 1) / (F + 1)); if (kOut < F) a = Math.min(a, (kOut + 1) / (F + 1));
        a = a * a * (3 - 2 * a);
        if (a < 1) await drawHead();
        const u = Math.min(1, lt / D), push = 1 + 0.05 * (u * u * (3 - 2 * u)); // slow push, matches the talking-head push-ins
        ctx.globalAlpha = a;
        if (bimg) {
          const k = Math.max(OUT_W / bimg.width, OUT_H / bimg.height) * push;
          ctx.drawImage(bimg, (OUT_W - bimg.width * k) / 2, (OUT_H - bimg.height * k) / 2, bimg.width * k, bimg.height * k);
        } else if (bsrc) {
          const f = await bsrc.frameAt(bplace.clipIn + lt), mv = bsrc.mv2;
          if (f) { const k = Math.max(OUT_W / mv.W, OUT_H / mv.H) * push; drawRotated(ctx, f, mv, k, (OUT_W - mv.W * k) / 2, (OUT_H - mv.H * k) / 2); }
        }
        ctx.globalAlpha = 1;
      } else {
        if (bplace) { bplace = null; if (bsrc) { bsrc.close(); bsrc = null; } bimg = null; }
        await drawHead();
      }
      for (const p of pops) { const st = popAlpha(oi, p, cfg); if (st && popImg.has(p)) drawPop(ctx, popImg.get(p), st, cfg); }
      if (!P.capsOff) { const c = capAt(t); drawCaption(ctx, c, cfg, t, c ? P.chunks[capK + 1] : null, captionLift(t, pops)); }
      const vf = new VideoFrame(canvas, { timestamp: Math.round(oi * 1e6 / FPS), duration: Math.round(1e6 / FPS) });
      venc.encode(vf, { keyFrame: oi % 60 === 0 }); vf.close();
      while (venc.encodeQueueSize > 4) await sleep(2);
      if (oi % 15 === 0) { onProgress && onProgress(oi / n, oi / ((performance.now() - t0) / 1000)); check(signal); await tick(); }
    }
    await venc.flush();
  } finally { src.close(); if (bsrc) bsrc.close(); if (venc.state !== 'closed') venc.close(); }
  if (encErr) throw encErr;
  mux.finalize();
  onProgress && onProgress(1);
  return new Blob([target.buffer], { type: 'video/mp4' });
}

// ------------------------------------------------------------------ edit log
function editLog(P) {
  const on = a => (a || []).filter(x => !x.off), name = p => (p.item && (p.item.subject || p.item.name)) || String(p.id).replace(/_/g, ' ');
  const fl = on(P.flubs), mine = on(P.manual), tr = on(P.trimmed), zs = on(P.zooms), br = on(P.broll), pl = on(P.pops);
  const L = ['EDIT SUMMARY', '', `Original ${fmt(P.total)}, reel ${fmt(P.outDur)} (${(P.total - P.outDur).toFixed(1)} s removed).`, '', 'CUTS'];
  fl.forEach(f => L.push(`- ${fmt(f.outJoin)}: retake before your clap removed${f.flubText ? ` ("${f.flubText}")` : ''}`));
  mine.filter(m => m.kind === 'cut').forEach(m => L.push(`- ${fmt(m.outJoin)}: you cut "${m.text}"`));
  if (!fl.length && !mine.some(m => m.kind === 'cut')) L.push('None.');
  L.push('', 'PAUSES', `${tr.length + mine.filter(m => m.kind !== 'cut').length} shortened.`);
  L.push('', 'ZOOMS'); zs.forEach(z => L.push(`- ${fmt(z.t0)}–${fmt(z.t1)}: ${z.kind === 'push' ? 'slow zoom' : 'punch-in'} on "${z.word}"`)); if (!zs.length) L.push('None.');
  L.push('', 'CLIPS'); br.forEach(p => L.push(`- ${fmt(p.s)}–${fmt(p.e)}: ${name(p)}, while you say "${p.text}"`)); if (!br.length) L.push('None.');
  L.push('', 'PHOTOS'); pl.forEach(p => L.push(`- ${fmt(p.s)}–${fmt(p.e)}: ${name(p)}, while you say "${p.text}"`)); if (!pl.length) L.push('None.');
  L.push('', 'MUSIC', P.music && P.music.kind !== 'none' ? P.music.name : 'None.');
  L.push('', 'TRANSCRIPT', P.outWords.map(w => w.w).join(' '), '');
  return L.join('\n');
}

root.ReelEngine = { DEFAULT_CORRECTIONS, retime, rebuildCaptions, setZooms, transcribe, MODELS, popAlpha, snapEnd, stem, isStop: w => STOP.has(norm(w)), norm, phrases, catalogLine, decodeAudioTrack, sanitizeAsc, decodeTrack, DIAG, verifyAudio, ensureMuxer, toWav, audioToM4a, drawRotated, defaults, analyze, buildCaptions, buildAudio, render, editLog, toSrt, composeMusic, openMovie, VideoSource,
  decodeTrack, loudness, drawCaption, capFont, fmt, pickHighlight, FPS, OUT_W, OUT_H, SR, MOODS, Cancelled,
  _t: { refineTiming, applyCorrections, parseCorrections, rhythmStart, planVisuals, deliveryRows, meaningSims, captionLift, detectClaps, vad, buildKeep, flubRegions, planZooms, chunkWords, parseNotes, findPhrase, mapWords, TimeMap, denoise, compress, limit, resample, highpass4, filt, biquad, addEmphasis, fitTrack, mixMusic, ratio } };
})(typeof window !== 'undefined' ? window : globalThis);
