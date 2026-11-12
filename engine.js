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
    broll: { min: 1.8, target: 2.0, max: 2.4, maxFrac: 0.25, autoMinStart: 1.5, gap: 2.8, amount: 1 },
    cap: { size: 86, accent: '#D6B06C', color: '#FFFFFF', style: 'outline', upper: false, baseline: 0.70, baselineUp: 0.468, maxWords: 4, side: 150 },
    pop: { on: true, x: 140, y: 985, w: 800, h: 520, radius: 28, min: 1.8, max: 2.4, gap: 2.8, inF: 13, outF: 11 },
    music: { mode: 'generated', genre: 'chill', level: -10 },
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

function keepFor(speech, removed, total, cfg, restored, range, hook) {
  const [a, b] = range || [0, total];
  const sp = speech.map(([s, e]) => [Math.max(s, a), Math.min(e, b)]).filter(([s, e]) => e - s > 0.04);
  const r = buildKeep(sp, removed, total, cfg, restored);
  const fa = Math.ceil(a * FPS), fb = Math.floor(b * FPS);
  let keep = r.keep.map(([f0, f1]) => [Math.max(f0, fa), Math.min(f1, fb)]).filter(([f0, f1]) => f1 - f0 >= 3);
  const trimmed = r.trimmed.filter(t => t[0] >= a - 0.01 && t[1] <= b + 0.01);
  if (hook && hook[0] > a + 0.3) {
    const h0 = Math.floor(hook[0] * FPS), h1 = Math.ceil(hook[1] * FPS), front = [], rest = [];
    for (const [f0, f1] of keep) {
      const x0 = Math.max(f0, h0), x1 = Math.min(f1, h1);
      if (x1 - x0 >= 3) front.push([x0, x1]);
      if (Math.min(f1, h0) - f0 >= 3) rest.push([f0, Math.min(f1, h0)]);
      if (f1 - Math.max(f0, h1) >= 3) rest.push([Math.max(f0, h1), f1]);
    }
    if (front.length && rest.length) keep = [...front, ...rest];
  }
  return { keep, trimmed };
}

class TimeMap {
  constructor(keep) {
    this.keep = keep; this.starts = []; let o = 0;
    for (const [f0, f1] of keep) { this.starts.push(o); o += f1 - f0; }
    this.nOut = o;
  }
  rawToOut(t) {
    const f = t * FPS; let next = -1;
    for (let i = 0; i < this.keep.length; i++) {
      const [f0, f1] = this.keep[i];
      if (f >= f0 && f < f1) return (this.starts[i] + f - f0) / FPS;
      if (f0 > f && (next < 0 || f0 < this.keep[next][0])) next = i;
    }
    return next >= 0 ? this.starts[next] / FPS : this.nOut / FPS;
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

function cutAudio(a, keep, base = 0) {
  const fade = Math.round(0.004 * SR); let n = 0;
  for (const [f0, f1] of keep) n += (f1 - f0) * SPF;
  const out = new Float32Array(n); let o = 0;
  for (const [f0, f1] of keep) {
    const len = (f1 - f0) * SPF;
    out.set(a.subarray(Math.max(0, f0 * SPF - base), Math.max(0, Math.min(a.length, f1 * SPF - base))), o);
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
  out.sort((a, b) => a.s - b.s);
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
/** The feel of a line or a shot, as a mix of these. Lines and visuals are placed in the same space by the sentence model,
 *  so a quiet line about patience lands near a slow ocean shot even when no word matches. */
const VIBES = {
  calm: ['be patient with yourself', 'slow down and breathe', 'calm, quiet, peaceful nature', 'trust the process, it takes time', 'rest and recover', 'zoom out and get perspective'],
  grind: ['hard work and sweat', 'push through when it gets hard', 'train hard, heavy weights', 'discipline and effort every day', 'the grind'],
  triumph: ['you did it, feeling proud', 'getting stronger, real progress', 'confidence and winning', 'the results paid off', 'feeling unstoppable'],
  warmth: ['my kids and my family', 'love and connection', 'grateful for the people around me', 'being there for the ones you love', 'home and togetherness'],
  struggle: ['stressed and overwhelmed', 'exhausted, burned out', 'doubting yourself, wanting to quit', 'feeling stuck and frustrated', 'it hurts and it is hard'],
  hype: ['influencers showing off', 'expensive gear and fancy gyms', 'quick fixes and shortcuts', 'social media fitness hype', 'perfect abs and flashy promises'],
  practical: ['what to eat and how to cook it', 'a simple daily routine', 'step by step how to do it', 'track it and write it down', 'meals, protein and groceries'],
  hope: ['a fresh start', 'a new morning, a new day', 'it is never too late to begin', 'looking forward to the future', 'possibility and new beginnings'],
  play: ['that is funny', 'just having fun with it', 'joking around, lighthearted', 'laughing at myself', 'playful and silly'],
};
const VIBE_ENERGY = { calm: -1, warmth: -0.5, hope: -0.3, practical: 0, play: 0.3, struggle: 0.1, triumph: 0.7, grind: 1, hype: 1 };
const VIBE_LABEL = { calm: 'calm', grind: 'effort', triumph: 'strong', warmth: 'warm', struggle: 'struggle', hype: 'hype', practical: 'everyday', hope: 'hopeful', play: 'light' };
function softmax(xs, T) { const m = Math.max(...xs), e = xs.map(x => Math.exp((x - m) * T)), z = e.reduce((a, b) => a + b, 0); return e.map(x => x / z); }
function cosv(a, b) { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.sqrt(x * y + 1e-12); }
/** For each line: what it means (the line, and the whole thought around it) and how it feels; same for every visual. */
async function vibeContext(words, items) {
  const ph = phrases(words, 10), names = Object.keys(VIBES);
  const sentStart = i => { while (i > 0 && !(/[.?!]["')]*$/.test(words[i - 1].w))) i--; return i; };
  const sentEnd = i => { while (i < words.length - 1 && !/[.?!]["')]*$/.test(words[i].w)) i++; return i; };
  const thoughts = ph.map(p => { const a = sentStart(p.a); let b = sentEnd(p.b); if (b - a < 10 && b + 1 < words.length) b = sentEnd(b + 1); return words.slice(a, Math.min(b + 1, a + 45)).map(w => w.w).join(' '); });
  const protos = names.flatMap(k => VIBES[k].map(t => [k, t])), pV = await embed(protos.map(x => x[1]));
  const [lineV, thV, itV] = [await embed(ph.map(p => p.text)), await embed(thoughts), await embed(items.map(itemText))];
  const vbs = v => names.map(k => { const xs = protos.map((x, i) => x[0] === k ? dot(v, pV[i]) : -1).filter(x => x > -1).sort((a, b) => b - a); return 0.7 * xs[0] + 0.3 * xs[1]; });
  const itemVibe = itV.map(v => softmax(vbs(v), 14));
  const lines = new Map();
  ph.forEach((p, k) => {
    const wl = p.b - p.a < 7 ? 0.65 : 0.45;                                        // a short line carries its own feel
    const lv = vbs(lineV[k]), tv = vbs(thV[k]), vib = softmax(lv.map((x, i) => wl * x + (1 - wl) * tv[i]), 14);
    const sem = new Float32Array(items.length); items.forEach((_, j) => { sem[j] = 0.55 * dot(lineV[k], itV[j]) + 0.45 * dot(thV[k], itV[j]); });
    lines.set(`${p.a}-${p.b}`, { vib, sem, energy: vib.reduce((a, x, i) => a + x * VIBE_ENERGY[names[i]], 0), top: names[vib.indexOf(Math.max(...vib))] });
  });
  const index = new Map(items.map((it, j) => [it.id, j]));
  const itemInfo = items.map((it, j) => ({ vib: itemVibe[j], top: names[itemVibe[j].indexOf(Math.max(...itemVibe[j]))],
    energy: (it.energy === 'high' ? 0.8 : it.energy === 'low' ? -0.8 : 0) * 0.5 + 0.5 * itemVibe[j].reduce((a, x, i) => a + x * VIBE_ENERGY[names[i]], 0) }));
  return { lines, index, itemInfo, names };
}
function targetVisuals(total, amount = 1) { return total < 9 ? 0 : clamp(Math.round(total / 7 * amount), 1, 12); }
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
    if (broll.filter(p => !p.off).reduce((a, p) => a + p.e - p.s, 0) + (r.e - r.s) > budget) return `would push full-screen clips past ${Math.round(B.maxFrac * 100)}% of the reel`;
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
/** Fill the reel to a natural rhythm with visuals that fit what you're saying and how it feels.
 *  Each line is scored against every clip and photo by: meaning (the line and the whole thought), feel (calm, effort,
 *  warm, hopeful...), energy (how you're saying it against how busy the shot is), and only lightly by matching words.
 *  Visuals go where they fit best, spaced out, never over the opening or closing line. */
function fillVisuals(words, items, photos, add, all, cuts, total, cfg, log, firstEnd, lastSentence, ctx = {}) {
  const B = cfg.broll, want = targetVisuals(total, B.amount || 1);
  if (all().length >= want || !words.length) return;
  const pool = [
    ...items.filter(it => !it.others_visible && it.kind !== 'photo' && (it.kind === 'still' || (it.duration || 0) >= B.min - 0.05)).map(it => ({ it, type: 'clip' })),
    ...(cfg.pop.on ? photos.filter(it => !it.others_visible).map(it => ({ it, type: 'photo' })) : []),
  ];
  if (!pool.length) return;
  const ph = phrases(words, 10), V = ctx.vibe;
  const terms = pool.map(({ it }) => { const t = itemTerms(it); String(it.subject || '').toLowerCase().split(/[^a-z']+/).forEach(x => { if (x.length >= 4 && !STOP.has(x)) t.words.push(stem(x)); }); return [...new Set(t.words)]; });
  const df = {}; for (const t of terms) for (const w of t) df[w] = (df[w] || 0) + 1;
  const idf = w => Math.log((pool.length + 1) / ((df[w] || 0) + 0.5));
  const cueSets = Object.fromEntries(Object.entries(CUES).map(([k, v]) => [k, new Set(v.map(stem))]));
  const THR = V ? 1.6 : 2.1;
  function scoreAll(sl) {
    const key = `${sl.a}-${sl.b}`, L = V && V.lines.get(key), dv = ctx.delivery && ctx.delivery.get(key);
    const inSlot = new Set(words.slice(sl.a, sl.b + 1).map(w => stem(w.w))), near = new Set(words.slice(Math.max(0, sl.a - 6), Math.min(words.length, sl.b + 7)).map(w => stem(w.w)));
    const cueHit = k => !cueSets[k] ? 0 : [...cueSets[k]].filter(c => inSlot.has(c)).length + 0.4 * [...cueSets[k]].filter(c => near.has(c) && !inSlot.has(c)).length;
    const prev = all().filter(p => p.e <= sl.s).sort((a, b) => b.e - a.e)[0];
    const said = t => { const w = words.slice(sl.a, sl.b + 1).find(x => stem(x.w) === t); return w ? norm(w.w) : t; };
    const lineEnergy = (L ? L.energy : 0) + (dv ? clamp(dv.loud / 4, -0.6, 0.6) + (dv.emphatic ? 0.3 : 0) - (dv.slow ? 0.3 : 0) : 0);
    return pool.map(({ it, type }, k) => {
      if (all().some(p => p.id === it.id)) return null;
      const hit = terms[k].filter(t => inSlot.has(t)), hitNear = terms[k].filter(t => near.has(t) && !inSlot.has(t));
      const lex = Math.min(1.6, hit.reduce((a, t) => a + idf(t), 0) * 0.35 + hitNear.reduce((a, t) => a + idf(t), 0) * 0.12);
      const cat = it.category; let sc, why, vtop = null;
      if (L) {
        const j = V.index.get(it.id), I = V.itemInfo[j], sem = L.sem[j], vm = cosv(L.vib, I.vib);
        const conf = clamp((Math.max(...L.vib) - 0.14) / 0.22, 0.2, 1);                   // a vague line can't lean on feel alone
        sc = 5 * Math.max(-0.1, sem - 0.16) + 2.6 * (vm - 0.45) * conf + lex - 0.45 * Math.abs(lineEnergy - I.energy);
        if (I.top === 'hype' && L.vib[V.names.indexOf('hype')] < 0.16) sc -= 2;              // flashy shots only when you're talking about that
        if (prev && prev.vibeTop && prev.vibeTop === I.top) sc -= 0.3;
        why = hit.length ? `fits "${hit.slice(0, 2).map(said).join('", "')}"` : sem > 0.35 ? 'matches what you mean' : `${VIBE_LABEL[L.top]} feel`;
        if (cueSets[cat]) sc += Math.min(0.9, cueHit(cat) * 0.45);                     // feeling words that fit this kind of shot
        vtop = I.top;
      } else {
        sc = hit.reduce((a, t) => a + idf(t), 0) * (type === 'photo' ? 1 : 0.8) + hitNear.reduce((a, t) => a + idf(t), 0) * 0.35;
        if (cueSets[cat]) sc += cueHit(cat) * 1.6;
        if (cat === 'contrast' || it.energy === 'high') sc += cueHit('contrast') > 0.9 ? 1 : -4;
        why = hit.length ? `fits "${hit.slice(0, 2).map(said).join('", "')}"` : `${String(cat).replace('_', ' ')} shot`;
      }
      if (it.source !== 'stock') sc += 0.5;                                                  // your own footage first
      if (dv && dv.emphatic) sc += type === 'photo' ? 0.35 : -0.35;                          // you're punching the line: stay on camera
      if (prev && prev.item.category === cat) sc -= 0.35;
      if (prev && (prev.item.kind === 'photo') === (type === 'photo')) sc -= 0.15;           // gently alternate clips and photos
      sc += ((k * 2654435761 + Math.round(sl.s * 10)) % 97) / 600;
      return { it, type, sc, why, vibeTop: vtop };
    }).filter(Boolean).sort((a, b) => b.sc - a.sc);
  }
  const slots = ph.filter(p => p.s >= Math.max(B.autoMinStart, firstEnd - 0.1) && p.s < lastSentence - 0.4 && p.e - p.s >= 0.6).map(p => {
    let sc = 0; const d = p.e - p.s;
    if (cuts.some(c => c >= p.s - 0.15 && c <= p.s + Math.min(2.4, d))) sc += 2.2;            // covers a jump cut
    if (d >= 1.2 && d <= 3.6) sc += 0.8;
    const r = scoreAll(p); const best = r.length ? r[0].sc : -9;
    return { ...p, base: sc + 1.4 * clamp(best - THR, -1, 2) };                               // go where something really fits
  });
  let guard = 0;
  while (all().length < want && slots.length && guard++ < 40) {
    const centers = all().map(p => (p.s + p.e) / 2);
    slots.forEach(sl => { const c = (sl.s + sl.e) / 2, d = centers.length ? Math.min(...centers.map(x => Math.abs(x - c))) : total / 2; sl.score = sl.base + Math.min(4, d / Math.max(2, total / (want + 1)) * 3) - (d < 3.5 ? 6 : 0); });
    slots.sort((a, b) => b.score - a.score);
    const sl = slots.shift(); if (sl.score < -2) break;
    const ranked = scoreAll(sl).filter(c => c.sc >= THR);
    for (const c of ranked.slice(0, 6)) {
      const err = add(c.type, c.it, sl.s, c.why, sl.text, true);
      if (!err) { const last = all().find(p => p.id === c.it.id); if (last) last.vibeTop = c.vibeTop; break; }
      if (/past \d+%/.test(err)) continue;
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
function capLum(h) { const n = parseInt(String(h).replace('#', ''), 16); if (isNaN(n)) return 1; const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(n >> 16 & 255) + 0.7152 * f(n >> 8 & 255) + 0.0722 * f(n & 255); }
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
  const sp = ctx.measureText(' ').width + 0.5, C = cfg.cap, style = C.style || 'outline';
  const ws = chunk.words.map(w => { w = C.upper ? w.toUpperCase() : w; return { w, wd: ctx.measureText(w).width }; });
  const total = ws.reduce((a, b) => a + b.wd, 0) + sp * (ws.length - 1);
  const lf = lift == null ? (chunk.up ? 1 : 0) : lift;
  const y = Math.round(OUT_H * (cfg.cap.baseline + (cfg.cap.baselineUp - cfg.cap.baseline) * lf)); let x = OUT_W / 2 - total / 2;
  const pos = ws.map(w => { const p = x; x += w.wd + sp; return p; });
  const fs = Math.round(C.size / 1.21), dark = capLum(C.color || '#FFFFFF') < 0.3;   // dark text gets a light outline / box
  const edge = a => dark ? `rgba(255,255,255,${a})` : `rgba(0,0,0,${a})`;
  if (style === 'box') {
    ctx.fillStyle = dark ? 'rgba(255,255,255,0.88)' : 'rgba(0,0,0,0.62)'; roundRect(ctx, pos[0] - 0.34 * fs, y - 0.92 * fs, total + 0.68 * fs, 1.24 * fs, 0.24 * fs); ctx.fill();
  } else if (style === 'shadow') {
    ctx.shadowColor = edge(0.8); ctx.shadowBlur = 14; ctx.shadowOffsetY = 3;
  } else { // soft drop shadow, then dark outline
    ctx.fillStyle = edge(0.45); ctx.strokeStyle = edge(0.45); ctx.lineWidth = 6.4;
    ws.forEach((w, k) => { ctx.strokeText(w.w, pos[k] + 2.2, y + 2.2); ctx.fillText(w.w, pos[k] + 2.2, y + 2.2); });
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.85)' : 'rgba(20,20,20,0.70)'; ctx.lineWidth = 6.4;
    ws.forEach((w, k) => ctx.strokeText(w.w, pos[k], y));
  }
  ws.forEach((w, k) => { ctx.fillStyle = k === chunk.hl ? C.accent : (C.color || '#FFFFFF'); ctx.fillText(w.w, pos[k], y); });
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

/** Sequential frame access by presentation time; frames are borrowed until the next call.
 *  Survives decoder failures: a corrupt packet, a decoder the browser reclaimed (another tab, low memory),
 *  or a hardware decoder that gives up. It rebuilds the decoder, switches to software decoding if the hardware
 *  one keeps failing, skips a damaged stretch to the next keyframe, and holds the last good picture meanwhile. */
class VideoSource {
  constructor(movie) {
    this.mv = movie; this.samples = movie.trakV.samples; this.ts = this.samples[0].timescale;
    this.fw = new FileWindow(movie.file);
    this.config = { codec: movie.v.codec, codedWidth: movie.codedW, codedHeight: movie.codedH, description: describe(movie, movie.trakV) };
    this.fails = 0; this.software = false; this.hold = null;
  }
  async open() {
    if (!('VideoDecoder' in root)) throw new Error('This browser cannot decode video frames (WebCodecs missing).');
    let sup = await VideoDecoder.isConfigSupported(this.config);
    if (!sup.supported) { sup = await VideoDecoder.isConfigSupported({ ...this.config, hardwareAcceleration: 'prefer-software' }); if (sup.supported) this.software = true; }
    if (!sup.supported) throw new Error(`This device can't decode ${this.config.codec} video.`);
    this._make(); this.i = 0;
  }
  _make() {
    this.out = []; this.cur = null; this.err = null; this.flushed = false; this.lastOut = performance.now();
    this.dec = new VideoDecoder({ output: f => { this.out.push(f); this.lastOut = performance.now(); }, error: e => { this.err = e || new Error('decoder error'); } });
    this.dec.configure(this.software ? { ...this.config, hardwareAcceleration: 'prefer-software' } : this.config);
  }
  _keyBefore(t) {
    const target = t * this.ts + this.mv.cts0; let k = 0;
    for (let i = 0; i < this.samples.length; i++) { if (this.samples[i].is_sync && this.samples[i].cts <= target) k = i; if (this.samples[i].cts > target + this.ts * 2) break; }
    return k;
  }
  _keyAfter(i) { for (let j = i + 1; j < this.samples.length; j++) if (this.samples[j].is_sync) return j; return this.samples.length; }
  _drop() { try { this.out.forEach(f => f.close()); if (this.cur) this.cur.close(); } catch (e) { /* already closed */ } this.out = []; this.cur = null; try { if (this.dec && this.dec.state !== 'closed') this.dec.close(); } catch (e) { /* closed */ } }
  seek(t) { this._drop(); this._make(); this.i = this._keyBefore(t); }
  /** The decoder failed while working toward time t: rebuild it and carry on. */
  async _recover(t) {
    this.fails++;
    const D = DIAG.videoRecover = DIAG.videoRecover || { count: 0, software: false, skipped: 0, last: '' };
    D.count++; D.last = String(this.err && (this.err.name + ': ' + this.err.message) || 'stalled').slice(0, 120);
    if (this.fails > 60) throw new Error("This video has too many damaged frames to read. Try exporting it again from your camera app.");
    if (!this.software && this.fails >= 2 && this.swOk !== false) {   // the hardware decoder keeps failing: use the software one, if this browser has it
      try { this.swOk = (await VideoDecoder.isConfigSupported({ ...this.config, hardwareAcceleration: 'prefer-software' })).supported; } catch (e) { this.swOk = false; }
      if (this.swOk) { this.software = true; D.software = true; }
    }
    const from = this.badAt === this._keyBefore(t) ? this._keyAfter(this._keyBefore(t)) : this._keyBefore(t);   // failed here before: skip this stretch
    if (from !== this._keyBefore(t)) D.skipped++;
    this.badAt = this._keyBefore(t);
    this._drop(); this._make(); this.i = from;
  }
  async _feed(n) {
    for (let c = 0; c < n && this.i < this.samples.length; c++) {
      const s = this.samples[this.i++];
      const data = await this.fw.read(s.offset, s.size);
      if (this.err || this.dec.state !== 'configured') return;
      this.dec.decode(new EncodedVideoChunk({ type: s.is_sync ? 'key' : 'delta', timestamp: Math.round((s.cts - this.mv.cts0) / this.ts * 1e6), duration: Math.round(s.duration / this.ts * 1e6), data }));
    }
  }
  async frameAt(t) {
    const last = this.lastT == null ? 0 : this.lastT;
    if (t < last - 0.5 / FPS || t > last + 3) this.seek(t);
    this.lastT = t;
    const lim = (t + 0.25 / FPS) * 1e6;
    for (let guard = 0; guard < 100000; guard++) {
      if (this.err || (this.dec && this.dec.state === 'closed')) { this.err = this.err || new Error('decoder closed'); await this._recover(t); continue; }
      this.out.sort((a, b) => a.timestamp - b.timestamp);
      while (this.out.length && this.out[0].timestamp <= lim) { if (this.cur) this.cur.close(); this.cur = this.out.shift(); }
      if (this.out.length) break;
      if (this.i >= this.samples.length) {
        if (!this.flushed) { this.flushed = true; try { await this.dec.flush(); } catch (e) { if (!this.err) this.err = e; } continue; }
        break;
      }
      if (this.dec.decodeQueueSize < 8) await this._feed(4);
      else if (performance.now() - this.lastOut > 8000) { this.err = new Error('decoder stalled'); continue; }   // fed but silent for 8 s
      else await sleep(1);
      if (!this.out.length) await sleep(0);
    }
    const f = this.cur || this.out[0] || null;
    if (f) { this.fails = Math.max(0, this.fails - 0.02); if (this.hold) this.hold.close(); try { this.hold = f.clone(); } catch (e) { this.hold = null; } return f; }
    return this.hold;   // a skipped stretch shows the last good picture instead of a black flash
  }
  close() { this._drop(); if (this.hold) { try { this.hold.close(); } catch (e) { /* closed */ } this.hold = null; } }
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
      const parts = []; let err = null, outSr = 0, bad = 0, lastOutTs = -Infinity;
      const onOut = d => {
        const n = d.numberOfFrames, nc = d.numberOfChannels, L = new Float32Array(n), R = new Float32Array(n), tmp = new Float32Array(n);
        if (!outSr) outSr = d.sampleRate || sr;
        if (opts.stereo) { d.copyTo(L, { planeIndex: 0, format: 'f32-planar' }); if (nc > 1) d.copyTo(R, { planeIndex: 1, format: 'f32-planar' }); else R.set(L); }
        else { for (let c = 0; c < nc; c++) { d.copyTo(tmp, { planeIndex: c, format: 'f32-planar' }); for (let i = 0; i < n; i++) L[i] += tmp[i] / nc; } }
        parts.push({ ts: d.timestamp - t0, L, R: opts.stereo ? R : null }); lastOutTs = Math.max(lastOutTs, d.timestamp); d.close();
      };
      // A damaged packet (or a decoder the browser reclaims) shouldn't sink a long video: rebuild the decoder,
      // leave that packet as a few milliseconds of silence, and keep going.
      const make = () => { const d = new AudioDecoder({ output: onOut, error: e => { err = e || new Error('audio decoder error'); } }); d.configure(cfg); return d; };
      let dec = make(), lastSent = -1, restartAt = -1;
      const tsOf = k => Math.round(trak.samples[k].cts / ts * 1e6);
      const fw = new FileWindow(movie.file);
      for (let i = 0; i < trak.samples.length; i++) {
        if (err || dec.state === 'closed') {
          bad++; DIAG.audioRecover = { count: bad, last: String(err && err.message || 'closed').slice(0, 100) };
          if (bad > 200 || (bad > 20 && bad > trak.samples.length * 0.05)) throw err || new Error('audio decoder kept failing');
          try { if (dec.state !== 'closed') dec.close(); } catch (e) { /* closed */ }
          err = null; dec = make();
          let from = lastSent; while (from > 0 && tsOf(from - 1) > lastOutTs) from--;   // resume after the last packet that came out
          if (from === restartAt) from++;                                               // failed at the same spot twice: skip that packet
          restartAt = from; i = Math.max(0, from);
          if (i >= trak.samples.length) break;
        }
        const s = trak.samples[i];
        dec.decode(new EncodedAudioChunk({ type: 'key', timestamp: Math.round(s.cts / ts * 1e6), duration: Math.round(s.duration / ts * 1e6), data: await fw.read(s.offset, s.size) }));
        lastSent = i;
        while (dec.decodeQueueSize > 64 && !err && dec.state !== 'closed') await sleep(1);
        if (i % 500 === 0) { check(signal); await tick(); }
      }
      try { await dec.flush(); } catch (e) { if (!err) err = e; }
      try { if (dec.state !== 'closed') dec.close(); } catch (e) { /* closed */ }
      if (err && !parts.length) throw err;
      if (!parts.length) throw new Error('the decoder returned no audio');
      outSr = outSr || sr;
      // timestamps are usable if they increase and roughly agree with the running sample count
      let run = parts[0].ts, tsOk = parts[0].ts >= 0 && parts[0].ts < 0.25e6;
      for (let k = 1; k < parts.length && tsOk; k++) { run += parts[k - 1].L.length / outSr * 1e6; if (bad && parts[k].ts > run && parts[k].ts - run < 3e6) run = parts[k].ts;   /* a skipped packet leaves a short gap */
        if (parts[k].ts < parts[k - 1].ts || Math.abs(parts[k].ts - run) > 0.25e6) { tsOk = false; break; } }
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
  if (!pcm) { // fallback: let the browser decode the whole file (only sensible for files that fit in memory)
    if (movie.file.size > 400e6) throw new Error("Couldn't read the sound in this video. Try exporting it again from your camera or editing app, or trim it into shorter parts.");
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
  let asr, used = which;   // asr is reloaded if the model's memory gets corrupted
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
  DIAG.asrFailed = []; DIAG.asrRecovered = 0;
  let calls = 0;
  const runAsr = (x, o) => { if (root.__asrFault && ++calls % root.__asrFault === 0) return Promise.reject(new Error('test: injected model failure')); return asr(x, o); };
  // The model can fail on a piece of audio: a word-timing step that trips over an odd segment, or its memory running
  // out partway through a long video. One bad piece shouldn't sink the whole video, so each piece is retried, the model
  // reloaded if its memory got corrupted, split in half, then read with sentence-level timing; only then is it skipped.
  const asrRange = async (t0, t1, depth = 0) => {
    const x = a16.subarray(Math.round(t0 * R16), Math.round(t1 * R16));
    let res;
    try { res = await runAsr(x, { return_timestamps: 'word' }); }
    catch (e) {
      if (e instanceof Cancelled) throw e;
      const msg = String(e && e.message || e); DIAG.asrRecovered++; DIAG.asrLastError = msg.slice(0, 160);
      if ((root.WebAssembly && e instanceof WebAssembly.RuntimeError) || /memory|out of bounds|abort|unreachable|OrtRun|session/i.test(msg)) {
        delete _whisperBy[used]; try { asr = await loadWhisper(null, used); } catch (e2) { /* keep the old one and let the retries decide */ }
      }
      check(signal);
      if (depth === 0) return asrRange(t0, t1, 1);
      if (t1 - t0 > 4 && depth < 3) { const m = splitAt(t0, t1); return [...await asrRange(t0, m, depth + 1), ...await asrRange(m, t1, depth + 1)]; }
      try {   // last try: sentence-level times, words spread across each sentence by length
        const r2 = await runAsr(x, { return_timestamps: true }), ws = [];
        for (const c of r2.chunks || []) {
          let [cs, ce] = c.timestamp || [0, (t1 - t0)]; if (cs == null) cs = 0; if (ce == null) ce = Math.min(t1 - t0, cs + 2);
          const words = String(c.text || '').trim().split(/\s+/).filter(Boolean), chars = words.reduce((a, w) => a + w.length + 1, 0) || 1; let at = cs;
          for (const w of words) { const d = (ce - cs) * (w.length + 1) / chars; ws.push({ w, s: +(t0 + at).toFixed(3), e: +Math.min(t1, t0 + at + d).toFixed(3) }); at += d; }
        }
        return ws;
      } catch (e2) { if (e2 instanceof Cancelled) throw e2; DIAG.asrFailed.push([+t0.toFixed(1), +t1.toFixed(1)]); return []; }
    }
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
  const speech = vad(audio48), redo = [];
  const spoken = (a, b) => speech.reduce((x, [s, e]) => x + Math.max(0, Math.min(b, e) - Math.max(a, s)), 0);
  const covered = (ws, a, b) => speech.reduce((x, [s, e]) => { const lo = Math.max(a, s), hi = Math.min(b, e); if (hi <= lo) return x;
    let c = 0, cur = lo; for (const w of ws) { const ws0 = Math.max(cur, w.s - 0.2), we = Math.min(hi, w.e + 0.2); if (we > ws0) { c += we - ws0; cur = we; } } return x + c; }, 0);
  // Whisper sometimes loops ("I was going to say that I was going to say that...") or skips a stretch.
  // Spot both, and transcribe that piece again in two halves.
  const broken = (ws, a, b) => {
    const sp = spoken(a, b); if (sp < 1) return false;
    let stacked = 0; for (let i = 1; i < ws.length; i++) if (ws[i].s - ws[i - 1].s < 0.02) stacked++;
    const grams = new Map(); let rep = 0;
    for (let i = 0; i + 4 <= ws.length; i++) { const g = ws.slice(i, i + 4).map(w => norm(w.w)).join(' '); const c = (grams.get(g) || 0) + 1; grams.set(g, c); if (c > 2) rep++; }
    return ws.length / Math.max(1, b - a) > 5.5 || stacked > 6 || rep > 2 || covered(ws, a, b) / sp < 0.55;
  };
  const splitAt = (a, b) => { let best = (a + b) / 2, bv = Infinity; for (let t = a + (b - a) * 0.3; t <= a + (b - a) * 0.7; t += 0.01) { const i = Math.round(t * 100); let m = -Infinity; for (let k = -15; k <= 15; k++) m = Math.max(m, db[clamp(i + k, 0, db.length - 1)]); if (m < bv) { bv = m; best = t; } } return best; };
  const fixLoops = ws => {
    const o = [];
    for (const w of ws) {
      if (o.length >= 2 && w.s - o[o.length - 1].s < 0.02 && o[o.length - 1].s - o[o.length - 2].s < 0.02) continue;   // words stacked on one instant
      o.push(w);
      for (let n = 1; n <= 6; n++) {                                                  // the same n words three times in a row: keep one
        if (o.length < 3 * n) break;
        const g = k => o.slice(o.length - k * n, o.length - (k - 1) * n).map(x => norm(x.w)).join(' ');
        if (g(1) === g(2) && g(2) === g(3)) { o.splice(o.length - 2 * n, 2 * n); break; }
      }
    }
    return o;
  };
  DIAG.asrRetries = [];
  const asrSafe = async (a, b, depth = 0) => {
    let ws = await asrRange(a, b);
    if (depth < 2 && b - a > 6 && broken(ws, a, b)) {
      DIAG.asrRetries.push([+a.toFixed(1), +b.toFixed(1)]);
      const m = splitAt(a, b), both = [...await asrSafe(a, m, depth + 1), ...await asrSafe(m, b, depth + 1)];
      if (!broken(both, a, b) || covered(both, a, b) >= covered(ws, a, b)) ws = both;
    }
    return fixLoops(ws);
  };
  const lost = () => DIAG.asrFailed.reduce((a, [x, y]) => a + spoken(x, y), 0);
  for (let k = 0; k < segs.length; k++) {
    check(signal);
    if (k >= 3 && lost() > 0.5 * spoken(0, segs[k][0])) throw new Error('The speech model keeps failing on this video. Try again, or switch Settings → Speech → "For long videos" to the other option.');
    onProgress && onProgress(0.58 + 0.36 * k / segs.length, segs.length > 1 ? `Transcribing what you said (part ${k + 1} of ${segs.length})` : 'Transcribing what you said');
    out.push(...await asrSafe(segs[k][0], segs[k][1]));
  }
  // Safety net: any stretch of clear speech that came back with no words gets transcribed again on its own.
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
  let detector = 'none', src = null;
  try {
    const { ort, sess } = await loadYunet(); detector = 'YuNet';
    const cv = new OffscreenCanvas(640, 640), ctx = cv.getContext('2d', { willReadFrequently: true });
    const k = 640 / Math.max(W, H), input = new Float32Array(3 * 640 * 640);
    src = new VideoSource(movie); await src.open();
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
  } catch (e) { if (e instanceof Cancelled) throw e; console.warn('face tracking failed', e); detector = 'none (' + e.message + ')'; }
  finally { if (src) src.close(); }
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
const GENRES = {
  lofi:      { label: 'Lo-fi', bpm: 78, prog: [[2, 5, 9, 12], [7, 11, 14, 17], [0, 4, 7, 11], [9, 12, 16, 19]], bars: 1, keys: 'rhodes', drums: 'lofi', swing: 0.16, bass: 'round', lp: 2600, crackle: true, about: 'Dusty keys, lazy drums' },
  chill:     { label: 'Chill', bpm: 84, prog: [[0, 4, 7, 11], [9, 12, 16, 19], [5, 9, 12, 16], [7, 11, 14, 17]], bars: 2, pad: 'warm', keys: 'piano', density: 0.35, drums: 'pulse', bass: 'sine', about: 'Soft pads, a gentle pulse' },
  piano:     { label: 'Piano', bpm: 68, prog: [[0, 4, 7, 11, 14], [5, 9, 12, 16], [9, 12, 16, 19], [5, 9, 12, 14]], bars: 2, pad: 'soft', keys: 'piano', density: 0.5, bass: 'sine', about: 'Felt piano, slow and thoughtful' },
  cinematic: { label: 'Cinematic', bpm: 72, minor: true, prog: [[0, 3, 7], [8, 12, 15], [3, 7, 10], [10, 14, 17]], bars: 2, pad: 'strings', keys: 'ostinato', drums: 'cinematic', bass: 'sine', about: 'Strings and big low hits' },
  acoustic:  { label: 'Acoustic', bpm: 96, prog: [[0, 4, 7], [7, 11, 14], [9, 12, 16], [5, 9, 12]], bars: 1, keys: 'guitar', drums: 'acoustic', bass: 'round', about: 'Strummed guitar, light percussion' },
  upbeat:    { label: 'Upbeat', bpm: 112, prog: [[0, 4, 7], [7, 11, 14], [9, 12, 16], [5, 9, 12]], bars: 1, pad: 'soft', keys: 'pluck', drums: 'pop', bass: 'eighths', about: 'Bright plucks, four-on-the-floor' },
  hiphop:    { label: 'Hip-hop', bpm: 88, minor: true, prog: [[0, 3, 7, 10], [8, 12, 15], [3, 7, 10, 14], [10, 14, 17]], bars: 1, keys: 'stabs', drums: 'boombap', swing: 0.1, bass: 'sub', about: 'Boom-bap drums, dark keys' },
};
const GENRE_ALIAS = { warm: 'chill', steady: 'lofi', reflective: 'piano', generated: 'chill' };
const MOODS = GENRES;
const DRUMS = { // 16 steps a bar
  lofi:      { k: [0, 7, 10], s: [4, 12], h: [0, 2, 4, 6, 8, 10, 12, 14] },
  boombap:   { k: [0, 3, 10], s: [4, 12], h: [0, 2, 4, 6, 8, 10, 12, 14], oh: [14] },
  pop:       { k: [0, 4, 8, 12], c: [4, 12], h: [2, 6, 10, 14], sh: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] },
  acoustic:  { k: [0, 8], r: [4, 12], sh: [0, 2, 4, 6, 8, 10, 12, 14] },
  pulse:     { p: [0, 8] },
  cinematic: { t: [0, 10] },
};
/** An original instrumental, written to the exact length of the reel, in the chosen genre. */
async function composeMusic(seconds, genre = 'chill', seed = Date.now()) {
  genre = GENRE_ALIAS[genre] || genre;
  const G = GENRES[genre] || GENRES.chill, r = rng(seed);
  const roots = [0, 2, 3, 5, 7, 9], key = roots[Math.floor(r() * roots.length)] + 48;
  const beat = 60 / G.bpm, bar = beat * 4, step = beat / 4, chordLen = bar * G.bars, len = Math.ceil((seconds + 1.5) * SR);
  const ac = new OfflineAudioContext(2, len, SR);
  const bus = ac.createGain(); bus.gain.value = 0.8;
  if (G.lp) { const f = ac.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = G.lp; f.Q.value = 0.5; bus.connect(f).connect(ac.destination); }
  else bus.connect(ac.destination);
  const ir = ac.createBuffer(2, Math.round(2.4 * SR), SR);
  for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); for (let i = 0; i < d.length; i++) d[i] = (r() * 2 - 1) * Math.pow(1 - i / d.length, 3.2) * 0.6; }
  const verb = ac.createConvolver(); verb.buffer = ir; const wet = ac.createGain(); wet.gain.value = genre === 'cinematic' ? 0.42 : 0.26;
  verb.connect(wet).connect(bus);
  const dry = ac.createGain(); dry.gain.value = 0.85; dry.connect(bus);
  const send = ac.createGain(); send.gain.value = 1; send.connect(verb); dry.connect(send);
  const drumBus = ac.createGain(); drumBus.gain.value = 0.9; drumBus.connect(bus); const drumSend = ac.createGain(); drumSend.gain.value = 0.12; drumBus.connect(drumSend).connect(verb);
  const hz = n => 440 * Math.pow(2, (n - 69) / 12);
  const noise = ac.createBuffer(1, SR * 2, SR); { const d = noise.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = r() * 2 - 1; }
  const panTo = (node, p, dest) => { const pn = ac.createStereoPanner(); pn.pan.value = clamp(p, -1, 1); node.connect(pn).connect(dest || dry); };
  const envG = (t, a, peak, d, hold = 0) => { const g = ac.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(peak, t + a); if (hold) g.gain.setValueAtTime(peak, t + a + hold); g.gain.exponentialRampToValueAtTime(0.0005, t + a + hold + d); return g; };
  const noiseHit = (t, vel, type, f, q, d, dest = drumBus, a = 0.001) => { const n = ac.createBufferSource(); n.buffer = noise; n.playbackRate.value = 0.9 + r() * 0.2; const bf = ac.createBiquadFilter(); bf.type = type; bf.frequency.value = f; bf.Q.value = q; const g = envG(t, a, vel, d); n.connect(bf).connect(g).connect(dest); n.start(t, r() * 1.5); n.stop(t + a + d + 0.05); return g; };
  const kick = (t, vel, low = 48) => { const o = ac.createOscillator(); o.frequency.setValueAtTime(140, t); o.frequency.exponentialRampToValueAtTime(low, t + 0.11); const g = envG(t, 0.003, vel, 0.42); o.connect(g).connect(drumBus); o.start(t); o.stop(t + 0.5); };
  const snare = (t, vel) => { noiseHit(t, vel * 0.8, 'bandpass', 1900, 0.7, 0.17); const o = ac.createOscillator(); o.type = 'triangle'; o.frequency.setValueAtTime(200, t); o.frequency.exponentialRampToValueAtTime(150, t + 0.08); const g = envG(t, 0.002, vel * 0.5, 0.09); o.connect(g).connect(drumBus); o.start(t); o.stop(t + 0.15); };
  const hat = (t, vel, open) => noiseHit(t, vel, 'highpass', 7500, 0.7, open ? 0.24 : 0.035);
  const clap = (t, vel) => { for (let i = 0; i < 3; i++) noiseHit(t + i * 0.011, vel * (i === 2 ? 1 : 0.6), 'bandpass', 1300, 1.1, i === 2 ? 0.14 : 0.012); };
  const shaker = (t, vel) => noiseHit(t, vel, 'highpass', 5200, 0.6, 0.05, drumBus, 0.012);
  const rim = (t, vel) => { const o = ac.createOscillator(); o.type = 'triangle'; o.frequency.value = 820; const g = envG(t, 0.001, vel, 0.035); o.connect(g).connect(drumBus); o.start(t); o.stop(t + 0.06); };
  const tom = (t, vel) => { kick(t, vel, 38); noiseHit(t, vel * 0.25, 'lowpass', 300, 0.5, 0.9); };
  const piano = (t, note, vel, d = 1.6) => {
    const o1 = ac.createOscillator(), o2 = ac.createOscillator(), f = ac.createBiquadFilter(), g = envG(t, 0.006, vel, d);
    o1.type = 'triangle'; o2.type = 'sine'; o1.frequency.value = hz(note); o2.frequency.value = hz(note) * 2.001; f.type = 'lowpass'; f.frequency.value = 2400;
    const g2 = ac.createGain(); g2.gain.value = 0.25; o2.connect(g2).connect(f); o1.connect(f); f.connect(g); panTo(g, (r() - 0.5) * 0.6);
    o1.start(t); o2.start(t); o1.stop(t + d + 0.1); o2.stop(t + d + 0.1);
  };
  const rhodes = (t, note, vel, d) => {
    const o1 = ac.createOscillator(), o2 = ac.createOscillator(), g = envG(t, 0.01, vel, d), trem = ac.createOscillator(), tg = ac.createGain(), am = ac.createGain();
    o1.frequency.value = hz(note); o2.frequency.value = hz(note) * 3.0; const b = ac.createGain(); b.gain.value = 0.12; o2.connect(b).connect(g); o1.connect(g);
    trem.frequency.value = 4.5; tg.gain.value = 0.18; am.gain.value = 0.82; trem.connect(tg).connect(am.gain); g.connect(am); panTo(am, (note % 7 - 3) * 0.08);
    [o1, o2, trem].forEach(o => { o.start(t); o.stop(t + d + 0.1); });
  };
  const pluck = (t, note, vel) => {
    const o = ac.createOscillator(), f = ac.createBiquadFilter(), g = envG(t, 0.003, vel, 0.28); o.type = 'square'; o.frequency.value = hz(note);
    f.type = 'lowpass'; f.Q.value = 2; f.frequency.setValueAtTime(3800, t); f.frequency.exponentialRampToValueAtTime(500, t + 0.18);
    o.connect(f).connect(g); panTo(g, (r() - 0.5) * 0.7); o.start(t); o.stop(t + 0.35);
  };
  const ks = new Map();
  const guitar = (t, note, vel) => {
    if (!ks.has(note)) {
      const n = Math.round(2.2 * SR), N = Math.max(2, Math.round(SR / hz(note))), buf = ac.createBuffer(1, n, SR), d = buf.getChannelData(0), ring = new Float32Array(N);
      for (let i = 0; i < N; i++) ring[i] = r() * 2 - 1;
      for (let i = 0, k = 0; i < n; i++) { const nx = (k + 1) % N, v = ring[k]; d[i] = v; ring[k] = 0.4985 * (v + ring[nx]); k = nx; }
      ks.set(note, buf);
    }
    const src = ac.createBufferSource(), f = ac.createBiquadFilter(), g = ac.createGain(); src.buffer = ks.get(note);
    f.type = 'lowpass'; f.frequency.value = 3200; g.gain.value = vel; src.connect(f).connect(g); panTo(g, (note % 12 - 6) * 0.05); src.start(t); src.stop(t + 2.2);
  };
  const pad = (t0, dur, notes, level, type, lpHz) => {
    const lp = ac.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 0.6;
    lp.frequency.setValueAtTime(lpHz * 0.75, t0); lp.frequency.linearRampToValueAtTime(lpHz * 1.2, t0 + dur * 0.6); lp.frequency.linearRampToValueAtTime(lpHz * 0.85, t0 + dur + 1.2);
    const atk = type === 'strings' ? Math.min(2, dur * 0.4) : 1.2;
    const pg = ac.createGain(); pg.gain.setValueAtTime(0, t0); pg.gain.linearRampToValueAtTime(level, t0 + atk); pg.gain.setValueAtTime(level, t0 + dur - 0.2); pg.gain.linearRampToValueAtTime(0, t0 + dur + 1.4);
    lp.connect(pg).connect(dry);
    notes.forEach((n, vi) => { for (const det of [-6, 7]) {
      const o = ac.createOscillator(); o.type = 'sawtooth'; o.frequency.value = hz(n); o.detune.value = det * 0.6 + (r() - 0.5) * 1.5;
      const pn = ac.createStereoPanner(); pn.pan.value = clamp((vi - 1.5) * 0.25 + det / 40, -1, 1); o.connect(pn).connect(lp); o.start(t0); o.stop(t0 + dur + 1.5);
    } });
  };
  const nChords = Math.ceil(seconds / chordLen) + 1, total = seconds + 1;
  const dr = DRUMS[G.drums] || null, drumsFrom = seconds > 9 ? bar : 0, drumsTo = seconds - bar * 0.5;
  for (let ci = 0; ci < nChords; ci++) {
    const t0 = ci * chordLen, chord = G.prog[ci % G.prog.length]; if (t0 > total) break;
    const tones = chord.map(iv => key + 12 + iv);
    if (G.pad) pad(t0, chordLen, tones.slice(0, 4), G.pad === 'strings' ? 0.05 : G.pad === 'soft' ? 0.028 : 0.05, G.pad, G.pad === 'strings' ? 1300 : 950);
    if (G.pad === 'strings') pad(t0, chordLen, [key + chord[0]], 0.035, 'strings', 600);
    for (let b = 0; b < G.bars; b++) {
      const tb = t0 + b * bar;
      // bass
      const root = key - 12 + (chord[0] % 12);
      const bassNote = (t, d, vel, wave = 'sine', lpf = 0) => { const o = ac.createOscillator(); o.type = wave; o.frequency.value = hz(root); const g = envG(t, 0.02, vel, d); if (lpf) { const f = ac.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = lpf; o.connect(f).connect(g); } else o.connect(g); g.connect(dry); o.start(t); o.stop(t + d + 0.1); };
      if (G.bass === 'sine') bassNote(tb, bar * 0.95, 0.16);
      else if (G.bass === 'round') { bassNote(tb, beat * 1.6, 0.2, 'triangle', 420); bassNote(tb + 2 * beat, beat * 1.6, 0.16, 'triangle', 420); }
      else if (G.bass === 'eighths') for (let e = 0; e < 8; e++) bassNote(tb + e * beat / 2, beat * 0.42, e % 2 ? 0.11 : 0.15, 'sawtooth', 520);
      else if (G.bass === 'sub') for (const st of (dr ? dr.k : [0])) bassNote(tb + st * step, beat * 1.2, 0.24);
      // keys
      if (G.keys === 'piano') for (let e8 = 0; e8 < 8; e8++) { if (r() > G.density) continue; piano(Math.max(0, tb + e8 * beat / 2 + (r() - 0.5) * 0.012), key + 24 + [...chord, chord[0] + 12][Math.floor(r() * (chord.length + 1))], 0.05 + r() * 0.035); }
      else if (G.keys === 'rhodes') { const sw = st => st * step + (st % 4 === 2 ? G.swing * beat : 0); tones.forEach(n => rhodes(tb, n, 0.035, beat * 2.2)); if (r() < 0.6) tones.forEach(n => rhodes(tb + sw(10), n, 0.025, beat * 1.2)); }
      else if (G.keys === 'ostinato') for (let e = 0; e < 8; e++) piano(tb + e * beat / 2, key + 24 + chord[[0, 1, 2, 1][e % 4] % chord.length], e % 4 === 0 ? 0.045 : 0.03, 0.8);
      else if (G.keys === 'pluck') for (let e = 0; e < 8; e++) pluck(tb + e * beat / 2, key + 24 + [...chord, chord[0] + 12][[0, 1, 2, 3, 2, 1, 2, 3][e] % (chord.length + 1)], e % 2 ? 0.028 : 0.04);
      else if (G.keys === 'stabs') for (const st of [0, 6, 10]) tones.forEach(n => piano(tb + st * step, n, 0.03, 0.45));
      else if (G.keys === 'guitar') {
        const strum = [key + chord[0], ...tones, key + 24 + chord[0]];
        [[0, 1], [4, 0.6], [6, -0.7], [10, -0.8], [12, 0.8], [14, -0.6]].forEach(([st, dir]) => {
          const order = dir > 0 ? strum : [...strum].reverse(), vel = 0.07 * Math.abs(dir);
          order.forEach((n, i) => guitar(tb + st * step + i * 0.012, n, vel * (0.8 + r() * 0.4)));
        });
      }
      // drums
      if (dr && tb >= drumsFrom && tb < drumsTo) {
        const sw = st => tb + st * step + (G.swing && st % 4 === 2 ? G.swing * beat : 0) + (r() - 0.5) * 0.006;
        (dr.k || []).forEach(st => kick(sw(st), 0.5));
        (dr.s || []).forEach(st => snare(sw(st), 0.32));
        (dr.c || []).forEach(st => clap(sw(st), 0.3));
        (dr.h || []).forEach(st => hat(sw(st), 0.06 + r() * 0.04, false));
        (dr.oh || []).forEach(st => { if (r() < 0.5) hat(sw(st), 0.05, true); });
        (dr.sh || []).forEach(st => shaker(sw(st), st % 2 ? 0.025 : 0.04));
        (dr.r || []).forEach(st => rim(sw(st), 0.12));
        (dr.p || []).forEach(st => kick(sw(st), 0.24, 42));
        if (dr.t && (b === 0)) dr.t.forEach(st => tom(sw(st), st ? 0.28 : 0.45));
      }
    }
  }
  // air, and vinyl crackle for lo-fi
  const nb = ac.createBuffer(2, len, SR);
  for (let c = 0; c < 2; c++) { const d = nb.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (r() * 2 - 1) * 0.005 + (G.crackle && r() < 0.0004 ? (r() - 0.5) * 0.5 : 0); }
  const ns = ac.createBufferSource(); ns.buffer = nb; const nf = ac.createBiquadFilter(); nf.type = 'bandpass'; nf.frequency.value = G.crackle ? 2200 : 3000; nf.Q.value = 0.4;
  ns.connect(nf).connect(dry); ns.start(0);
  const buf = await ac.startRendering();
  const names = ['C', 'D', 'E♭', 'F', 'G', 'A'];
  return { chs: [buf.getChannelData(0), buf.getChannelData(1)], genre, name: `${G.label} · ${names[roots.indexOf(key - 48)]}${G.minor ? 'm' : ''} · ${G.bpm} bpm` };
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

// ------------------------------------------------------------------ long form: find the best reels and their hooks
/** Hook formats that hold attention on short-form video. Weights can be refreshed from trends.json. */
const HOOK_WEIGHTS = { question: 0.55, contrarian: 0.8, curiosity: 0.75, number: 0.6, conditional: 0.6, story: 0.5, result: 0.65, bold: 0.4, you: 0.35 };
const HOOK_RX = {
  question: /\?["')]*$|^(why|how|what|when|who|which|do you|did you|are you|is it|have you|ever|can you|should you|would you)\b/i,
  contrarian: /\b(stop|don'?t|never|nobody|no one|wrong|myths?|lies?|lied|overrated|mistakes?|truth|quit|worst|waste\w*|ruin\w*|unpopular|forget)\b/i,
  curiosity: /\b(secret|here'?s (why|how|what|the)|the reason|nobody tells|no one tells|what (nobody|no one)|this is (why|how|what)|the one thing|most people|the real|turns out|i wish i|what happens|nobody talks)\b/i,
  number: /\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(things?|ways?|reasons?|mistakes?|tips?|rules?|steps?|habits?|exercises?|minutes?|days?|weeks?|months?|years?|pounds?|lbs|reps?|sets?|percent)\b/i,
  conditional: /^(if you|if your|when you|whether you|unless you|anyone who|for anyone)\b/i,
  story: /\b(i used to|when i was|years ago|last (week|year|month)|the day i|i remember|my (first|biggest)|i finally|i almost|i never thought)\b/i,
  result: /\b(lost|gained|dropped|doubled|tripled|changed my|transformed|went from|in \d+ (days|weeks|months))\b/i,
  bold: /\b(best|biggest|only|easiest|fastest|simplest|most important|every single|always|guaranteed|game.?changer|nobody)\b/i,
};
const HOOK_LABEL = { question: 'Question', contrarian: 'Contrarian', curiosity: 'Curiosity', number: 'Number', conditional: 'Speaks to "you if…"', story: 'Story', result: 'Result', bold: 'Bold claim', you: 'Talks to the viewer', topic: 'Your topic', trend: 'Trending topic' };
const WEAK_START = /^(and|but|so|or|because|um+|uh+|like|yeah|okay|ok|alright|anyway|also|then|well|plus)\b/i;
const BACKREF = /^(that|it|they|those|these|which|he|she|them|this)\b(?!\s+(is|was|'s)\s+(why|how|what|the))/i;
const FILLER = /^(um+|uh+|erm|hmm+|mm+)[,.]?$/i;
const TRANSITION = /^(so,? anyway|anyway|now,? (a lot|let'?s|another|the next)|let'?s (talk|move|switch|get into)|moving on|next up|okay,? so|alright,? (so|that'?s)|the (next|other) thing|speaking of|on (that|a different) note|which brings me)\b|\blet'?s talk about\b/i;
const PAYOFF = /\b(that'?s (why|it|how|the)|remember|the point|bottom line|trust me|do this|try this|that'?s all|simple as that|so stop|so start|start today|you got this|let'?s go)\b/i;
function sentencesOf(words) {
  const out = []; let a = 0;
  for (let i = 0; i < words.length; i++) {
    const end = /[.?!]["')]*$/.test(words[i].w) || i === words.length - 1 || (words[i + 1].s - words[i].e > 0.9) || i - a >= 40;
    if (end) { out.push({ a, b: i, s: words[a].s, e: words[i].e, text: words.slice(a, i + 1).map(w => w.w).join(' ') }); a = i + 1; }
  }
  return out;
}
function hookScore(text, W, topics, trends, loud = 0) {
  const t = text.trim(), lower = t.toLowerCase(), n = t.split(/\s+/).length, hits = [];
  for (const k of Object.keys(HOOK_RX)) if (HOOK_RX[k].test(t)) hits.push(k);
  if (/\b(you|your|you'?re|you'?ll)\b/i.test(t.split(/\s+/).slice(0, 8).join(' '))) hits.push('you');
  const ws = hits.map(k => W[k] ?? HOOK_WEIGHTS[k] ?? 0.3).sort((a, b) => b - a);
  let sc = ws.reduce((a, w, i) => a + w * Math.pow(0.5, i), 0);
  const has = list => (list || []).find(x => x && new RegExp(`\\b${x.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'i').test(lower));
  const tp = has(topics), tr = has(trends);
  if (tp) { sc += 0.3; hits.push('topic'); }
  if (tr) { sc += 0.25; hits.push('trend'); }
  if (n < 4) sc *= 0.5; else if (n > 22) sc *= 22 / n;
  if (WEAK_START.test(t)) sc -= 0.35;
  if (BACKREF.test(t)) sc -= 0.3;
  sc += clamp(loud / 8, -0.15, 0.2);
  return { score: clamp(sc / 1.3, 0, 1), tags: hits };
}
/** Read a long video's transcript and pick the stretches that work best as reels on their own:
 *  a strong opening line (or a strong line moved to the front), a complete thought, a payoff, and good pace. */
async function findReels(prep, opts = {}, onStage, signal) {
  const words = prep.words, total = prep.total;
  const minLen = opts.minLen || 20, maxLen = opts.maxLen || 45, target = (minLen + maxLen) / 2;
  const W = Object.assign({}, HOOK_WEIGHTS, (opts.trends && opts.trends.hooks) || {});
  const topics = (opts.topics || []).map(x => x.trim()).filter(Boolean), trends = opts.useTrends === false ? [] : ((opts.trends && opts.trends.topics) || []);
  const sents = sentencesOf(words); if (!sents.length) return [];
  onStage && onStage('score', 0.1, 'Reading every line');
  const raw = prep.raw, rms = (s, e) => { const a = Math.floor(s * SR), b = Math.min(raw.length, Math.max(a + 1, Math.floor(e * SR))); let x = 0, c = 0; for (let i = a; i < b; i += 4) { x += raw[i] * raw[i]; c++; } return 10 * Math.log10(x / Math.max(1, c) + 1e-10); };
  sents.forEach(z => { z.db = rms(z.s, z.e); z.n = z.b - z.a + 1; z.wps = z.n / Math.max(0.5, z.e - z.s); z.fill = words.slice(z.a, z.b + 1).filter(w => FILLER.test(w.w)).length; });
  const mdb = median(sents.map(z => z.db)), mwps = median(sents.map(z => z.wps));
  sents.forEach(z => { Object.assign(z, { hook: hookScore(z.text, W, topics, trends, z.db - mdb) }); });
  let vecs = null;
  try { onStage && onStage('score', 0.3, 'Following the topic'); vecs = await embed(sents.map(z => z.text)); } catch (e) { if (e instanceof Cancelled) throw e; }
  check(signal);
  const cands = [];
  for (let i = 0; i < sents.length; i++) {
    for (let j = i; j < sents.length; j++) {
      const d = sents[j].e - sents[i].s; if (d > maxLen) break; if (d < minLen) continue;
      const span = sents.slice(i, j + 1), first = sents[i], last = sents[j];
      // a hook later in the stretch can open the reel (a cold open), if it's clearly stronger and short
      let hk = first.hook.score, hookAt = i, cold = null;
      if (opts.coldOpen !== false) for (let k = i + 1; k <= j; k++) { const z = sents[k]; if (z.e - z.s <= 7 && z.n >= 4 && z.hook.score > hk + 0.18 && z.hook.score > first.hook.score * 1.3) { hk = z.hook.score * 0.92; hookAt = k; } }
      if (hookAt !== i) { const z = sents[hookAt], pw = words[z.a - 1], nw = words[z.b + 1]; cold = [Math.max(z.s - 0.08, pw ? pw.e + 0.02 : 0), Math.min(z.e + 0.12, nw ? nw.s - 0.02 : z.e + 0.12)]; }
      const standalone = clamp(1 - (WEAK_START.test(first.text) ? 0.5 : 0) - (BACKREF.test(first.text) ? 0.4 : 0) + (/[.?!]["')]*$/.test(last.text) ? 0.1 : -0.2), 0, 1);
      const payoff = clamp((PAYOFF.test(last.text) ? 0.85 : 0.4) + (last.n <= 12 ? 0.15 : 0) + (last.hook.tags.includes('contrarian') || last.hook.tags.includes('bold') ? 0.1 : 0), 0, 1);
      const nW = span.reduce((a, z) => a + z.n, 0), fill = span.reduce((a, z) => a + z.fill, 0), wps = nW / Math.max(1, d);
      const gaps = span.slice(1).reduce((a, z, k) => a + Math.max(0, z.s - span[k].e - 0.4), 0);
      const pace = clamp(0.55 + (wps - mwps) / (2 * mwps) - 4 * fill / Math.max(1, nW) - gaps / d, 0, 1);
      let flow = 0.6;
      if (vecs && span.length > 1) { const dim = vecs[0].length, c = new Float32Array(dim); span.forEach((_, k) => { const v = vecs[i + k]; for (let q = 0; q < dim; q++) c[q] += v[q]; }); flow = clamp((span.reduce((a, _, k) => a + cosv(vecs[i + k], c), 0) / span.length - 0.38) / 0.32, 0, 1); }
      const fit = clamp(1 - Math.abs(d - target) / Math.max(10, maxLen - minLen), 0, 1);
      const crosses = span.slice(1).some(z => TRANSITION.test(z.text));                 // runs into the next topic
      const score = 0.42 * hk + 0.18 * standalone + 0.14 * flow + 0.12 * payoff + 0.08 * pace + 0.06 * fit - (crosses ? 0.22 : 0) - (TRANSITION.test(last.text) ? 0.15 : 0);
      let f0 = first.a; while (f0 < Math.min(first.a + 3, first.b) && /^(and|so|but|um+|uh+|like|yeah|okay|ok|alright|well|anyway)[,.]?$/i.test(words[f0].w)) f0++;   // drop a leading "and um,"
      const pw = words[f0 - 1], nw = words[last.b + 1];
      cands.push({ a: Math.max(0, words[f0].s - 0.15, pw ? pw.e + 0.02 : 0), b: Math.min(total, last.e + 0.35, nw ? nw.s - 0.02 : total), i, j, d, score, hookAt, hook: cold,
        hookText: sents[hookAt].text, firstText: first.text, text: span.map(z => z.text).join(' '), tags: sents[hookAt].hook.tags,
        parts: { hook: hk, standalone, flow, payoff, pace } });
    }
  }
  cands.sort((x, y) => y.score - x.score);
  const want = opts.count === 'auto' || !opts.count ? 10 : +opts.count, pick = [];
  for (const c of cands) {
    if (pick.length >= want) break;
    if (pick.some(p => c.a < p.b - 1 && c.b > p.a + 1)) continue;
    if (opts.count === 'auto' || !opts.count) { if (pick.length && c.score < 0.42) break; }
    pick.push(c);
  }
  onStage && onStage('score', 1);
  return pick.map((c, k) => ({ ...c, rank: k + 1, why: c.tags.map(t => HOOK_LABEL[t]).filter(Boolean) }));
}

// ------------------------------------------------------------------ analysis: everything before the render
function cleanVoice(raw, cfg, signal) {
  return denoise(highpass4(raw, 80), cfg.nr, signal).then(p => compress(filt(filt(filt(p, biquad('peak', 220, 1.0, -2.5)), biquad('peak', 450, 1.4, -1.5)), biquad('peak', 3200, 1.0, 1.5))));
}
/** Read the video, find claps, clean the sound (short clips) and transcribe it. Long videos skip the whole-file cleanup;
 *  each reel cleans only its own stretch. */
async function prepareTake({ file, cfg, onStage, signal, model, long = false }) {
  cfg = cfg || defaults();
  const stage = (id, pct, msg) => onStage && onStage(id, pct, msg);
  const t00 = performance.now(), times = {};
  const mark = k => { times[k] = Math.round((performance.now() - t00) / 100) / 10; };
  stage('read', 0, 'Reading your video');
  const movie = await openMovie(file);
  const nFrames = Math.floor(movie.duration * FPS), total = nFrames / FPS;
  let raw = await decodeAudioTrack(movie, signal);
  if (!raw) throw new Error('This video has no sound.');
  const off = Math.round(movie.audioOffset * SR), need = nFrames * SPF, aligned = new Float32Array(need);
  for (let i = 0; i < need; i++) { const j = i - off; aligned[i] = j >= 0 && j < raw.length ? raw[j] : 0; }
  raw = aligned; mark('read'); stage('read', 1);

  stage('claps', 0, 'Listening for claps');
  const { claps, near } = cfg.clap.off ? { claps: [], near: [] } : detectClaps(raw, cfg); await tick(); check(signal);
  let proc = null;
  if (!long) { stage('claps', 0.4, 'Cleaning up the sound'); proc = await cleanVoice(raw, cfg, signal); }
  mark('audio'); stage('claps', 1);

  stage('transcribe', 0, 'Loading the speech model');
  const words = await transcribe(proc || raw, (p, msg) => stage('transcribe', p, msg || undefined), signal, model || 'small');
  refineTiming(words, proc || raw);
  applyCorrections(words, parseCorrections(cfg.corrections ?? DEFAULT_CORRECTIONS));
  mark('transcribe'); check(signal);
  return { movie, total, raw, proc, words, claps, near, times, model: words.model, long };
}
/** Build one reel from the take (or from a stretch of it, with an optional hook line moved to the front). */
async function buildReel(prep, { items, photos, notes, cfg, onStage, signal, range, hook }) {
  cfg = cfg || defaults();
  const stage = (id, pct, msg) => onStage && onStage(id, pct, msg);
  const { movie, total } = prep;
  const [ra, rb] = range ? [Math.max(0, range[0]), Math.min(total, range[1])] : [0, total];
  const base = Math.round(ra * FPS) * SPF, a0 = base / SR;
  let raw = prep.raw, proc = prep.proc;
  if (range || !proc) {
    stage('cut', 0, 'Cleaning up the sound');
    raw = prep.raw.slice(base, Math.min(prep.raw.length, Math.round(rb * FPS) * SPF));
    proc = cfg.cleanup === false ? Float32Array.from(raw) : await cleanVoice(raw, cfg, signal);
  }
  const words = prep.words.filter(w => w.s >= ra - 0.05 && w.e <= rb + 0.05).map(w => ({ ...w }));
  stage('cut', 0.3, 'Trimming pauses and retakes');
  const speech = addMissedWords(vad(proc).map(([s, e]) => [s + a0, e + a0]), words);
  const claps = prep.claps.filter(c => c.start >= ra && c.end <= rb), near = (prep.near || []).filter(c => c.t >= ra && c.t <= rb);
  const flubs = flubRegions(claps, speech, words, cfg);
  DIAG.claps = { found: claps.map(c => `${c.t}s L${c.level} R${c.rise} I${c.iso}`), cuts: flubs.map(f => `${f.cut[0].toFixed(2)}-${f.cut[1].toFixed(2)} ${f.method}`) };
  const { keep, trimmed } = keepFor(speech, flubs.map(f => f.cut), total, cfg, [], range ? [ra, rb] : null, hook);
  if (!keep.length) throw new Error("Couldn't hear any speech in this video.");
  const tmap = new TimeMap(keep), outDur = tmap.nOut / FPS;
  flubs.forEach(f => { f.outJoin = tmap.rawToOut(f.cut[1]); });
  const voice = cutAudio(proc, keep, base);
  const inLufs = loudness([voice]);
  const g = Math.pow(10, ((cfg.lufs - 3) - inLufs) / 20); // -17 LUFS mono = -14 LUFS once doubled to stereo
  for (let i = 0; i < voice.length; i++) voice[i] *= g;
  const outWords = mapWords(words, tmap);
  const timing = refineTiming(outWords, voice);
  addEmphasis(outWords, voice);
  check(signal); stage('cut', 1);

  stage('plan', 0, 'Choosing clips and photos');
  const ctx = { delivery: deliveryRows(outWords, voice, phrases(outWords, 10)) };
  if (cfg.meaning !== false && ((items || []).length || (photos || []).length)) {
    try { stage('plan', 0.2, 'Matching visuals to the feel of each line'); ctx.vibe = await vibeContext(outWords, [...(items || []), ...(photos || [])]); }
    catch (e) { if (e instanceof Cancelled) throw e; DIAG.vibeError = String(e.message || e); }
  }
  const { broll, pops, log: brollLog, mode: brollMode } = planVisuals(outWords, items || [], photos || [], notes, tmap, outDur, cfg, null, ctx);
  const zooms = cfg.zoom.off ? [] : planZooms(outWords, tmap.cutPoints(), broll, outDur, cfg);
  const zc = zoomCurve(zooms, tmap.nOut, cfg);
  stage('plan', 1);

  stage('face', 0, 'Framing you');
  const track = await faceTrack(movie, tmap, broll, cfg, p => stage('face', p), signal);
  stage('face', 1);
  // everything below lets each edit be undone later without redoing the whole analysis
  const surv = outWords.map(w => {   // back to take time; a word never spans a cut (its end stays in the piece its start is in)
    const k = tmap.segIndex(Math.floor(w.s * FPS)), segEnd = tmap.keep[k][1] / FPS, s0 = tmap.outToRaw(w.s);
    return { ...w, s: +s0.toFixed(3), e: +Math.min(segEnd, Math.max(s0 + 0.05, tmap.outToRaw(Math.max(w.s, w.e - 0.001)))).toFixed(3) };
  });
  const cutWords = words.filter(w => !surv.some(o => o.e > w.s + 0.02 && o.s < w.e - 0.02)).map(w => ({ ...w, cut: true }));
  return { movie, cfg, total, outDur, claps, near, flubs, keep, trimmed, tmap, words, outWords, voice, inLufs,
    speechOut: vad(voice), broll, brollLog, brollMode, pops, popLog: [], zooms, timing, proof: { applied: [] }, model: prep.model, zc, track, times: prep.times, chunks: null, music: null, notes,
    proc, raw, speech, rawWords: [...surv, ...cutWords].sort((x, y) => x.s - y.s), audioBase: base, range: range ? [ra, rb] : null, hook: hook || null, cleanup: cfg.cleanup !== false, capsOff: false };
}
async function analyze({ file, items, notes, cfg, onStage, signal, photos, model }) {
  cfg = cfg || defaults();
  const prep = await prepareTake({ file, cfg, onStage, signal, model });
  return buildReel(prep, { items, photos, notes, cfg, onStage, signal });
}


/** Redo the cut after edits were switched off or back on (a clap cut, a trimmed pause, voice cleanup):
 *  rebuild the timeline and move every other edit (B-roll, photos, zooms, caption fixes) to follow its words. */
async function retime(P, onStage, signal) {
  const cfg = P.cfg, old = P.tmap, stage = (id, p, m) => onStage && onStage(id, p, m);
  stage('cut', 0, 'Rebuilding the cut');
  const restored = P.trimmed.filter(t => t.off).map(t => [t[0], t[1]]);
  const removed = [...P.flubs.filter(f => !f.off).map(f => f.cut), ...(P.manual || []).filter(m => !m.off).map(m => m.range)].sort((a, b) => a[0] - b[0]);
  const { keep, trimmed } = keepFor(P.speech, removed, P.total, cfg, restored, P.range, P.hook);
  if (!keep.length) throw new Error('Nothing left to show with these edits removed.');
  const tmap = new TimeMap(keep), outDur = tmap.nOut / FPS;
  const move = t => tmap.rawToOut(old.outToRaw(t));
  // pauses: keep the ones the user put back listed (marked off) so they can be trimmed again
  const tr = trimmed.map(t => t); for (const r of P.trimmed.filter(t => t.off)) { const x = [r[0], r[1], r[2]]; x.off = true; tr.push(x); }
  tr.sort((a, b) => a[0] - b[0]);
  P.flubs.forEach(f => { f.outJoin = tmap.rawToOut(f.off ? f.cut[0] : f.cut[1]); });
  (P.manual || []).forEach(m => { m.outJoin = tmap.rawToOut(m.off ? m.range[0] : m.range[1]); });
  // voice
  let voice = cutAudio(P.cleanup ? P.proc : P.raw, keep, P.audioBase || 0);
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
  outWords.sort((x, y) => x.s - y.s);
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
  P.chunks = chunkWords(P.outWords, s => c.measureText(P.cfg.cap.upper ? s.toUpperCase() : s).width, OUT_W - 2 * P.cfg.cap.side - 20 - (P.cfg.cap.style === 'box' ? 0.7 * P.cfg.cap.size : 0), P.cfg);
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
  try {
    if (M.mode === 'mine') {
      const pool = M.tracks || [];
      const pick = (M.trackId && pool.find(t => t.id === M.trackId)) || pool[Math.floor(Math.random() * pool.length)];
      if (pick) {
        onStage && onStage('music', 0.3, `Fitting "${pick.name}"`);
        try { music = fitTrack(await decodeTrack(pick.url), P.outDur); chosen = { kind: 'track', name: pick.name, id: pick.id, any: !M.trackId }; }
        catch (e) { DIAG.trackError = String(e.message || e); M.fellBack = `couldn't read "${pick.name}"`; }
      } else M.fellBack = 'no tracks added yet';
      if (!music) M.mode = 'generated';
    }
    if (!music && M.mode !== 'off') {
      onStage && onStage('music', 0.3, 'Writing the music');
      const seed = M.seed || Math.floor(Math.random() * 1e9);
      const g = await composeMusic(P.outDur, M.genre || M.mood || 'chill', seed); music = g.chs;
      chosen = { kind: 'generated', name: g.name + (M.fellBack ? ` (${M.fellBack})` : ''), seed, genre: g.genre };
    }
  } catch (e) { if (e instanceof Cancelled) throw e; console.warn(e); music = null; chosen = { kind: 'none', name: 'No music (' + e.message + ')' }; }
  check(signal);
  let L, R;
  if (music) [L, R] = mixMusic(P.voice, music, P.speechOut, M.level);
  else { L = Float32Array.from(P.voice); R = Float32Array.from(P.voice); }
  const lu = loudness([L, R]), gg = Math.pow(10, (P.cfg.lufs - lu) / 20);
  for (let i = 0; i < L.length; i++) { L[i] *= gg; R[i] *= gg; }
  limit([L, R], -1.2);
  P.mix = [L, R]; P.music = Object.assign(chosen, { level: M.level });
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
/** Render the reel. If the hardware video encoder fails partway (it happens on some devices, more on long jobs), try once more in software. */
async function render(P, onProgress, signal) {
  try { return await renderOnce(P, onProgress, signal, false); }
  catch (e) {
    if (e instanceof Cancelled) throw e;
    const msg = String(e && (e.name + ' ' + e.message) || e);
    if (!/encod|codec|closed|quota|operation|invalidstate/i.test(msg)) throw e;
    const vcfg = { codec: 'avc1.640028', width: OUT_W, height: OUT_H, bitrate: P.cfg.videoBitrate, framerate: FPS, hardwareAcceleration: 'prefer-software' };
    let ok = false; try { ok = (await VideoEncoder.isConfigSupported(vcfg)).supported; } catch (x) { ok = false; }
    if (!ok) throw e;
    DIAG.renderRetry = msg.slice(0, 160);
    return renderOnce(P, onProgress, signal, true);
  }
}
async function renderOnce(P, onProgress, signal, soft) {
  await ensureMuxer();
  const Mx = root.Mp4Muxer, cfg = P.cfg, n = P.tmap.nOut;
  const vcfg = { codec: 'avc1.640028', width: OUT_W, height: OUT_H, bitrate: cfg.videoBitrate, framerate: FPS, avc: { format: 'avc' } };
  if (soft) vcfg.hardwareAcceleration = 'prefer-software';
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
        if (bsrc) { bsrc.close(); bsrc = null; } if (bimg) { bimg.close(); bimg = null; }
        try {   // a clip that won't load (offline, removed) is left out rather than failing the reel
          const r = await fetch(p.item.url); if (!r.ok) throw new Error('HTTP ' + r.status); const blob = await r.blob();
          if (p.item.kind === 'still') bimg = await createImageBitmap(blob);
          else { const mv = await openMovie(blob); bsrc = new VideoSource(mv); bsrc.mv2 = mv; await bsrc.open(); if (p.clipIn > 0.05) bsrc.seek(p.clipIn); }
        } catch (e) { if (e instanceof Cancelled) throw e; (DIAG.brollSkipped = DIAG.brollSkipped || []).push(String(p.item.id || p.item.name || '') + ': ' + String(e.message || e).slice(0, 80)); bplace = null; bsrc = null; bimg = null; }
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
        if (bplace) { bplace = null; if (bsrc) { bsrc.close(); bsrc = null; } if (bimg) { bimg.close(); bimg = null; } }
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
  } finally { src.close(); if (bsrc) bsrc.close(); if (bimg) bimg.close(); popImg.forEach(b => b.close()); if (venc.state !== 'closed') venc.close(); }
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

root.ReelEngine = { GENRES, VIBES, prepareTake, buildReel, findReels, hookScore, sentencesOf, HOOK_LABEL, DEFAULT_CORRECTIONS, retime, rebuildCaptions, setZooms, transcribe, MODELS, popAlpha, snapEnd, stem, isStop: w => STOP.has(norm(w)), norm, phrases, catalogLine, decodeAudioTrack, sanitizeAsc, decodeTrack, DIAG, verifyAudio, ensureMuxer, toWav, audioToM4a, drawRotated, defaults, analyze, buildCaptions, buildAudio, render, editLog, toSrt, composeMusic, openMovie, VideoSource,
  decodeTrack, loudness, drawCaption, capFont, fmt, pickHighlight, FPS, OUT_W, OUT_H, SR, MOODS, Cancelled,
  _t: { refineTiming, applyCorrections, parseCorrections, rhythmStart, planVisuals, deliveryRows, vibeContext, captionLift, detectClaps, vad, buildKeep, flubRegions, planZooms, chunkWords, parseNotes, findPhrase, mapWords, TimeMap, denoise, compress, limit, resample, highpass4, filt, biquad, addEmphasis, fitTrack, mixMusic, ratio } };
})(typeof window !== 'undefined' ? window : globalThis);
