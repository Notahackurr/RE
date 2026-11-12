// Free stock search for Reel Studio: Pexels videos (portrait first) and photos, normalized for the app.
// The Pexels key stays here on the server. Results are cached for a week per search, so the free
// Pexels limits (200 searches an hour) go a long way. Deploy with JWT verification OFF.
// Secrets: PEXELS_API_KEY (free at pexels.com/api).
import { createClient } from 'npm:@supabase/supabase-js@2';

const ORIGINS = ['https://reeleditapp.com', 'https://www.reeleditapp.com', 'http://localhost:8765'];
const cors = (req: Request) => {
  const o = req.headers.get('origin') ?? '';
  return { 'Access-Control-Allow-Origin': ORIGINS.includes(o) ? o : ORIGINS[0], 'Access-Control-Allow-Headers': 'authorization, apikey, content-type, x-client-info',
    'Access-Control-Allow-Methods': 'GET, OPTIONS', Vary: 'Origin' };
};
const json = (req: Request, body: unknown, status = 200, maxAge = 0) => new Response(JSON.stringify(body), {
  status, headers: { ...cors(req), 'Content-Type': 'application/json', ...(maxAge ? { 'Cache-Control': `public, max-age=${maxAge}` } : {}) } });

type Item = { id: string; kind: 'clip' | 'photo'; url: string; thumb: string; subject: string; tags: string[]; duration?: number; w: number; h: number; credit: string; page: string };
const titleFrom = (page: string) => {
  const slug = (page.match(/\/(?:video|photo)\/([a-z0-9-]+?)-\d+\/?$/) || [])[1] || '';
  const t = slug.replace(/-/g, ' ').trim(); return t ? t[0].toUpperCase() + t.slice(1) : '';
};
const words = (s: string) => s.toLowerCase().split(/[^a-z]+/).filter((w) => w.length > 3 && !['with', 'from', 'that', 'this', 'their', 'into', 'over', 'stock', 'footage', 'video', 'photo'].includes(w));

async function pexels(path: string) {
  const r = await fetch('https://api.pexels.com' + path, { headers: { Authorization: Deno.env.get('PEXELS_API_KEY') ?? '' } });
  if (!r.ok) throw new Error(`Pexels ${r.status}`);
  return r.json();
}
async function videos(q: string, n: number): Promise<Item[]> {
  const j = await pexels(`/videos/search?query=${encodeURIComponent(q)}&orientation=portrait&size=medium&per_page=${n}`);
  return (j.videos || []).map((v: any) => {
    const files = (v.video_files || []).filter((f: any) => f.link && f.width && f.height && /mp4/.test(f.file_type || 'video/mp4'));
    const portrait = files.filter((f: any) => f.height > f.width);
    const pool = portrait.length ? portrait : files;
    // closest to 720 x 1280: sharp enough for a 1080 x 1920 reel, light enough to download quickly
    const f = pool.sort((a: any, b: any) => Math.abs(Math.min(a.width, a.height) - 720) - Math.abs(Math.min(b.width, b.height) - 720))[0];
    if (!f) return null;
    const subject = titleFrom(v.url) || q;
    return { id: `pexels_v_${v.id}`, kind: 'clip', url: f.link, thumb: v.image, subject, tags: [q, ...words(subject)].slice(0, 8), duration: v.duration, w: f.width, h: f.height,
      credit: `Video by ${v.user?.name || 'a Pexels creator'} on Pexels`, page: v.url } as Item;
  }).filter(Boolean);
}
async function photos(q: string, n: number): Promise<Item[]> {
  const j = await pexels(`/v1/search?query=${encodeURIComponent(q)}&orientation=landscape&size=medium&per_page=${n}`);
  return (j.photos || []).map((p: any) => {
    const subject = (p.alt || '').trim() || titleFrom(p.url) || q;
    return { id: `pexels_p_${p.id}`, kind: 'photo', url: p.src?.large2x || p.src?.large, thumb: p.src?.medium, subject, tags: [q, ...words(subject)].slice(0, 8), w: p.width, h: p.height,
      credit: `Photo by ${p.photographer || 'a Pexels photographer'} on Pexels`, page: p.url } as Item;
  });
}

const memo = new Map<string, { at: number; items: Item[] }>();
Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: cors(req) });
  const u = new URL(req.url), q = (u.searchParams.get('q') || '').trim().toLowerCase().replace(/\s+/g, ' ').slice(0, 60);
  const kind = ['clip', 'photo'].includes(u.searchParams.get('kind') || '') ? u.searchParams.get('kind')! : 'both';
  const n = Math.max(1, Math.min(15, +(u.searchParams.get('n') || 8)));
  if (q.length < 2) return json(req, { items: [] });
  if (!Deno.env.get('PEXELS_API_KEY')) return json(req, { error: 'Free stock search is not set up yet.' }, 503);
  const key = `${kind}|${n}|${q}`, week = 7 * 864e5;
  const hit = memo.get(key); if (hit && Date.now() - hit.at < week) return json(req, { items: hit.items }, 200, 3600);
  const db = createClient(Deno.env.get('SUPABASE_URL')!, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, { auth: { persistSession: false } });
  try {   // the shared cache (a missing table just means no cache)
    const { data } = await db.from('stock_cache').select('items,at').eq('key', key).maybeSingle();
    if (data && Date.now() - new Date(data.at).getTime() < week) { memo.set(key, { at: Date.now(), items: data.items }); return json(req, { items: data.items }, 200, 3600); }
  } catch { /* no cache table */ }
  try {
    const [v, p] = await Promise.all([kind === 'photo' ? [] : videos(q, n), kind === 'clip' ? [] : photos(q, Math.ceil(n / 2))]);
    const items = [...v, ...p];
    memo.set(key, { at: Date.now(), items });
    try { await db.from('stock_cache').upsert({ key, items, at: new Date().toISOString() }); } catch { /* no cache table */ }
    return json(req, { items }, 200, 3600);
  } catch (e) {
    return json(req, { error: /429/.test(String(e)) ? 'Free stock is busy right now. Try again in a few minutes.' : "Couldn't reach free stock right now." }, 502);
  }
});
