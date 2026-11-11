// Pull free stock clips and photos from Pexels or Pixabay into stock/ (downloaded, trimmed to 8 s, cropped to 9:16, tagged).
// Runs in GitHub Actions (see .github/workflows/fetch-stock.yml) or locally: node tools/fetch-stock.mjs "ocean waves" nature clip 3 pexels
// Keys: PEXELS_API_KEY / PIXABAY_API_KEY (free: pexels.com/api, pixabay.com/api/docs).
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';

const [query = 'ocean waves slow', category = 'nature', kind = 'clip', n = '3', provider = 'pexels'] = process.argv.slice(2);
const STOCK = path.resolve('stock'), TMP = path.join(STOCK, '_dl');
const slugTags = url => ((url || '').match(/\/(?:video|photo)\/([a-z0-9-]+?)-\d+\/?$/) || [, ''])[1].split('-').filter(w => w.length > 2);
const ff = args => execFileSync('ffmpeg', ['-y', '-v', 'error', ...args], { stdio: 'inherit' });

async function getJson(url, headers = {}) {
  const r = await fetch(url, { headers });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} from ${new URL(url).host}`);
  return r.json();
}
async function download(url, dst) {
  const r = await fetch(url); if (!r.ok) throw new Error(`download failed: ${r.status}`);
  await writeFile(dst, Buffer.from(await r.arrayBuffer())); return dst;
}

async function searchPexels(key, want) {
  const h = { Authorization: key };
  if (kind === 'clip') {
    const j = await getJson(`https://api.pexels.com/videos/search?${new URLSearchParams({ query, per_page: want, orientation: 'portrait', size: 'medium' })}`, h);
    return (j.videos || []).map(v => {
      const f = (v.video_files || []).filter(x => x.file_type === 'video/mp4' && (x.height || 0) >= 1080).sort((a, b) => a.height - b.height)[0];
      return f && { srcId: `pexels_${v.id}`, url: f.link, duration: +v.duration || 8, tags: slugTags(v.url), subject: slugTags(v.url).join(' '), page: v.url, credit: `${v.user?.name || ''} on Pexels`, license: 'Pexels License' };
    }).filter(Boolean);
  }
  const j = await getJson(`https://api.pexels.com/v1/search?${new URLSearchParams({ query, per_page: want, orientation: 'landscape' })}`, h);
  return (j.photos || []).map(p => ({ srcId: `pexels_${p.id}`, url: p.src.large, tags: [...slugTags(p.url), ...((p.alt || '').toLowerCase().match(/[a-z]{3,}/g) || [])],
    subject: p.alt || slugTags(p.url).join(' '), page: p.url, credit: `${p.photographer || ''} on Pexels`, license: 'Pexels License' }));
}
async function searchPixabay(key, want) {
  if (kind === 'clip') {
    const j = await getJson(`https://pixabay.com/api/videos/?${new URLSearchParams({ key, q: query, per_page: Math.max(3, want), safesearch: 'true' })}`);
    return (j.hits || []).map(v => { const f = v.videos?.large?.url ? v.videos.large : v.videos?.medium; const tags = (v.tags || '').split(',').map(t => t.trim()).filter(Boolean);
      return f?.url && { srcId: `pixabay_${v.id}`, url: f.url, duration: +v.duration || 8, tags, subject: tags.slice(0, 4).join(', '), page: v.pageURL, credit: `${v.user} on Pixabay`, license: 'Pixabay Content License' }; }).filter(Boolean);
  }
  const j = await getJson(`https://pixabay.com/api/?${new URLSearchParams({ key, q: query, per_page: Math.max(3, want), image_type: 'photo', orientation: 'horizontal', safesearch: 'true' })}`);
  return (j.hits || []).map(p => { const tags = (p.tags || '').split(',').map(t => t.trim()).filter(Boolean);
    return { srcId: `pixabay_${p.id}`, url: p.largeImageURL || p.webformatURL, tags, subject: tags[0] || query, page: p.pageURL, credit: `${p.user} on Pixabay`, license: 'Pixabay Content License' }; });
}

const key = provider === 'pixabay' ? process.env.PIXABAY_API_KEY : process.env.PEXELS_API_KEY;
if (!key) { console.error(`Missing ${provider === 'pixabay' ? 'PIXABAY_API_KEY' : 'PEXELS_API_KEY'}. Add it as a repository secret.`); process.exit(1); }
const want = +n;
const found = await (provider === 'pixabay' ? searchPixabay : searchPexels)(key, want * 2);
const manPath = path.join(STOCK, 'manifest.json');
const m = existsSync(manPath) ? JSON.parse(readFileSync(manPath, 'utf8')) : { clips: [], photos: [] };
m.clips ||= []; m.photos ||= [];
const have = new Set([...m.clips, ...m.photos].map(x => x.src_id || x.id));
mkdirSync(TMP, { recursive: true }); ['clips', 'thumbs', 'photos'].forEach(d => mkdirSync(path.join(STOCK, d), { recursive: true }));
let added = 0;
for (const r of found) {
  if (added >= want || have.has(r.srcId)) continue;
  try {
    if (kind === 'clip') {
      const raw = await download(r.url, path.join(TMP, r.srcId + '.mp4'));
      const start = r.duration > 8 ? (r.duration - 8) / 2 : 0;
      ff(['-ss', start.toFixed(2), '-i', raw, '-t', '8', '-an', '-vf', 'fps=30,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path.join(STOCK, 'clips', r.srcId + '.mp4')]);
      ff(['-ss', '1', '-i', path.join(STOCK, 'clips', r.srcId + '.mp4'), '-frames:v', '1', '-vf', 'scale=240:-2', path.join(STOCK, 'thumbs', r.srcId + '.jpg')]);
      m.clips.push({ id: r.srcId, src_id: r.srcId, file: `clips/${r.srcId}.mp4`, thumb: `thumbs/${r.srcId}.jpg`, duration_s: Math.min(8, r.duration), category,
        subject: r.subject, tags: [...r.tags, ...query.split(' ')], energy: 'medium', shirtless: false, others_visible: false,
        source: provider, source_page: r.page, credit: r.credit, license: r.license });
    } else {
      const raw = await download(r.url, path.join(TMP, r.srcId + '.jpg')), id = 'photo_' + r.srcId;
      ff(['-i', raw, '-vf', 'scale=1200:-2', '-q:v', '3', path.join(STOCK, 'photos', id + '.jpg')]);
      ff(['-i', path.join(STOCK, 'photos', id + '.jpg'), '-vf', 'scale=320:-2', path.join(STOCK, 'photos', id + '_t.jpg')]);
      m.photos.push({ id, src_id: r.srcId, file: `photos/${id}.jpg`, thumb: `photos/${id}_t.jpg`, category, subject: r.subject,
        tags: [...r.tags, ...query.split(' ')], source: provider, source_page: r.page, credit: r.credit, license: r.license });
    }
    added++; console.log(`Added ${r.srcId}: ${r.subject}`);
  } catch (e) { console.log(`Skipped ${r.srcId}: ${e.message}`); }
}
rmSync(TMP, { recursive: true, force: true });
writeFileSync(manPath, JSON.stringify(m, null, 1));
console.log(`Added ${added} from ${provider} for "${query}".`);
