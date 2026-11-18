// Keeps the blog's index, sitemap and menu link in step with the articles in blog/ (written by Marblism, or by hand):
//   blog/<slug>.html or blog/<slug>/index.html   an article page, left exactly as written
//   blog/index.html                               every article, newest first (or "No articles yet")
//   sitemap.xml                                   blog URLs with lastmod
//   "Blog" nav link                               on the public pages, only while there are articles
// Title, date, read time and description are read from each article page. Runs from .github/workflows/blog.yml
// whenever something in blog/ changes. No API key, nothing to install.
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const SITE = 'https://reeleditapp.com';
const EMAIL = (() => { try { const e = String(JSON.parse(fs.readFileSync(path.join(root, 'site/contact.json'), 'utf8')).email || '').trim(); return /^[^\s@<>"]+@[^\s@<>"]+\.[a-z]{2,}$/i.test(e) ? e : ''; } catch (e) { return ''; } })();

const esc = (x) => String(x ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const unesc = (x) => String(x ?? '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
const day = (d) => { const t = new Date(d); return isNaN(t) ? '' : t.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' }); };
const iso = (d) => { const t = new Date(d); return isNaN(t) ? '' : t.toISOString(); };

// ---- the articles in blog/
const blogDir = path.join(root, 'blog');
const files = !fs.existsSync(blogDir) ? [] : fs.readdirSync(blogDir, { withFileTypes: true }).flatMap((d) => {
  if (d.isFile() && d.name.endsWith('.html') && d.name !== 'index.html') return [{ file: path.join(blogDir, d.name), slug: d.name.slice(0, -5), href: `/blog/${d.name}` }];
  if (d.isDirectory() && fs.existsSync(path.join(blogDir, d.name, 'index.html'))) return [{ file: path.join(blogDir, d.name, 'index.html'), slug: d.name, href: `/blog/${d.name}/` }];
  return [];
});
const posts = files.map(({ file, slug, href }) => {
  const html = fs.readFileSync(file, 'utf8'), pick = (re) => (html.match(re) || [])[1] || '';
  const title = unesc(pick(/<h1[^>]*>([\s\S]*?)<\/h1>/).replace(/<[^>]+>/g, '').trim() || pick(/<title>([^<|]+)/).trim());
  const published = pick(/article:published_time" content="([^"]+)"/) || pick(/"datePublished"\s*:\s*"([^"]+)"/) || pick(/<time[^>]*datetime="([^"]+)"/);
  const updated = pick(/article:modified_time" content="([^"]+)"/) || pick(/"dateModified"\s*:\s*"([^"]+)"/) || published;
  const draft = /<meta name="robots" content="[^"]*noindex/i.test(html);
  return { slug, href, title, published, updated, mins: +pick(/(\d+)\s*min read/) || null, draft };
}).filter((a) => a.title && !a.draft).sort((x, y) => String(y.published).localeCompare(String(x.published)));

// ---- the index, in the site's own design
const meta = (a) => [a.published && `<time datetime="${esc(iso(a.published))}">${esc(day(a.published))}</time>`, a.mins && `${a.mins} min read`].filter(Boolean).join(' · ');
fs.mkdirSync(blogDir, { recursive: true });
fs.writeFileSync(path.join(blogDir, 'index.html'), `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Blog | Reel Studio by reeleditapp</title>
<meta name="description" content="Articles from Reel Studio on editing talking-head reels, short-form video and posting without the editing evening.">
<link rel="canonical" href="${SITE}/blog/">
<meta name="theme-color" content="#07090F">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Reel Studio by reeleditapp">
<meta property="og:url" content="${SITE}/blog/">
<meta property="og:title" content="The Reel Studio blog">
<meta property="og:image" content="${SITE}/site/img/og.jpg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="${SITE}/site/img/og.jpg">
<link rel="icon" href="/icons/icon-192.png?v=3">
<link rel="apple-touch-icon" href="/icons/icon-180.png?v=3">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@108..125,800&family=Inter:wght@400;500;600;700;800&display=swap">
<link rel="stylesheet" href="/site/site.css">
</head>
<body>
<div class="glow" aria-hidden="true"></div>
<header class="top">
  <div class="wrap">
    <a class="brand" href="/"><img src="/icons/icon-192.png?v=3" alt="" width="30" height="30">Reel <span>Studio</span></a>
    <nav aria-label="Main">
      <a href="/#how">How it works</a>
      <a href="/pricing.html">Pricing</a>
      ${posts.length ? '<a href="/blog/" aria-current="page">Blog</a>\n      ' : ''}<a href="/#faq">Questions</a>
    </nav>
    <a class="btn sm primary" href="/app.html">Open the editor</a>
  </div>
</header>
<main>
  <section style="border:0">
    <div class="wrap">
      <div class="head"><span class="eyebrow">Blog</span><h1 style="font-size:clamp(34px,5vw,54px)">The Reel Studio blog</h1></div>
${posts.length ? `      <ul class="post-list">
${posts.map((a) => `        <li><a href="${esc(a.href)}"><h2>${esc(a.title)}</h2><span class="muted">${meta(a)}</span></a></li>`).join('\n')}
      </ul>` : '      <p class="lede">No articles yet.</p>'}
    </div>
  </section>
</main>
<footer>
  <div class="wrap">
    <a class="brand" href="/" style="font-size:16px"><img src="/icons/icon-192.png?v=3" alt="" width="24" height="24">Reel <span>Studio</span></a>
    <span class="sp"></span>
    <a href="/app.html">Editor</a><a href="/pricing.html">Pricing</a>${posts.length ? '<a href="/blog/">Blog</a>' : ''}<a href="/data.html">Your data</a><a href="/#faq">Questions</a><a href="/about">About Reel Studio</a><a href="/contact">Contact</a><a href="/privacy">Privacy</a><a href="/terms">Terms</a>${EMAIL ? `<a href="mailto:${esc(EMAIL)}">Email</a>` : ''}
  </div>
</footer>
</body>
</html>
`);

// ---- sitemap: the site's own pages, plus the blog
const smf = path.join(root, 'sitemap.xml');
let sm = fs.readFileSync(smf, 'utf8').replace(/\s*<url><loc>https:\/\/reeleditapp\.com\/blog\/[\s\S]*?<\/url>/g, '');
const lastmod = (d) => (iso(d) ? `<lastmod>${iso(d).slice(0, 10)}</lastmod>` : '');
const latest = posts.map((a) => a.updated).filter(Boolean).sort().pop();
const urls = [`  <url><loc>${SITE}/blog/</loc>${lastmod(latest)}<changefreq>weekly</changefreq><priority>0.6</priority></url>`,
  ...posts.map((a) => `  <url><loc>${SITE}${esc(a.href)}</loc>${lastmod(a.updated)}<priority>0.6</priority></url>`)];
sm = sm.replace('</urlset>', urls.join('\n') + '\n</urlset>');
fs.writeFileSync(smf, sm);

// ---- "Blog" in the main menu, only while there are articles
for (const f of ['index.html', 'pricing.html', 'data.html', 'about.html', 'refund.html', 'vs/opus-clip.html', 'contact.html', 'privacy.html', 'terms.html']) {
  const p = path.join(root, f); if (!fs.existsSync(p)) continue;
  const html = fs.readFileSync(p, 'utf8');
  if (!/<!--bake:blognav-->/.test(html)) { console.warn(`blog: ${f} has no menu marker, skipped`); continue; }
  fs.writeFileSync(p, html.replace(/<!--bake:blognav-->[\s\S]*?<!--\/bake:blognav-->/, `<!--bake:blognav-->${posts.length ? '<a href="/blog/">Blog</a>' : ''}<!--/bake:blognav-->`));
}
console.log(`blog: ${posts.length} article${posts.length === 1 ? '' : 's'}`);
