// Builds the blog from the HeyCatch API into static pages (the site is static, so this is its server-side render):
//   blog/index.html          every article, newest first (or "No articles yet")
//   blog/<slug>/index.html   one page per article: title, date, read time, table of contents, the article
//   sitemap.xml              blog URLs with lastmod
//   "Blog" nav link          on the public pages, only while there are articles
// Needs HEYCATCH_API_KEY in the environment (a GitHub Actions secret). Run by .github/workflows/blog.yml every few minutes.
// Any API problem (bad key, revoked, rate limit, network) leaves the current pages as they are and exits cleanly.
import fs from 'node:fs';
import path from 'node:path';
import sanitizeHtml from 'sanitize-html';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const SITE = 'https://reeleditapp.com';
const API = process.env.HEYCATCH_API_BASE || 'https://api.heycatch.ai/v1';   // the base can be overridden for tests
const KEY = process.env.HEYCATCH_API_KEY || '';
const RTL = new Set(['ar', 'he', 'fa', 'ur', 'ps', 'yi', 'dv', 'ckb']);

const esc = (x) => String(x ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const day = (d) => { const t = new Date(d); return isNaN(t) ? '' : t.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }); };
const iso = (d) => { const t = new Date(d); return isNaN(t) ? '' : t.toISOString(); };

async function api(p) {
  const r = await fetch(`${API}${p}${p.includes('?') ? '&' : '?'}key=${encodeURIComponent(KEY)}`, { headers: { accept: 'application/json' } });
  if (!r.ok) { const e = new Error(`HeyCatch API ${r.status} for ${p.replace(/key=[^&]+/, 'key=…')}`); e.status = r.status; throw e; }
  return r.json();
}

// ---- fetch everything (pages of 50)
let list = [];
try {
  if (!KEY) throw Object.assign(new Error('HEYCATCH_API_KEY is not set'), { status: 0 });
  for (let offset = 0; ; offset += 50) {
    const j = await api(`/articles?limit=50&offset=${offset}`);
    list.push(...(j.articles || []));
    if (!j.articles || !j.articles.length || list.length >= (j.total ?? 0)) break;
  }
} catch (e) {
  const why = { 0: 'no key', 401: 'wrong key', 403: 'key revoked or project deleted', 429: 'rate limited' }[e.status] || e.message;
  console.warn(`blog: left as is (${why})`);
  process.exit(0);
}
list = list.filter((a) => a && a.slug && a.published !== false);

const posts = [];
for (const a of list) {
  try { posts.push({ ...a, ...(await api(`/articles/${encodeURIComponent(a.slug)}`)) }); }
  catch (e) { if (e.status === 404) continue; console.warn(`blog: skipped ${a.slug} (${e.message})`); const old = path.join(root, 'blog', a.slug, 'index.html'); if (fs.existsSync(old)) posts.push({ ...a, keep: true }); }
}

// ---- shared page parts (the site's own header, footer and styles)
const head = ({ title, description, url, type = 'website', extra = '' }) => `<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>${esc(title)}</title>
${description ? `<meta name="description" content="${esc(description)}">\n` : ''}<link rel="canonical" href="${esc(url)}">
<meta name="theme-color" content="#07090F">
<meta property="og:type" content="${type}">
<meta property="og:site_name" content="Reel Studio by reeleditapp">
<meta property="og:url" content="${esc(url)}">
<meta property="og:title" content="${esc(title)}">
${description ? `<meta property="og:description" content="${esc(description)}">\n` : ''}<meta property="og:image" content="${SITE}/site/img/og.jpg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="${SITE}/site/img/og.jpg">
<link rel="icon" href="/icons/icon-192.png?v=3">
<link rel="apple-touch-icon" href="/icons/icon-180.png?v=3">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@108..125,800&family=Inter:wght@400;500;600;700;800&display=swap">
<link rel="stylesheet" href="/site/site.css">
<script type="module" src="/site/analytics.js"></script>${extra}`;
const header = `<div class="glow" aria-hidden="true"></div>
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
</header>`;
const footer = `<footer>
  <div class="wrap">
    <a class="brand" href="/" style="font-size:16px"><img src="/icons/icon-192.png?v=3" alt="" width="24" height="24">Reel <span>Studio</span></a>
    <span class="sp"></span>
    <a href="/app.html">Editor</a><a href="/pricing.html">Pricing</a>${posts.length ? '<a href="/blog/">Blog</a>' : ''}<a href="/data.html">Your data</a><a href="/#faq">Questions</a>
  </div>
</footer>`;
const meta = (a) => [a.published_at && `<time datetime="${esc(iso(a.published_at))}">${esc(day(a.published_at))}</time>`, a.read_time && `${esc(a.read_time)} min read`].filter(Boolean).join(' · ');
const write = (rel, html) => { const f = path.join(root, rel); fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, html); };

// ---- article pages
const clean = (html) => sanitizeHtml(String(html || ''), {
  allowedTags: ['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr', 'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption',
    'a', 'blockquote', 'strong', 'b', 'em', 'i', 'code', 'pre', 'span', 'sup', 'sub', 'section', 'div', 'details', 'summary', 'dl', 'dt', 'dd'],
  allowedAttributes: { a: ['href', 'title', 'rel', 'target'], h2: ['id'], h3: ['id'], h4: ['id'], th: ['scope', 'colspan', 'rowspan'], td: ['colspan', 'rowspan'], section: ['id'], div: ['id'] },
  allowedSchemes: ['http', 'https', 'mailto'],
  allowProtocolRelative: false,
  transformTags: { a: (tag, attribs) => ({ tagName: 'a', attribs: /^https?:/i.test(attribs.href || '') && !(attribs.href || '').startsWith(SITE) ? { ...attribs, rel: 'noopener' } : attribs }) },
});
const slugs = new Set();
for (const a of posts) {
  slugs.add(a.slug);
  if (a.keep) continue;   // a fetch hiccup: keep yesterday's page rather than a broken one
  const body = clean(a.content_html);
  const toc = [...body.matchAll(/<h2 id="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g)].filter((m) => m[1] !== 'faq')
    .map((m) => `<li><a href="#${esc(m[1])}">${m[2].replace(/<[^>]+>/g, '')}</a></li>`);
  const lang = String(a.language || 'en').toLowerCase().split(/[-_]/)[0];
  const langAttr = lang && lang !== 'en' ? ` lang="${esc(a.language)}"${RTL.has(lang) ? ' dir="rtl"' : ''}` : '';
  const url = `${SITE}/blog/${encodeURIComponent(a.slug)}/`;
  const ld = a.jsonLd ? `\n<script type="application/ld+json">${JSON.stringify(a.jsonLd).replace(/</g, '\\u003c')}</script>` : '';
  write(`blog/${a.slug}/index.html`, `<!doctype html>
<html lang="${lang === 'en' ? 'en' : esc(a.language)}"${RTL.has(lang) ? ' dir="rtl"' : ''}>
<head>
${head({ title: `${a.title} | Reel Studio blog`, description: a.meta_description, url, type: 'article',
  extra: `${a.published_at ? `\n<meta property="article:published_time" content="${esc(iso(a.published_at))}">` : ''}${a.updated_at ? `\n<meta property="article:modified_time" content="${esc(iso(a.updated_at))}">` : ''}${ld}` })}
</head>
<body>
${header}
<main>
  <section style="border:0">
    <article class="wrap post"${langAttr}>
      <p class="eyebrow"><a href="/blog/">Blog</a></p>
      <h1>${esc(a.title)}</h1>
      <p class="post-meta muted">${meta(a)}</p>
${toc.length ? `      <nav class="toc" aria-label="On this page"><b>On this page</b><ol>${toc.join('')}</ol></nav>\n` : ''}      <div class="post-body">
${body}
      </div>
      <p class="post-end"><a href="/blog/">← All articles</a></p>
    </article>
  </section>
</main>
${footer}
</body>
</html>
`);
}
// articles that were taken down
const dir = path.join(root, 'blog');
if (fs.existsSync(dir)) for (const d of fs.readdirSync(dir, { withFileTypes: true })) if (d.isDirectory() && !slugs.has(d.name)) fs.rmSync(path.join(dir, d.name), { recursive: true });

// ---- the index
write('blog/index.html', `<!doctype html>
<html lang="en">
<head>
${head({ title: 'Blog | Reel Studio by reeleditapp', description: 'Articles from Reel Studio on editing talking-head reels, short-form video and posting without the editing evening.', url: `${SITE}/blog/` })}
</head>
<body>
${header}
<main>
  <section style="border:0">
    <div class="wrap">
      <div class="head"><span class="eyebrow">Blog</span><h1 style="font-size:clamp(34px,5vw,54px)">The Reel Studio blog</h1></div>
${posts.length ? `      <ul class="post-list">
${posts.map((a) => `        <li><a href="/blog/${esc(encodeURIComponent(a.slug))}/"><h2>${esc(a.title)}</h2><span class="muted">${meta(a)}</span></a></li>`).join('\n')}
      </ul>` : '      <p class="lede">No articles yet.</p>'}
    </div>
  </section>
</main>
${footer}
</body>
</html>
`);

// ---- sitemap: the site's own pages, plus the blog
const smf = path.join(root, 'sitemap.xml');
let sm = fs.readFileSync(smf, 'utf8').replace(/\s*<url><loc>https:\/\/reeleditapp\.com\/blog\/[\s\S]*?<\/url>/g, '');
const latest = posts.map((a) => a.updated_at || a.published_at).filter(Boolean).sort().pop();
const urls = [`  <url><loc>${SITE}/blog/</loc>${latest ? `<lastmod>${iso(latest).slice(0, 10)}</lastmod>` : ''}<changefreq>daily</changefreq><priority>0.6</priority></url>`,
  ...posts.map((a) => `  <url><loc>${SITE}/blog/${esc(encodeURIComponent(a.slug))}/</loc>${(a.updated_at || a.published_at) ? `<lastmod>${iso(a.updated_at || a.published_at).slice(0, 10)}</lastmod>` : ''}<priority>0.6</priority></url>`)];
sm = sm.replace('</urlset>', urls.join('\n') + '\n</urlset>');
fs.writeFileSync(smf, sm);

// ---- "Blog" in the main menu, only while there are articles
for (const f of ['index.html', 'pricing.html', 'data.html', 'about.html', 'refund.html', 'vs/opus-clip.html']) {
  const p = path.join(root, f); let html = fs.readFileSync(p, 'utf8');
  html = html.replace(/<!--bake:blognav-->[\s\S]*?<!--\/bake:blognav-->/, `<!--bake:blognav-->${posts.length ? '<a href="/blog/">Blog</a>' : ''}<!--/bake:blognav-->`);
  fs.writeFileSync(p, html);
}
console.log(`blog: ${posts.length} article${posts.length === 1 ? '' : 's'}`);
