// Writes the site's proof straight into the HTML, so it's in the page itself (search engines, link previews and site
// audits read the served markup, not what a script draws later):
//   site/founder.json → the "Who's behind it" card and footer link on the home page, and the About page
//   site/proof.json   → quotes from named users, dated mentions elsewhere, early-user faces, the reels-made count
//   live count        → reels_made from Supabase, once billing.json has the project URL and anon key
// Runs from .github/workflows/bake-proof.yml whenever those files change, and once a day for the count. Safe to run again.
import fs from 'node:fs';

const root = new URL('..', import.meta.url).pathname;
const read = (f) => JSON.parse(fs.readFileSync(root + f, 'utf8'));
const esc = (x) => String(x ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const url = (u) => /^(https:\/\/|site\/|\/)/.test(String(u || '')) ? esc(u) : '';
const MIN_COUNT = 50;   // a count shows once it's more than a handful

const founder = read('site/founder.json'), proof = read('site/proof.json'), bill = read('billing.json'), contact = read('site/contact.json');
const email = /^[^\s@<>"]+@[^\s@<>"]+\.[a-z]{2,}$/i.test(String(contact.email || '').trim()) ? String(contact.email).trim() : '';
const name = String(founder.name || '').trim(), first = name.split(/\s+/)[0];
const quotes = (proof.quotes || []).filter((q) => q && q.name && q.text);
const mentions = (proof.mentions || []).filter((m) => m && m.name && /^https:\/\//.test(m.link || '') && m.date);
const avatars = (proof.avatars || []).filter(Boolean);

let count = Number(proof.reelsMade) || 0;
if (bill.supabaseUrl && bill.supabaseAnonKey) {
  try {
    const r = await fetch(String(bill.supabaseUrl).replace(/\/+$/, '') + '/rest/v1/rpc/reels_made', {
      method: 'POST', headers: { apikey: bill.supabaseAnonKey, 'Content-Type': 'application/json' }, body: '{}' });
    if (r.ok) count = Math.max(count, Number(await r.json()) || 0);
  } catch (e) { console.warn('live count unavailable:', e.message); }
}

const month = (d) => { const t = new Date(String(d).length === 7 ? d + '-01T12:00:00' : d + 'T12:00:00'); return isNaN(t) ? esc(d) : t.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }); };

// ---- home page
const usage = (() => {
  const faces = avatars.length ? `<span class="faces">${avatars.slice(0, 6).map((a) => `<img src="${url(a)}" alt="" width="30" height="30">`).join('')}</span>` : '';
  const parts = [];
  if (count >= MIN_COUNT) parts.push(`<b data-n>${count.toLocaleString('en-US')}</b> reels made so far`);
  if (proof.cohort) parts.push(esc(proof.cohort));
  const show = faces || parts.length;
  return `<p class="usage" id="usage" data-count="${count >= MIN_COUNT ? count : 0}" data-min="${MIN_COUNT}"${show ? '' : ' hidden'}>${faces}${parts.length ? `<span>${parts.join(' · ')}</span>` : ''}</p>`;
})();

// Each quote shows the person's own photo when they've given one; otherwise their initials (never a stand-in face).
const face = (q) => {
  if (q.photo) return `<img src="${url(q.photo)}" alt="" width="44" height="44" class="qface">`;
  const ini = String(q.name).replace(/[^\p{L}\s]/gu, ' ').trim().split(/\s+/).map((w) => w[0]).join('').slice(0, 2).toUpperCase();
  let h = 0; for (const c of String(q.name)) h = (h * 31 + c.codePointAt(0)) % 360;
  return `<span class="qface ini" style="--h:${h}" aria-hidden="true">${esc(ini)}</span>`;
};
const proofwall = !quotes.length ? '' : `<section id="proofwall">
    <div class="wrap">
      <div class="head"><span class="eyebrow">From people using it</span><h2>What EDDIE's doing for them</h2></div>
      <div class="quotes">
${quotes.map((q) => `        <figure><blockquote>“${esc(q.text)}”</blockquote><figcaption>${face(q)}<span><b>${esc(q.name)}</b>${q.role ? `, ${esc(q.role)}` : ''}${q.link ? ` · <a href="${url(q.link)}" rel="nofollow noopener">${esc(q.linkLabel || 'Original post')}</a>` : ''}</span></figcaption></figure>`).join('\n')}
      </div>
    </div>
  </section>`;

const elsewhere = !mentions.length ? '' : `<section id="elsewhere" class="elsewhere">
    <div class="wrap">
      <span class="eyebrow">Elsewhere</span>
      <ul class="mentions">
${mentions.map((m) => `        <li>${m.badge ? `<img src="${url(m.badge)}" alt="" height="40">` : ''}<span><a href="${url(m.link)}" rel="noopener">${esc(m.name)}</a>${m.note ? `: ${esc(m.note)}` : ''}</span><time datetime="${esc(m.date)}">${month(m.date)}</time></li>`).join('\n')}
      </ul>
    </div>
  </section>`;

const founderCard = name
  ? `<div class="maker">${founder.photo ? `<img src="${url(founder.photo)}" alt="${esc(name)}" width="56" height="56">` : ''}<div><b>${esc(name)}</b>${founder.role ? `<span>${esc(founder.role)}</span>` : ''}</div></div>
          <p class="muted">Reel Studio is made by ${esc(first)}, an independent creator who edits his own reels with it.</p>
          <p class="muted">It's one product, with a free plan that stays free and no ads.</p>
          <p><a href="about.html">More about who makes it →</a></p>`
  : `<p class="muted">Reel Studio is made by an independent creator who edits his own reels with it.</p>
          <p class="muted">It's one product, with a free plan that stays free and no ads.</p>`;

// ---- About page
const links = (founder.links || []).filter((l) => l && /^https:\/\//.test(l.url || ''));
const about = name
  ? `  <span class="eyebrow">About</span>
  <h1 style="font-size:clamp(32px,4.6vw,48px);margin-top:10px">Hi, I'm ${esc(first.replace(/\.$/, ''))}.</h1>
  <div class="founder">${founder.photo ? `<img src="${url(founder.photo)}" alt="${esc(name)}" width="180" height="180">` : '<span></span>'}
    <div>${String(founder.story || '').split(/\n\s*\n/).filter(Boolean).map((p) => `<p class="lede">${esc(p.trim())}</p>`).join('')}
      <p class="muted"><b style="color:var(--fg)">${esc(name)}</b>${founder.role ? ` · ${esc(founder.role)}` : ''}</p>
      ${founder.since || founder.based ? `<p class="muted">${[founder.since && `Building Reel Studio since ${esc(founder.since)}`, founder.based && `from ${esc(founder.based)}`].filter(Boolean).join(' ')}.</p>` : ''}
      <p class="muted">Questions or feedback? ${email ? `Write to <a href="mailto:${esc(email)}">${esc(email)}</a>, or see the <a href="/contact">Contact page</a>.` : 'See the <a href="/contact">Contact page</a>.'}</p>
      ${links.length ? `<p>${links.map((l) => `<a href="${url(l.url)}" rel="me noopener">${esc(l.label || l.url)}</a>`).join(' · ')}</p>` : ''}
    </div>
  </div>`
  : `  <span class="eyebrow">About</span>
  <h1 style="font-size:clamp(32px,4.6vw,48px);margin-top:10px">Who makes Reel Studio</h1>
  <p class="lede" style="margin-top:16px">Reel Studio is made by an independent creator who edits his own reels with it.</p>`;

function bake(file, regions) {
  let html = fs.readFileSync(root + file, 'utf8');
  for (const [key, inner] of Object.entries(regions)) {
    const re = new RegExp(`<!--bake:${key}-->[\\s\\S]*?<!--/bake:${key}-->`);
    if (!re.test(html)) throw new Error(`${file}: missing <!--bake:${key}--> marker`);
    html = html.replace(re, () => `<!--bake:${key}-->${inner}<!--/bake:${key}-->`);
  }
  fs.writeFileSync(root + file, html);
}
bake('index.html', {
  usage, proofwall, elsewhere, founder: '\n          ' + founderCard + '\n          ',
});
const footemail = email ? `<a href="mailto:${esc(email)}">Email</a>` : '';
for (const f of ['index.html', 'pricing.html', 'data.html', 'about.html', 'refund.html', 'vs/opus-clip.html', '404.html', 'contact.html']) bake(f, { footemail });
bake('contact.html', { contact: '\n' + (email
  ? `  <div class="card" style="margin:28px 0"><h2 style="margin:0">Email</h2><p style="margin:0;font-size:20px"><a href="mailto:${esc(email)}">${esc(email)}</a></p><p class="muted" style="margin:0">Usually answered within a day or two.</p></div>`
  : `  <div class="card" style="margin:28px 0"><h2 style="margin:0">Email</h2><p class="muted" style="margin:0">The contact address is being set up and will be posted here shortly.</p></div>`) + '\n' });
bake('about.html', {
  robots: name ? '' : '<meta name="robots" content="noindex">',
  title: `<title>${name ? `About ${esc(name)}` : 'About'} | Reel Studio by reeleditapp</title>`,
  about: '\n' + about + '\n',
});
console.log(`baked: founder ${name ? 'named' : 'not set'}, ${quotes.length} quotes, ${mentions.length} mentions, ${avatars.length} faces, count ${count}${count < MIN_COUNT ? ' (hidden under ' + MIN_COUNT + ')' : ''}`);
