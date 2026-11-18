// Before anything is published: every public page must be whole. A page that's been cut off (no footer, no closing tags)
// stops the publish, so the live site keeps its last good version. Exit code 1 = don't publish.
import fs from 'node:fs';
import path from 'node:path';
const root = path.resolve(new URL('..', import.meta.url).pathname);
const pages = ['index.html', 'pricing.html', 'data.html', 'about.html', 'refund.html', 'contact.html', 'vs/opus-clip.html', '404.html', 'blog/index.html'];
const blog = path.join(root, 'blog');
if (fs.existsSync(blog)) for (const d of fs.readdirSync(blog, { withFileTypes: true })) {
  if (d.isFile() && d.name.endsWith('.html') && d.name !== 'index.html') pages.push(`blog/${d.name}`);
  if (d.isDirectory() && fs.existsSync(path.join(blog, d.name, 'index.html'))) pages.push(`blog/${d.name}/index.html`);
}
const bad = [];
for (const f of pages) {
  const p = path.join(root, f); if (!fs.existsSync(p)) { if (!f.startsWith('blog/')) bad.push(`${f}: missing`); continue; }
  const h = fs.readFileSync(p, 'utf8'), miss = ['</main>', '<footer', '</footer>', '</body>', '</html>'].filter((t) => !h.includes(t));
  if (miss.length) bad.push(`${f}: cut off (no ${miss.join(', ')})`);
}
const app = fs.readFileSync(path.join(root, 'app.html'), 'utf8');
if (!app.includes('</html>') || !app.includes('id="view-batch"') || !app.includes('window.__S = S')) bad.push('app.html: cut off');
const home = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
for (const id of ['id="jobs"', 'id="pricing"', 'id="faq"']) if (!home.includes(id)) bad.push(`index.html: the ${id.slice(4, -1)} section is gone`);
if (bad.length) { console.error('Not publishing. These pages look broken:\n  ' + bad.join('\n  ')); process.exit(1); }
console.log(`pages ok (${pages.length + 1} checked)`);
