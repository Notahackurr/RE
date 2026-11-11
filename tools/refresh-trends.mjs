// Refresh trends.json: today's trending searches (Google Trends, US) as topics that boost a matching hook line.
// Hook weights are kept as they are. Runs daily from .github/workflows/refresh-trends.yml; no key needed.
import fs from 'node:fs';
const file = new URL('../trends.json', import.meta.url);
const cur = JSON.parse(fs.readFileSync(file, 'utf8'));
const res = await fetch('https://trends.google.com/trending/rss?geo=US', { headers: { 'user-agent': 'Mozilla/5.0 (reeleditapp trends refresh)' } });
if (!res.ok) { console.log(`Trends feed returned ${res.status}; keeping the current file.`); process.exit(0); }
const xml = await res.text();
const titles = [...xml.matchAll(/<item>[\s\S]*?<title>(?:<!\[CDATA\[)?([^<\]]+)(?:\]\]>)?<\/title>/g)].map(m => m[1].trim().toLowerCase()).filter(t => t && t.length <= 40);
if (!titles.length) { console.log('No topics found in the feed; keeping the current file.'); process.exit(0); }
cur.topics = [...new Set(titles)].slice(0, 30);
cur.updated = new Date().toISOString().slice(0, 10);
fs.writeFileSync(file, JSON.stringify(cur, null, 2) + '\n');
console.log(`Saved ${cur.topics.length} topics:`, cur.topics.join(', '));
