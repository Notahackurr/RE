# Reel Studio

Reel editing in the browser. Your videos never leave your device.

- **Clip → Reel**: one take in, one finished vertical reel out. Pauses and clap-marked retakes trimmed, captions, clips and photo cards matched to the meaning and feel of each line, music, 1080×1920 export.
- **Long form → Reels**: a long video in, a batch of reels out. Every line is scored as a hook (question, contrarian, curiosity, number, result, story…), the best self-contained stretches are picked, and the strongest line can be moved to the front.

Live at https://reeleditapp.com (editor at https://reeleditapp.com/app.html)

## Files

```
index.html       public home page        app.html      the editor app (engine.js does the editing)
pricing.html, data.html                  pricing and how your data is handled
about.html, refund.html                  founder page and refund policy (both stay hidden until site/founder.json / billing.json refundDays are filled)
vs/opus-clip.html                        honest comparison page
llms.txt                                 plain description for AI assistants
site/analytics.js                        HeyCatch analytics: public pages and the editor (one init; editor queues identity and events)
a/ … z/, 0/ … 9/                         HeyCatch short links: each forwards to /?utm_source=heycatch&utm_campaign=<char>
site/            styles, screenshots, share image; proof.json (reel count, testimonials, mentions) and founder.json, empty until you fill them;
                 put a real before/after at site/demo/before.mp4 + after.mp4 and it replaces the hero screenshots
robots.txt, sitemap.xml                  for search engines
stock/           free stock clips and photos (Mixkit Free License) + manifest.json
models/          face detector (YuNet, MIT)
trends.json      hook weights and trending topics (refreshed daily by .github/workflows/refresh-trends.yml)
icons/, manifest.webmanifest              app icon
tools/, .github/workflows/                stock fetcher (Pexels / Pixabay) and trends refresh
```

To run it locally, serve the folder over http (for example `npx serve .`).
