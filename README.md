# Reel Studio

Reel editing in the browser. Your videos never leave your device.

- **Clip → Reel**: one take in, one finished vertical reel out. Pauses and clap-marked retakes trimmed, captions, clips and photo cards matched to the meaning and feel of each line, music, 1080×1920 export.
- **Long form → Reels**: a long video in, a batch of reels out. Every line is scored as a hook (question, contrarian, curiosity, number, result, story…), the best self-contained stretches are picked, and the strongest line can be moved to the front.

Live at https://reeleditapp.com (editor at https://reeleditapp.com/app.html)

## Files

```
index.html       public home page        app.html      the editor app (engine.js does the editing)
pricing.html, data.html                  pricing and how your data is handled
site/            styles, screenshots, share image, proof.json (real user proof, empty until collected)
robots.txt, sitemap.xml                  for search engines
stock/           free stock clips and photos (Mixkit Free License) + manifest.json
models/          face detector (YuNet, MIT)
trends.json      hook weights and trending topics (refreshed daily by .github/workflows/refresh-trends.yml)
icons/, manifest.webmanifest              app icon
tools/, .github/workflows/                stock fetcher (Pexels / Pixabay) and trends refresh
```

To run it locally, serve the folder over http (for example `npx serve .`).
