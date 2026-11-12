// HeyCatch analytics for reeleditapp.com: the public pages and the editor (app.html) are one project, so one init, no `site` tag.
// Pageviews and clicks are captured automatically. Version pinned to the latest stable release.
import { analytics } from 'https://esm.sh/@heycatch/sdk@0.8.5';

// Sign-in links from Supabase emails land with the session in the address (#access_token=…). Never send that anywhere.
const SECRET = /\b(access_token|refresh_token|provider_token|provider_refresh_token|token_hash)=[^&#\s"]*/gi;
const scrub = (o) => {
  if (!o || typeof o !== 'object') return;
  for (const k of Object.keys(o)) {
    const v = o[k];
    if (typeof v === 'string') { if (v.includes('_token')) o[k] = v.replace(SECRET, '$1=redacted'); }
    else if (k === '$set' || k === '$set_once') scrub(v);
  }
};

analytics.init({
  projectKey: 'hck_pk_5nW7o4oHXVFis5fCSEQjTfBG0Hy6w1y4',
  install: { framework: 'web', agent: 'claude-code' },
  requestBatching: false,   // the public pages are full page loads
  beforeSend: (event) => { if (event) scrub(event.properties); return event; },
});

// The editor is a classic script, so it queues sign-in, sign-out and business events in window.__hcq until this module has loaded.
const CALLS = ['setIdentity', 'resetIdentity', 'setPersonProperties', 'trackEvent'];
window.__hcRun = () => {
  const q = window.__hcq || []; window.__hcq = [];
  for (const [fn, args] of q) if (CALLS.includes(fn)) { try { analytics[fn](...args); } catch (e) { /* analytics never breaks the app */ } }
};
window.__hcRun();
