// HeyCatch analytics for the public pages (home, pricing, comparison, about, data, refund). The editor (app.html) doesn't load it.
// Pageviews and clicks are captured automatically. Version pinned to the latest stable release.
import { analytics } from 'https://esm.sh/@heycatch/sdk@0.8.5';

analytics.init({
  projectKey: 'hck_pk_5nW7o4oHXVFis5fCSEQjTfBG0Hy6w1y4',
  install: { framework: 'web', agent: 'claude-code' },
  site: 'landing',
  requestBatching: false,
});
