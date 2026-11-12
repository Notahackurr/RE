// Reel Studio account actions that need the server:
//   { action: 'link-key', key }  link a license key to the signed-in account, once (it can't be linked to anyone else after)
//   { action: 'delete' }         delete the signed-in account (purchases stay on record and come back if you sign up again with that email)
// Secrets: LS_STORE_ID (your Lemon Squeezy store number), optional LIFETIME_VARIANT_IDS.
import { createClient } from 'npm:@supabase/supabase-js@2';

const ORIGINS = ['https://reeleditapp.com', 'https://www.reeleditapp.com', 'http://localhost:8765'];
const cors = (req: Request) => {
  const o = req.headers.get('origin') ?? '';
  return { 'Access-Control-Allow-Origin': ORIGINS.includes(o) ? o : ORIGINS[0], 'Access-Control-Allow-Headers': 'authorization, apikey, content-type, x-client-info',
    'Access-Control-Allow-Methods': 'POST, OPTIONS', Vary: 'Origin' };
};
const reply = (req: Request, body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { ...cors(req), 'Content-Type': 'application/json' } });

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: cors(req) });
  if (req.method !== 'POST') return reply(req, { error: 'Use POST.' }, 405);
  const url = Deno.env.get('SUPABASE_URL')!;
  const asUser = createClient(url, Deno.env.get('SUPABASE_ANON_KEY')!, { global: { headers: { Authorization: req.headers.get('Authorization') ?? '' } }, auth: { persistSession: false } });
  const { data: { user } } = await asUser.auth.getUser();
  if (!user) return reply(req, { error: 'Sign in again, then try that once more.' }, 401);
  const admin = createClient(url, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, { auth: { persistSession: false } });
  let body: { action?: string; key?: string } = {};
  try { body = await req.json(); } catch { /* empty */ }

  if (body.action === 'delete') {
    const { error } = await admin.auth.admin.deleteUser(user.id);
    return error ? reply(req, { error: "Couldn't delete the account. Try again in a moment." }, 500) : reply(req, { ok: true });
  }

  if (body.action === 'link-key') {
    const key = String(body.key ?? '').trim();
    if (!key) return reply(req, { error: 'Paste your license key first.' }, 400);
    if (!/^[A-Za-z0-9-]{8,80}$/.test(key)) return reply(req, { error: "That doesn't look like a license key." }, 400);
    const r = await fetch('https://api.lemonsqueezy.com/v1/licenses/validate', { method: 'POST', headers: { Accept: 'application/json' }, body: new URLSearchParams({ license_key: key }) });
    const j = await r.json().catch(() => ({}));
    if (!j.license_key) return reply(req, { error: "That key wasn't recognized. Copy it again from your receipt email." }, 400);
    const m = j.meta ?? {}, store = Deno.env.get('LS_STORE_ID');
    if (store && String(m.store_id) !== store.replace('#', '')) return reply(req, { error: "That key is for a different store's product." }, 400);
    const ids = (Deno.env.get('LIFETIME_VARIANT_IDS') || '').split(',').map((s) => s.trim()).filter(Boolean);
    const lifetime = ids.includes(String(m.variant_id)) || /life|forever|one[- ]?time/i.test(`${m.product_name ?? ''} ${m.variant_name ?? ''}`);

    // already recorded (by the webhook)? attach it, unless it's someone else's
    const { data: rows } = await admin.from('entitlements').select('id,user_id').or(`license_key.eq.${key},ls_order_id.eq.${m.order_id}`);
    const theirs = rows?.find((x) => x.user_id && x.user_id !== user.id);
    if (theirs) return reply(req, { error: 'That key is already linked to another Reel Studio account. Sign in to that account instead.' }, 409);
    if (rows?.length) {
      await admin.from('entitlements').update({ user_id: user.id, license_key: key, updated_at: new Date().toISOString() }).in('id', rows.map((x) => x.id));
      return reply(req, { ok: true });
    }
    if (!lifetime) return reply(req, { error: "We can't see that subscription yet. Give it a minute after paying, then try again." }, 404);
    const st = j.license_key.status;
    const { error } = await admin.from('entitlements').insert({ id: `order:${m.order_id}`, user_id: user.id, email: m.customer_email ?? user.email, plan: 'lifetime',
      status: st === 'disabled' ? 'refunded' : st === 'expired' ? 'expired' : 'active', license_key: key, ls_order_id: String(m.order_id) });
    return error ? reply(req, { error: "Couldn't link that key. Try again in a moment." }, 500) : reply(req, { ok: true });
  }
  return reply(req, { error: 'Unknown action.' }, 400);
});
