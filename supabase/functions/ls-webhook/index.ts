// Lemon Squeezy → Reel Studio. Every order, subscription change and license key lands here and is
// written to the buyer's account. Deploy with JWT verification OFF (Lemon Squeezy can't sign in).
// Secrets: LS_SIGNING_SECRET (the signing secret you typed when creating the webhook).
import { createClient } from 'npm:@supabase/supabase-js@2';
import { analytics } from 'npm:@heycatch/sdk@0.8.5/server';

// HeyCatch: purchases are business events only this server knows for sure. Same account id the app identifies with.
analytics.init({ projectKey: 'hck_pk_5nW7o4oHXVFis5fCSEQjTfBG0Hy6w1y4', install: { framework: 'web', agent: 'claude-code' } });

const db = () => createClient(Deno.env.get('SUPABASE_URL')!, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, { auth: { persistSession: false } });
const enc = new TextEncoder();

async function validSignature(raw: string, sig: string | null): Promise<boolean> {
  const secret = Deno.env.get('LS_SIGNING_SECRET');
  if (!secret || !sig) return false;
  const key = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const mac = new Uint8Array(await crypto.subtle.sign('HMAC', key, enc.encode(raw)));
  const hex = [...mac].map((b) => b.toString(16).padStart(2, '0')).join('');
  if (hex.length !== sig.length) return false;
  let diff = 0; for (let i = 0; i < hex.length; i++) diff |= hex.charCodeAt(i) ^ sig.charCodeAt(i);   // constant time
  return diff === 0;
}

// A one-time order is the lifetime plan (monthly orders are tracked through their subscription instead).
const isLifetime = (item: Record<string, unknown> | undefined) => {
  const ids = (Deno.env.get('LIFETIME_VARIANT_IDS') || '').split(',').map((s) => s.trim()).filter(Boolean);
  if (item && ids.includes(String(item.variant_id))) return true;
  return /life|forever|one[- ]?time/i.test(`${item?.product_name ?? ''} ${item?.variant_name ?? ''}`);
};
const uuid = (v: unknown) => typeof v === 'string' && /^[0-9a-f-]{36}$/i.test(v) ? v : null;

Deno.serve(async (req) => {
  if (req.method !== 'POST') return new Response('ok');
  const raw = await req.text();
  if (!(await validSignature(raw, req.headers.get('x-signature')))) return new Response('bad signature', { status: 401 });
  const body = JSON.parse(raw), event: string = body.meta?.event_name ?? '', a = body.data?.attributes ?? {};
  const userId = uuid(body.meta?.custom_data?.user_id), test = !!body.meta?.test_mode, sb = db();
  const now = new Date().toISOString();
  const keep = (row: Record<string, unknown>) => Object.fromEntries(Object.entries(row).filter(([, v]) => v !== undefined));
  const upsert = async (row: Record<string, unknown>) => {
    // never take a purchase away from the account it's already on
    const { data: cur } = await sb.from('entitlements').select('user_id').eq('id', row.id).maybeSingle();
    if (cur?.user_id) delete row.user_id;
    const { error } = await sb.from('entitlements').upsert(keep({ ...row, updated_at: now }), { onConflict: 'id' });
    if (error) throw error;
  };

  try {
    if (event === 'order_created' || event === 'order_refunded') {
      const item = a.first_order_item;
      if (isLifetime(item)) {
        await upsert({ id: `order:${body.data.id}`, user_id: userId ?? undefined, email: a.user_email, plan: 'lifetime',
          status: event === 'order_refunded' || a.refunded ? 'refunded' : a.status === 'paid' ? 'active' : a.status,
          ls_order_id: String(body.data.id), portal_url: a.urls?.receipt ?? undefined, test_mode: test });
        if (userId && event === 'order_refunded') await analytics.trackEvent('purchase_refunded', { plan: 'lifetime', test_mode: test }, { userId });
        else if (userId && a.status === 'paid') await analytics.trackEvent('purchase_completed', { plan: 'lifetime', test_mode: test }, { userId, set: { plan: 'lifetime' } });
      }
    } else if (event.startsWith('subscription_')) {
      const id = event.startsWith('subscription_payment_') ? a.subscription_id : body.data.id;
      if (id) {
        const row: Record<string, unknown> = { id: `sub:${id}`, user_id: userId ?? undefined, email: a.user_email, plan: 'monthly',
          ls_subscription_id: String(id), test_mode: test };
        if (!event.startsWith('subscription_payment_')) Object.assign(row, { status: a.status, renews_at: a.renews_at, ends_at: a.ends_at,
          portal_url: a.urls?.customer_portal, ls_order_id: a.order_id != null ? String(a.order_id) : undefined });
        else {   // payment events carry no subscription status; record the row if we've never seen it
          const { data: cur } = await sb.from('entitlements').select('id').eq('id', row.id).maybeSingle();
          if (!cur) row.status = event === 'subscription_payment_success' ? 'active' : 'past_due'; else delete row.email;
        }
        if (row.status !== undefined || event.startsWith('subscription_payment_')) await upsert(row);
        if (userId && event === 'subscription_created') await analytics.trackEvent('subscription_started', { plan: 'monthly', test_mode: test }, { userId, set: { plan: 'monthly' } });
        else if (userId && event === 'subscription_cancelled') await analytics.trackEvent('subscription_cancelled', { plan: 'monthly', test_mode: test }, { userId });
        else if (userId && event === 'subscription_expired') await analytics.trackEvent('subscription_ended', { plan: 'monthly', test_mode: test }, { userId });
      }
    } else if (event === 'license_key_created' || event === 'license_key_updated') {
      // the key belongs to an order we've (usually) already recorded; store it there so it can't be linked to anyone else
      const { data: rows } = await sb.from('entitlements').select('id').eq('ls_order_id', String(a.order_id));
      if (!rows?.length) return new Response('order not recorded yet', { status: 409 });   // arrived first: Lemon Squeezy retries shortly
      await sb.from('entitlements').update({ license_key: a.key, updated_at: now }).in('id', rows.map((r) => r.id));
    }
  } catch (e) {
    console.error(event, e);
    return new Response('error', { status: 500 });   // Lemon Squeezy retries
  }
  return new Response('ok');
});
