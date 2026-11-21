// Stripe → Reel Studio. Checkout (Stripe Managed Payments, through payment links), subscription changes and refunds
// land here and are written to the buyer's account. Deploy with JWT verification OFF (Stripe can't sign in).
// Secrets: STRIPE_WEBHOOK_SECRET (the webhook's signing secret, whsec_…).
// Events to send: checkout.session.completed, checkout.session.async_payment_succeeded,
//   customer.subscription.created, customer.subscription.updated, customer.subscription.deleted, charge.refunded
// The app opens payment links with ?client_reference_id=<account id>&prefilled_email=<email>, so every purchase
// arrives with the account it belongs to. Purchases made some other way attach by email once that email is confirmed
// (claim_purchases). Customers manage or cancel their orders at Link (app.link.com).
import { createClient } from 'npm:@supabase/supabase-js@2';

const db = () => createClient(Deno.env.get('SUPABASE_URL')!, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, { auth: { persistSession: false } });
const enc = new TextEncoder();
const PORTAL = 'https://app.link.com';
const TOLERANCE = 300;   // seconds a signed event stays valid

async function validSignature(raw: string, header: string | null): Promise<boolean> {
  const secret = Deno.env.get('STRIPE_WEBHOOK_SECRET');
  if (!secret || !header) return false;
  const parts = header.split(',').map((p) => p.split('=') as [string, string]);
  const t = parts.find(([k]) => k === 't')?.[1], sigs = parts.filter(([k]) => k === 'v1').map(([, v]) => v);
  if (!t || !sigs.length || Math.abs(Date.now() / 1000 - Number(t)) > TOLERANCE) return false;
  const key = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const mac = new Uint8Array(await crypto.subtle.sign('HMAC', key, enc.encode(`${t}.${raw}`)));
  const hex = [...mac].map((b) => b.toString(16).padStart(2, '0')).join('');
  return sigs.some((s) => { if (s.length !== hex.length) return false; let d = 0; for (let i = 0; i < hex.length; i++) d |= hex.charCodeAt(i) ^ s.charCodeAt(i); return d === 0; });
}

const uuid = (v: unknown) => typeof v === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(v) ? v : null;
const idOf = (v: unknown) => typeof v === 'string' ? v : (v && typeof v === 'object' && 'id' in v ? String((v as { id: unknown }).id) : null);
const iso = (sec: unknown) => typeof sec === 'number' && sec > 0 ? new Date(sec * 1000).toISOString() : undefined;

// Stripe subscription status → the app's vocabulary (active | on_trial | past_due | cancelled (still running until ends_at) | expired)
function subState(s: Record<string, any>) {
  const end = s.items?.data?.[0]?.current_period_end ?? s.current_period_end;
  const ending = s.cancel_at_period_end || s.cancel_at;
  const base: Record<string, string> = { active: 'active', trialing: 'on_trial', past_due: 'past_due', unpaid: 'expired', canceled: 'expired', incomplete_expired: 'expired', incomplete: 'pending', paused: 'paused' };
  let status = base[s.status] ?? s.status;
  if ((status === 'active' || status === 'on_trial') && ending) status = 'cancelled';
  return { status, renews_at: ['active', 'on_trial', 'past_due'].includes(status) ? iso(end) : undefined, ends_at: status === 'cancelled' ? iso(s.cancel_at ?? end) : status === 'expired' ? iso(s.ended_at ?? s.canceled_at) : undefined };
}

Deno.serve(async (req) => {
  if (req.method !== 'POST') return new Response('ok');
  const raw = await req.text();
  if (!(await validSignature(raw, req.headers.get('stripe-signature')))) return new Response('bad signature', { status: 401 });
  const ev = JSON.parse(raw), o = ev.data?.object ?? {}, test = ev.livemode === false, sb = db(), now = new Date().toISOString();
  const keep = (row: Record<string, unknown>) => Object.fromEntries(Object.entries(row).filter(([, v]) => v !== undefined && v !== null));
  const upsert = async (row: Record<string, unknown>) => {
    // never move a purchase off the account it's already on, and never let a late, stale event downgrade a known status
    const { data: cur } = await sb.from('entitlements').select('user_id,status').eq('id', row.id).maybeSingle();
    if (cur?.user_id) delete row.user_id;
    if (cur && row.status === 'active' && ev.type.startsWith('checkout.')) delete row.status;   // the subscription event knows better
    const { error } = await sb.from('entitlements').upsert(keep({ ...row, updated_at: now }), { onConflict: 'id' });
    if (error) throw error;
  };

  try {
    if (ev.type === 'checkout.session.completed' || ev.type === 'checkout.session.async_payment_succeeded') {
      if (o.payment_status !== 'paid' && o.payment_status !== 'no_payment_required') return new Response('waiting for payment');
      const base = { user_id: uuid(o.client_reference_id) ?? undefined, email: o.customer_details?.email ?? o.customer_email, stripe_customer_id: idOf(o.customer), portal_url: PORTAL, test_mode: test };
      if (o.mode === 'subscription' && o.subscription) {
        await upsert({ ...base, id: `sub:${idOf(o.subscription)}`, plan: 'monthly', status: 'active', stripe_subscription_id: idOf(o.subscription) });
      } else if (o.mode === 'payment' && o.payment_intent) {
        await upsert({ ...base, id: `pi:${idOf(o.payment_intent)}`, plan: 'lifetime', status: 'active', stripe_payment_intent: idOf(o.payment_intent) });
      }
    } else if (ev.type.startsWith('customer.subscription.')) {
      const st = subState(o);
      await upsert({ id: `sub:${o.id}`, plan: 'monthly', ...st, stripe_subscription_id: o.id, stripe_customer_id: idOf(o.customer), portal_url: PORTAL, test_mode: test });
    } else if (ev.type === 'charge.refunded') {
      // a full refund ends a lifetime purchase; a refunded first month is ended by cancelling the subscription in Stripe
      const pi = idOf(o.payment_intent);
      if (pi && o.refunded) await sb.from('entitlements').update({ status: 'refunded', updated_at: now }).eq('stripe_payment_intent', pi);
    }
  } catch (e) {
    console.error(ev.type, e);
    return new Response('error', { status: 500 });   // Stripe retries
  }
  return new Response('ok');
});
