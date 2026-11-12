# Reel Studio accounts and payments

Premium belongs to a Reel Studio account (email + password), not to a device.

1. Someone taps a plan. If they're signed out, they create an account (or sign in) first.
2. Checkout opens with their account attached (`checkout[custom][user_id]`).
3. Lemon Squeezy tells `ls-webhook` about the purchase, which writes it to their account.
4. The app notices within a few seconds and unlocks. Any device they sign in to has Premium too.

Fallbacks:
- **Bought with the same email before signing up:** the purchase attaches itself once that email is confirmed (`claim_purchases`).
- **Something went missing:** they link their license key to their account, once (`account` → `link-key`). After that the key can't be linked to anyone else.

## Files
- `schema.sql`: the purchases table, its security rules, and the claim function.
- `functions/ls-webhook/index.ts`: receives Lemon Squeezy events. Deploy with JWT verification **off**.
- `functions/account/index.ts`: links a license key, or deletes an account. Leave JWT verification **on**.

## Secrets (Supabase → Edge Functions → Secrets)
- `LS_SIGNING_SECRET`: the same random string you type into the Lemon Squeezy webhook.
- `LS_STORE_ID`: `521572`
- `LIFETIME_VARIANT_IDS`: comma-separated lifetime variant IDs. Test mode is `2365610`; add the live one after you copy to live mode.

## Lemon Squeezy webhook (Settings → Webhooks → +, once in test mode and again in live mode)
- **URL:** `https://YOUR-PROJECT.supabase.co/functions/v1/ls-webhook`
- **Events:**
  - Orders: `order_created`, `order_refunded`
  - Subscriptions: `subscription_created`, `subscription_updated`, `subscription_cancelled`, `subscription_resumed`, `subscription_expired`, `subscription_paused`, `subscription_unpaused`
  - Subscription payments: `subscription_payment_success`, `subscription_payment_failed`
  - License keys: `license_key_created`

## App config (`billing.json`)
`supabaseUrl` and `supabaseAnonKey` come from Project Settings → API. The anon key is public by design; never put the `service_role` key anywhere in the site.
