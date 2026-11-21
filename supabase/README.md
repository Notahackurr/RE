# Reel Studio accounts and payments

Premium belongs to a Reel Studio account (email + password), not to a device. Checkout is Stripe (Managed Payments: Stripe is the seller of record, through its Link checkout).

1. Someone taps a plan. If they're signed out, they create an account (or sign in) first.
2. The app opens the Stripe payment link with `?client_reference_id=<account id>&prefilled_email=<email>`.
3. Stripe tells `stripe-webhook` about the purchase, which writes it to their account.
4. The app notices within a few seconds and unlocks. Any device they sign in to has Premium too.
5. Customers manage or cancel at app.link.com; refunds are issued in the Stripe Dashboard (or by Link support).

Fallback: a purchase paid with the same email as the account attaches itself once that email is confirmed (`claim_purchases`).

## Files
- `schema.sql`: purchases, free-plan usage, the reel counter, and their security rules.
- `functions/stripe-webhook/index.ts`: receives Stripe events. Deploy with JWT verification **off**.
- `functions/account/index.ts`: deletes an account. Leave JWT verification **on**.
- `functions/stock/index.ts`: free stock search (needs `PEXELS_API_KEY`). JWT **off**. Optional.
- `functions/ls-webhook/index.ts`: the old Lemon Squeezy webhook (Lemon Squeezy is shutting down). Unused.

## Secrets (Supabase → Edge Functions → Secrets)
- `STRIPE_WEBHOOK_SECRET`: the signing secret (`whsec_…`) of the Stripe webhook below.
- `PEXELS_API_KEY`: optional, for the `stock` function.

## Stripe
- Settings → Managed Payments: turn it on.
- Products: "Reel Studio Premium (monthly)" $10/month recurring, and "Reel Studio Premium (lifetime)" $100 one time, each with a software/SaaS tax code.
- Payment links: one per product, with **Enable Managed Payments** ticked. Paste both links into `billing.json` (`monthly`, `lifetime`).
- Developers → Webhooks → Add destination: `https://<project>.supabase.co/functions/v1/stripe-webhook`, events
  `checkout.session.completed`, `checkout.session.async_payment_succeeded`, `customer.subscription.created`,
  `customer.subscription.updated`, `customer.subscription.deleted`, `charge.refunded`. Do it once in test mode and once in live mode.
