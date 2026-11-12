-- Reel Studio accounts and purchases. Paste this whole file into Supabase → SQL Editor → New query → Run.
-- Safe to run again: everything is "if not exists" / "or replace".

-- One row per purchase (a lifetime order, a monthly subscription, or the creator's own access).
create table if not exists public.entitlements (
  id                 text primary key,               -- 'order:<id>' | 'sub:<id>' | 'creator:<user id>'
  user_id            uuid references auth.users (id) on delete set null,
  email              text,                           -- the checkout email, used to attach purchases made before signing in
  plan               text not null check (plan in ('monthly', 'lifetime', 'creator')),
  status             text not null,                  -- active | on_trial | past_due | paused | cancelled | expired | refunded ...
  renews_at          timestamptz,
  ends_at            timestamptz,                    -- when a cancelled subscription stops
  portal_url         text,                           -- the customer's own manage/cancel page
  license_key        text unique,
  ls_order_id        text,
  ls_subscription_id text,
  test_mode          boolean default false,
  updated_at         timestamptz not null default now()
);
create index if not exists entitlements_user on public.entitlements (user_id);
create index if not exists entitlements_email on public.entitlements (lower(email));
create index if not exists entitlements_order on public.entitlements (ls_order_id);

-- People can only ever read their own purchases. Nobody can write from the app: only the server functions can.
alter table public.entitlements enable row level security;
drop policy if exists "read own purchases" on public.entitlements;
create policy "read own purchases" on public.entitlements for select to authenticated using (user_id = auth.uid());

-- Attach purchases made with your email (before you signed in, or from another device) to your account.
-- Only works once the account's email is confirmed, so nobody can claim someone else's purchase.
create or replace function public.claim_purchases()
returns integer language plpgsql security definer set search_path = public as $$
declare n integer := 0; em text;
begin
  select u.email into em from auth.users u where u.id = auth.uid() and u.email_confirmed_at is not null;
  if em is null then return 0; end if;
  update public.entitlements set user_id = auth.uid(), updated_at = now()
    where user_id is null and lower(email) = lower(em);
  get diagnostics n = row_count;
  return n;
end $$;
revoke all on function public.claim_purchases() from public, anon;
grant execute on function public.claim_purchases() to authenticated;

-- Creator access. After you've created your own account in the app, run this one line with your email:
--   insert into public.entitlements (id, user_id, email, plan, status)
--   select 'creator:' || id, id, email, 'creator', 'active' from auth.users where lower(email) = lower('YOU@EXAMPLE.COM')
--   on conflict (id) do nothing;

-- Shared cache for free stock searches (the stock function writes it; nobody reads it from the app).
create table if not exists public.stock_cache (key text primary key, items jsonb not null, at timestamptz not null default now());
alter table public.stock_cache enable row level security;

-- Public count of reels made (for the "X reels made with Reel Studio" line on the home page).
-- The app adds 1 when a reel finishes; nothing else is stored: no video, no words, no user or device ID.
create table if not exists public.counters (name text primary key, n bigint not null default 0);
alter table public.counters enable row level security;
insert into public.counters (name, n) values ('reels', 0) on conflict (name) do nothing;
create or replace function public.count_reel() returns void language sql security definer set search_path = public as
  $$ update public.counters set n = n + 1 where name = 'reels'; $$;
create or replace function public.reels_made() returns bigint language sql stable security definer set search_path = public as
  $$ select n from public.counters where name = 'reels'; $$;
grant execute on function public.count_reel() to anon, authenticated;
grant execute on function public.reels_made() to anon, authenticated;
