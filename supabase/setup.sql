-- Hybrid Face Recognition — Supabase setup
-- Run in Supabase Dashboard → SQL Editor

-- 1. Table for face encodings
create table if not exists public.registered_faces (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  num_images int not null default 1,
  avg_encoding jsonb not null,
  image_folder text,
  created_at timestamptz not null default now()
);

create index if not exists registered_faces_name_idx on public.registered_faces (name);

-- 2. Row Level Security (open policies for demo — tighten for production)
alter table public.registered_faces enable row level security;

drop policy if exists "faces_select" on public.registered_faces;
drop policy if exists "faces_insert" on public.registered_faces;
drop policy if exists "faces_update" on public.registered_faces;

create policy "faces_select" on public.registered_faces for select using (true);
create policy "faces_insert" on public.registered_faces for insert with check (true);
create policy "faces_update" on public.registered_faces for update using (true);

-- 3. Storage bucket (create "faces" in Dashboard → Storage if SQL insert fails)
insert into storage.buckets (id, name, public)
values ('faces', 'faces', true)
on conflict (id) do update set public = true;

-- 4. Storage policies — public read, anon upload/update for demo
drop policy if exists "faces_storage_select" on storage.objects;
drop policy if exists "faces_storage_insert" on storage.objects;
drop policy if exists "faces_storage_update" on storage.objects;

create policy "faces_storage_select" on storage.objects
  for select using (bucket_id = 'faces');

create policy "faces_storage_insert" on storage.objects
  for insert with check (bucket_id = 'faces');

create policy "faces_storage_update" on storage.objects
  for update using (bucket_id = 'faces');
