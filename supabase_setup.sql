-- Run this once in your Supabase project:
-- Dashboard → SQL Editor → New Query → paste & run

CREATE TABLE IF NOT EXISTS exec_profiles (
  id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id     TEXT        NOT NULL,
  exec_name   TEXT        NOT NULL,
  company     TEXT        NOT NULL,
  profile_data JSONB      NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exec_profiles_user_id    ON exec_profiles (user_id);
CREATE INDEX IF NOT EXISTS idx_exec_profiles_created_at ON exec_profiles (created_at DESC);
