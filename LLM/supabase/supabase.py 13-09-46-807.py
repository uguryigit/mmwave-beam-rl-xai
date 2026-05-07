from dotenv import load_dotenv
import os, json, math
import psycopg2
from psycopg2.extras import execute_batch

load_dotenv(".env")

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    sslmode="require",
)
with conn, conn.cursor() as cur:
    cur.execute("SELECT 1;")
    print("OK (pooler)")

# --------------------------
# 1) Schema Migration (PK/FK + algo kolonları)
# --------------------------
MIGRATION_SQL = """
-- algo TEXT sütunlarının varlığını garanti et
ALTER TABLE IF EXISTS public.runs
  ADD COLUMN IF NOT EXISTS algo TEXT;
ALTER TABLE IF EXISTS public.run_seeds
  ADD COLUMN IF NOT EXISTS algo TEXT;
ALTER TABLE IF EXISTS public.run_best_actions
  ADD COLUMN IF NOT EXISTS algo TEXT;

-- PK/FK'leri kompozit hale getir (algo, run_id)

-- runs: eski PK'yi düşür → yeni PK (algo, run_id)
DO $$
DECLARE pkname text;
BEGIN
  SELECT conname INTO pkname
  FROM pg_constraint
  WHERE conrelid = 'public.runs'::regclass AND contype='p'
  LIMIT 1;
  IF pkname IS NOT NULL THEN
    EXECUTE 'ALTER TABLE public.runs DROP CONSTRAINT ' || quote_ident(pkname);
  END IF;
END$$;

ALTER TABLE public.runs
  ADD CONSTRAINT runs_pk PRIMARY KEY (algo, run_id);

-- run_seeds: eski PK/FK'yi düşür → yeni PK/FK
DO $$
DECLARE pkname text;
BEGIN
  -- FK'yi düşür
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid='public.run_seeds'::regclass
      AND contype='f'
  ) THEN
    EXECUTE (
      SELECT 'ALTER TABLE public.run_seeds DROP CONSTRAINT ' || quote_ident(conname)
      FROM pg_constraint
      WHERE conrelid='public.run_seeds'::regclass AND contype='f'
      LIMIT 1
    );
  END IF;

  -- PK'yi düşür
  SELECT conname INTO pkname
  FROM pg_constraint
  WHERE conrelid='public.run_seeds'::regclass AND contype='p'
  LIMIT 1;
  IF pkname IS NOT NULL THEN
    EXECUTE 'ALTER TABLE public.run_seeds DROP CONSTRAINT ' || quote_ident(pkname);
  END IF;
END$$;

ALTER TABLE public.run_seeds
  ADD CONSTRAINT run_seeds_pk PRIMARY KEY (algo, run_id);

ALTER TABLE public.run_seeds
  ADD CONSTRAINT run_seeds_runs_fk
  FOREIGN KEY (algo, run_id) REFERENCES public.runs (algo, run_id) ON DELETE CASCADE;

-- run_best_actions: eski PK/FK'yi düşür → yeni PK/FK
DO $$
DECLARE pkname text;
BEGIN
  -- FK
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid='public.run_best_actions'::regclass
      AND contype='f'
  ) THEN
    EXECUTE (
      SELECT 'ALTER TABLE public.run_best_actions DROP CONSTRAINT ' || quote_ident(conname)
      FROM pg_constraint
      WHERE conrelid='public.run_best_actions'::regclass AND contype='f'
      LIMIT 1
    );
  END IF;

  -- PK
  SELECT conname INTO pkname
  FROM pg_constraint
  WHERE conrelid='public.run_best_actions'::regclass AND contype='p'
  LIMIT 1;
  IF pkname IS NOT NULL THEN
    EXECUTE 'ALTER TABLE public.run_best_actions DROP CONSTRAINT ' || quote_ident(pkname);
  END IF;
END$$;

ALTER TABLE public.run_best_actions
  ADD CONSTRAINT run_best_actions_pk PRIMARY KEY (algo, run_id, rank);

ALTER TABLE public.run_best_actions
  ADD CONSTRAINT run_best_actions_runs_fk
  FOREIGN KEY (algo, run_id) REFERENCES public.runs (algo, run_id) ON DELETE CASCADE;

-- Indeksler
CREATE INDEX IF NOT EXISTS runs_reward_idx
  ON public.runs (algo, final_eval_mean_reward DESC);

CREATE INDEX IF NOT EXISTS rba_dir_ang_idx
  ON public.run_best_actions (algo, dir_deg, ang_deg);
"""

with conn, conn.cursor() as cur:
    cur.execute(MIGRATION_SQL)

with conn, conn.cursor() as cur:
    cur.execute("UPDATE public.runs SET algo='dqn' WHERE algo IS NULL;")
    cur.execute("UPDATE public.run_seeds SET algo='dqn' WHERE algo IS NULL;")
    cur.execute("UPDATE public.run_best_actions SET algo='dqn' WHERE algo IS NULL;")

print("Migration bitti / kolonlar, PK-FK ve indexler ayarlandı.")

# --------------------------
# 2) UPSERT queries (yalnızca mevcut runs sütunları)
# --------------------------
INSERT_RUN = """
INSERT INTO public.runs
(algo, run_id, start_time, end_time, final_eval_mean_reward,
 position_x, position_y, top1_action_freq, action_entropy,
 run_dir, final_model_path, early_stopped)
VALUES (%(algo)s, %(run_id)s, %(start_time)s, %(end_time)s, %(final_eval_mean_reward)s,
        %(position_x)s, %(position_y)s, %(top1_action_freq)s, %(action_entropy)s,
        %(run_dir)s, %(final_model_path)s, %(early_stopped)s)
ON CONFLICT (algo, run_id) DO UPDATE SET
 start_time = EXCLUDED.start_time,
 end_time = EXCLUDED.end_time,
 final_eval_mean_reward = EXCLUDED.final_eval_mean_reward,
 position_x = EXCLUDED.position_x,
 position_y = EXCLUDED.position_y,
 top1_action_freq = EXCLUDED.top1_action_freq,
 action_entropy = EXCLUDED.action_entropy,
 run_dir = EXCLUDED.run_dir,
 final_model_path = EXCLUDED.final_model_path,
 early_stopped = EXCLUDED.early_stopped;
"""

INSERT_SEEDS = """
INSERT INTO public.run_seeds
(algo, run_id, master, np_seed, py_seed, torch_seed, env_seed, action_seed, replay_seed, eval_seed)
VALUES (%(algo)s, %(run_id)s, %(master)s, %(np_seed)s, %(py_seed)s, %(torch_seed)s, %(env_seed)s, %(action_seed)s, %(replay_seed)s, %(eval_seed)s)
ON CONFLICT (algo, run_id) DO UPDATE SET
 master = EXCLUDED.master,
 np_seed = EXCLUDED.np_seed,
 py_seed = EXCLUDED.py_seed,
 torch_seed = EXCLUDED.torch_seed,
 env_seed = EXCLUDED.env_seed,
 action_seed = EXCLUDED.action_seed,
 replay_seed = EXCLUDED.replay_seed,
 eval_seed = EXCLUDED.eval_seed;
"""

INSERT_BEST = """
INSERT INTO public.run_best_actions
(algo, run_id, rank, dir_idx, dir_deg, ang_idx, ang_deg, count, freq)
VALUES (%(algo)s, %(run_id)s, %(rank)s, %(dir_idx)s, %(dir_deg)s, %(ang_idx)s, %(ang_deg)s, %(count)s, %(freq)s)
ON CONFLICT (algo, run_id, rank) DO UPDATE SET
 dir_idx = EXCLUDED.dir_idx,
 dir_deg = EXCLUDED.dir_deg,
 ang_idx = EXCLUDED.ang_idx,
 ang_deg = EXCLUDED.ang_deg,
 count   = EXCLUDED.count,
 freq    = EXCLUDED.freq;
"""

# --------------------------
# 3) JSON uploader helpers
# --------------------------
def compute_action_entropy(best_actions):
    p = [a.get("freq", 0.0) for a in (best_actions or [])]
    p = [x for x in p if x and x > 0]
    if not p:
        return None
    return -sum(pi * math.log(pi + 1e-12) for pi in p)

def load_runs_json(json_path, algo):
    """all_runs_summary.json formatını okuyup rows listeleri döndürür."""
    with open(json_path, "r") as f:
        data = json.load(f)

    runs_rows, seeds_rows, best_rows = [], [], []
    for r in data["runs"]:
        pos = r.get("position") or [None, None]

        # top-1 freq
        t1 = None
        for a in r.get("best_actions", []):
            if a.get("rank") == 1:
                t1 = a.get("freq")
                break

        runs_rows.append(dict(
            algo=algo,
            run_id=r["run_id"],
            start_time=r.get("start_time"),
            end_time=r.get("end_time"),
            final_eval_mean_reward=r.get("final_eval_mean_reward") or r.get("final_eval_reward"),
            position_x=(int(pos[0]) if isinstance(pos[0], (int, float)) else None),
            position_y=(int(pos[1]) if isinstance(pos[1], (int, float)) else None),
            top1_action_freq=t1,
            action_entropy=compute_action_entropy(r.get("best_actions")),
            run_dir=r.get("run_dir"),
            final_model_path=r.get("final_model_path"),
            early_stopped=r.get("early_stopped", False),
        ))

        s = r.get("seed") or {}
        seeds_rows.append(dict(
            algo=algo,
            run_id=r["run_id"],
            master=s.get("master"),
            np_seed=s.get("np_seed"),
            py_seed=s.get("py_seed"),
            torch_seed=s.get("torch_seed"),
            env_seed=s.get("env_seed"),
            action_seed=s.get("action_seed"),
            replay_seed=s.get("replay_seed"),
            eval_seed=s.get("eval_seed"),
        ))

        for a in r.get("best_actions", []):
            best_rows.append(dict(
                algo=algo,
                run_id=r["run_id"],
                rank=a.get("rank"),
                dir_idx=a.get("dir_idx"),
                dir_deg=a.get("dir_deg"),
                ang_idx=a.get("ang_idx"),
                ang_deg=a.get("ang_deg"),
                count=a.get("count"),
                freq=a.get("freq"),
            ))

    return runs_rows, seeds_rows, best_rows

def upsert_all(runs_rows, seeds_rows, best_rows):
    with conn, conn.cursor() as cur:
        if runs_rows:
            execute_batch(cur, INSERT_RUN, runs_rows, page_size=500)
        if seeds_rows:
            execute_batch(cur, INSERT_SEEDS, seeds_rows, page_size=500)
        if best_rows:
            execute_batch(cur, INSERT_BEST, best_rows, page_size=1000)

# --------------------------
# 4) Example usage
# --------------------------
DQN_JSON = "/Users/otisvaliant/Desktop/supabase_presentation/all_runs_summary_dqn.json"
QF_JSON  = "/Users/otisvaliant/Desktop/supabase_presentation/all_summary_q_factorized.json"

if os.path.exists(DQN_JSON):
    runs_rows, seeds_rows, best_rows = load_runs_json(DQN_JSON, algo="dqn")
    upsert_all(runs_rows, seeds_rows, best_rows)
    print(f"DQN yükleme: runs={len(runs_rows)}, seeds={len(seeds_rows)}, best={len(best_rows)}")

if os.path.exists(QF_JSON):
    runs_rows, seeds_rows, best_rows = load_runs_json(QF_JSON, algo="q_factorized")
    upsert_all(runs_rows, seeds_rows, best_rows)
    print(f"QF yükleme: runs={len(runs_rows)}, seeds={len(seeds_rows)}, best={len(best_rows)}")