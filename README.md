# AI Explainability for Adaptive mmWave Beam Configuration in Dynamic Vehicular Environments

Companion code for the paper **"AI Explainability for Adaptive mmWave Beam Configuration in Dynamic Vehicular Environments"** (Yigit *et al.*).
The full PDF is included under [`paper/`](paper/Yigit2026_mmWave_BeamRL_XAI.pdf).

This repository couples two components:

1. A reproducible **mmWave vehicular simulator + reinforcement-learning trainer** that jointly optimises base-station placement, beam azimuth, and beamwidth to maximise system sum-rate.
2. A multi-agent **LLM-based explainability layer** that turns numerical training logs, scenario rules, and visual outputs into structured natural-language explanations.

> **Why?** In safety-critical 5G/6G O-RAN deployments, operators need to understand *why* a learned policy chose a given beam — not only *that* it works. We close that gap by combining RL-based optimisation with evidence-grounded LLM reasoning.

---

## Table of contents

- [System overview](#system-overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Configuration (.env)](#configuration-env)
- [Running the simulator and training](#running-the-simulator-and-training)
- [The LLM explainability layer](#the-llm-explainability-layer)
- [Reproducing the paper figures](#reproducing-the-paper-figures)
- [Citation](#citation)

---

## System overview

### Scenario
- Urban map: **500 m × 500 m** with a sparse two-way road grid and a central **roundabout** at (125 m, 125 m), `r = 25 m`.
- **N = 20** kinematic vehicles spawn at map edges and circulate at per-segment speeds (30, 50 km/h on roads; 20 km/h on the roundabout).
- **Three rectangular blockages** create realistic LOS / NLOS transitions.
- **75 candidate BS placements** distributed across three sectors (Left / Center / Right, 25 m spacing).
- The agent controls **one** mmWave sector; **two fixed interfering BSs** sit at the lower corners.

### Radio model
- Carrier 26 GHz (3GPP n258), bandwidth 100 MHz.
- **Close-In path loss** with LOS / NLOS exponents, 1 m reference:
  $\text{PL}(d) = \text{FSPL}(f_c, d_0) + 10 n \log_{10}(d/d_0) + X_\sigma$
- Link budget: EIRP = 55 dBm, $G_\text{rx}$ = 8 dBi, $L_\text{misc}$ = 15 dB, $L_\text{blockage}$ from 3GPP-style attenuation per blockage.
- ULA array factor with Hamming taper; sidelobe floor at –13 dB.
- SINR aggregates the **top-K = 3** non-serving beams; capacity follows Shannon $C = B \log_2(1 + \text{SINR})$.

### MDP
| Component | Definition |
|---|---|
| **State** $s_t$ | $(G_t, \theta_t, \omega_t)$: 25×25 vehicle-density grid (20 m bins) + current beam azimuth & width index |
| **Action** $a_t$ | 24 discrete configurations = **8 azimuths** {0°, 45°, …, 315°} × **3 HPBW** {65°, 90°, 120°} |
| **Reward** $r_t$ | $\dfrac{C^\text{srv}_t - C^\text{int}_t}{\Delta t} - \lambda_\theta \Delta\theta - \lambda_\omega \Delta\omega$ with $\lambda_\theta = 200$ Mbps, $\lambda_\omega = 100$ Mbps |
| **Horizon** | up to 1 800 steps (180 s), $\Delta t = 0.1$ s, $\gamma = 0.99$ |

### Two RL strategies (compared head-to-head)
| | DQN | Factorised tabular Q-learning |
|---|---|---|
| Model | MLP Q-network (Stable-Baselines3) | Single Q-table shared across BS placements |
| State featurisation | Flattened density vector + one-hot beam | (COM x-bin, COM y-bin, peak-density bin, beam dir, beam ang) |
| Strength | Sample efficiency / fast convergence | Higher asymptotic performance with longer training |
| Hyperparameters | $\gamma = 0.99$, buffer 1e5, batch 256, $\epsilon$: 1.0 → 0.05 | $\gamma = 0.97$, $\alpha_0 = 0.10$ with $1/\sqrt{1+e/1500}$ decay |
| Early stopping | Rolling-window 400-episode reward delta < 0.01 | Rolling-window 500-episode reward delta < 0.02 |

### Multi-agent LLM explainability layer
A user query is routed via **n8n** to four specialised agents that run in parallel and feed their evidence to a final synthesiser:

| Agent | Backend | Evidence type |
|---|---|---|
| **SQL-AI** | Supabase / Postgres on AWS Bedrock Claude 3.5 | Quantitative metrics from the run database (rewards, action frequencies, BS positions) |
| **Vector-AI** (RAG) | Pinecone vector store + Bedrock Titan embeddings | Domain rules: RF model, reward design, action-space constraints |
| **Visual-AI** | OpenAI GPT-5 multimodal | Plots, heatmaps, HUD screenshots |
| **Analysis-AI** | Bedrock Claude 3.5 | Synthesises the above into **Summary / Interpretation / Recommendations** |

A Streamlit chat UI sits in front (`streamlit_app4.py`); responses are returned via S3 polling.

---

## Repository layout

```
.
├── paper/                          # Conference paper PDF
│   └── Yigit2026_mmWave_BeamRL_XAI.pdf
│
├── layout.py                       # Gymnasium env: roads, roundabout, vehicles, RF, render
├── beam.py                         # ULA array factor + ray-traced beam rendering
├── car.py                          # Vehicle kinematics + per-link SINR / capacity
├── pathloss.py                     # Close-In path-loss model (LOS/NLOS + shadowing)
├── seed.py                         # Deterministic seed bundles for reproducibility
├── utility.py                      # Bootstrap installer for first-run dependencies
│
├── dqn_all.py                      # Train DQN across the 75 BS placements (with early stop + multi-seed eval)
├── q_l_all.py                      # Train factorised tabular Q-learning across placements
├── dqn_test.py / q_l_test.py       # Standalone evaluation entry points
├── dqn_torch.py / dqn.py           # Standalone DQN demos
├── q_l.py                          # Standalone Q-learning demo
│
├── streamlit_app4.py               # Streamlit chat UI for the explainability layer
├── LLM/
│   ├── n8n/DQN Explain.json        # n8n workflow (importable)
│   ├── pinecone/index_docx_pinecone.py   # Build the RAG vector index from .docx specs
│   └── supabase/supabase.py        # SQL ingest helper for Postgres run logs
│
├── airflow/                        # Distributed / Airflow-friendly variant of the trainer
├── req.txt                         # Minimal pip requirements
├── .env.example                    # Template for required environment variables
└── .gitignore
```

`logs/` (TensorBoard events, checkpoints, per-run summaries) is **git-ignored** — train locally to regenerate.

---

## Installation

```bash
# 1) Clone
git clone https://github.com/uguryigit/mmwave-beam-rl-xai.git
cd mmwave-beam-rl-xai

# 2) Create a venv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r req.txt
# Streamlit, boto3, pinecone-client, psycopg2-binary are needed for the
# explainability layer; they are auto-installed by utility.py on first run
# of streamlit_app4.py, or you can pip-install them manually.
```

> **Heads-up:** [`layout.py`](layout.py) ships with a self-bootstrap (`ensure_packages()`) that creates `.venv/` and installs `pygame`, `numpy`, `gymnasium` automatically on first import — it then re-execs the script under that venv.

Dependencies (see [`req.txt`](req.txt)):
`torch>=2.2.0`, `gymnasium>=0.29.1`, `gymnasium[box2d]`, `numpy>=1.26.0`, `pygame>=2.5.2`, `matplotlib`, `stable_baselines3`, `tensorboard`.

---

## Configuration (`.env`)

Copy [`.env.example`](.env.example) to `.env` and fill in real values, **or** export the variables before launching the apps:

| Variable | Used by | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | `streamlit_app4.py`, `LLM/pinecone/index_docx_pinecone.py` | S3 polling + Bedrock embeddings. Prefer an IAM role on EC2/ECS over long-lived keys. |
| `AWS_REGION` | both | defaults to `us-east-1` |
| `S3_BUCKET` | `streamlit_app4.py` | defaults to `dqn-simulation` |
| `PINECONE_API_KEY` | `LLM/pinecone/index_docx_pinecone.py` | required to build / query the RAG index |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | `LLM/supabase/supabase.py` | Postgres / Supabase log store |
| `N8N_WEBHOOK_URL` | `streamlit_app4.py` | endpoint for the n8n explainability pipeline |

**Never commit a populated `.env`** — it is in `.gitignore`. AWS credentials previously hardcoded in source were rotated and removed before this repo was first published.

---

## Running the simulator and training

### Interactive demo (rendered)
The env is a standard `gymnasium.Env`:
```python
from layout import RoundaboutSegmentedEnv
env = RoundaboutSegmentedEnv(render_mode="human", seed=42)
obs, _ = env.reset(pairIndex=0, seed=42)
for _ in range(1800):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc: break
env.close()
```
Press `1`/`2`/`3` to switch HPBW, `←`/`→` to steer, `m` to toggle BS-placement markers, `ESC`/`ENTER` to pause/resume.

### Full DQN sweep (75 BS placements)
```bash
python dqn_all.py
# Per-run logs / TB events / checkpoints land under logs/roundabout_dqn/run_NN/
# All-runs aggregate JSON + CSV under logs/roundabout_dqn/
```

### Full factorised Q-learning sweep
```bash
python q_l_all.py
# Logs under logs/roundabout_q_factorized/
```

Both scripts auto-resume: on restart they detect the last completed run (`run_summary.json` present) and continue from the next index.

### TensorBoard
```bash
tensorboard --logdir logs/roundabout_dqn
tensorboard --logdir logs/roundabout_q_factorized
```

---

## The LLM explainability layer

```bash
# 1) Build / refresh the Pinecone RAG index from the simulation-rules .docx
python LLM/pinecone/index_docx_pinecone.py

# 2) Push run summaries into Postgres / Supabase
python LLM/supabase/supabase.py

# 3) Import the n8n workflow
#    LLM/n8n/DQN Explain.json  →  n8n UI  →  "Import from file"

# 4) Launch the Streamlit chat UI
streamlit run streamlit_app4.py
```
The UI accepts a free-form question (and optional images: learning curves, HUD screenshots, heat-maps). Behind the scenes:

```
User query ──▶ n8n webhook
                  ├─▶ SQL-AI    (Postgres run logs)
                  ├─▶ Vector-AI (Pinecone RAG over simulation rules)
                  └─▶ Visual-AI (image analysis)
                              │
                              ▼
                      Analysis-AI ──▶ Summary / Interpretation / Recommendations ──▶ S3 ──▶ UI
```

---

## Reproducing the paper figures

| Paper figure | Produced by |
|---|---|
| Fig. 1 — simulation topology / HUD | `layout.py` render with `render_mode="human"` |
| Fig. 2 — n8n explainability workflow | `LLM/n8n/DQN Explain.json` (import into n8n) |
| Fig. 3 — Streamlit chatbot UI | `streamlit run streamlit_app4.py` |
| Fig. 4 — Q-factorised episode-reward curves | `tensorboard --logdir logs/roundabout_q_factorized` after `python q_l_all.py` |
| Fig. 5 — DQN episode-reward curves | `tensorboard --logdir logs/roundabout_dqn` after `python dqn_all.py` |

Multi-seed evaluation: every trained policy is tested over **100 deterministically-seeded episodes** (`make_seeds(SEED0 + i)` for `i ∈ {0..99}`), and `eval_mean_reward` / `eval_std_reward` are written to each `run_summary.json`.

---

## Citation

```bibtex
@inproceedings{yigit2026mmwave,
  author    = {Yigit, Ugur and Akbas, Ayhan and Kose, Abdulkadir and
               Foh, Chuan Heng and Shojafar, Mohammad},
  title     = {AI Explainability for Adaptive {mmWave} Beam Configuration
               in Dynamic Vehicular Environments},
  booktitle = {Proc. IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2026}
}
```

---

## Acknowledgements

Developed at the **5G/6GIC, Institute for Communication Systems, University of Surrey** in collaboration with **Ankara Bilim University**, **Abdullah Gül University**, and **Ransight Technology Ltd.**

## License

This repository is released for academic and research use accompanying the paper above. Please contact the authors before redistributing or building commercial products on top of it.
