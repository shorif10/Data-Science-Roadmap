# Data Science Roadmap: the phased plan

The folders are numbered in **learning order**. Work top to bottom inside a phase; do the phase
milestone project before moving on. Hours are rough estimates for someone who already programs.

```
Phase 0  Foundations     00 Setup · 01 Python · 02 Math · 03 Statistics
Phase 1  Analysis        04 NumPy · 05 Pandas · 06 SQL · 07 Matplotlib · 08 Seaborn
                         09 Snowflake · 10 Power BI · 11 Tableau        (tools: learn on demand)
Phase 2  Machine Learning 12 Machine Learning
Phase 3  Deep Learning   13 Deep Learning · 14 PyTorch · 15 TensorFlow · 16 Keras · 17 NLP
Phase 4  Engineering     18 Data Engineering · 19 Data Pipelines · 20 MLOps
Phase 5  AI Systems      21 LLMs & RAG · 22 Agentic AI · 23 Agent Development · 24 MCP
Capstone                 25 End-to-End Project
```

## Why this order

| Change from the original list | Reason |
|---|---|
| **Statistics + Math added (02, 03)** | ML and A/B testing are applied statistics. Without them you can run models but cannot tell a real result from noise. |
| **Setup/Git/CLI added (00)** | Every later module assumes you can manage environments and version your work. |
| **SQL moved next to pandas** | SQL is the most-asked data skill in interviews and jobs; learn it while pandas is fresh so you see both side by side. |
| **BI tools (Snowflake, Power BI, Tableau) marked "on demand"** | They are tools, not concepts. Learn the one your target job or project needs; the concepts transfer. |
| **Machine Learning expanded to 16 notebooks** | Feature engineering, evaluation, leakage, imbalance, interpretation and forecasting are where real projects succeed or fail. |
| **PyTorch before TensorFlow/Keras** | One framework deeply beats three shallowly. Pick PyTorch first; TF/Keras are there when a team uses them. |
| **NLP added (17)** | It is the bridge from classic ML to LLMs. |
| **LLMs & RAG added (21)** | Agents (22-24) sit on top of LLM and retrieval fundamentals. |
| **Data Engineering and Pipelines before MLOps** | You cannot operate a model reliably on top of fragile data plumbing. |

## Phase milestones (do these before moving on)

| Phase | You are ready to move on when you can... | Milestone project |
|---|---|---|
| **0 Foundations** | write and test a small Python program; use git branches; explain a p-value and a confidence interval to a colleague; read a gradient-descent loop | Analyse `orders.csv` with only the standard library, and put it in a git repo |
| **1 Analysis** | clean a messy table, join four tables, aggregate, answer business questions in both pandas and SQL, and chart the answer clearly | The e-commerce EDA in `05_Pandas/09_eda_project_ecommerce` plus a written 1-page findings memo |
| **2 Machine Learning** | frame a problem, avoid leakage, build a pipeline, evaluate with CV, pick a threshold from costs, explain the model | `12_Machine_Learning/16_ml_capstone_end_to_end_churn` |
| **3 Deep Learning** | train a network in PyTorch, debug a run that does not learn, and say honestly when *not* to use deep learning | Beat (or fairly lose to) the gradient-boosting baseline on one dataset, with the comparison written up |
| **4 Engineering** | build an idempotent pipeline, track experiments, serve a model behind an API, detect drift | Serve the churn model with tests and a drift report |
| **5 AI Systems** | build a RAG system and a tool-using agent, evaluate them, secure them, expose tools via MCP | A support agent that answers from the knowledge base and queries the shop database |
| **Capstone** | deliver the full lifecycle in one repo | `25_End_to_End_Project` |

## Suggested pacing

Total is roughly **250-300 focused hours**. That is about 6-8 months at 10 hours a week, or 3-4 months full-time.
Do not try to "finish" everything before applying for roles. Pick a track:

| Goal | Path |
|---|---|
| **Data analyst** | Phase 0 (skip Math 04-06) → Phase 1 in full (+ one BI tool) → ML 01-08 |
| **Data scientist** | Phase 0 → Phase 1 (BI optional) → Phase 2 in full → NLP → Capstone |
| **ML engineer** | Phase 0 → Phase 1 (core) → Phase 2 → PyTorch → Phase 4 → Capstone |
| **AI / LLM engineer** | Phase 0 → Phase 1 (core) → ML 01-08 → NLP → Phase 5 → Phase 4 (MLOps + serving) |

## How to study a notebook (30-60 min each)

1. **Read a section, then type the code yourself** in a scratch cell. Do not paste.
2. Change one thing and predict the result before you run it.
3. Do every exercise before opening the collapsed solution. If you peek, redo the exercise the next day.
4. Write three sentences in your own words at the end ("what I learned / what surprised me / what I'd ask").
5. Weekly: revisit the *Interview corner* questions aloud, without looking at the answers.
6. Between phases: build the milestone project **without copying**, then compare with the reference.

## What "working through this" does not replace

- Practice on **real, messy, public data** (each module README lists some). The datasets here are synthetic on purpose so that
  every result is reproducible and checkable, but real data is harder.
- Reading library documentation directly. The notebooks point to it.
- Talking to people who do this job. Ask what they use and what they wish they had learned first.
