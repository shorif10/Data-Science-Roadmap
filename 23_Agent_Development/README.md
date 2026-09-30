# 23 Agent Development

**Goal:** turn the agent ideas of module 22 into software you would be willing to run for customers: a production-shaped loop, a data agent with a genuinely safe SQL tool, context and memory management with measured trade-offs, and the testing, tracing and deployment practices around it. All notebooks run offline (deterministic mock model, simulated clock where latency matters); the faithful Claude SDK versions are in `no-run` cells.

**Prerequisites:** module 22 (especially notebooks 02 and 04), module 21 notebook 02 (the `complete()` interface), SQL basics (06), pytest basics helpful. FastAPI knowledge is not assumed.

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_build_an_agent_loop_from_scratch` | Tool registry from type hints, valid-by-construction conversation state, every stop reason, bounded retries with backoff and jitter, structured logs, SDK twin | 1.5 |
| 02 | `02_a_data_agent_with_sql_and_knowledge_base_tools` | A read-only SQL tool with five independent layers, a KB search tool, scripted scenarios asserting tool trajectory and answer numbers against pandas | 2 |
| 03 | `03_memory_state_and_context_management` | Context budget accounting, five strategies compared on a 30-turn session (cost and what each forgets), retrieval memory and its limits | 1.5 |
| 04 | `04_testing_observability_and_deploying_agents` | pytest for tools with a mutation check, record/replay of model responses, JSON tracing spans, cost and latency percentiles, a FastAPI endpoint tested with `TestClient`, deployment checklist | 2 |

**Total: about 7 hours** of reading, typing and exercises.

Data: `../datasets/shop.db` (read-only; notebook 02 hashes it before and after to prove it is untouched) and `../datasets/knowledge_base/`; see `../docs/DATASETS.md`. Notebook 04 writes temporary files to a temp folder and deletes it in its last cell.

## You are done when you can...

- [ ] generate a tool's JSON Schema from a function signature, register tools with a decorator, and make the dispatcher return errors instead of raising
- [ ] write an agent loop that names every way a run can end (done, truncated, refused, step limit, repeated tool errors, model API unavailable) and keeps the conversation valid after each
- [ ] retry transient model-API failures with capped exponential backoff and jitter, and explain why not to retry 4xx errors or blindly retry side effects
- [ ] build a read-only SQL tool with a validator, read-only connection, authorizer, row/size cap and timeout, and show that hostile SQL fails at each layer alone
- [ ] test a data agent by asserting its tool trajectory and its numbers against an independent pandas computation
- [ ] budget a context window, pick a context strategy from measurements, and test what it forgets with probe questions
- [ ] unit-test tools, replay recorded trajectories with request fingerprints, and confirm your tests can fail
- [ ] emit tracing spans, compute cost per run and p95 latency, and serve an agent behind an endpoint with fail-closed auth, validation, rate limiting and honest status codes
- [ ] say plainly what is not solved: hallucination, prompt injection, non-determinism and cost

## Resources

- Anthropic API documentation (Messages API, tool use, errors and rate limits, token counting, prompt caching): https://docs.anthropic.com
- Anthropic engineering article, "Building effective agents": https://www.anthropic.com/engineering/building-effective-agents
- FastAPI documentation, testing section: https://fastapi.tiangolo.com
- pytest documentation: https://docs.pytest.org
- SQLite documentation on the authorizer, URI filenames and the progress handler: https://www.sqlite.org
- OpenTelemetry documentation (the tracing vocabulary notebook 04 uses): https://opentelemetry.io

**Next module:** [24 MCP](../24_MCP/README.md)
