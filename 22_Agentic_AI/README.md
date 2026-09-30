# 22 Agentic AI

**Goal:** understand agents as engineering, not magic: the loop, tool use and its exact message protocol, the workflow patterns that most "agents" really are, and how to make an agent safe and **evaluate it on its trajectory**. Every notebook runs offline with a deterministic mock model behind the same `complete()` interface as the real client; the real Claude version is shown in `no-run` cells.

**Prerequisites:** module 21 (LLMs and RAG, especially notebooks 01 and 02), Python functions and classes (01), basic pandas (05), basic SQL (06).

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_what_is_an_agent` | Workflow vs agent, write the loop, read a trajectory, cost growth and compounding unreliability | 1 |
| 02 | `02_tool_use_and_function_calling` | Tool schemas, the `tool_use`/`tool_result` flow and its rules (with a checker), parallel calls, error results, repairing a broken history | 1.5 |
| 03 | `03_agent_patterns_planning_reflection_and_multi_agent` | Chaining with gates, routing, sectioning and voting, orchestrator-workers, evaluator-optimiser, plan-and-execute; a measured cost/latency/reliability table | 1.5 |
| 04 | `04_agent_safety_guardrails_and_evaluation` | Prompt injection through a tool result (safely, in a mock), least privilege, confirmation gates, output validation, budgets, trajectory evaluation, red-team checklist | 1.5 |

**Total: about 5.5 hours** of reading, typing and exercises.

Data: `../datasets/shop.db` (opened read-only) and `../datasets/knowledge_base/`; see `../docs/DATASETS.md`. The "attack" in notebook 04 uses fake tools that only write to in-memory lists and an in-memory copy of one document. Nothing in `datasets/` is modified.

## You are done when you can...

- [ ] decide between a single prompt, a workflow and an agent using cost, latency, testability and the cost of mistakes, and explain the choice
- [ ] write a tool definition (name, description, JSON Schema), validate the model's input, and return errors the model can recover from
- [ ] write out the request / `tool_use` / `tool_result` message flow from memory, including parallel calls and the rules that make the API return a 400
- [ ] implement prompt chaining with gates, routing with a human fallback, voting (and say when it does not help), orchestrator-workers and a bounded evaluator-optimiser loop
- [ ] reproduce a prompt injection through a tool result against a mock and stop it with least privilege, a confirmation gate and validation, in code rather than in the prompt
- [ ] audit a tool set for the "lethal trifecta" (private data, untrusted content, external communication)
- [ ] evaluate an agent on required and forbidden tool calls, side effects, steps, and computed answer checks, and report attack-success and benign-pass rates
- [ ] say plainly what remains unsolved: hallucination, prompt injection, non-determinism and cost

## Resources

- Anthropic API documentation, tool-use section: https://docs.anthropic.com
- Anthropic engineering article, "Building effective agents": https://www.anthropic.com/engineering/building-effective-agents
- JSON Schema reference: https://json-schema.org
- OWASP Top 10 for LLM Applications (search the title at https://owasp.org)
- Model Context Protocol documentation (next but one module): https://modelcontextprotocol.io

**Next module:** [23 Agent Development](../23_Agent_Development/README.md)
