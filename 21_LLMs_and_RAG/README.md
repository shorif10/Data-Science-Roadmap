# 21 LLMs and RAG

**Goal:** understand what a language model actually does (tokens, context, sampling), control it with prompts and validated structured output, retrieve knowledge for it with embeddings, assemble and **measure** a retrieval-augmented generation (RAG) system, and evaluate LLM applications honestly. Everything runs offline against a deterministic mock model; the real Claude client is a one-line swap shown in `no-run` cells.

**Prerequisites:** Python and NumPy (modules 01, 04), basic pandas (05), a little probability (03). Helpful: NLP basics (17).

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_how_llms_work` | Train a BPE tokenizer, build a tiny language model, see temperature/top-k/top-p, do cost and latency arithmetic | 1.5 |
| 02 | `02_prompting_and_structured_outputs` | Prompt anatomy, few-shot (measured), JSON validated with pydantic, retry with the error fed back, mock-vs-real client | 1.5 |
| 03 | `03_embeddings_and_vector_search` | TF-IDF and hashing embeddings, cosine similarity, an exact NumPy index, approximate search by hyperplane hashing | 1.5 |
| 04 | `04_build_rag_from_scratch` | Chunking strategies, prompt with citations, grounded generation, separate retrieval failures from generation failures | 2 |
| 05 | `05_evaluating_llm_applications` | Golden sets, per-case metrics, paired bootstrap, LLM-as-judge bias checks, prompt regression tests, cost per correct answer | 2 |

**Total: about 8.5 hours** of reading, typing and exercises.

Data: `../datasets/knowledge_base/*.md` (ten short synthetic company policy documents) and `../datasets/shop.db`; see `../docs/DATASETS.md`. No network access or API key is needed to run anything except the cells tagged `no-run`.

## You are done when you can...

- [ ] explain tokens, the context window, and why cost and latency grow with conversation length, and estimate the cost of a workload before building it
- [ ] say what temperature, top-k and top-p change, and why the same prompt can give different answers (non-determinism)
- [ ] write a prompt with clear parts, test few-shot examples on a labelled set, and get **validated** JSON out of a model with bounded retries
- [ ] build and compare embedding-based and keyword retrieval with `hit@k`, and explain what approximate nearest-neighbour search trades away
- [ ] build a RAG pipeline with chunking, citations and an abstention path, and tell a retrieval failure from a generation failure
- [ ] build a golden set, report metrics with uncertainty, compare two versions without chasing noise, and use an LLM judge only after checking its bias
- [ ] state plainly what these systems cannot guarantee: hallucination, prompt injection through retrieved text, non-determinism, and cost

## Resources

- Anthropic API documentation: https://docs.anthropic.com (Messages API, prompting, tool use, token counting)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020), arXiv:2005.11401
- Vaswani et al., "Attention Is All You Need" (2017), arXiv:1706.03762
- Sentence-Transformers documentation for real embedding models: https://www.sbert.net
- Andrej Karpathy, "Neural Networks: Zero to Hero" video series (includes building a GPT and a tokenizer)

**Next module:** [22 Agentic AI](../22_Agentic_AI/README.md)
