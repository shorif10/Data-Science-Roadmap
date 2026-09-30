# 24 MCP (Model Context Protocol)

**Goal:** understand MCP as a protocol you can read on the wire, build a server with the official Python SDK, connect to it from a client, and secure it. You write a JSON-RPC server by hand first, then use the SDK, then attack and harden a SQL-backed server. Everything runs locally: stdio servers are launched as subprocesses with timeouts and cleanup (notebooks prove no process or temp file is left behind), and one notebook runs a streamable-HTTP server on localhost.

**Prerequisites:** module 22 (tool use, agent safety), Python async basics (`async`/`await`, used in the SDK client), SQL basics (06). No API key is needed.

**SDK note:** the notebooks target the installed `mcp` Python SDK **2.x** (the notebooks print the version and assert on real behaviour). In this version the high-level server class is `MCPServer` (`from mcp.server.mcpserver import MCPServer`); older tutorials call it `FastMCP` from `mcp.server.fastmcp`, which does not exist here. The protocol and SDK evolve quickly, so when the notebooks and the official documentation disagree, trust what the installed version does and check the docs.

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_what_is_mcp` | Hosts/clients/servers, a hand-written JSON-RPC server and client, capture a real session, tools/resources/prompts, stdio vs streamable HTTP, protocol versions | 1.5 |
| 02 | `02_build_an_mcp_server_with_fastmcp` | Decorators and type hints to schemas, a shop-data server over `shop.db` and the knowledge base, `stdio_client` + `ClientSession`, tool errors vs protocol errors, clean shutdown | 2 |
| 03 | `03_connect_clients_and_secure_your_server` | Host config (`mcpServers`), why stdout must stay clean (tested on the SDK and on a hand-written server), tool poisoning and rug pulls, confused deputy, a layered SQL defence | 2 |

**Total: about 5.5 hours** of reading, typing and exercises.

Data: `../datasets/shop.db` and `../datasets/knowledge_base/`; see `../docs/DATASETS.md`. Destructive demos run on a copy in a temp folder; the original file's hash is checked before and after.

## You are done when you can...

- [ ] explain hosts, clients and servers, the N x M problem MCP removes, and what a session looks like on the wire (`initialize`, `tools/list`, `tools/call`)
- [ ] name the three server primitives (tools, resources, prompts) and who controls each
- [ ] build an MCP server with decorators, inspect the generated input schemas, and call it from a client session
- [ ] tell a tool error (`isError`) from a protocol error and handle both in client code
- [ ] start a stdio server as a subprocess with a timeout, and terminate it so that nothing is left running
- [ ] explain why a stdio server must never write to stdout except through the protocol, and what this SDK version does about stray prints (and why you should not rely on it)
- [ ] describe tool poisoning, rug pulls, indirect prompt injection through results, the confused deputy, and over-broad permissions, with a defence for each
- [ ] harden a SQL-backed tool with validation, a read-only connection, an allow-list, a row cap and a timeout
- [ ] say what MCP does **not** do: it standardises plumbing, not safety

## Resources

- Model Context Protocol documentation and specification: https://modelcontextprotocol.io
- Official Python SDK repository: https://github.com/modelcontextprotocol/python-sdk
- Anthropic API documentation (tool use, and the MCP connector): https://docs.anthropic.com
- JSON-RPC 2.0 specification: https://www.jsonrpc.org/specification
- OWASP Top 10 for LLM Applications (search the title at https://owasp.org)

**Next module:** [25 End-to-End Project](../25_End_to_End_Project/README.md)
