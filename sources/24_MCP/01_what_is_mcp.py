# %% [markdown]
# # What Is MCP? The Protocol Behind Tool-Using AI Hosts
#
# **Why this matters.** Every team that wires an AI assistant to its own systems (a database, a wiki, a ticket
# tracker) hits the same wall: each assistant has its own plug-in format, so every integration is written again
# for every host. The Model Context Protocol (MCP) is an open standard that lets you write the integration once
# as a *server* and use it from any *host* that speaks the protocol. You will meet it in job descriptions for
# agent engineers, and you will be asked to build and debug servers, which starts with knowing what actually
# travels over the wire.
#
# **You will learn**
# - explain hosts, clients and servers, and the N x M integration problem MCP removes
# - write a working JSON-RPC 2.0 server and client in about 30 lines, to see how simple the wire format is
# - capture the real messages of an MCP session (`initialize`, `tools/list`, `tools/call`, ...) and read them
# - name the three server primitives (tools, resources, prompts) and who controls each
# - compare the two transports, stdio and streamable HTTP, on a live localhost server
#
# **Prerequisites:** [Build an agent loop from scratch](../23_Agent_Development/01_build_an_agent_loop_from_scratch.ipynb),
# [Tool use and function calling](../22_Agentic_AI/02_tool_use_and_function_calling.ipynb), and
# [Testing, observability and deploying agents](../23_Agent_Development/04_testing_observability_and_deploying_agents.ipynb).
#
# > Everything here runs offline. The servers are tiny Python files that this notebook writes to a temp folder,
# > starts as subprocesses, talks to, and shuts down again. Installed SDK: the `mcp` Python package
# > (version printed below). No Claude API key is needed.

# %%
import json
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from importlib.metadata import version
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="mcp_nb1_"))  # removed in the last code cell
print("mcp SDK version:", version("mcp"))
print("python:", sys.version.split()[0])

# %% [markdown]
# ## 1. The problem: N x M integrations
#
# Suppose your company has **N** AI hosts (a chat app, an IDE assistant, a support bot) and **M** systems the AI
# should reach (database, docs, tickets, calendar). Without a shared protocol each pairing is custom glue code.
# With a protocol, each host implements the client side once and each system implements the server side once.

# %%
def glue_pieces(n_hosts: int, m_systems: int, with_protocol: bool) -> int:
    return n_hosts + m_systems if with_protocol else n_hosts * m_systems

for n, m in [(2, 3), (5, 8), (10, 20)]:
    print(f"{n:>2} hosts x {m:>2} systems: bespoke = {glue_pieces(n, m, False):>3} integrations, "
          f"with a protocol = {glue_pieces(n, m, True):>3}")

# %% [markdown]
# *What just happened?* Multiplication turned into addition. That is the whole economic argument (the same one
# behind USB and the Language Server Protocol for editors). The catch: a protocol only pays off if it is widely
# implemented, and it adds a layer to debug. Teams still write direct API calls when there is exactly one host
# and one system.
#
# **The three roles**
#
# | Role | What it is | Example |
# |---|---|---|
# | **Host** | The AI application the user talks to; owns the model and decides what to expose to it | a desktop chat app, a coding assistant |
# | **Client** | A connector *inside* the host; one client holds one connection to one server | the `ClientSession` object in your Python code |
# | **Server** | A program that exposes capabilities (tools, resources, prompts) | the shop-database server you build in the next notebook |
#
# One host runs many clients, one per server. The model never speaks MCP itself: the host translates between
# the model's tool-call format and MCP requests (you built that translation loop in module 23).

# %% [markdown]
# ## 2. JSON-RPC 2.0 in 30 lines
#
# MCP messages are **JSON-RPC 2.0**: a tiny convention for remote procedure calls in JSON. There are only
# four message shapes:
#
# | Shape | Has | Meaning |
# |---|---|---|
# | Request | `jsonrpc`, `id`, `method`, optional `params` | "do this and answer me" |
# | Success response | `jsonrpc`, same `id`, `result` | the answer |
# | Error response | `jsonrpc`, same `id`, `error: {code, message, data?}` | it failed |
# | Notification | `jsonrpc`, `method`, **no `id`** | fire and forget, never answered |
#
# The `id` is how a client matches a reply to its request when several are in flight. Let us build a server
# that handles one message (a Python `dict`) and returns the reply.

# %%
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602

METHODS = {
    "add": lambda p: p["a"] + p["b"],
    "echo": lambda p: p,
}

def handle(msg: dict):
    """Return the reply dict, or None for a notification."""
    is_notification = "id" not in msg
    try:
        fn = METHODS[msg["method"]]
    except KeyError:
        return None if is_notification else {"jsonrpc": "2.0", "id": msg["id"],
                                             "error": {"code": METHOD_NOT_FOUND, "message": "Method not found"}}
    result = fn(msg.get("params", {}))
    return None if is_notification else {"jsonrpc": "2.0", "id": msg["id"], "result": result}

print(handle({"jsonrpc": "2.0", "id": 1, "method": "add", "params": {"a": 2, "b": 3}}))
print(handle({"jsonrpc": "2.0", "id": 2, "method": "nope"}))
print(handle({"jsonrpc": "2.0", "method": "add", "params": {"a": 1, "b": 1}}))  # notification -> None

# %% [markdown]
# *What just happened?* A request got a `result`, an unknown method got an `error` with the standard code
# `-32601`, and a notification (no `id`) got **no reply at all**. Notice this toy version does not validate
# `params`; a missing key would raise `KeyError` and crash it. Real servers must turn that into `-32602`
# (invalid params). You will fix exactly this in Exercise 1.

# %% [markdown]
# ### From function to process: JSON-RPC over stdio
#
# The most common MCP transport is **stdio**: the host starts your server as a child process and exchanges
# one JSON message per line through the child's `stdin` (host to server) and `stdout` (server to host).
# Here is a tiny server file that does exactly that, using our `handle` idea.

# %%
MINI_SERVER = textwrap.dedent('''
    import json, sys
    METHODS = {"add": lambda p: p["a"] + p["b"]}
    for line in sys.stdin:                       # one JSON message per line
        msg = json.loads(line)
        if msg["method"] in METHODS:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": METHODS[msg["method"]](msg["params"])}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "Method not found"}}
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()                       # without flush the client waits forever
''')
(WORK / "mini_server.py").write_text(MINI_SERVER, encoding="utf-8")

# %% [markdown]
# Now a small client that launches the process. Two engineering details matter and recur in every later
# notebook: reads go through a **thread and a queue with a timeout** (so a silent server can never hang your
# notebook), and `close()` always **closes stdin, waits, then kills** as a last resort.

# %%
class RawStdioClient:
    """Minimal newline-delimited JSON-RPC client for a subprocess server."""

    def __init__(self, args, env=None, stderr_path=None):
        self._err = open(stderr_path, "w", encoding="utf-8") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([sys.executable, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._err, text=True, encoding="utf-8", env=env)
        self.lines = queue.Queue()
        self.log = []            # every line, with direction: ("->" sent, "<-" received)
        self._next_id = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)     # EOF marker

    def send_line(self, text):
        self.log.append(("->", text))
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def send(self, msg):
        self.send_line(json.dumps(msg, separators=(",", ":")))

    def recv_line(self, timeout=5.0):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return "<timeout>"
        if line is None:
            return "<eof>"
        self.log.append(("<-", line))
        return line

    def recv(self, timeout=5.0):
        return json.loads(self.recv_line(timeout))

    def request(self, method, params=None, timeout=5.0):
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        while True:              # skip any notifications until our reply arrives
            reply = self.recv(timeout)
            if reply.get("id") == self._next_id:
                return reply

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)

    def close(self):
        try:
            self.proc.stdin.close()          # EOF on stdin is the polite "please exit"
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=3)
        if self._err is not subprocess.DEVNULL:
            self._err.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


with RawStdioClient([str(WORK / "mini_server.py")]) as c:
    print(c.request("add", {"a": 20, "b": 22}))
    print(c.request("multiply", {"a": 2, "b": 3}))
print("server exited with code", c.proc.returncode)

# %% [markdown]
# *What just happened?* We spoke JSON-RPC to a subprocess through pipes. There is no HTTP, no schema, no
# framework. **MCP is this, plus an agreed vocabulary of method names** (`initialize`, `tools/list`,
# `tools/call`, ...) and payload shapes. Everything else in the SDK is convenience around that.

# %% [markdown]
# ## 3. A real MCP server, spoken to by hand
#
# Now the real thing. We write a minimal server with the official SDK. In `mcp` 2.x the high-level server
# class is `MCPServer` (it was called `FastMCP` in `mcp` 1.x; the module `mcp.server.fastmcp` no longer
# works in the installed version, so old tutorials fail with `ModuleNotFoundError`).
# It exposes one tool, one resource and one prompt.

# %%
DEMO_SERVER = textwrap.dedent('''
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("demo")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @mcp.resource("note://welcome")
    def welcome() -> str:
        """A static welcome note."""
        return "Hello from the demo server"

    @mcp.prompt()
    def review(code: str) -> str:
        """Ask for a code review."""
        return f"Please review this code:\\n{code}"

    if __name__ == "__main__":
        mcp.run("stdio")
''')
(WORK / "demo_server.py").write_text(DEMO_SERVER, encoding="utf-8")

def show(msg, width=110):
    print(json.dumps(msg, indent=1)[:width * 12])

# %% [markdown]
# The sequence every session follows in the classic handshake style: the client sends `initialize` (its protocol
# version, capabilities and identity), the server replies with what *it* supports, then the client sends the
# notification `notifications/initialized`. Only after that does normal traffic begin.

# %%
with RawStdioClient([str(WORK / "demo_server.py")], stderr_path=WORK / "demo_stderr.txt") as c:
    init = c.request("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                    "clientInfo": {"name": "notebook", "version": "0.1"}})
    c.notify("notifications/initialized")
    show(init)

    tools = c.request("tools/list")
    call = c.request("tools/call", {"name": "add", "arguments": {"a": 20, "b": 22}})
    resources = c.request("resources/list")
    prompts = c.request("prompts/list")

assert init["result"]["serverInfo"]["name"] == "demo"
assert [t["name"] for t in tools["result"]["tools"]] == ["add"]
assert call["result"]["structuredContent"] == {"result": 42}

# %% [markdown]
# Read this reply like a contract. `capabilities` lists which primitives the server offers (`tools`,
# `resources`, `prompts`), `protocolVersion` is the version the server picked, `serverInfo` names it.
# Now the two messages you will read most often when debugging:

# %%
show(tools["result"])
print("-" * 60)
show(call["result"])

# %% [markdown]
# `tools/list` returns, per tool, a `name`, a `description` and an **`inputSchema`** (JSON Schema, generated by the
# SDK from the Python type hints `a: int, b: int`). This is exactly the information a host hands to the model
# so it can decide to call the tool. `tools/call` returns `content` (blocks meant for the model, here one text
# block containing `"42"`), `structuredContent` (the same value as machine-readable JSON, matching the tool's
# `outputSchema`) and `isError`.
#
# The other two listings, for completeness:

# %%
print("resources:", [(r["uri"], r["name"]) for r in resources["result"]["resources"]])
print("prompts:  ", [(p["name"], [a["name"] for a in p["arguments"]]) for p in prompts["result"]["prompts"]])

# %% [markdown]
# ### Two kinds of failure
# JSON-RPC gives you a *protocol* error channel (the `error` member). MCP tool calls have a second, separate
# channel: a normal `result` with `isError: true`. Let us provoke both and compare.

# %%
with RawStdioClient([str(WORK / "demo_server.py")]) as c:
    c.request("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                             "clientInfo": {"name": "notebook", "version": "0.1"}})
    c.notify("notifications/initialized")
    unknown_method = c.request("does/not/exist")
    bad_args = c.request("tools/call", {"name": "add", "arguments": {"a": "not a number", "b": 1}})
    unknown_tool = c.request("tools/call", {"name": "subtract", "arguments": {}})
    c.send_line("{this is not json")
    garbage_reply = c.recv_line(timeout=1.0)

print("unknown method  ->", unknown_method.get("error"))
print("bad arguments   -> isError =", bad_args["result"]["isError"], "|", bad_args["result"]["content"][0]["text"][:70], "...")
print("unknown tool    -> isError =", unknown_tool["result"]["isError"], "|", unknown_tool["result"]["content"][0]["text"])
assert unknown_method["error"]["code"] == METHOD_NOT_FOUND
assert bad_args["result"]["isError"] is True and unknown_tool["result"]["isError"] is True
print("malformed JSON line ->", garbage_reply)
assert garbage_reply == "<timeout>", "this SDK version stays silent on a malformed line"

# %% [markdown]
# *What just happened?* A wrong **method** is a protocol error (`error`, code `-32601`). A wrong **tool call**
# (bad arguments, unknown tool, an exception inside the tool) comes back as a successful JSON-RPC response with
# `isError: true`. That is deliberate: tool failures are *content for the model*, which can read the message and
# retry with corrected arguments. A protocol error is for the client's code, not for the model.
#
# A behaviour worth knowing: this SDK's stdio server does not answer a line that is not valid JSON; it stays
# silent. Our client's timeout would catch that; a client without one would hang.

# %% [markdown]
# ## 4. Protocol versions and the two connection styles
#
# The `initialize` message carries a `protocolVersion` string (a date). Let us ask the installed SDK which
# versions it knows, and what a server does when a client asks for a version it has never heard of.

# %%
import mcp.types.version as V

print("supported:       ", V.SUPPORTED_PROTOCOL_VERSIONS)
print("latest handshake:", V.LATEST_HANDSHAKE_VERSION, "| latest overall:", V.LATEST_PROTOCOL_VERSION)

with RawStdioClient([str(WORK / "demo_server.py")]) as c:
    odd = c.request("initialize", {"protocolVersion": "1999-01-01", "capabilities": {},
                                   "clientInfo": {"name": "notebook", "version": "0.1"}})
print("client asked for 1999-01-01, server answered:", odd["result"]["protocolVersion"])
assert odd["result"]["protocolVersion"] in V.SUPPORTED_PROTOCOL_VERSIONS

# %% [markdown]
# *What just happened?* Version negotiation: the client proposes, the server answers with a version it
# supports, and the client must then decide whether it can live with that. Never hard-code the version in your
# own code paths; let the SDK do it.
#
# The list above shows the SDK also knows a newer revision than its handshake versions. To see what changes on
# the wire, we need to capture real traffic from the SDK's own client. Rather than patching the SDK, we insert a
# **tap**: a tiny process that sits between client and server, copies every line both ways, and logs it.

# %%
TAP = textwrap.dedent('''
    import json, subprocess, sys, threading
    log = open(sys.argv[1], "a", encoding="utf-8")
    lock = threading.Lock()
    child = subprocess.Popen([sys.executable, *sys.argv[2:]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def pump(src, dst, tag):
        for line in src:
            with lock:
                log.write(json.dumps({"dir": tag, "line": line.decode("utf-8").rstrip("\\r\\n")}) + "\\n")
                log.flush()
            dst.write(line)
            dst.flush()
        if tag == "c2s":
            child.stdin.close()          # client hung up -> tell the real server (EOF)

    threading.Thread(target=pump, args=(sys.stdin.buffer, child.stdin, "c2s"), daemon=True).start()
    pump(child.stdout, sys.stdout.buffer, "s2c")
    child.wait()
    log.close()
''')
(WORK / "tap.py").write_text(TAP, encoding="utf-8")

# %% [markdown]
# The SDK has two client classes. `ClientSession` is the classic one: it always performs the
# `initialize` handshake. `Client` is the newer high-level one; its default `mode="auto"` first tries a newer
# style and falls back to the handshake for old servers. Both are used with `async with`. A notebook cell can
# `await` directly because Jupyter already runs an event loop.

# %%
from mcp import Client, ClientSession, StdioServerParameters, stdio_client


async def capture(style: str):
    """Run one short session through the tap; return the logged messages as dicts."""
    log_path = WORK / f"tap_{style}.jsonl"
    log_path.unlink(missing_ok=True)
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(WORK / "tap.py"), str(log_path), str(WORK / "demo_server.py")])
    with open(WORK / f"stderr_{style}.txt", "w") as errlog:
        # A notebook's sys.stderr has no file descriptor, so we pass our own errlog file to the stdio transport.
        if style == "modern":
            async with Client(stdio_client(params, errlog=errlog)) as client:
                await client.list_tools()
                await client.call_tool("add", {"a": 1, "b": 2})
        else:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.list_tools()
                    await session.call_tool("add", {"a": 1, "b": 2})
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines()]
    return [{"dir": r["dir"], **json.loads(r["line"])} for r in rows]


legacy = await capture("legacy")
modern = await capture("modern")

for name, msgs in [("ClientSession (handshake)", legacy), ("Client (auto)", modern)]:
    print(name)
    for m in msgs:
        kind = m.get("method") or ("result" if "result" in m else "error")
        print(f"   {m['dir']}  id={m.get('id', '-')!s:<2} {kind}")

# %% [markdown]
# Compare the two sequences. The classic style needs `initialize` plus the `notifications/initialized`
# notification before the first real call. The newer style in the installed SDK starts with a
# `server/discover` request and then has **no separate handshake**: each later request carries the protocol
# version, client identity and capabilities in its own `params._meta`, so a server can treat every request
# independently. Let us verify that claim on the captured data instead of trusting the prose:

# %%
legacy_methods = [m["method"] for m in legacy if m["dir"] == "c2s"]
modern_methods = [m["method"] for m in modern if m["dir"] == "c2s"]
assert legacy_methods[:2] == ["initialize", "notifications/initialized"]
assert modern_methods[0] == "server/discover" and "initialize" not in modern_methods
per_request_meta = [m["params"]["_meta"] for m in modern if m["dir"] == "c2s" and m.get("method") == "tools/call"][0]
print("keys inside the tools/call _meta:", sorted(per_request_meta))
print("negotiated version in that request:", per_request_meta["io.modelcontextprotocol/protocolVersion"])

# %% [markdown]
# *What just happened?* Both sessions performed the same work (list tools, call `add`) but the wire format
# differs by protocol revision. This is a **moving target**: the protocol is versioned by date, the SDK supports
# several revisions, and your server must keep working with hosts that speak older ones. In practice you use the
# SDK on both sides and read the specification at modelcontextprotocol.io for anything version-specific;
# we only claim here what the capture shows.
#
# **Client-side capabilities.** Servers can also ask things *of the client*. The client advertises which of these
# it supports during setup; the field names below come straight from the SDK's type definitions:

# %%
import mcp_types as types

print("client capability fields: ", list(types.ClientCapabilities.model_fields))
print("server capability fields: ", list(types.ServerCapabilities.model_fields))

async def one_root(ctx):
    return types.ListRootsResult(roots=[types.Root(uri=WORK.as_uri(), name="scratch")])

log_path = WORK / "tap_roots.jsonl"
params = StdioServerParameters(command=sys.executable,
                               args=[str(WORK / "tap.py"), str(log_path), str(WORK / "demo_server.py")])
with open(WORK / "stderr_roots.txt", "w") as errlog:
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write, list_roots_callback=one_root) as session:
            await session.initialize()
first = json.loads(json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])["line"])
print("capabilities the client announced:", first["params"]["capabilities"])
assert "roots" in first["params"]["capabilities"]

# %% [markdown]
# The client advertised **roots** only because we passed a `list_roots_callback`. Capabilities are opt-in and
# negotiated. What the client-side features mean:
#
# - **Roots**: the client tells the server which folders or URIs it may work in (a scoping hint, not a sandbox).
# - **Sampling**: the server asks the *host's model* to produce a completion, so a server can use an LLM without
#   holding an API key. The host stays in control and should let the user review the request.
# - **Elicitation**: the server asks the user a question through the host (for example "which account?").
#
# The SDK's `ClientSession` takes a callback for each (`list_roots_callback`, `sampling_callback`,
# `elicitation_callback`). We only verified roots on the wire above; sampling and elicitation follow the same
# advertise-then-call pattern, and you should consult the specification before relying on them.

# %% [markdown]
# ## 5. The three primitives and who controls them
#
# A server offers up to three kinds of things. The important design idea is **who decides when they are used**.
#
# | Primitive | Meaning | Controlled by | Real listing method | Example |
# |---|---|---|---|---|
# | **Tool** | a function with typed arguments that *does* something | the **model** decides to call it | `tools/list`, `tools/call` | `run_sql_query`, `create_ticket` |
# | **Resource** | read-only data addressed by a URI | the **application** decides what to load into context | `resources/list`, `resources/read` | `kb://returns_policy`, a file |
# | **Prompt** | a reusable message template with arguments | the **user** picks it (like a slash command) | `prompts/list`, `prompts/get` | `/review code=...` |
#
# Newcomers over-use tools. If the data is static and you know which item is needed, a resource is
# cheaper and safer: no model decision, no side effects. If a user should trigger a canned workflow, that is a prompt.
# Use a tool when the *model* must choose to act or query, based on the conversation.
#
# Let us read a resource and a prompt through the SDK client, to see the shapes (the next notebook builds a
# far richer server):

# %%
async def use_demo_server():
    params = StdioServerParameters(command=sys.executable, args=[str(WORK / "demo_server.py")])
    with open(WORK / "stderr_sdk.txt", "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.read_resource("note://welcome")
                prm = await session.get_prompt("review", {"code": "print(1)"})
                return res, prm

res, prm = await use_demo_server()
print(res.contents[0].text)
print(prm.messages[0].role, "|", prm.messages[0].content.text)
assert res.contents[0].text == "Hello from the demo server"

# %% [markdown]
# ## 6. The second transport: streamable HTTP
#
# stdio is for **local** servers the host launches itself. For **remote** servers (shared by a team, behind
# authentication) MCP uses **streamable HTTP**: the client sends each JSON-RPC message as an HTTP `POST` to one
# endpoint, and the server answers either with plain JSON or with a Server-Sent-Events stream. The messages
# inside are the same JSON-RPC you just saw. We start a real one on `127.0.0.1` and poke it with `httpx`.

# %%
HTTP_SERVER = textwrap.dedent('''
    import sys
    import anyio
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("demo-http")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    if __name__ == "__main__":
        anyio.run(lambda: mcp.run_streamable_http_async(host="127.0.0.1", port=int(sys.argv[1])))
''')
(WORK / "http_server.py").write_text(HTTP_SERVER, encoding="utf-8")

import httpx

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

port = free_port()
server = subprocess.Popen([sys.executable, str(WORK / "http_server.py"), str(port)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    url = f"http://127.0.0.1:{port}/mcp"
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    init_body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                            "clientInfo": {"name": "httpx", "version": "0"}}}
    for _ in range(60):                      # wait for the server to start listening (max ~12 s)
        try:
            r1 = httpx.post(url, headers=headers, json=init_body, timeout=5)
            break
        except httpx.TransportError:
            time.sleep(0.2)
    else:
        raise RuntimeError("HTTP server did not start")

    session_id = r1.headers.get("mcp-session-id")
    h2 = {**headers, "mcp-session-id": session_id, "mcp-protocol-version": "2025-11-25"}
    r2 = httpx.post(url, headers=h2, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout=5)
    r3 = httpx.post(url, headers=h2, timeout=5, json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                                      "params": {"name": "add", "arguments": {"a": 20, "b": 22}}})
    r4 = httpx.post(url, headers=headers, timeout=5, json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)

print("initialize   ->", r1.status_code, r1.headers["content-type"], "| session id header present:", bool(session_id))
print("notification ->", r2.status_code)
print("tools/call   ->", r3.status_code, "|", r3.text.strip().splitlines()[-1][:110])
print("no session id->", r4.status_code, "|", r4.text[:100])
assert r2.status_code == 202 and r3.status_code == 200 and r4.status_code == 400
assert server.poll() is not None, "server should be stopped"

# %% [markdown]
# *What just happened?* Each message was one HTTP `POST`. The response to a request came back as a
# `text/event-stream` body (the JSON-RPC reply is in a `data:` line). The server issued an `mcp-session-id`
# header on `initialize`; the client sends it back on every later request, and without it the server answered
# `400`. A notification got `202 Accepted` with no body, the HTTP twin of "no reply". Same JSON-RPC, different pipe.
# (This was the classic handshake style over HTTP; the newer per-request style does not rely on that session
# header, another reason to use the SDK client rather than hand-rolled HTTP in production.)
#
# | | **stdio** | **streamable HTTP** |
# |---|---|---|
# | Where the server runs | a child process on the user's machine | anywhere reachable by URL |
# | Who starts it | the host | you (a normal web service) |
# | Auth | inherits the user's OS permissions and environment | needs real auth (OAuth, tokens): notebook 03 |
# | Typical use | local files, dev tools, personal databases | shared team services, SaaS integrations |
# | Watch out for | anything printed to `stdout` corrupts the protocol | exposed network surface, sessions, timeouts |

# %% [markdown]
# ## Common mistakes
#
# 1. **Writing to `stdout` in a stdio server.** `print("debug")` is not a log line, it is a corrupt protocol
#    message. Log to `stderr` (notebook 03 shows the corruption on a hand-written server, and that the 2.x Python SDK diverts stray prints to stderr, which you should not rely on).
# 2. **Forgetting to flush** when writing your own line-based server: the client waits for a reply that is
#    sitting in a buffer.
# 3. **Reading a pipe without a timeout.** One silent server and your notebook or agent hangs for ever.
# 4. **Treating `isError: true` as an exception.** It is a normal result that the *model* should see; a protocol
#    `error` is a different thing. Handle both.
# 5. **Following a v1 tutorial on a v2 SDK** (`FastMCP` versus `MCPServer`): check the installed version first.
# 6. **Using a tool for static data** that a resource would serve more safely and cheaply.
# 7. **Leaving the subprocess running** after an exception: always `try/finally` or use the SDK's `async with`.

# %%
# Demonstration of mistake 3 and why the timeout matters: a server that never answers
SILENT = "import sys\nfor line in sys.stdin:\n    pass\n"
(WORK / "silent.py").write_text(SILENT, encoding="utf-8")
with RawStdioClient([str(WORK / "silent.py")]) as c:
    t0 = time.perf_counter()
    c.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    print(repr(c.recv_line(timeout=0.5)), f"after {time.perf_counter() - t0:.1f}s (we did not hang)")

# %% [markdown]
# ## Interview corner
#
# **1. What problem does MCP solve, and what does it not solve?**
# <details><summary>Answer</summary>
# It standardises how AI hosts discover and call external capabilities, turning N x M custom integrations into
# N + M. It does <b>not</b> make tools safe or reliable: authentication, authorisation, input validation,
# prompt-injection defence and good tool descriptions are still your job. It also does not replace a plain API
# call when you have one host and one system.
# </details>
#
# **2. Explain host, client and server. Where does the LLM sit?**
# <details><summary>Answer</summary>
# The host is the AI application; it owns the model and the user relationship and runs one client per server.
# A client maintains one connection to one server. The server exposes tools, resources and prompts. The model is
# inside the host and never speaks MCP directly: the host converts the model's tool calls into MCP requests and
# the results back into model input.
# </details>
#
# **3. Tools, resources and prompts: what is the difference and who controls each?**
# <details><summary>Answer</summary>
# Tools are model-controlled actions or queries with typed arguments. Resources are application-controlled,
# read-only data addressed by URI. Prompts are user-controlled templates. Choosing the right primitive is a
# safety and cost decision, not only a style one.
# </details>
#
# **4. stdio versus streamable HTTP: how do you choose?**
# <details><summary>Answer</summary>
# stdio for local servers the host launches: simple, no network exposure, runs with the user's permissions.
# Streamable HTTP for remote or shared servers: requires real authentication (typically OAuth), TLS, and thought
# about sessions and timeouts. The JSON-RPC messages are the same in both.
# </details>
#
# **5. A tool call fails. Where does the error appear?**
# <details><summary>Answer</summary>
# Failures inside tool execution (bad arguments, exceptions) are reported in a normal result with
# <code>isError: true</code> so the model can read and react. Protocol-level problems (unknown method, malformed
# request) use the JSON-RPC <code>error</code> member.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. `RawStdioClient`, `WORK`, `handle`
# and the captured `legacy` / `modern` lists from above are available.

# %% [markdown]
# ### Exercise 1: Harden the hand-rolled server
# Rewrite `handle_line(line: str)` so it takes a **raw text line** and returns a **JSON string** reply (or `None`
# for notifications). Required behaviour:
# - invalid JSON -> error code `PARSE_ERROR` with `id` `null`
# - unknown method -> `METHOD_NOT_FOUND`
# - `add` with missing or non-numeric `a`/`b` -> `INVALID_PARAMS`
# - success -> `result`; notifications (no `id`) never get a reply
#
# *Hint: `json.JSONDecodeError`, `isinstance(x, (int, float))`.*

# %% tags=["starter"]
def handle_line(line: str):
    ...

# %%
def _reply(line):
    out = handle_line(line)
    return None if out is None else json.loads(out)

r = _reply("{not json")
assert r and r["error"]["code"] == PARSE_ERROR and r["id"] is None, "invalid JSON must give -32700 with id null"
r = _reply('{"jsonrpc":"2.0","id":7,"method":"nope"}')
assert r["id"] == 7 and r["error"]["code"] == METHOD_NOT_FOUND
for bad in ('{"jsonrpc":"2.0","id":8,"method":"add","params":{"a":1}}',
            '{"jsonrpc":"2.0","id":9,"method":"add","params":{"a":"1","b":2}}',
            '{"jsonrpc":"2.0","id":10,"method":"add"}'):
    r = _reply(bad)
    assert r["error"]["code"] == INVALID_PARAMS, f"expected -32602 for {bad}"
r = _reply('{"jsonrpc":"2.0","id":11,"method":"add","params":{"a":2.5,"b":4}}')
assert r == {"jsonrpc": "2.0", "id": 11, "result": 6.5}, r
assert _reply('{"jsonrpc":"2.0","method":"add","params":{"a":1,"b":2}}') is None, "notifications get no reply"
print("Exercise 1 passed")

# %% tags=["solution"]
def handle_line(line: str):
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": PARSE_ERROR, "message": "Parse error"}})
    is_notification = "id" not in msg

    def err(code, text):
        return None if is_notification else json.dumps(
            {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": code, "message": text}})

    if msg.get("method") != "add":
        return err(METHOD_NOT_FOUND, "Method not found")
    p = msg.get("params")
    ok = isinstance(p, dict) and all(isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool) for k in ("a", "b"))
    if not ok:
        return err(INVALID_PARAMS, "Invalid params")
    if is_notification:
        return None
    return json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": p["a"] + p["b"]})

# %% [markdown]
# ### Exercise 2: Classify captured messages
# Using the `legacy` capture (a list of dicts, each with a `dir` key plus the JSON-RPC fields), count the messages
# of each shape across **both directions**: store `n_requests`, `n_notifications`, `n_responses` (successes and
# errors together). *Hint: a request has `method` and `id`; a notification has `method` but no `id`; a response has
# `result` or `error`.*

# %% tags=["starter"]
n_requests = ...
n_notifications = ...
n_responses = ...

# %%
assert n_requests + n_notifications + n_responses == len(legacy), "every message should fall in exactly one class"
expected_requests = sum(1 for m in legacy if m["dir"] == "c2s" and "id" in m)
assert n_requests == expected_requests, f"expected {expected_requests} requests"
assert n_responses == sum(1 for m in legacy if m["dir"] == "s2c"), "every server line here is a response"
assert n_requests == n_responses, "each request in this session got exactly one response"
assert n_notifications == 1, "only notifications/initialized was sent"
print(f"Exercise 2 passed ({n_requests} requests, {n_notifications} notification, {n_responses} responses)")

# %% tags=["solution"]
n_requests = sum(1 for m in legacy if "method" in m and "id" in m)
n_notifications = sum(1 for m in legacy if "method" in m and "id" not in m)
n_responses = sum(1 for m in legacy if "result" in m or "error" in m)

# %% [markdown]
# ### Exercise 3: Speak the protocol yourself
# Start `demo_server.py` with `RawStdioClient`, perform the handshake, then read the resource `note://welcome`
# with a raw `resources/read` request. Store the text in `welcome_text`, and the tool names from `tools/list`
# in `tool_names`. Always close the client (use `with`).
# *Hint: the reply is `result.contents[0].text`; the handshake is `initialize` then the `notifications/initialized` notification.*

# %% tags=["starter"]
welcome_text = ...
tool_names = ...

# %%
assert welcome_text == "Hello from the demo server", f"got {welcome_text!r}"
assert tool_names == ["add"], f"got {tool_names!r}"
print("Exercise 3 passed")

# %% tags=["solution"]
with RawStdioClient([str(WORK / "demo_server.py")]) as c:
    c.request("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                             "clientInfo": {"name": "exercise", "version": "0"}})
    c.notify("notifications/initialized")
    welcome_text = c.request("resources/read", {"uri": "note://welcome"})["result"]["contents"][0]["text"]
    tool_names = [t["name"] for t in c.request("tools/list")["result"]["tools"]]

# %% [markdown]
# ### Exercise 4: Count the wire bytes
# The modern-style session carries the client identity and capabilities in **every** request, the handshake style
# once. Compute `overhead`: total characters of the client-to-server lines in `modern` divided by the same total
# for `legacy` (both include `dir == "c2s"` messages only; serialise each message dict *without* the `dir` key using
# `json.dumps(..., separators=(",", ":"))`). Then set `modern_is_bigger` to whether that ratio is above 1.
# Do not guess: the two sessions did the same work but differ in message count, so measure.

# %% tags=["starter"]
overhead = ...
modern_is_bigger = ...

# %%
def _chars(msgs):
    return sum(len(json.dumps({k: v for k, v in m.items() if k != "dir"}, separators=(",", ":")))
               for m in msgs if m["dir"] == "c2s")
assert abs(overhead - _chars(modern) / _chars(legacy)) < 1e-9, "ratio of c2s characters, modern / legacy"
assert modern_is_bigger == (overhead > 1)
print(f"Exercise 4 passed (modern/legacy client bytes = {overhead:.2f})")

# %% tags=["solution"]
def _chars_sol(msgs):
    return sum(len(json.dumps({k: v for k, v in m.items() if k != "dir"}, separators=(",", ":")))
               for m in msgs if m["dir"] == "c2s")
overhead = _chars_sol(modern) / _chars_sol(legacy)
modern_is_bigger = bool(overhead > 1)

# %% [markdown]
# ### Exercise 5 (open-ended): Read a protocol trace
# Look at the `legacy` and `modern` captures once more (print the server's `initialize` / `server/discover` reply in full).
# In a markdown cell write three observations: what the server tells the client about itself, which fields you
# would have to change if you renamed the server, and one thing in the trace that would worry you from a
# security or privacy point of view (for example, what identifying information the client sends on every request).

# %% [markdown]
# ## Clean up
# Remove the temp folder created at the top (all subprocesses are already stopped).

# %%
shutil.rmtree(WORK, ignore_errors=True)
print("temp dir removed:", not WORK.exists())

# %% [markdown]
# ## Recap
# - MCP replaces N x M bespoke integrations with N + M protocol implementations: hosts run clients, clients
#   connect to servers, the model stays inside the host.
# - The wire format is JSON-RPC 2.0: requests, responses, errors and id-less notifications. Over stdio it is one
#   JSON message per line on stdin/stdout.
# - A session performs version negotiation and capability exchange; the installed SDK supports both a classic
#   `initialize` handshake and a newer per-request style, so trust the SDK and the capture, not memory.
# - Servers offer tools (model-controlled), resources (application-controlled) and prompts (user-controlled).
#   Tool failures come back as `isError: true` results, not protocol errors.
# - stdio is for local servers, streamable HTTP for remote ones; the messages are identical.
# - Always read pipes with timeouts and always shut subprocesses down in `finally` / `async with`.
#
# **Go deeper:** the specification and guides at modelcontextprotocol.io (architecture, transports, base
# protocol), and the JSON-RPC 2.0 specification at jsonrpc.org.
#
# **Next:** [Build an MCP server with FastMCP / MCPServer](02_build_an_mcp_server_with_fastmcp.ipynb)
