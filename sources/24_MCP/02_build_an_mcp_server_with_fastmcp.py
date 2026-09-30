# %% [markdown]
# # Build an MCP Server with FastMCP (`MCPServer` in mcp 2.x)
#
# **Why this matters.** The most common MCP task at work is: "expose our database and our documentation to the
# assistant, safely." That means a server with well-typed tools, read-only data resources, a good error story,
# and tests that prove it behaves. In this notebook you build one over the shop database and the company
# knowledge base, launch it exactly the way a real host would (as a subprocess over stdio), and verify every
# answer against an independent pandas computation.
#
# **You will learn**
# - declare tools, resources (including URI templates) and prompts with decorators and type hints
# - see how the SDK turns type hints into `inputSchema` / `outputSchema` (and how to constrain arguments)
# - launch the server from Python with `stdio_client` + `ClientSession`, list and call everything
# - tell apart tool errors (`isError`) from protocol errors, and handle both in client code
# - shut the subprocess down cleanly and prove it is gone
#
# **Prerequisites:** [What is MCP?](01_what_is_mcp.ipynb) (JSON-RPC, primitives, stdio) and
# [Tool use and function calling](../22_Agentic_AI/02_tool_use_and_function_calling.ipynb).
#
# > **A naming note.** The high-level server API you may know as `FastMCP` (`from mcp.server.fastmcp import FastMCP`,
# > `mcp` 1.x) is called **`MCPServer`** in the installed `mcp` 2.x (`from mcp.server.mcpserver import MCPServer`);
# > importing the old module raises `ModuleNotFoundError`. The decorators (`@tool`, `@resource`, `@prompt`) work the
# > same way, so everything you learn here transfers. If a tutorial says FastMCP, translate the import.
#
# > Data used: `datasets/shop.db` (SQLite, synthetic) and `datasets/knowledge_base/*.md`. See `docs/DATASETS.md`.

# %%
import asyncio
import hashlib
import inspect
import json
import sqlite3
import sys
import tempfile
import shutil
import textwrap
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import psutil
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

DATA = Path("../datasets").resolve()
DB = DATA / "shop.db"
KB = DATA / "knowledge_base"
WORK = Path(tempfile.mkdtemp(prefix="mcp_nb2_"))  # removed in the last code cell
print("mcp", version("mcp"), "| db exists:", DB.exists(), "| kb docs:", len(list(KB.glob("*.md"))))

# %% [markdown]
# ## 1. Check the API you actually have
# SDKs change; tutorials go stale. Before writing code, ask the installed package.

# %%
from mcp.server.mcpserver import MCPServer

for name in ["tool", "resource", "prompt"]:
    print(f"MCPServer.{name}{inspect.signature(getattr(MCPServer, name))}\n")
print("constructor params:", list(inspect.signature(MCPServer.__init__).parameters)[1:12], "...")

# %% [markdown]
# `tool(name, title, description, annotations, ..., structured_output)`, `resource(uri, *, name, description,
# mime_type, ...)` and `prompt(name, title, description, ...)`. When `name` / `description` are omitted the SDK
# uses the function name and its **docstring**, so the docstring is the text the model reads when deciding
# whether to call your tool. Write it for the model, not for yourself.

# %% [markdown]
# ## 2. From type hints to schemas (in-process)
# You do not need a subprocess to see what the SDK generates: a server object can list and call its own tools.

# %%
from typing import Annotated, Literal

from pydantic import BaseModel, Field

scratch = MCPServer("scratch")

class Point(BaseModel):
    x: int
    y: int

@scratch.tool()
def bounded(n: Annotated[int, Field(ge=1, le=5, description="how many, 1 to 5")] = 2) -> Point:
    """Return a point whose coordinates are n and n squared."""
    return Point(x=n, y=n * n)

@scratch.tool()
def shout(text: str, style: Literal["upper", "title"] = "upper") -> str:
    """Change the case of text."""
    return text.upper() if style == "upper" else text.title()

@scratch.tool()
def untyped_dict() -> dict:
    """Return a bare dict."""
    return {"x": 1}

tools = {t.name: t for t in await scratch.list_tools()}
print(json.dumps(tools["bounded"].input_schema, indent=1))
print("bounded output schema keys:", sorted(tools["bounded"].output_schema["properties"]))
print("shout   output schema     :", tools["shout"].output_schema["properties"])
print("untyped_dict output schema:", tools["untyped_dict"].output_schema)

# %% [markdown]
# *What just happened?* Four rules to remember:
# 1. **Argument types become JSON Schema.** `Literal[...]` becomes an `enum`; `Field(ge=1, le=5)` becomes
#    `minimum`/`maximum`; the default makes the argument optional; `Field(description=...)` documents it for the model.
# 2. A **pydantic model** return type becomes the `outputSchema`, and the result carries matching `structuredContent`.
# 3. A **plain return type** such as `str` or `list[...]` is wrapped as `{"result": ...}`.
# 4. A bare `dict` return has no output schema at all: the model gets text only. Prefer a model when the shape matters.
#
# And the schema is enforced. Calling with a value outside the constraint:

# %%
from mcp.server.mcpserver.exceptions import ToolError

ok = await scratch.call_tool("bounded", {"n": 3})
print("valid  :", ok.is_error, ok.structured_content)
try:
    await scratch.call_tool("bounded", {"n": 99})
    raise AssertionError("expected a validation failure")
except ToolError as exc:
    message = str(exc)
    print("invalid:", message.splitlines()[0], "|", message.splitlines()[2].strip())
assert ok.structured_content == {"x": 3, "y": 9} and "less than or equal to 5" in message

# %% [markdown]
# Calling the server object directly raises `ToolError` for the out-of-range `n=99`: the SDK validated the
# arguments against the schema *before* running your function. When the same call arrives over the wire, the SDK
# turns that `ToolError` into a normal result with `is_error=True` and the message as content (you will see this
# in section 6), so the model that sent `n=99` can read it and try again. This also shows why testing through a real
# client matters: the in-process call and the wire call behave differently at exactly this seam.

# %% [markdown]
# ## 3. The shop server
# Now the real server. It is a normal Python file; we write it into a temp folder so that the notebook can launch it
# as a subprocess. Design decisions worth noticing (each is a habit for production servers):
#
# - the database is opened with SQLite's **`mode=ro`** URI, so even a bug cannot write;
# - all SQL uses **`?` parameters**, never string formatting;
# - tools return **pydantic models** so clients get a typed contract;
# - the knowledge-base resource checks that the resolved path stays **inside** the KB folder;
# - file paths come from **environment variables** set by whoever launches the server, not from the model;
# - logs go to **stderr**, never to stdout (the protocol channel).

# %%
SERVER_BASE = textwrap.dedent('''
    import logging
    import os
    import sqlite3
    import sys
    from pathlib import Path
    from typing import Annotated, Literal

    from pydantic import BaseModel, Field
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)   # stderr, never stdout
    log = logging.getLogger("shop")

    DB = Path(os.environ["SHOP_DB"]).resolve()
    KB = Path(os.environ["KB_DIR"]).resolve()
    ALLOWED_TABLES = ("customers", "products", "orders", "order_items")

    mcp = MCPServer("shop", instructions="Read-only access to the shop database and the company knowledge base.")


    def connect() -> sqlite3.Connection:
        con = sqlite3.connect(f"{DB.as_uri()}?mode=ro", uri=True)   # read-only at the SQLite level
        con.row_factory = sqlite3.Row
        return con


    class CategoryRevenue(BaseModel):
        category: str
        revenue: float = Field(description="sum(unit_price * quantity) over delivered orders")
        units: int


    class Customer(BaseModel):
        customer_id: int
        first_name: str
        country: str
        signup_date: str
        age: int | None
        is_premium: bool


    class ProductUnits(BaseModel):
        product_name: str
        units: int


    @mcp.tool()
    def list_tables() -> list[str]:
        """List the tables that can be queried."""
        return list(ALLOWED_TABLES)


    @mcp.tool()
    def revenue_by_category(
        country: Annotated[str | None, Field(description="ISO country code such as US; omit for all countries")] = None,
    ) -> list[CategoryRevenue]:
        """Revenue and units sold per product category, delivered orders only."""
        sql = """SELECT p.category AS category, ROUND(SUM(p.unit_price * oi.quantity), 2) AS revenue,
                        SUM(oi.quantity) AS units
                 FROM order_items oi
                 JOIN orders o USING (order_id) JOIN products p USING (product_id) JOIN customers c USING (customer_id)
                 WHERE o.status = 'delivered' AND (? IS NULL OR c.country = ?)
                 GROUP BY p.category ORDER BY p.category"""
        with connect() as con:
            return [CategoryRevenue(**dict(r)) for r in con.execute(sql, (country, country))]


    @mcp.tool()
    def order_count(status: Literal["delivered", "shipped", "cancelled", "returned"]) -> int:
        """Number of orders that currently have the given status."""
        with connect() as con:
            return con.execute("SELECT COUNT(*) FROM orders WHERE status = ?", (status,)).fetchone()[0]


    @mcp.tool()
    def customer_lookup(customer_id: int) -> Customer:
        """Fetch one customer row by id. Fails with a clear message if the id does not exist."""
        with connect() as con:
            row = con.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
        if row is None:
            raise ToolError(f"no customer with id {customer_id}")
        return Customer(**dict(row))


    @mcp.tool()
    async def top_products(ctx: Context, limit: Annotated[int, Field(ge=1, le=20)] = 5) -> list[ProductUnits]:
        """The best-selling products by units across all orders."""
        await ctx.info(f"computing top {limit} products")          # a log message sent to the client
        with connect() as con:
            rows = con.execute(
                """SELECT p.product_name AS product_name, SUM(oi.quantity) AS units
                   FROM order_items oi JOIN products p USING (product_id)
                   GROUP BY p.product_id ORDER BY units DESC, product_name LIMIT ?""", (limit,)).fetchall()
        return [ProductUnits(**dict(r)) for r in rows]


    @mcp.tool()
    def server_pid() -> int:
        """Diagnostic: the operating-system process id of this server."""
        return os.getpid()


    @mcp.resource("schema://shop", mime_type="text/plain")
    def schema() -> str:
        """The CREATE TABLE statements of the shop database."""
        with connect() as con:
            return "\\n".join(r[0] for r in con.execute("SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name"))


    @mcp.resource("kb://index", mime_type="application/json")
    def kb_index() -> list[str]:
        """Names of all knowledge-base documents."""
        return sorted(p.stem for p in KB.glob("*.md"))


    @mcp.resource("kb://{doc}", mime_type="text/markdown")
    def kb_doc(doc: str) -> str:
        """One knowledge-base document by name (without the .md extension)."""
        path = (KB / f"{doc}.md").resolve()
        if path.parent != KB or not path.is_file():                 # stay inside the KB folder
            raise ValueError(f"unknown document {doc!r}")
        return path.read_text(encoding="utf-8")


    @mcp.prompt()
    def support_answer(question: str) -> str:
        """Template: answer a customer question using only knowledge-base documents."""
        return ("You are a support agent. Read kb://index, open the relevant documents, and answer using only "
                f"what they say. If they do not cover it, say so.\\nCustomer question: {question}")
''')

SERVER_FOOTER = textwrap.dedent('''

    if __name__ == "__main__":
        mcp.run("stdio")
''')
print(len(SERVER_BASE.splitlines()), "lines of server source")

# %% [markdown]
# ## 4. Launch it like a host would
# `stdio_client(params)` starts the subprocess and yields a pair of streams; `ClientSession` speaks MCP over them;
# `session.initialize()` performs the handshake. Both are **async context managers**: leaving the `async with`
# block closes stdin, waits for the server, and force-kills it if it does not exit, so cleanup is automatic
# even when your code raises.
#
# We wrap the boilerplate once. Three details: the environment passed to the server is explicit (a launched
# server does not inherit your whole environment, only a safe subset plus what you add); the server's stderr goes to a
# file (a notebook's own `stderr` has no file descriptor to hand to a child); and every request has a read timeout,
# so a stuck server fails the cell instead of hanging it.

# %%
@asynccontextmanager
async def shop_session(extra_src: str = "", logging_callback=None):
    """Write the server (base + optional extra tool code), launch it, yield an initialised ClientSession."""
    source = SERVER_BASE + extra_src + SERVER_FOOTER
    tag = hashlib.md5(source.encode()).hexdigest()[:8]
    path = WORK / f"shop_server_{tag}.py"
    path.write_text(source, encoding="utf-8")
    params = StdioServerParameters(command=sys.executable, args=[str(path)],
                                   env={"SHOP_DB": str(DB), "KB_DIR": str(KB)})
    with open(WORK / f"stderr_{tag}.txt", "a", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=30,
                                     logging_callback=logging_callback) as session:
                await session.initialize()
                yield session


async with shop_session() as s:
    print("server:", s.server_info.name, "| protocol:", s.protocol_version)
    tools = (await s.list_tools()).tools
    resources = (await s.list_resources()).resources
    templates = (await s.list_resource_templates()).resource_templates
    prompts = (await s.list_prompts()).prompts

pd.DataFrame([{"tool": t.name, "required": t.input_schema.get("required", []),
               "args": list(t.input_schema["properties"]), "has_output_schema": t.output_schema is not None}
              for t in tools])

# %% [markdown]
# *What just happened?* We started a process, did the handshake, listed the four things a server can offer, and
# left the block, which stopped the process. The table is what a host would turn into the tool list it shows the model.

# %%
print("resources:", [(r.uri, r.mime_type) for r in resources])
print("templates:", [(t.uri_template, t.mime_type) for t in templates])
print("prompts:  ", [(p.name, [a.name for a in p.arguments]) for p in prompts])
assert {t.name for t in tools} >= {"revenue_by_category", "order_count", "customer_lookup", "top_products"}
assert [t.uri_template for t in templates] == ["kb://{doc}"]

# %% [markdown]
# Resources come in two flavours. **Static resources** (`schema://shop`, `kb://index`) have a fixed URI and appear in
# `resources/list`. **Resource templates** (`kb://{doc}`) are URI patterns: the client fills the placeholder and reads.
# You cannot list every possible document from a template, so also offer an index resource that says what exists.

# %% [markdown]
# ## 5. Call the tools and verify the numbers
# A server you cannot test is a liability. We compute the expected answers **independently**, straight from the
# database with pandas, then compare with what the server returned.

# %%
con = sqlite3.connect(f"{DB.as_uri()}?mode=ro", uri=True)
customers = pd.read_sql("SELECT * FROM customers", con)
products = pd.read_sql("SELECT * FROM products", con)
orders = pd.read_sql("SELECT * FROM orders", con)
items = pd.read_sql("SELECT * FROM order_items", con)
con.close()

def expected_revenue(country=None) -> pd.DataFrame:
    df = (items.merge(orders, on="order_id").merge(products, on="product_id").merge(customers, on="customer_id"))
    df = df[df["status"] == "delivered"]
    if country:
        df = df[df["country"] == country]
    df = df.assign(revenue=df["unit_price"] * df["quantity"])
    return df.groupby("category").agg(revenue=("revenue", "sum"), units=("quantity", "sum")).round(2).reset_index()

async with shop_session() as s:
    r_us = await s.call_tool("revenue_by_category", {"country": "US"})
    r_all = await s.call_tool("revenue_by_category", {})
    cnt = await s.call_tool("order_count", {"status": "returned"})

got_us = pd.DataFrame(r_us.structured_content["result"])
exp_us = expected_revenue("US")
pd.testing.assert_frame_equal(got_us.sort_values("category").reset_index(drop=True),
                              exp_us.sort_values("category").reset_index(drop=True), check_dtype=False)
assert len(r_all.structured_content["result"]) == products["category"].nunique()
assert cnt.structured_content["result"] == int((orders["status"] == "returned").sum())
got_us

# %% [markdown]
# The server and the independent computation agree, to the cent. Notice we compared **`structured_content`**,
# not the text. Every result has two views of the same data: `content` (text blocks, what the model reads) and
# `structured_content` (JSON matching the output schema, what code should read). Never parse the text in code.

# %%
print("content[0].text  :", r_us.content[0].text.replace("\n", " ")[:80], "...")
print("number of blocks :", len(r_us.content), "| structured keys:", list(r_us.structured_content))
print("order_count text :", repr(cnt.content[0].text), "| structured:", cnt.structured_content)

# %% [markdown]
# ### Server-to-client logging
# `top_products` is an `async` tool that accepts a `Context` parameter. The SDK injects it (it is not part of the tool's
# input schema), and `await ctx.info(...)` sends a log notification to the client while the tool runs. Clients receive
# them through a `logging_callback`.

# %%
received_logs = []

async def on_log(params):
    received_logs.append((params.level, params.data))

async with shop_session(logging_callback=on_log) as s:
    top = await s.call_tool("top_products", {"limit": 3})
    tools_now = {t.name: t for t in (await s.list_tools()).tools}

exp_top = (items.merge(products, on="product_id").groupby(["product_id", "product_name"])["quantity"].sum()
           .reset_index().sort_values(["quantity", "product_name"], ascending=[False, True]).head(3))
assert [(p["product_name"], p["units"]) for p in top.structured_content["result"]] == \
       list(zip(exp_top["product_name"], exp_top["quantity"]))
assert "ctx" not in tools_now["top_products"].input_schema["properties"], "Context must not leak into the schema"
print("top 3:", [(p["product_name"], p["units"]) for p in top.structured_content["result"]])
print("logs received by the client:", received_logs)

# %% [markdown]
# ## 6. Two kinds of errors, handled properly
# From notebook 01: a failing **tool call** is a normal result with `is_error=True` (for the model to read); a
# problem with the **request itself** raises `MCPError` in the SDK client. Here are the cases you will meet:

# %%
async with shop_session() as s:
    bad_enum = await s.call_tool("order_count", {"status": "lost"})           # violates the Literal
    too_many = await s.call_tool("top_products", {"limit": 500})              # violates le=20
    missing = await s.call_tool("customer_lookup", {"customer_id": 10**9})    # the tool raised ToolError
    no_tool = await s.call_tool("drop_everything", {})                        # unknown tool
    try:
        await s.read_resource("kb://does_not_exist")
        resource_error = None
    except MCPError as exc:
        resource_error = exc

for label, res in [("bad enum", bad_enum), ("limit 500", too_many), ("missing id", missing), ("unknown tool", no_tool)]:
    print(f"{label:<13} is_error={res.is_error} | {res.content[0].text.splitlines()[0][:70]}")
print("unknown doc   ->", type(resource_error).__name__, "|", str(resource_error)[:70])
assert all(r.is_error for r in [bad_enum, too_many, missing, no_tool]) and resource_error is not None
assert "no customer" in missing.content[0].text

# %% [markdown]
# *What just happened?* Four different mistakes, one shape. `ToolError` is the way to say "this is a business-level
# failure, here is a message for the model" (the id does not exist). Note the resource failure is different: it
# surfaced as an exception on the client and its message does **not** contain our `unknown document` text, because
# the SDK wraps exceptions raised inside resource functions. Design resource errors accordingly (an index resource
# helps the caller avoid them).
#
# A client helper that gives your own code a single, sane failure path:

# %%
class ToolFailed(Exception):
    pass

async def call_checked(session: ClientSession, name: str, args: dict | None = None):
    """Return structured content, or raise ToolFailed with the message the model would have seen."""
    res = await session.call_tool(name, args or {})
    if res.is_error:
        raise ToolFailed(res.content[0].text if res.content else "tool failed")
    return res.structured_content

async with shop_session() as s:
    try:
        await call_checked(s, "customer_lookup", {"customer_id": 10**9})
    except ToolFailed as exc:
        print("caught:", exc)
    who = await call_checked(s, "customer_lookup", {"customer_id": 1})
row1 = customers.loc[customers["customer_id"] == 1].iloc[0]
assert who["customer_id"] == 1 and who["first_name"] == row1["first_name"] and who["country"] == row1["country"]        and who["is_premium"] == bool(row1["is_premium"])
print(who)

# %% [markdown]
# ## 7. Resources and prompts
# Reading a resource returns `contents`. We check that the template returns exactly the file's bytes, and that
# the index matches the folder.

# %%
async with shop_session() as s:
    index = await s.read_resource("kb://index")
    doc = await s.read_resource("kb://returns_policy")
    schema_text = await s.read_resource("schema://shop")
    prompt = await s.get_prompt("support_answer", {"question": "Can I return headphones?"})

names = json.loads(index.contents[0].text)
assert names == sorted(p.stem for p in KB.glob("*.md"))
assert doc.contents[0].text == (KB / "returns_policy.md").read_text(encoding="utf-8")
assert "CREATE TABLE" in schema_text.contents[0].text
print(names[:4], "...")
print(doc.contents[0].text.splitlines()[0], "|", doc.contents[0].mime_type)
print(prompt.messages[0].role, "|", prompt.messages[0].content.text[:100], "...")

# %% [markdown]
# A **prompt** is not run by the server: `prompts/get` just returns ready-made messages that a host shows to the
# user (as a slash command, say) or injects into the conversation. Here it tells the model to *use the resources*.
# Servers, resources and prompts compose: the prompt is the workflow, the index and template are the data access.

# %% [markdown]
# ## 8. Prove the subprocess is gone
# "It exited when the block ended" is a claim; we can test it. The `server_pid` tool tells us the process id while
# the session is open, then `psutil` tells us whether that process still exists afterwards.

# %%
async with shop_session() as s:
    pid = (await call_checked(s, "server_pid"))["result"]
    alive_inside = psutil.pid_exists(pid)

for _ in range(50):                 # allow a moment for the operating system to reap it
    if not psutil.pid_exists(pid):
        break
    await asyncio.sleep(0.1)
print(f"pid {pid}: alive inside block = {alive_inside}, alive after block = {psutil.pid_exists(pid)}")
assert alive_inside and not psutil.pid_exists(pid)

# %% [markdown]
# ## Common mistakes
#
# 1. **`print()` in a stdio server.** It writes into the protocol stream. Use `logging` to stderr (notebook 03 demonstrates the breakage).
# 2. **Returning bare `dict`s / untyped values** from tools: no output schema, the model gets an unstructured blob. Use pydantic models or typed returns.
# 3. **Building SQL with f-strings** from tool arguments. Use `?` parameters (and, for the general case, the hardening in notebook 03).
# 4. **Vague docstrings.** The docstring is the tool description the model uses to choose tools. "Gets data" loses to "Revenue and units per product category, delivered orders only".
# 5. **Trusting relative paths.** Servers are launched by the host in *its* working directory. Pass absolute paths via environment variables or arguments.
# 6. **Parsing `content[0].text` in code.** Use `structured_content`.
# 7. **Not testing through a real client.** Calling your function directly skips schema generation, validation and serialisation, where many bugs live.
# 8. **Assuming the server inherits your environment.** `stdio_client` passes a restricted default environment plus `StdioServerParameters(env=...)`; a missing variable shows up as a server crash at start-up.

# %%
# Demonstration of mistake 8: the server needs SHOP_DB, we forgot to pass it
broken = WORK / "broken_server.py"
broken.write_text(SERVER_BASE + SERVER_FOOTER, encoding="utf-8")
params = StdioServerParameters(command=sys.executable, args=[str(broken)], env={"KB_DIR": str(KB)})
try:
    with open(WORK / "broken_stderr.txt", "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=5) as session:
                await session.initialize()
    print("unexpectedly worked")
except BaseException as exc:                     # noqa: BLE001 - we only want to show that it fails
    print("client-side failure:", type(exc).__name__)
print("server's stderr says:", (WORK / "broken_stderr.txt").read_text(encoding="utf-8").strip().splitlines()[-1])

# %% [markdown]
# ## Interview corner
#
# **1. How does an MCP server know what JSON Schema to advertise for a tool?**
# <details><summary>Answer</summary>
# High-level SDKs inspect the function signature: argument type hints become <code>inputSchema</code> (pydantic
# validation is used at call time), the return annotation becomes <code>outputSchema</code>, the docstring becomes the
# description. Constraints go in <code>Annotated[..., Field(...)]</code>. The schema is the contract both the model
# and the SDK's validator use.
# </details>
#
# **2. When would you expose something as a resource rather than a tool?**
# <details><summary>Answer</summary>
# When it is read-only data identified by a URI that the application (or user) chooses to load into context: files,
# documents, schemas. A tool is for something the <i>model</i> decides to invoke with arguments, especially when it
# computes or has side effects. Resources avoid a model round-trip and cannot mutate anything.
# </details>
#
# **3. A tool fails on bad input. Raise an exception, return an error string, or use `isError`?**
# <details><summary>Answer</summary>
# Signal it with <code>isError: true</code> (in the SDK, raise <code>ToolError</code> or let validation fail) and put an
# actionable message in the content, so the model can correct itself. Do not return an error as normal-looking data,
# and do not crash the server process.
# </details>
#
# **4. How do you test an MCP server?**
# <details><summary>Answer</summary>
# Through a real client over the real transport: launch the server as a subprocess, list capabilities, call tools,
# assert on <code>structured_content</code> against values computed independently, include error cases, and always
# clean the subprocess up. Unit-test the pure functions underneath separately.
# </details>

# %% [markdown]
# ## Exercises
#
# `shop_session`, `call_checked`, the `orders` / `customers` / `products` / `items` DataFrames and the KB path are
# available. Each check launches the server for real.

# %% [markdown]
# ### Exercise 1: Add a tool
# Write the source of a **new tool** `orders_by_channel(channel: Literal["web", "mobile", "store"]) -> int` that returns
# how many orders came through that channel. Store the code as a string in `extra_tool_src`; it is inserted into the
# server file *after* the base tools (so `mcp`, `connect`, `Literal` are already defined). Use a `?` parameter.
# *Hint: copy the shape of `order_count`.*

# %% tags=["starter"]
extra_tool_src = '''
'''

# %%
assert "orders_by_channel" in extra_tool_src, "define a function called orders_by_channel"
async with shop_session(extra_src="\n" + extra_tool_src) as s:
    listed = {t.name: t for t in (await s.list_tools()).tools}
    assert "orders_by_channel" in listed, "the tool was not registered: did you use @mcp.tool()?"
    assert listed["orders_by_channel"].input_schema["properties"]["channel"].get("enum") == ["web", "mobile", "store"], \
        "use Literal['web', 'mobile', 'store'] so the schema carries an enum"
    for ch in ["web", "mobile", "store"]:
        got = (await call_checked(s, "orders_by_channel", {"channel": ch}))["result"]
        assert got == int((orders["channel"] == ch).sum()), f"wrong count for {ch}: {got}"
    rejected = await s.call_tool("orders_by_channel", {"channel": "fax"})
    assert rejected.is_error, "an unknown channel must be rejected by the schema"
print("Exercise 1 passed")

# %% tags=["solution"]
extra_tool_src = '''
@mcp.tool()
def orders_by_channel(channel: Literal["web", "mobile", "store"]) -> int:
    """Number of orders placed through a sales channel."""
    with connect() as con:
        return con.execute("SELECT COUNT(*) FROM orders WHERE channel = ?", (channel,)).fetchone()[0]
'''

# %% [markdown]
# ### Exercise 2: A client that uses resources
# Write `async def kb_search(session, word)` that returns the **sorted list of document names** in the knowledge base
# whose text contains `word` (case-insensitive). It must discover the documents through the `kb://index` resource and read
# each through the `kb://{doc}` template, not from the disk.
# *Hint: `json.loads(result.contents[0].text)` for the index; `read_resource(f"kb://{name}")` for each doc.*

# %% tags=["starter"]
async def kb_search(session, word):
    ...

# %%
async with shop_session() as s:
    for word in ["refund", "PREMIUM", "warehouse", "zzz-not-there"]:
        got = await kb_search(s, word)
        want = sorted(p.stem for p in KB.glob("*.md") if word.lower() in p.read_text(encoding="utf-8").lower())
        assert got == want, f"{word!r}: expected {want}, got {got}"
print("Exercise 2 passed")

# %% tags=["solution"]
async def kb_search(session, word):
    names = json.loads((await session.read_resource("kb://index")).contents[0].text)
    hits = []
    for name in names:
        text = (await session.read_resource(f"kb://{name}")).contents[0].text
        if word.lower() in text.lower():
            hits.append(name)
    return sorted(hits)

# %% [markdown]
# ### Exercise 3: Cross-check the server against pandas
# Using only MCP calls, find which **country** has the highest delivered revenue. Call `revenue_by_category` once
# per country (get the country codes from the `customers` DataFrame) and add up the revenue. Store the country in
# `top_country` and its total revenue (rounded to 2 decimals) in `top_revenue`.
# *Hint: loop inside one `async with shop_session() as s:` block and sum `structured_content["result"]`.*

# %% tags=["starter"]
top_country = ...
top_revenue = ...

# %%
exp_tbl = {c: expected_revenue(c)["revenue"].sum() for c in sorted(customers["country"].unique())}
exp_country = max(exp_tbl, key=exp_tbl.get)
assert top_country == exp_country, f"expected {exp_country}"
assert abs(top_revenue - round(exp_tbl[exp_country], 2)) < 0.02, f"expected about {exp_tbl[exp_country]:.2f}"
print(f"Exercise 3 passed ({top_country}: {top_revenue:,.2f})")

# %% tags=["solution"]
totals = {}
async with shop_session() as s:
    for country in sorted(customers["country"].unique()):
        rows = (await call_checked(s, "revenue_by_category", {"country": country}))["result"]
        totals[country] = sum(r["revenue"] for r in rows)
top_country = max(totals, key=totals.get)
top_revenue = round(totals[top_country], 2)

# %% [markdown]
# ### Exercise 4: One safe wrapper for the model
# Write `async def safe_call(session, name, args)` returning a tuple `(ok: bool, text: str)`:
# - `(True, <text of the first content block>)` on success
# - `(False, <message>)` when the tool result has `is_error`
# - `(False, <message>)` when the SDK raises `MCPError` (a protocol-level failure)
# It must never raise for those cases. This is the shape of the glue in an agent loop that feeds results back to a model.
# *Hint: `try/except MCPError` around `call_tool`, then check `res.is_error`.*

# %% tags=["starter"]
async def safe_call(session, name, args):
    ...

# %%
async with shop_session() as s:
    ok, text = await safe_call(s, "order_count", {"status": "delivered"})
    assert ok is True and text.strip() == str(int((orders["status"] == "delivered").sum())), (ok, text)
    ok, text = await safe_call(s, "order_count", {"status": "nope"})
    assert ok is False and text, "schema violations must come back as (False, message)"
    ok, text = await safe_call(s, "no_such_tool", {})
    assert ok is False and text, "unknown tools must come back as (False, message)"
    ok, text = await safe_call(s, "customer_lookup", {"customer_id": 10**9})
    assert ok is False and "no customer" in text
print("Exercise 4 passed")

# %% tags=["solution"]
async def safe_call(session, name, args):
    try:
        res = await session.call_tool(name, args)
    except MCPError as exc:
        return False, str(exc)
    text = res.content[0].text if res.content else ""
    return (not res.is_error), text

# %% [markdown]
# ### Exercise 5 (open-ended): Design the next capability
# A support lead asks for "look up my order status by order id, and explain the return policy for it". Decide which
# parts should be a **tool**, a **resource** and a **prompt**, and what could go wrong (an id that is not the caller's;
# stale data). Write your design in a markdown cell (5-8 lines). Then implement one of the pieces with `extra_src`
# in `shop_session` and test it against the database.

# %% [markdown]
# ## Clean up

# %%
shutil.rmtree(WORK, ignore_errors=True)
print("temp dir removed:", not WORK.exists())

# %% [markdown]
# ## Recap
# - In `mcp` 2.x the FastMCP-style server class is `MCPServer`; decorate functions with `@mcp.tool()`, `@mcp.resource(uri)`, `@mcp.prompt()`.
# - Type hints and `Annotated[..., Field(...)]` generate `inputSchema`; return types (pydantic models best) generate `outputSchema` and `structuredContent`.
# - Resources are read-only data by URI (static or `{template}`), prompts are user-selected templates, tools are model-selected actions.
# - `stdio_client` + `ClientSession` launch and talk to a server; use timeouts, explicit env, an `errlog` file, and let `async with` clean up.
# - Tool failures are `isError` results with messages for the model; protocol failures raise `MCPError`. Handle both.
# - Test through a real client against independently computed expectations, and prove the subprocess exits.
#
# **Go deeper:** the MCP server guides at modelcontextprotocol.io (tools, resources, prompts) and the `mcp` Python SDK
# documentation. The specification lists the exact message shapes we inspected in notebook 01.
#
# **Next:** [Connect clients and secure your server](03_connect_clients_and_secure_your_server.ipynb)
