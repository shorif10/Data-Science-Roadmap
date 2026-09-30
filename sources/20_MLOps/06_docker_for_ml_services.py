# %% [markdown]
# # Docker for ML Services
#
# **Why this matters.** "It works on my machine" is the most expensive sentence in ML deployment, because ML environments are fragile: a particular scikit-learn version, a system library, a model file in a particular place. A **container image** freezes the whole environment, so the thing you tested is the thing that runs, on a teammate's laptop, in CI, and in production.
# Most ML engineers do not need to be Docker experts, but they must be able to read a Dockerfile, spot the mistakes that cost minutes per build or leak secrets, and structure an image so rebuilding after a code change takes seconds.
#
# **Docker is not running in this environment** (no daemon). So this notebook teaches with what *can* run: a **Dockerfile parser and linter** that flags the common mistakes, a **build-cache simulator** that shows why instruction order matters, a `.dockerignore` measurement, and a **compose file checker**. The real `docker` commands are shown, not executed.
#
# **You will learn**
# - what images, containers and layers are, and how the **build cache** decides what is rebuilt
# - write a production-minded Dockerfile for the FastAPI churn service (non-root user, pinned base, health check, exec-form `CMD`)
# - lint a Dockerfile for the classic mistakes and see the cache impact of instruction order
# - describe a multi-service stack (API + tracking server + database) in `docker-compose.yaml` and validate it
# - keep the build context small with `.dockerignore`
#
# **Prerequisites:** [Serving models with FastAPI](04_serving_models_with_fastapi.ipynb). Helpful: [The terminal](../00_Setup_and_Tools/01_terminal_and_files.ipynb).

# %%
import atexit
import fnmatch
import hashlib
import os
import re
import shutil
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TMP = Path(tempfile.mkdtemp(prefix="docker_nb_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

# %% [markdown]
# ## 1. The mental model
# * An **image** is a read-only template: a stack of **layers**, each the result of one Dockerfile instruction (`FROM`, `COPY`, `RUN`, ...).
# * A **container** is a running instance of an image (a process with an isolated filesystem), cheap to start and throw away.
# * The **build cache**: Docker reuses a layer if the instruction **and everything before it** are unchanged (for `COPY`, also the content of the copied files). The first changed layer invalidates **all layers after it**.
#
# That last rule is the single most useful thing to know. It means: put slow, rarely-changing steps (installing dependencies) **early** and fast, frequently-changing steps (your code) **late**.

# %% [markdown]
# ## 2. Two Dockerfiles for the same service
# First the version most people write on day one, then the one you would review approvingly. Read both before we lint them.

# %%
BAD = textwrap.dedent("""
    FROM python:latest
    ENV API_TOKEN=abc123secret
    RUN apt-get update
    RUN apt-get install -y curl build-essential
    COPY . /app
    WORKDIR /app
    RUN pip install pandas scikit-learn fastapi uvicorn
    ADD model/model.joblib /app/model/model.joblib
    EXPOSE 8000
    CMD uvicorn churn_service:app --host 0.0.0.0 --port 8000
""").lstrip()

GOOD = textwrap.dedent("""
    # syntax=docker/dockerfile:1
    FROM python:3.11-slim

    ENV PYTHONDONTWRITEBYTECODE=1 \\
        PYTHONUNBUFFERED=1
    WORKDIR /app
    RUN useradd --create-home --uid 10001 appuser

    # 1) dependencies first: this layer is rebuilt only when requirements.txt changes
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # 2) the model changes less often than the code
    COPY model/ ./model/
    # 3) application code last: the fast, frequently changing layer
    COPY src/ ./src/

    USER appuser

    EXPOSE 8000
    HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \\
      CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8000/health')"]
    CMD ["uvicorn", "src.churn_service:app", "--host", "0.0.0.0", "--port", "8000"]
""").lstrip()
print(GOOD)

# %% [markdown]
# Why each choice in `GOOD`:
# * **`python:3.11-slim`, not `python:latest`**: `latest` silently changes under you (a rebuild next month may be a different Python). `slim` is far smaller than the full image. For strict reproducibility pin the digest (`python:3.11-slim@sha256:...`).
# * **`requirements.txt` first, pinned versions inside it** (from `pip freeze` or a lock file): the dependency layer is cached until dependencies actually change.
# * **`--no-cache-dir`**: pip's download cache would otherwise be baked into the layer, making the image bigger for no benefit.
# * **Non-root `USER`**: if the service is compromised, the attacker is not root inside the container.
# * **`HEALTHCHECK`**: lets Docker/compose/Kubernetes know the service is actually answering, not just that the process exists (this is the `/health` endpoint from the last notebook).
# * **Exec-form `CMD ["uvicorn", ...]`**: the server becomes PID 1 and receives `SIGTERM` directly, so it shuts down gracefully. The shell form (`CMD uvicorn ...`) wraps it in `/bin/sh -c`, which does not forward signals.
# * **No secrets in `ENV`/`ARG`**: anything in an instruction is readable by whoever can run `docker history` on the image. Pass secrets at run time (env var, mounted secret).
#
# Next, a parser and a linter so a machine can check what a reviewer would.

# %% [markdown]
# ## 3. A Dockerfile parser and linter
# A Dockerfile is a list of instructions, one per logical line, where a trailing `\` continues the line and `#` starts a comment. Parsing it is 12 lines:

# %%
def parse_dockerfile(text):
    """-> list of {"line": first line number, "op": INSTRUCTION, "args": str}"""
    instrs, buf, start = [], "", None
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not buf and (not s or s.startswith("#")):
            continue
        if start is None:
            start = n
        if s.endswith("\\"):
            buf += s[:-1].rstrip() + " "
            continue
        buf += s
        op, _, args = buf.partition(" ")
        instrs.append({"line": start, "op": op.upper(), "args": args.strip()})
        buf, start = "", None
    return instrs

for i in parse_dockerfile(GOOD)[:5]:
    print(i)

# %% [markdown]
# Now the rules. Each returns findings with a severity (`error` = fix before shipping, `warning` = probably wrong, `info` = worth knowing). Read the rule set as a review checklist:

# %%
SECRET_WORDS = re.compile(r"(SECRET|PASSWORD|PASSWD|TOKEN|API_?KEY|PRIVATE_?KEY)", re.I)

def lint_dockerfile(text):
    ins = parse_dockerfile(text)
    out = []
    add = lambda sev, line, code, msg: out.append({"severity": sev, "line": line, "code": code, "message": msg})

    froms = [i for i in ins if i["op"] == "FROM"]
    if not froms:
        add("error", 1, "DL000", "no FROM instruction")
    for f in froms:
        image = f["args"].split()[0]
        if image.lower() != "scratch" and "$" not in image and (":" not in image.split("/")[-1] and "@" not in image or image.endswith(":latest")):
            add("error", f["line"], "DL001", f"base image '{image}' is unpinned (no tag or :latest): builds are not reproducible")

    last_from = max((i["line"] for i in froms), default=0)
    users = [i for i in ins if i["op"] == "USER" and i["line"] > last_from]
    if not users or all(u["args"].split()[0] in ("root", "0") for u in users):
        add("warning", last_from or 1, "DL002", "no non-root USER in the final stage: the service runs as root")

    def copies_everything(i):
        sources = [t for t in i["args"].split() if not t.startswith("--")][:-1]
        return i["op"] == "COPY" and any(t in (".", "./") for t in sources)
    first_copy_all = next((i for i in ins if copies_everything(i)), None)
    first_pip = next((i for i in ins if i["op"] == "RUN" and "pip install" in i["args"]), None)
    if first_copy_all and first_pip and first_copy_all["line"] < first_pip["line"]:
        add("warning", first_copy_all["line"], "DL003", "COPY of the whole context happens before pip install: every code edit re-installs all dependencies")

    for i in ins:
        a = i["args"]
        if i["op"] == "RUN" and "pip install" in a:
            if "--no-cache-dir" not in a and "PIP_NO_CACHE_DIR" not in text:
                add("warning", i["line"], "DL004", "pip install without --no-cache-dir bakes the pip cache into the layer")
            after = a.split("pip install", 1)[1].split("&&")[0].split()
            skip_next = False
            unpinned = []
            for tok in after:
                if skip_next:
                    skip_next = False
                    continue
                if tok in ("-r", "--requirement", "-c", "--constraint", "-e"):
                    skip_next = True
                    continue
                if tok.startswith("-") or tok.startswith((".", "/")):
                    continue
                if "==" not in tok and "@" not in tok:
                    unpinned.append(tok)
            if unpinned:
                add("warning", i["line"], "DL010", f"unpinned packages {unpinned}: use a pinned requirements file")
        if i["op"] == "RUN" and "apt-get" in a:
            if re.fullmatch(r"apt-get update", a.strip()):
                add("error", i["line"], "DL005", "'apt-get update' on its own line is cached stale: combine it with install in one RUN")
            if "apt-get install" in a:
                if "--no-install-recommends" not in a:
                    add("warning", i["line"], "DL006", "apt-get install without --no-install-recommends pulls extra packages")
                if "/var/lib/apt/lists" not in a and "apt-get update" in a:
                    add("warning", i["line"], "DL006", "apt lists are not removed in the same RUN: the layer keeps the package index")
        if i["op"] == "RUN" and re.search(r"curl[^|]*\|\s*(sudo\s+)?(ba)?sh", a):
            add("error", i["line"], "DL011", "piping curl into a shell runs unverified remote code at build time")
        if i["op"] == "RUN" and re.search(r"\bsudo\b", a):
            add("warning", i["line"], "DL012", "sudo in a Dockerfile is unnecessary (you are already root during the build)")
        if i["op"] == "ADD" and not re.match(r"https?://", a.split()[0]) and not re.search(r"\.(tar|tgz|gz|xz|bz2)\b", a.split()[0]):
            add("warning", i["line"], "DL007", "use COPY for local files; ADD has surprising extra behaviour (URLs, auto-extraction)")
        if i["op"] in ("CMD", "ENTRYPOINT") and not a.startswith("["):
            add("warning", i["line"], "DL008", f"{i['op']} in shell form: the process is not PID 1 and will not receive SIGTERM")
        if i["op"] in ("ENV", "ARG"):
            for key, val in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)", a):
                if SECRET_WORDS.search(key) and not val.startswith("$"):
                    add("error", i["line"], "DL009", f"{i['op']} {key} holds a literal secret: it is readable in the image history")
    if not any(i["op"] == "HEALTHCHECK" for i in ins):
        add("info", last_from or 1, "DL013", "no HEALTHCHECK: the orchestrator cannot tell a stuck service from a healthy one")
    return sorted(out, key=lambda f: (f["line"], f["code"]))

def show(findings):
    return pd.DataFrame(findings) if findings else "no findings"

print("BAD Dockerfile:")
show(lint_dockerfile(BAD))

# %%
print("GOOD Dockerfile:", show(lint_dockerfile(GOOD)))

# %% [markdown]
# *What just happened?* The `BAD` file triggers a finding for nearly every habit listed above (unpinned `latest`, a literal secret, stale `apt-get update`, whole-context `COPY` before installs, `ADD`, shell-form `CMD`, root user, no health check), and `GOOD` is clean.
# A linter like this is the idea behind real tools (for example `hadolint`, which you can run in CI); ours covers a small subset, so a clean result here is not a guarantee.

# %% [markdown]
# ## 4. Why order matters: a build-cache simulator
# We model Docker's rule directly: the cache key of a layer is a hash of the **previous layer's key**, the **instruction text**, and, for `COPY`/`ADD`, the **content of the files it copies**. If a key was seen in an earlier build, the layer is cached; otherwise it is rebuilt, and (because keys chain) so is everything after it.
# This is a model of the rule, not Docker itself, but it reproduces the behaviour you will see in real builds.

# %%
def files_for_copy(args, context):
    toks = [t for t in args.split() if not t.startswith("--")]
    srcs = toks[:-1]
    chosen = {}
    for s in srcs:
        s = s.lstrip("./") if s not in (".", "./") else "."
        for path, content in context.items():
            if s == "." or path == s or (path.startswith(s.rstrip("/") + "/")):
                chosen[path] = content
    return chosen

def simulate_build(dockerfile_text, context, seen):
    """Return [(instruction, 'CACHED'|'REBUILT')]. `seen` is the set of cache keys from earlier builds (updated in place)."""
    key, report = "", []
    for i in parse_dockerfile(dockerfile_text):
        material = f"{key}|{i['op']} {i['args']}"
        if i["op"] in ("COPY", "ADD"):
            material += "|" + "|".join(f"{p}:{hashlib.sha256(c.encode()).hexdigest()}" for p, c in sorted(files_for_copy(i["args"], context).items()))
        key = hashlib.sha256(material.encode()).hexdigest()
        report.append((f"{i['op']} {i['args'][:38]}", "CACHED" if key in seen else "REBUILT"))
        seen.add(key)
    return report

CONTEXT = {"requirements.txt": "scikit-learn==1.9.1\nfastapi\n", "model/model.joblib": "weights-v1", "src/churn_service.py": "print('v1')\n"}

def compare_orderings():
    results = {}
    for label, df_text in (("BAD order (COPY . first)", BAD.replace("python:latest", "python:3.11-slim").replace("ENV API_TOKEN=abc123secret\n", "")), ("GOOD order", GOOD)):
        seen, ctx = set(), dict(CONTEXT)
        simulate_build(df_text, ctx, seen)                                   # initial build fills the cache
        ctx["src/churn_service.py"] = "print('v2')\n"                        # the developer edits one line of application code
        results[label] = simulate_build(df_text, ctx, seen)
    return results

for label, report in compare_orderings().items():
    rebuilt = [name for name, status in report if status == "REBUILT"]
    pip_reran = any("pip install" in n for n in rebuilt)
    print(f"{label:26s} layers rebuilt after a code edit: {len(rebuilt)} of {len(report)}; dependency install re-ran: {pip_reran}")

# %% [markdown]
# With the bad ordering, a one-line edit re-runs the dependency install; with the good ordering, only the layers from `COPY src/` onward are rebuilt. Multiply by every commit, every CI build and every teammate, and the difference is hours of waiting per week (we do not quote a number: it depends on your dependencies, so measure your own build with `docker build --progress=plain`).
# And when `requirements.txt` itself changes? Everything from that point on is rebuilt, as it should be:

# %%
seen, ctx = set(), dict(CONTEXT)
simulate_build(GOOD, ctx, seen)
ctx["requirements.txt"] += "joblib==1.5.0\n"
for name, status in simulate_build(GOOD, ctx, seen):
    print(f"{status:8s} {name}")

# %% [markdown]
# ## 5. Keep the build context small: `.dockerignore`
# `docker build .` first sends the whole **context** (the folder) to the Docker daemon. If your project holds a virtualenv, `.git`, raw datasets and old model files, you ship gigabytes before the first instruction runs, and `COPY . .` bakes them into the image. A `.dockerignore` file lists what to leave out.
# We build a small synthetic project and measure the context with and without ignore rules (sizes here are created by the notebook, so read them as a demonstration of the mechanism, not of your project):

# %%
proj = TMP / "project"
rng = np.random.default_rng(42)
layout = {"src/churn_service.py": 2_000, "requirements.txt": 200, "model/model.joblib": 400_000, "data/raw/events.csv": 6_000_000,
          "mlruns/0/model_old1.pkl": 1_500_000, "mlruns/0/model_old2.pkl": 1_500_000, ".git/objects/pack/p.pack": 3_000_000,
          "notebooks/explore.ipynb": 900_000, ".venv/lib/site.bin": 8_000_000}
for rel, size in layout.items():
    f = proj / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(rng.bytes(size))

def context_size(root, ignore_patterns):
    """Bytes sent to the daemon. A pattern matches a path or any of its parent directories (simplified .dockerignore)."""
    total, n = 0, 0
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        prefixes = ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]
        if any(fnmatch.fnmatch(pref, pat.rstrip("/")) for pref in prefixes for pat in ignore_patterns):
            continue
        total += p.stat().st_size
        n += 1
    return total, n

DOCKERIGNORE = [".git", ".venv", "mlruns", "data", "notebooks", "*.pyc", "__pycache__"]
for label, pats in (("no .dockerignore", []), ("with .dockerignore", DOCKERIGNORE)):
    size, n = context_size(proj, pats)
    print(f"{label:20s} {n} files, {size / 1e6:6.2f} MB sent to the daemon")

# %% [markdown]
# Notice we ignored `data` and `mlruns` but **kept** `model/` (the one model the image needs). Prefer an allow-list mindset: the image should contain only what the service needs to run. Never let `.env` files or credentials into the context either.

# %% [markdown]
# ## 6. Several services: `docker-compose.yaml`
# A realistic local stack has more than the API: a tracking server, a database. **Compose** describes them in one YAML file and starts them together with `docker compose up`. We keep it as a string and validate it, since YAML mistakes are the usual failure.
# (It uses `build:` for our own services and the official `postgres:16` image. The password comes from the environment, never the file.)

# %%
COMPOSE = textwrap.dedent("""
    services:
      db:
        image: postgres:16
        environment:
          POSTGRES_USER: mlflow
          POSTGRES_PASSWORD: ${DB_PASSWORD}
          POSTGRES_DB: mlflow
        volumes:
          - dbdata:/var/lib/postgresql/data
        healthcheck:
          test: ["CMD-SHELL", "pg_isready -U mlflow"]
          interval: 5s
          timeout: 3s
          retries: 10

      mlflow:
        build: ./docker/mlflow
        command: >
          mlflow server --host 0.0.0.0 --port 5000
          --backend-store-uri postgresql://mlflow:${DB_PASSWORD}@db:5432/mlflow
          --default-artifact-root /artifacts
        ports:
          - "5000:5000"
        volumes:
          - artifacts:/artifacts
        depends_on:
          db:
            condition: service_healthy

      api:
        build: .
        ports:
          - "8000:8000"
        environment:
          MODEL_PATH: /app/model/model.joblib
          MLFLOW_TRACKING_URI: http://mlflow:5000
        depends_on:
          - mlflow
        restart: unless-stopped

    volumes:
      dbdata: {}
      artifacts: {}
""").lstrip()

compose = yaml.safe_load(COMPOSE)
print("services:", list(compose["services"]), "| volumes:", list(compose["volumes"]))

# %% [markdown]
# Compose encodes the things you would otherwise do by hand: **networking by service name** (`api` reaches the tracker at `http://mlflow:5000`, no IP addresses), **named volumes** so the database survives container restarts, **health-gated start-up** (`service_healthy` means "wait until Postgres actually accepts connections", which is different from "the container started"), and restart policies.
# A checker for the mistakes that bite:

# %%
def check_compose(c):
    problems = []
    services = c.get("services", {})
    for name, svc in services.items():
        if "image" not in svc and "build" not in svc:
            problems.append(f"{name}: needs either image or build")
        for port in svc.get("ports", []):
            if not isinstance(port, str):
                problems.append(f"{name}: port {port!r} parsed as {type(port).__name__}; quote port mappings like \"8000:8000\" (YAML reads 22:22 as a number)")
        deps = svc.get("depends_on", [])
        dep_items = deps.items() if isinstance(deps, dict) else [(d, {}) for d in deps]
        for dep, cfg in dep_items:
            if dep not in services:
                problems.append(f"{name}: depends_on unknown service '{dep}'")
            elif cfg.get("condition") == "service_healthy" and "healthcheck" not in services[dep]:
                problems.append(f"{name}: waits for {dep} to be healthy but {dep} defines no healthcheck")
        for v in svc.get("volumes", []):
            src = v.split(":")[0] if isinstance(v, str) else None
            if src and not src.startswith((".", "/", "~")) and src not in c.get("volumes", {}):
                problems.append(f"{name}: named volume '{src}' is not declared under top-level volumes")
        env = svc.get("environment", {})
        env_items = env.items() if isinstance(env, dict) else [e.split("=", 1) for e in env if "=" in e]
        for k, v in env_items:
            if SECRET_WORDS.search(str(k)) and v not in (None, "") and not str(v).startswith("${"):
                problems.append(f"{name}: environment {k} contains a literal secret; use ${{VAR}} from the environment or a secrets mechanism")
    return problems

print("our compose file:", check_compose(compose))
broken = yaml.safe_load("""
services:
  api:
    build: .
    ports: [22:22]
    depends_on:
      db: {condition: service_healthy}
    volumes: ["cache:/cache"]
    environment: {API_TOKEN: hunter2}
  db:
    image: postgres:16
""")
for p in check_compose(broken):
    print("-", p)

# %% [markdown]
# ## 7. The real commands (shown, not run)
# Everything below needs a Docker daemon; run it on your machine when you have Docker Desktop or Docker Engine installed.

# %% tags=["no-run"]
# docker build -t churn-service:0.1.0 .                     # build using the Dockerfile in this folder
# docker build --progress=plain -t churn-service:0.1.0 .    # see which layers were CACHED
# docker run --rm -p 8000:8000 churn-service:0.1.0          # run it; Ctrl+C stops it (exec-form CMD handles the signal)
# docker run --rm -e LOG_LEVEL=debug -v "$PWD/model:/app/model:ro" -p 8000:8000 churn-service:0.1.0   # mount the model read-only instead of baking it in
# docker ps; docker logs -f <container>; docker exec -it <container> sh    # inspect a running container
# docker images; docker history churn-service:0.1.0          # size per layer (this is where a leaked ENV would show up)
# docker system prune                                        # reclaim disk from unused images/containers
#
# docker compose up --build -d     # start the whole stack in the background
# docker compose ps; docker compose logs -f api
# docker compose down              # stop and remove containers (add -v to delete named volumes, which DELETES the data)
#
# hadolint Dockerfile              # a real Dockerfile linter; run it in CI

# %% [markdown]
# ## Common mistakes
#
# 1. **`FROM python:latest`** (or no tag). Builds stop being reproducible and images are huge. Pin a version, prefer `-slim`, pin the digest for strict reproducibility.
# 2. **`COPY . .` before installing dependencies.** Every code edit re-installs everything. Copy the requirements file first.
# 3. **Running as root.** Add a non-root `USER`.
# 4. **Secrets in `ENV`, `ARG`, or the build context.** They live in image layers forever. Inject at run time; add `.env` to `.dockerignore`.
# 5. **Shell-form `CMD`** so the app misses `SIGTERM` and is killed after a timeout instead of finishing in-flight requests.
# 6. **Baking big, frequently changing models into the code layer.** Order layers by change frequency, or mount the model / pull it from the registry at start-up.
# 7. **Forgetting `.dockerignore`.** Gigabytes of context, slow builds, leaked files.
# 8. **`apt-get update` in its own `RUN`.** A cached stale package index makes the next install fail weeks later. Update and install in one `RUN`, and clean the lists.
# 9. **Treating containers as pets.** Anything written inside a container disappears with it. Keep state in volumes or external services.
# 10. **Assuming the image is secure because it runs.** Scan images and update base images regularly.

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between an image and a container?**
# <details><summary>Answer</summary>
# An image is an immutable, layered template containing the filesystem and metadata; a container is a running (or stopped) instance of an image with a thin writable layer, its own process tree, network and filesystem view. Many containers can run from one image.
# </details>
#
# **2. A Docker build takes 10 minutes for a one-line code change. What do you check?**
# <details><summary>Answer</summary>
# Layer order and cache invalidation: is the dependency install placed after <code>COPY . .</code>, so any change busts it? Copy the requirements file first, install, then copy the source. Also check a missing <code>.dockerignore</code> (huge context), unpinned downloads that change every time, and whether CI preserves the layer cache.
# </details>
#
# **3. How do you handle secrets in containers?**
# <details><summary>Answer</summary>
# Never bake them into the image (ENV/ARG/COPY stay in layers and history). Provide them at run time through the orchestrator's secret mechanism, environment variables injected from a secret store, or mounted files, and keep the build context free of credential files. Use build secrets (BuildKit) when a build step needs one.
# </details>
#
# **4. Would you put the model inside the image or load it at start-up?**
# <details><summary>Answer</summary>
# Baking it in gives an immutable, self-contained release that is easy to roll back and needs no network at start-up; the cost is bigger images and a rebuild per model. Pulling from a registry or mounting a volume decouples model and code releases but adds a start-up dependency and needs version pinning and integrity checks. Many teams bake small models and fetch large ones, pinned by version or hash.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Catch leaked secrets
# Write `leaked_secrets(dockerfile_text)` returning a **sorted list of (line_number, variable_name)** for every `ENV` or `ARG` assignment whose variable name contains a secret word (`SECRET_WORDS`) **and** has a literal value (not empty, not a `$VAR` / `${VAR}` reference). An `ARG NAME` with no value, or `ENV TOKEN=$BUILD_TOKEN`, is not a leak. Handle several assignments on one `ENV` line.
# *Hint: `parse_dockerfile` gives you logical lines; `re.findall(r"(\w+)=(\S+)", args)` finds assignments.*

# %% tags=["starter"]
def leaked_secrets(dockerfile_text):
    ...

# %%
sample = textwrap.dedent("""
    FROM python:3.11-slim
    ARG BUILD_TOKEN
    ENV MODE=prod DB_PASSWORD=hunter2
    ENV API_KEY=$INJECTED
    ARG PIP_INDEX_PASSWORD=abc
    ENV GREETING=hello
""").lstrip()
assert leaked_secrets(sample) == [(3, "DB_PASSWORD"), (5, "PIP_INDEX_PASSWORD")], leaked_secrets(sample)
assert leaked_secrets(GOOD) == []
assert [v for _, v in leaked_secrets(BAD)] == ["API_TOKEN"]
print("Exercise 1 passed:", leaked_secrets(sample))

# %% tags=["solution"]
def leaked_secrets(dockerfile_text):
    found = []
    for i in parse_dockerfile(dockerfile_text):
        if i["op"] in ("ENV", "ARG"):
            for key, val in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)", i["args"]):
                if SECRET_WORDS.search(key) and not val.startswith("$"):
                    found.append((i["line"], key))
    return sorted(found)

# %% [markdown]
# ### Exercise 2: Write a Dockerfile that passes review
# Write `MY_DOCKERFILE` (a string) for a **training job** image that runs `python train.py`. Requirements: pinned base image tag (not latest), dependencies installed from `requirements.txt` with `--no-cache-dir`, a non-root user, exec-form `CMD`, and **layer order such that editing only `train.py` rebuilds only the `COPY train.py` layer and later layers, never the `pip install`**. The linter must report **no errors and no warnings** (an `info` about a missing `HEALTHCHECK` is fine for a batch job).
# Build context for the check: `requirements.txt`, `train.py`, `config.yaml`.

# %% tags=["starter"]
MY_DOCKERFILE = """
"""

# %%
findings = [f for f in lint_dockerfile(MY_DOCKERFILE) if f["severity"] in ("error", "warning")]
assert findings == [], f"the linter still reports: {findings}"
ctx = {"requirements.txt": "scikit-learn==1.9.1\n", "train.py": "print('v1')\n", "config.yaml": "C: 1.0\n"}
seen = set()
first = simulate_build(MY_DOCKERFILE, ctx, seen)
assert all(status == "REBUILT" for _, status in first)
ctx["train.py"] = "print('v2')\n"
second = simulate_build(MY_DOCKERFILE, ctx, seen)
cached = [name for name, st in second if st == "CACHED"]
rebuilt = [name for name, st in second if st == "REBUILT"]
assert any("pip install" in n for n in cached), "the dependency install must stay cached when only train.py changes"
assert any("train.py" in n for n in rebuilt), "the COPY of train.py must be rebuilt"
assert not any("pip install" in n for n in rebuilt)
assert any(i["op"] == "CMD" and i["args"].startswith("[") and "train.py" in i["args"] for i in parse_dockerfile(MY_DOCKERFILE)), "use exec-form CMD running train.py"
print("Exercise 2 passed:", len(cached), "layers cached,", len(rebuilt), "rebuilt")

# %% tags=["solution"]
MY_DOCKERFILE = """
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 trainer
COPY config.yaml .
COPY train.py .
USER trainer
CMD ["python", "train.py"]
"""

# %% [markdown]
# ### Exercise 3: Port clashes in compose
# Write `duplicate_host_ports(compose_dict)` returning the **sorted list of host ports** (as strings) that are published by more than one service in a compose dict. Port entries look like `"8000:8000"` or `"127.0.0.1:5432:5432"` (host ip, host port, container port) or `"9090"` (no host port, so ignore it).

# %% tags=["starter"]
def duplicate_host_ports(compose_dict):
    ...

# %%
assert duplicate_host_ports(compose) == [], "our stack publishes 5000 and 8000 only once"
clash = {"services": {"a": {"image": "x", "ports": ["8000:8000"]}, "b": {"image": "y", "ports": ["8000:9000", "127.0.0.1:5432:5432"]},
                      "c": {"image": "z", "ports": ["5432:5432", "9090"]}, "d": {"image": "w"}}}
assert duplicate_host_ports(clash) == ["5432", "8000"], duplicate_host_ports(clash)
print("Exercise 3 passed")

# %% tags=["solution"]
def duplicate_host_ports(compose_dict):
    from collections import Counter
    hosts = []
    for svc in compose_dict.get("services", {}).values():
        for port in svc.get("ports", []):
            parts = str(port).split(":")
            if len(parts) >= 2:                           # "host:container" or "ip:host:container"; a bare container port has no fixed host port
                hosts.append(parts[-2])
    return sorted(h for h, n in Counter(hosts).items() if n > 1)

# %% [markdown]
# ### Exercise 4 (open-ended): Review a Dockerfile you have
# Take a Dockerfile from your work (or the one in this repository's `docker/` folder if you have one). Run `lint_dockerfile` on its text, and write down which findings you agree with, which you would waive and why, and one risk the linter cannot see (for example a base image with known vulnerabilities, or a model file that is never verified).

# %% [markdown]
# ## Recap
# - An **image** is layered; a **container** is a running instance. The **build cache** reuses a layer only if the instruction, everything before it, and (for `COPY`) the file contents are unchanged, so **order layers from least to most frequently changing**: base, dependencies, model, code.
# - A production-minded Dockerfile pins the base tag, installs from a pinned requirements file with `--no-cache-dir`, runs as a **non-root user**, declares a **HEALTHCHECK**, uses **exec-form `CMD`**, and never embeds secrets.
# - Lint Dockerfiles and compose files automatically; YAML gotchas (unquoted `22:22`) and missing health checks are common, cheap to catch.
# - `.dockerignore` keeps the build context (and the image) small and keeps credentials out.
# - **Compose** runs multi-service stacks with service-name networking, named volumes and health-gated start-up.
# - The Docker commands themselves need a daemon; practise them locally with the files you wrote here.
#
# **Go deeper:** Docker documentation, "Dockerfile best practices" and "Build cache" (docs.docker.com); the Compose file reference (docs.docker.com/compose); the `hadolint` project (github.com/hadolint/hadolint) for a production Dockerfile linter.
#
# **Next:** [Monitoring and drift detection](07_monitoring_and_drift_detection.ipynb)
