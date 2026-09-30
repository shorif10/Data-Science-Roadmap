# %% [markdown]
# # Tensors and Autograd
#
# **Why this matters.** Every PyTorch bug report you will ever write starts with one of three things: a wrong
# **shape**, a wrong **dtype**, or a gradient that is not what you think it is. A model that "trains" but
# learns nothing is very often a silent broadcasting bug or a forgotten `zero_grad()`. This notebook gives you the
# working vocabulary (tensors, views, broadcasting, `requires_grad`, `backward`) and the habit of *verifying* gradients
# against an independent calculation, the same habit you built by hand in the backpropagation notebook.
#
# **You will learn**
# - create tensors, read their `shape` / `dtype`, and move between PyTorch and NumPy (and know when memory is shared)
# - predict the result shape of a broadcast and spot the silent `(n,) - (n, 1)` bug
# - compute gradients with `requires_grad` and `backward()`, and explain `no_grad`, `detach` and gradient accumulation
# - cross-check autograd against a hand-derived gradient and with `torch.autograd.gradcheck` in float64
#
# **Prerequisites:** [Backpropagation from scratch](../13_Deep_Learning/02_backpropagation_from_scratch.ipynb) and the
# [deep learning debugging checklist](../13_Deep_Learning/05_deep_learning_debugging_checklist.ipynb).
# NumPy arrays (module 04) are assumed.
#
# > No dataset here: we use tiny hand-made tensors and random data from a seeded generator. Everything runs on CPU.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.gradcheck import GradcheckError

torch.manual_seed(42)
torch.set_num_threads(2)
rng = np.random.default_rng(42)
print("PyTorch", torch.__version__)

# %% [markdown]
# ## 1. A tensor is an array with a dtype, a shape and (optionally) a gradient
#
# If you know NumPy, a `torch.Tensor` is an `ndarray` that can also remember how it was computed, so PyTorch can differentiate
# through it. The three attributes you check constantly: `shape`, `dtype`, `device` (CPU only in this course).

# %%
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", a.shape, " ndim:", a.ndim, " numel:", a.numel())
print("dtype:", a.dtype, " device:", a.device)
print("zeros :", torch.zeros(2, 2).tolist())
print("arange:", torch.arange(0, 10, 3).tolist(), " linspace:", torch.linspace(0, 1, 5).tolist())
print("random normal:", torch.randn(2, 3).shape, " (seeded, so repeatable)")

# %% [markdown]
# ## 2. dtypes: the default differs from NumPy
#
# | Source | Default dtype |
# |---|---|
# | `torch.tensor([1.0, 2.0])` | `float32` |
# | `torch.tensor([1, 2])` | `int64` |
# | `np.array([1.0, 2.0])` | `float64` |
#
# Neural networks use `float32` (half the memory, faster, precise enough for training). Mixing dtypes is the most common first error.

# %%
print(torch.tensor([1.0, 2.0]).dtype, torch.tensor([1, 2]).dtype, np.array([1.0, 2.0]).dtype)
x32 = torch.ones(2, 2)
x64 = torch.ones(2, 2, dtype=torch.float64)
try:
    x32 @ x64
except RuntimeError as e:
    print("RuntimeError:", str(e)[:80])
print("after .to(float32):", (x32 @ x64.to(torch.float32)).dtype)

# %% [markdown]
# *What just happened?* `@` refuses two different float dtypes. Cast explicitly with `.to(dtype)`. Note that operations
# between an integer and a float tensor *do* promote silently, so a wrong dtype does not always announce itself:

# %%
print((torch.tensor([1, 2]) + torch.tensor([0.5, 0.5])).dtype, " <- int64 + float32 promoted to float32")
print("int / int ->", (torch.tensor([3]) / torch.tensor([2])).item(), "(true division)   int // int ->", (torch.tensor([3]) // torch.tensor([2])).item())

# %% [markdown]
# ## 3. Shapes: reshape, view, transpose, unsqueeze
#
# Most PyTorch code is shape bookkeeping. The moves you need:
#
# - `t.reshape(2, -1)`: new shape (`-1` = infer). Returns a **view** (shared memory) when possible, otherwise a copy.
# - `t.T` / `t.transpose(0, 1)`: swap axes. The result is a view and is usually not contiguous in memory.
# - `t.unsqueeze(k)` / `t.squeeze(k)`: add / remove a length-1 axis.
# - `a @ b`: matrix multiply; `(n, k) @ (k, m) -> (n, m)`.

# %%
t = torch.arange(6)
m = t.reshape(2, 3)
m[0, 0] = 100                      # write through the view...
print("original changed too:", t.tolist())
print("unsqueeze(1):", t.unsqueeze(1).shape, "  unsqueeze(0):", t.unsqueeze(0).shape)
W = torch.randn(3, 4)
print("(2,3) @ (3,4) ->", (m.float() @ W).shape)

# %% [markdown]
# ## 4. Broadcasting, and the bug it hides
#
# Broadcasting aligns shapes from the **right**; two dimensions are compatible if they are equal or one of them is 1, and the 1 is
# stretched. It is great for "subtract the column means" and dangerous when you did not mean it.

# %%
X = torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 9.0]])            # (2, 3)
col_mean = X.mean(dim=0, keepdim=True)                            # (1, 3)
print("centred columns:", (X - col_mean).mean(dim=0).tolist(), "  shapes:", X.shape, col_mean.shape)

# %% [markdown]
# Now the classic silent bug: predictions of shape `(n,)` and targets of shape `(n, 1)`. The subtraction does not fail; it
# builds an `(n, n)` table of all pairs.

# %%
pred = torch.tensor([1.0, 2.0, 3.0])            # (3,)
target = torch.tensor([[1.0], [2.0], [3.0]])    # (3, 1)   <- a perfect model should give loss 0
print("(pred - target).shape:", (pred - target).shape)
print("buggy MSE :", ((pred - target) ** 2).mean().item())
print("fixed MSE :", ((pred - target.squeeze(1)) ** 2).mean().item())

# %% [markdown]
# A perfect model reports a non-zero loss, and training would quietly optimise the wrong objective. **Print shapes, and assert them**
# at the boundary between your data code and your model code.

# %% [markdown]
# ## 5. NumPy interop and shared memory
#
# `torch.from_numpy(arr)` and `tensor.numpy()` share memory with the original (no copy). `torch.tensor(arr)` copies.

# %%
arr = np.arange(3, dtype=np.float64)
shared = torch.from_numpy(arr)
copied = torch.tensor(arr)
arr[0] = 99.0
print("from_numpy sees the change:", shared[0].item(), "  torch.tensor does not:", copied[0].item())
print("from_numpy kept dtype:", shared.dtype, "(float64, not the float32 the network will want)")
print("tensor -> numpy:", torch.tensor([1.5, 2.5]).numpy())

# %% [markdown]
# ## 6. Autograd: gradients for free
#
# Mark a tensor with `requires_grad=True` and PyTorch records every operation on it in a graph. `loss.backward()` walks that
# graph backwards (exactly the chain rule you coded by hand) and stores the result in `.grad`.
#
# Here is the single neuron from the backpropagation notebook: `z = w x + b`, `a = sigmoid(z)`, `loss = (a - t)^2`,
# with analytic gradient `2(a - t) * a(1 - a) * x`.

# %%
x_in, t_out = 1.5, 1.0
w = torch.tensor(0.4, requires_grad=True)
b = torch.tensor(-0.2, requires_grad=True)

z = w * x_in + b
a = torch.sigmoid(z)
loss = (a - t_out) ** 2
loss.backward()

a_val = a.item()
analytic_dw = 2 * (a_val - t_out) * a_val * (1 - a_val) * x_in
print(f"autograd  dL/dw = {w.grad.item():.6f}")
print(f"analytic  dL/dw = {analytic_dw:.6f}")
print("loss.grad_fn:", loss.grad_fn.name(), " | w.is_leaf:", w.is_leaf, " z.is_leaf:", z.is_leaf)

# %% [markdown]
# *What just happened?* Autograd reproduced our hand-derived number. `w` and `b` are **leaf** tensors (created by you, they receive `.grad`);
# `z`, `a` and `loss` are intermediate results that remember their `grad_fn`. Only leaves that require grad keep a `.grad` by default.
#
# `backward()` needs a **scalar**; for a vector output you must reduce it (`.sum()`, `.mean()`) or pass a `gradient` argument.

# %% [markdown]
# ## 7. `no_grad` and `detach`: stepping outside the graph
#
# - `with torch.no_grad():` runs code without recording a graph. Use it for evaluation and for parameter updates.
# - `t.detach()` returns a tensor that shares data but is cut off from the graph. Use it before converting to NumPy or logging.

# %%
p = torch.tensor([1.0, 2.0], requires_grad=True)
with torch.no_grad():
    q = p * 2
print("inside no_grad -> requires_grad:", q.requires_grad)
r = p * 2
print("normal op      -> requires_grad:", r.requires_grad, " | r.detach():", r.detach().requires_grad)
try:
    r.numpy()
except RuntimeError as e:
    print("RuntimeError:", str(e)[:70])
print("detach().numpy():", r.detach().numpy())

# %% [markdown]
# ## 8. The gradient accumulation gotcha
#
# `backward()` **adds** to `.grad`; it does not overwrite it. That is deliberate (it lets you sum gradients over several
# mini-batches), but if you forget to clear gradients between training steps, every step uses the sum of all previous gradients.

# %%
w_acc = torch.tensor(2.0, requires_grad=True)
for step in range(3):
    (w_acc * 3).backward()                  # d/dw (3w) = 3 every time
    print(f"after backward #{step + 1}: w.grad = {w_acc.grad.item()}")

opt = torch.optim.SGD([w_acc], lr=0.1)
opt.zero_grad()                              # the standard way to clear
print("after zero_grad: w.grad =", w_acc.grad, "(PyTorch 2 sets it to None by default)")

# %% [markdown]
# Three correct gradients of 3 became 3, 6, 9. In a real loop the loss still goes down for a while, so nothing crashes, which is why this bug survives code review.
#
# ## 9. Cross-checking autograd on a real computation
#
# Logistic regression in tensors: `p = sigmoid(X w + b)`, mean binary cross-entropy. The hand-derived gradient (from the
# backprop notebook) is `dw = X^T (p - y) / n`, `db = mean(p - y)`. We compare it with autograd on random data in float64.

# %%
n, d = 60, 4
Xn = rng.normal(size=(n, d))
yn = (Xn[:, 0] - 0.5 * Xn[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.float64)
Xt, yt = torch.from_numpy(Xn), torch.from_numpy(yn)            # float64, shares memory

wt = torch.zeros(d, dtype=torch.float64, requires_grad=True)
bt = torch.zeros((), dtype=torch.float64, requires_grad=True)
logits = Xt @ wt + bt
loss_t = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
loss_t.backward()

p_np = 1.0 / (1.0 + np.exp(-(Xn @ wt.detach().numpy() + bt.item())))
dw_manual = Xn.T @ (p_np - yn) / n
db_manual = np.mean(p_np - yn)
print("max |autograd dw - manual dw|:", np.abs(wt.grad.numpy() - dw_manual).max())
print("|autograd db - manual db|    :", abs(bt.grad.item() - db_manual))

# %% [markdown]
# ## 10. `gradcheck`: let PyTorch verify a gradient for you
#
# `torch.autograd.gradcheck(fn, inputs)` compares autograd's gradient against finite differences (the numerical check you wrote earlier).
# It needs **float64** inputs that require grad. This matters most for custom `autograd.Function`s, where *you* write `backward`.

# %%
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * 2 * x


class BuggySquare(Square):
    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * x                      # BUG: forgot the factor 2


xg = torch.randn(5, dtype=torch.float64, requires_grad=True)
print("Square gradcheck      :", torch.autograd.gradcheck(Square.apply, (xg,)))
try:
    torch.autograd.gradcheck(BuggySquare.apply, (xg,))
except GradcheckError as e:
    print("BuggySquare           : GradcheckError ->", str(e).splitlines()[0][:70])

# %% [markdown]
# `gradcheck` passes the correct `backward` and raises on the buggy one. If you feed it `float32` inputs the finite differences are too noisy
# and PyTorch warns you; always use `dtype=torch.float64` for the check and keep `float32` for training.

# %% [markdown]
# ## 11. Putting it together: gradient descent by hand with autograd
#
# Fit `y = slope * x + intercept` to noisy data. The update `param -= lr * param.grad` must happen under `no_grad` (we do not want the update itself in the graph),
# and gradients must be cleared each step. We compare with the closed-form least-squares answer from NumPy.

# %%
xs = rng.uniform(-2, 2, size=100)
ys = 3.0 * xs + 1.0 + 0.3 * rng.normal(size=100)
xs_t, ys_t = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

slope = torch.zeros((), requires_grad=True)
icpt = torch.zeros((), requires_grad=True)
losses = []
for step in range(150):
    pred = slope * xs_t + icpt
    loss = ((pred - ys_t) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        slope -= 0.1 * slope.grad
        icpt -= 0.1 * icpt.grad
    slope.grad = None
    icpt.grad = None
    losses.append(loss.item())

ls_slope, ls_icpt = np.polyfit(xs, ys, 1)
print(f"gradient descent: slope={slope.item():.4f}  intercept={icpt.item():.4f}")
print(f"least squares   : slope={ls_slope:.4f}  intercept={ls_icpt:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(losses)
ax[0].set_xlabel("step")
ax[0].set_ylabel("mean squared error")
ax[0].set_title("Loss during manual gradient descent")
ax[1].scatter(xs, ys, s=10, alpha=0.6, label="data")
grid = np.linspace(-2, 2, 2)
ax[1].plot(grid, slope.item() * grid + icpt.item(), color="C3", label="fitted line")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Fit found by autograd")
ax[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* Autograd plus a two-line update rule is all of "training". In the next notebook `nn.Module` and an optimizer replace the
# hand-written parameters and update; the loop structure stays identical.

# %% [markdown]
# ## Common mistakes
#
# 1. **Not clearing gradients** between steps: they accumulate (section 8). Use `optimizer.zero_grad()` every iteration.
# 2. **Updating parameters without `no_grad`:** an in-place change to a leaf that requires grad raises an error (shown below); the fix is `with torch.no_grad():`.
# 3. **`.numpy()` on a tensor with a graph:** call `.detach().numpy()` (shown in section 7).
# 4. **Mismatched shapes that broadcast:** `(n,)` vs `(n, 1)` gives an `(n, n)` loss (section 4). Assert shapes.
# 5. **Logging `loss` instead of `loss.item()`:** appending the tensor to a list keeps every iteration's graph alive and memory grows.
# 6. **Wrong dtype:** `float64` from NumPy into a `float32` model, or integer class labels in a float loss. Cast at the data boundary.

# %%
leaf = torch.tensor([1.0, 2.0], requires_grad=True)
try:
    leaf -= 0.1                                   # in-place op on a leaf that requires grad
except RuntimeError as e:
    print("RuntimeError:", str(e)[:75])
try:
    torch.tensor([1, 2, 3], requires_grad=True)   # integers have no gradient
except RuntimeError as e:
    print("RuntimeError:", str(e)[:75])

# %% [markdown]
# ## Interview corner
#
# **1. What does `loss.backward()` do, and what does `optimizer.step()` do?**
# <details><summary>Answer</summary>
# <code>backward()</code> computes the gradient of the (scalar) loss with respect to every leaf tensor with <code>requires_grad=True</code> and <em>adds</em> it to <code>.grad</code>.
# <code>step()</code> reads those <code>.grad</code> values and updates the parameters. They are separate so you can, for example, accumulate gradients over several mini-batches or clip them before stepping.
# </details>
#
# **2. Why must you call `zero_grad()` every iteration?**
# <details><summary>Answer</summary>
# Because gradients accumulate in <code>.grad</code>. Without clearing, each update uses the sum of all previous gradients. The same accumulation is used deliberately for gradient accumulation
# (several small batches to mimic a large one), where you call <code>step()</code> and <code>zero_grad()</code> only every k batches.
# </details>
#
# **3. `torch.no_grad()` vs `detach()`: what is the difference?**
# <details><summary>Answer</summary>
# <code>no_grad()</code> is a context that stops recording operations inside it (evaluation, parameter updates, saving memory). <code>detach()</code> acts on one existing tensor, returning a view of its data
# that is cut from the graph. Both avoid building or keeping graphs; <code>detach</code> is what you use before <code>.numpy()</code> or when you want a value treated as a constant.
# </details>
#
# **4. How would you test a custom autograd function or loss?**
# <details><summary>Answer</summary>
# Use <code>torch.autograd.gradcheck</code> with small float64 inputs that require grad: it compares your <code>backward</code> with finite differences. Also compare with a hand-derived gradient on a tiny example,
# and assert output shapes.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Standardise columns with broadcasting
# Write `standardize(t)` for a 2-D float tensor: subtract each **column** mean and divide by each column **standard deviation** (use the population
# std, `unbiased=False`), without a Python loop. The result must keep shape `(rows, cols)`.
# *Hint: `keepdim=True` keeps the reduced axis so broadcasting lines up.*

# %% tags=["starter"]
def standardize(t):
    ...

# %%
rng_e1 = np.random.default_rng(7)
raw = rng_e1.normal(loc=[10, -3, 50], scale=[2, 0.5, 20], size=(200, 3)).astype(np.float32)
out = standardize(torch.from_numpy(raw))
assert out.shape == (200, 3), f"shape should stay (200, 3), got {tuple(out.shape)}"
expected = (raw - raw.mean(axis=0)) / raw.std(axis=0)
assert np.allclose(out.numpy(), expected, atol=1e-4), "compare with (raw - mean) / std computed per column"
assert np.allclose(out.mean(dim=0).numpy(), 0, atol=1e-4), "each column should have mean 0"
print("Exercise 1 passed")

# %% tags=["solution"]
def standardize(t):
    mean = t.mean(dim=0, keepdim=True)
    std = t.std(dim=0, unbiased=False, keepdim=True)
    return (t - mean) / std

# %% [markdown]
# ### Exercise 2: The gradient of a linear model's MSE
# For `X2` (50 x 3), `y2` (50,) and weights `w0` (3,) below, compute `loss = mean((X2 @ w - y2)^2)` with `w` a tensor that requires grad, call `backward()`,
# and store the gradient as a NumPy array in `w_grad`. Also store the loss value as a Python float in `loss_value`.
# *Hint: create `w` from `w0` with `torch.tensor(w0, requires_grad=True)`; the analytic answer is `2/n * X^T (Xw - y)`.*

# %% tags=["starter"]
rng_e2 = np.random.default_rng(11)
X2 = torch.tensor(rng_e2.normal(size=(50, 3)))                       # float64
y2 = torch.tensor(rng_e2.normal(size=50))
w0 = np.array([0.5, -1.0, 2.0])

w_grad = ...
loss_value = ...

# %%
Xn2, yn2 = X2.numpy(), y2.numpy()
resid = Xn2 @ w0 - yn2
assert abs(loss_value - np.mean(resid ** 2)) < 1e-9, "loss_value should be the mean squared residual"
assert isinstance(w_grad, np.ndarray) and w_grad.shape == (3,), "w_grad should be a NumPy array of shape (3,)"
assert np.allclose(w_grad, 2 / len(yn2) * Xn2.T @ resid), "compare with 2/n * X.T @ (X w - y)"
print("Exercise 2 passed")

# %% tags=["solution"]
rng_e2 = np.random.default_rng(11)
X2 = torch.tensor(rng_e2.normal(size=(50, 3)))
y2 = torch.tensor(rng_e2.normal(size=50))
w0 = np.array([0.5, -1.0, 2.0])

w = torch.tensor(w0, requires_grad=True)
loss = ((X2 @ w - y2) ** 2).mean()
loss.backward()
w_grad = w.grad.numpy()
loss_value = loss.item()

# %% [markdown]
# ### Exercise 3: A correct SGD step
# Write `sgd_step(params, lr)` that, for every tensor in `params`, subtracts `lr * grad` **without recording the update in the graph** and then clears the
# gradient (set `.grad` to `None`). Then use it to fit `y = 2x - 1` for 200 steps; store the learned values in `fit_slope` and `fit_icpt` (Python floats).
# *Hint: `with torch.no_grad():` and loop over `params`; build the fit loop like section 11.*

# %% tags=["starter"]
def sgd_step(params, lr):
    ...


fit_slope = ...
fit_icpt = ...

# %%
p_a = torch.tensor([1.0, 2.0], requires_grad=True)
(p_a ** 2).sum().backward()                      # grad = 2 * p
before = p_a.detach().clone()
sgd_step([p_a], 0.1)
assert torch.allclose(p_a.detach(), before - 0.1 * 2 * before), "update should be p - lr * grad"
assert p_a.grad is None, "gradients must be cleared after the step"
assert p_a.requires_grad and p_a.is_leaf, "the parameter must still be a trainable leaf (update under no_grad)"
assert abs(fit_slope - 2.0) < 0.05 and abs(fit_icpt + 1.0) < 0.05, f"expected roughly slope 2 and intercept -1, got {fit_slope:.3f}, {fit_icpt:.3f}"
print(f"Exercise 3 passed (slope {fit_slope:.3f}, intercept {fit_icpt:.3f})")

# %% tags=["solution"]
def sgd_step(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad = None


xe = torch.linspace(-1, 1, 80)
ye = 2.0 * xe - 1.0
s = torch.zeros((), requires_grad=True)
c = torch.zeros((), requires_grad=True)
for _ in range(200):
    loss = ((s * xe + c - ye) ** 2).mean()
    loss.backward()
    sgd_step([s, c], 0.3)
fit_slope, fit_icpt = s.item(), c.item()

# %% [markdown]
# ### Exercise 4 (open-ended): Measure float32 vs float64
# Multiply two random `1000 x 1000` matrices in `float32` and in `float64` and time each (`time.perf_counter`, a few repeats). Also compare the memory each
# tensor needs (`t.element_size() * t.numel()`). Write two sentences: which is faster on your machine, by roughly how much, and why do deep learning frameworks default to `float32`?
# No assert: measure and report.

# %% [markdown]
# ## Recap
# - A tensor has `shape`, `dtype`, `device`; the default float is `float32` (NumPy's is `float64`). `from_numpy` and `.numpy()` share memory; `torch.tensor` copies.
# - Broadcasting aligns shapes from the right. Use `keepdim=True`, and watch for `(n,)` vs `(n, 1)` silently producing `(n, n)`.
# - `requires_grad` + `backward()` fills `.grad` on leaf tensors; gradients **accumulate**, so clear them every step.
# - Use `no_grad` for updates and evaluation, `detach` before NumPy.
# - Verify gradients: against a hand derivation, and with `gradcheck` in float64.
#
# **Go deeper:** the PyTorch docs pages on tensors, broadcasting semantics and "Autograd mechanics" (pytorch.org/docs), and the
# "Introduction to PyTorch" tutorials (pytorch.org/tutorials).
#
# **Next:** [nn.Module and the training loop](02_nn_module_and_training_loop.ipynb)
