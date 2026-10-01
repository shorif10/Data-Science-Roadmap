# %% [markdown]
# # Calculus for ML: Derivatives, Gradients and the Chain Rule
#
# **Why this matters.** Training a model means "nudge the parameters so the error goes down". A **derivative** tells you
# which way to nudge and how sensitive the error is to each parameter. Every deep learning framework you will use
# (PyTorch, Keras) is an automatic derivative machine built on one rule: the chain rule. When training
# blows up or a gradient is `NaN`, you debug with exactly the ideas below.
#
# **You will learn**
# - read a derivative as a slope, and compute it numerically and symbolically (SymPy)
# - see why the step size `h` of a numerical derivative has a sweet spot
# - use partial derivatives and the gradient vector for functions of many inputs
# - apply the chain rule to a tiny "neuron" and verify it with *gradient checking* (the seed of backprop)
#
# **Prerequisites:** [Vectors and matrices](01_vectors_and_matrices.ipynb).
#
# > No dataset here: everything is a small function you can plot. The next notebook uses `housing.csv`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 1. The derivative is a slope
#
# Take `f(x) = x**2`. Between `x` and `x + h`, the function rises by `f(x+h) - f(x)`. The slope of that secant line is
# `(f(x+h) - f(x)) / h`. Shrink `h` towards 0 and the secant becomes the tangent: that limit is the **derivative** `f'(x)`.
# Positive slope: going right increases `f`. Zero slope: a flat spot, maybe a minimum. That last fact is why we care.

# %%
def f(x):
    return x**2

x0 = 1.5
for h in [1.0, 0.1, 0.01, 0.001]:
    slope = (f(x0 + h) - f(x0)) / h
    print(f"h={h:<6} secant slope = {slope:.4f}")
print("exact derivative of x^2 is 2x =", 2 * x0)

# %% [markdown]
# The slopes approach 3.0 as `h` shrinks. Plot the tangent line to see what the number means:

# %%
xs = np.linspace(-0.5, 3.5, 200)
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(xs, f(xs), label="f(x) = x^2")
ax.plot(xs, f(x0) + 2 * x0 * (xs - x0), "--", label=f"tangent at x={x0} (slope {2*x0})")
ax.plot([x0], [f(x0)], "ko")
ax.set_ylim(-1, 8); ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("The derivative is the slope of the tangent")
ax.grid(alpha=0.3); ax.legend()
plt.show()

# %% [markdown]
# ## 2. Numerical derivatives: how small should `h` be?
#
# The forward difference above has error proportional to `h`. The **central difference** `(f(x+h) - f(x-h)) / (2h)`
# has error proportional to `h**2`, so it is far more accurate for the same `h`. But making `h` tiny is not free: computers
# store about 16 digits, and `f(x+h) - f(x-h)` subtracts two nearly equal numbers, destroying precision.

# %%
def num_deriv(func, x, h=1e-5):
    """Central difference approximation of func'(x)."""
    return (func(x + h) - func(x - h)) / (2 * h)

g = np.exp                        # derivative of e^x is e^x, so we know the truth
x0 = 1.0
truth = np.exp(x0)
print(f"{'h':>8} {'forward err':>13} {'central err':>13}")
for h in [1e-1, 1e-3, 1e-5, 1e-8, 1e-11, 1e-14]:
    fwd = (g(x0 + h) - g(x0)) / h
    ctr = (g(x0 + h) - g(x0 - h)) / (2 * h)
    print(f"{h:8.0e} {abs(fwd - truth):13.2e} {abs(ctr - truth):13.2e}")

# %% [markdown]
# *What just happened?* Errors shrink as `h` shrinks, hit a floor, then grow again from floating-point round-off. Central differences
# reach a much lower floor near `h ~ 1e-5`. Rule of thumb: use central differences with `h` around 1e-5 to 1e-6 for double precision.
# Numerical derivatives are slow (two function calls *per parameter*), so we use them to **check** correct formulas, not to train.

# %% [markdown]
# ## 3. Exact derivatives with SymPy
#
# SymPy manipulates formulas as objects, so it can differentiate them exactly. Useful for deriving a gradient you then code in NumPy.

# %%
x = sp.symbols("x")
expr = x**3 * sp.sin(x)
d_expr = sp.diff(expr, x)
print("f(x)  =", expr)
print("f'(x) =", d_expr)

f_prime = sp.lambdify(x, d_expr, "numpy")       # turn the formula into a fast numeric function
print("f'(2) exact via sympy:", f_prime(2.0))
print("f'(2) via central diff:", num_deriv(sp.lambdify(x, expr, "numpy"), 2.0))

# %% [markdown]
# A short table worth memorising, because these show up in every loss function and activation:
#
# | f(x) | f'(x) | where you meet it |
# |---|---|---|
# | `x**n` | `n*x**(n-1)` | squared error |
# | `exp(x)` | `exp(x)` | softmax, sigmoid |
# | `log(x)` | `1/x` | log-loss, entropy |
# | `sigmoid(x) = 1/(1+exp(-x))` | `sigmoid(x)*(1-sigmoid(x))` | logistic regression |
# | `tanh(x)` | `1 - tanh(x)**2` | RNNs, older nets |
# | `relu(x) = max(0,x)` | `1` if x>0 else `0` | most modern nets |

# %%
sig = 1 / (1 + sp.exp(-x))
print("sympy d/dx sigmoid, simplified:", sp.simplify(sp.diff(sig, x)))
xv = 0.7
s = 1 / (1 + np.exp(-xv))
print("s*(1-s) =", s * (1 - s), " sympy value:", float(sp.diff(sig, x).subs(x, xv)))

# %% [markdown]
# ## 4. Many inputs: partial derivatives and the gradient
#
# A loss depends on many parameters. The **partial derivative** `df/dx` treats every other variable as a constant and asks: how
# does `f` change if I wiggle only `x`? Stack the partials into a vector and you get the **gradient**
# `grad f = [df/dx, df/dy, ...]`. Its key property: it points in the direction of **steepest increase**, and its length is the
# steepness. To *minimise* a loss, step against it.

# %%
xs_, ys_ = sp.symbols("x y")
fxy = xs_**2 + 3 * ys_**2 + xs_ * ys_
grad_sym = [sp.diff(fxy, v) for v in (xs_, ys_)]
print("df/dx =", grad_sym[0], "   df/dy =", grad_sym[1])

f_num = sp.lambdify((xs_, ys_), fxy, "numpy")
g_num = sp.lambdify((xs_, ys_), grad_sym, "numpy")

gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
qx, qy = np.meshgrid(np.linspace(-2.5, 2.5, 9), np.linspace(-2.5, 2.5, 9))
gu, gv = g_num(qx, qy)

fig, ax = plt.subplots(figsize=(5.5, 5))
cs = ax.contour(gx, gy, f_num(gx, gy), levels=15, cmap="viridis")
ax.quiver(qx, qy, gu, gv, color="tab:red", alpha=0.7)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Gradient arrows are perpendicular to contours and point uphill")
plt.show()

# %% [markdown]
# The contours are lines of equal `f`; the red gradient arrows cross them at right angles, pointing uphill and getting longer
# where the surface is steeper. Walking *opposite* the arrows takes you down: that is gradient descent, the topic of the next notebook.

# %% [markdown]
# ## 5. The chain rule: derivatives of compositions
#
# If `y = g(u)` and `u = h(x)`, then `dy/dx = dy/du * du/dx`. "Multiply the local slopes along the path." A neural network is a long
# chain of simple functions, so its gradient is a long product of simple derivatives. Let us do one neuron by hand:
#
# ```
# z = w*x + b          (linear)
# a = tanh(z)          (activation)
# L = (a - y)^2        (squared error)
# ```
#
# Then `dL/dw = dL/da * da/dz * dz/dw = 2(a-y) * (1 - a^2) * x` and `dL/db = 2(a-y) * (1 - a^2) * 1`.

# %%
def forward(w, b, x, y):
    z = w * x + b
    a = np.tanh(z)
    return (a - y) ** 2, a

def backward(w, b, x, y):
    _, a = forward(w, b, x, y)
    dL_da = 2 * (a - y)
    da_dz = 1 - a**2
    dL_dz = dL_da * da_dz          # the running product along the chain
    return dL_dz * x, dL_dz * 1.0  # dz/dw = x, dz/db = 1

w, b, x_in, y_true = 0.8, -0.2, 1.5, 0.5
dw, db = backward(w, b, x_in, y_true)
print(f"chain rule    : dL/dw = {dw:.6f}, dL/db = {db:.6f}")

# %% [markdown]
# ### Gradient checking: trust, but verify
# Compare against central differences on the loss. If both agree to several digits, your derivation and code are right.
# This is how people debug hand-written backprop, and how you unit-test a custom layer.

# %%
loss = lambda w_, b_: forward(w_, b_, x_in, y_true)[0]
dw_num = (loss(w + 1e-6, b) - loss(w - 1e-6, b)) / 2e-6
db_num = (loss(w, b + 1e-6) - loss(w, b - 1e-6)) / 2e-6
print(f"numerical     : dL/dw = {dw_num:.6f}, dL/db = {db_num:.6f}")
print("agree:", np.allclose([dw, db], [dw_num, db_num], atol=1e-6))

# %% [markdown]
# *What just happened?* We computed the forward pass storing intermediate values (`a`), then walked *backwards*
# multiplying local derivatives. That is backpropagation for one neuron; for a real network the only differences are more
# nodes and matrices instead of scalars. PyTorch's `autograd` does the same bookkeeping automatically:

# %%
import torch

wt = torch.tensor(0.8, requires_grad=True)
bt = torch.tensor(-0.2, requires_grad=True)
Lt = (torch.tanh(wt * 1.5 + bt) - 0.5) ** 2
Lt.backward()
print(f"autograd      : dL/dw = {wt.grad.item():.6f}, dL/db = {bt.grad.item():.6f}")

# %% [markdown]
# ## 6. Preview: the gradient of the linear-regression loss
#
# For `L(w) = mean((X w - y)^2)` the gradient has a compact form: `grad = (2/n) * X.T @ (X w - y)`. Check it against numerical
# derivatives on random data, because we will rely on it in the next notebook.

# %%
rng = np.random.default_rng(42)
Xr = rng.normal(size=(50, 3)); yr = rng.normal(size=50); wr = rng.normal(size=3)

mse = lambda w_: np.mean((Xr @ w_ - yr) ** 2)
analytic = 2 / len(yr) * Xr.T @ (Xr @ wr - yr)
numeric = np.array([(mse(wr + e) - mse(wr - e)) / 2e-6 for e in np.eye(3) * 1e-6])
print("analytic:", analytic)
print("numeric :", numeric)
print("match:", np.allclose(analytic, numeric, atol=1e-6))

# %% [markdown]
# ## Common mistakes
#
# 1. **A tiny `h` for numerical derivatives** (`1e-12`): round-off error dominates. Use about `1e-5` with central differences.
# 2. **Stepping along the gradient instead of against it** when minimising. Gradient = uphill.
# 3. **Forgetting a factor in the chain.** The most common backprop bug. Gradient-check every custom formula.
# 4. **Trusting derivatives at kinks.** `abs(x)` and `relu(x)` have no derivative at 0; frameworks pick a convention (0 or 1).

# %%
h = 1e-13
print("central diff of exp at 1 with h=1e-13:", (np.exp(1 + h) - np.exp(1 - h)) / (2 * h), " truth:", np.exp(1.0))

# %% [markdown]
# ## Interview corner
#
# **1. What does the gradient tell you, geometrically?**
# <details><summary>Answer</summary>
# It is the vector of partial derivatives; it points in the direction of steepest ascent and its norm is the rate of increase in that direction.
# Gradient descent steps in the opposite direction.
# </details>
#
# **2. Explain backpropagation in two sentences.**
# <details><summary>Answer</summary>
# Run the network forward and cache intermediate values; then apply the chain rule from the loss backwards, reusing each layer's upstream gradient
# to get the gradient with respect to its inputs and parameters. It computes all parameter gradients in roughly the cost of one extra forward pass.
# </details>
#
# **3. Symbolic vs numerical vs automatic differentiation?**
# <details><summary>Answer</summary>
# Symbolic (SymPy) manipulates formulas and can blow up in size; numerical (finite differences) is approximate and costs a function call per parameter, good for checking;
# automatic differentiation records elementary operations and applies the chain rule exactly (to machine precision) at low cost, which is what PyTorch/TensorFlow do.
# </details>
#
# **4. What is the vanishing gradient problem?**
# <details><summary>Answer</summary>
# Multiplying many local derivatives smaller than 1 (the sigmoid slope is at most 0.25; tanh at most 1 but below 1 almost everywhere) shrinks the gradient towards zero in deep chains, so early layers barely learn.
# ReLU-type activations, residual connections and careful initialisation mitigate it.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Write a reusable gradient checker
# Write `num_grad(func, point, h=1e-6)` that takes a function of a NumPy vector and returns the central-difference gradient as an array
# the same length as `point`. *Hint: loop over coordinates, perturbing one at a time with `np.eye(len(point))[i] * h`.*

# %% tags=["starter"]
def num_grad(func, point, h=1e-6):
    ...

# %%
fq = lambda p: p[0] ** 2 * p[1] + np.sin(p[1])
pt = np.array([1.0, 2.0])
exact = np.array([2 * pt[0] * pt[1], pt[0] ** 2 + np.cos(pt[1])])
got = num_grad(fq, pt)
assert isinstance(got, np.ndarray) and got.shape == (2,), "return a NumPy array of shape (2,)"
assert np.allclose(got, exact, atol=1e-6), f"expected about {exact}, got {got}"
assert np.allclose(num_grad(lambda p: np.sum(p**2), np.array([1.0, -2.0, 3.0])), [2, -4, 6], atol=1e-6)
print("Exercise 1 passed:", got)

# %% tags=["solution"]
def num_grad(func, point, h=1e-6):
    point = np.asarray(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(len(point)):
        step = np.zeros_like(point)
        step[i] = h
        grad[i] = (func(point + step) - func(point - step)) / (2 * h)
    return grad

# %% [markdown]
# ### Exercise 2: Derive with SymPy
# Let `L(w) = log(1 + exp(-y * w * x))` (the logistic loss for one sample, label `y` in {-1, +1}). Using SymPy, differentiate with respect to `w`
# and store the numeric value at `w=0.5, x=2.0, y=1.0` in `dL_dw`. Then store in `matches_formula` whether it equals `-y*x*sigmoid(-y*w*x)`.
# *Hint: `sp.symbols("w x y")`, `sp.diff`, `.subs({...})`, `float(...)`.*

# %% tags=["starter"]
dL_dw = ...
matches_formula = ...

# %%
sig_ = lambda t: 1 / (1 + np.exp(-t))
w_, x_, y_ = 0.5, 2.0, 1.0
formula = -y_ * x_ * sig_(-y_ * w_ * x_)
assert abs(dL_dw - formula) < 1e-9, f"expected {formula:.6f}, got {dL_dw}"
assert matches_formula is True or matches_formula == True, "the SymPy derivative should match -y*x*sigmoid(-y*w*x)"
print("Exercise 2 passed:", round(dL_dw, 6))

# %% tags=["solution"]
w_s, x_s, y_s = sp.symbols("w x y")
L_s = sp.log(1 + sp.exp(-y_s * w_s * x_s))
d_s = sp.diff(L_s, w_s)
dL_dw = float(d_s.subs({w_s: 0.5, x_s: 2.0, y_s: 1.0}))
matches_formula = bool(abs(dL_dw - (-1.0 * 2.0 / (1 + np.exp(-(-1.0 * 0.5 * 2.0))))) < 1e-9)

# %% [markdown]
# ### Exercise 3: Backprop a two-parameter neuron
# For the neuron `a = sigmoid(w*x + b)` with loss `L = (a - y)^2`, write `grads(w, b, x, y)` returning `(dL_dw, dL_db)` using the chain rule
# (sigmoid derivative is `a*(1-a)`). The check compares to numerical gradients at several random points.

# %% tags=["starter"]
def grads(w, b, x, y):
    ...

# %%
sig_ = lambda t: 1 / (1 + np.exp(-t))
L_ = lambda w, b, x, y: (sig_(w * x + b) - y) ** 2
r = np.random.default_rng(0)
for _ in range(5):
    w, b, x, y = r.normal(size=4)
    y = float(y > 0)
    e = 1e-6
    num_w = (L_(w + e, b, x, y) - L_(w - e, b, x, y)) / (2 * e)
    num_b = (L_(w, b + e, x, y) - L_(w, b - e, x, y)) / (2 * e)
    gw, gb = grads(w, b, x, y)
    assert abs(gw - num_w) < 1e-6 and abs(gb - num_b) < 1e-6, f"mismatch at w={w:.2f}, b={b:.2f}: got ({gw:.5f}, {gb:.5f}) vs numeric ({num_w:.5f}, {num_b:.5f})"
print("Exercise 3 passed")

# %% tags=["solution"]
def grads(w, b, x, y):
    a = 1 / (1 + np.exp(-(w * x + b)))
    dL_dz = 2 * (a - y) * a * (1 - a)
    return dL_dz * x, dL_dz

# %% [markdown]
# ### Exercise 4 (open-ended): Where does the gradient vanish?
# Plot `sigmoid(x)` and its derivative for `x` in [-8, 8]. In a markdown cell, state the largest value of the derivative (read it off your
# computed array with `.max()`), and for which range of `x` the derivative is below 0.01. Explain what that means for a neuron whose
# input `z` is very large or very small during training.

# %% [markdown]
# ## Recap
# - A derivative is a slope; zero slope marks candidate minima; central differences with `h ~ 1e-5` approximate it well.
# - The gradient stacks the partial derivatives, points uphill, and we step against it to reduce a loss.
# - The chain rule multiplies local derivatives along a computation path; backprop is that, done systematically backwards.
# - Always gradient-check hand-derived gradients against numerical ones. Autograd (PyTorch) does this bookkeeping for you.
#
# **Go deeper:** SymPy tutorial section "Calculus"; the PyTorch tutorial "A Gentle Introduction to torch.autograd" (pytorch.org/tutorials).
#
# **Next:** [Gradient descent from scratch](04_gradient_descent_from_scratch.ipynb)
