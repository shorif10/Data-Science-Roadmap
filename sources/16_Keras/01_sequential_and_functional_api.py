# %% [markdown]
# # Keras: Sequential and Functional APIs
#
# **Why this matters.** Keras is the layer at which most practitioners actually build neural networks: you stack layers, call
# `compile`, `fit`, `evaluate`, `predict`, and you have a working model in twenty lines. The skill that separates a beginner
# from someone you can hand a project to is knowing **when a plain stack is not enough** (several inputs, several outputs,
# shared layers, skip connections) and how to read `model.summary()` to spot a wiring bug before spending an hour training.
#
# **You will learn**
# - build a regression network and a classifier with `keras.Sequential`, and read `summary()` (shapes and parameter counts)
# - compile, fit, evaluate and compare against honest baselines (mean predictor, linear/logistic regression)
# - use the Functional API to build a model with **two inputs** (numeric features + categorical embeddings)
# - know how Keras 3 chooses its backend (TensorFlow, PyTorch or JAX)
#
# **Prerequisites:** [TensorFlow module](../15_TensorFlow/), [Deep Learning basics](../13_Deep_Learning/).
#
# > Data used: `datasets/housing.csv` (regression) and `datasets/churn.csv` (classification): synthetic tables described in `docs/DATASETS.md`.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score

DATA = Path("../datasets")
keras.utils.set_random_seed(42)
print("Keras", keras.__version__, "| backend:", keras.backend.backend())

# %% [markdown]
# ## 1. Regression with `Sequential`
#
# **Intuition.** A `Sequential` model is a pipe: the output of each layer is the input of the next. A `Dense(32, activation="relu")` layer
# computes `relu(x @ W + b)`: 32 weighted sums, each passed through a kink that lets the network bend. Stack a few and the model
# can represent curved relationships that a straight line cannot.
#
# **Data first.** Predict `price` (in $1000s) from the property features. The dataset has missing values and ~15 absurd prices
# (see `DATASETS.md`); we remove the impossible prices, and fill missing numbers with the **training** median.

# %%
h = pd.read_csv(DATA / "housing.csv")
ok = h["price"].between(30_000, 1_000_000)  # planted absurd values fall outside; decided after looking at describe()
print(f"dropping {(~ok).sum()} rows with price outside $30k-$1M of {len(h)}")
h = h[ok].reset_index(drop=True)
y_price = (h["price"] / 1000).to_numpy("float32")  # target in $1000s

num_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "distance_to_center_km", "quality_score"]
perm = np.random.default_rng(42).permutation(len(h))
n_tr = int(0.8 * len(h))
tr, te = perm[:n_tr], perm[n_tr:]

medians = h.loc[tr, num_cols].median()  # from TRAIN only
Xh = pd.concat([h[num_cols].fillna(medians), pd.get_dummies(h["neighborhood"], dtype="float32")], axis=1).astype("float32")
Xh_tr, Xh_te = Xh.iloc[tr].to_numpy(), Xh.iloc[te].to_numpy()
print(Xh.shape, "features:", list(Xh.columns))

# %% [markdown]
# Baselines first: a model that cannot beat "predict the average price" or a plain linear regression has no business being deployed.

# %%
mae_mean = mean_absolute_error(y_price[te], np.full(len(te), y_price[tr].mean()))
lin = LinearRegression().fit(Xh_tr, y_price[tr])
mae_lin = mean_absolute_error(y_price[te], lin.predict(Xh_te))
print(f"MAE (in $1000s)  mean predictor: {mae_mean:.1f}   linear regression: {mae_lin:.1f}")

# %% [markdown]
# Now the network. The first layer is `Normalization`: it standardises inputs using the mean/variance it learns from the training data with `adapt`.
# Neural nets train poorly on raw columns of very different scale (area ~1500 vs bathrooms ~2).

# %%
norm = keras.layers.Normalization()
norm.adapt(Xh_tr)

reg = keras.Sequential([
    keras.layers.Input((Xh_tr.shape[1],)),
    norm,
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),  # regression: one linear output, no activation
])
reg.summary()

# %% [markdown]
# Read the summary as a wiring check. Parameter counts are `inputs * units + units` (weights + biases): for the first Dense layer
# `(13 inputs * 32) + 32 = 448`. The `Normalization` layer's 27 parameters are its stored mean, variance and a count: they are **not trainable**.
# If a number here surprises you, your shapes are not what you think.

# %%
reg.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
history = reg.fit(Xh_tr, y_price[tr], validation_split=0.2, epochs=60, batch_size=64, verbose=0)
print("epochs run:", len(history.epoch), "| final train MAE:", round(history.history["mae"][-1], 1), "| val MAE:", round(history.history["val_mae"][-1], 1))

# %%
loss, mae_nn = reg.evaluate(Xh_te, y_price[te], verbose=0)
print(f"TEST MAE ($1000s): network {mae_nn:.1f} | linear {mae_lin:.1f} | mean predictor {mae_mean:.1f}")

# %% [markdown]
# *What just happened?* `compile` chose the optimiser and the loss; `fit` looped over mini-batches, and `validation_split=0.2` held out 20% of the
# training rows to watch for overfitting; `evaluate` scored data the model never trained on. The network improves on the mean predictor by
# a lot. Against linear regression the gap is smaller and depends on how non-linear the data is, which is exactly why you always
# print both: on a tabular problem a simple model is a serious competitor.

# %% [markdown]
# ## 2. Classification with `Sequential`
#
# Same recipe, different last layer and loss: **one sigmoid unit** (probability of churn) and **binary cross-entropy**. For more
# than two classes it becomes `Dense(k, activation="softmax")` with `sparse_categorical_crossentropy`.

# %%
c = pd.read_csv(DATA / "churn.csv")
c["contract"] = c["contract"].str.lower()
c["internet_service"] = c["internet_service"].fillna("None")
c["total_charges"] = c["total_charges"].fillna(c["monthly_charges"] * c["tenure_months"])
for col in ["paperless_billing", "autopay"]:
    c[col] = (c[col] == "Yes").astype("int64")
num_c = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "paperless_billing", "autopay", "senior_citizen"]
Xc = pd.concat([c[num_c], pd.get_dummies(c[["contract", "internet_service"]], dtype="float32")], axis=1).astype("float32")
yc = c["churned"].to_numpy("float32")
pc = np.random.default_rng(42).permutation(len(c))
trc, tec = pc[:4000], pc[4000:]
Xc_tr, Xc_te = Xc.iloc[trc].to_numpy(), Xc.iloc[tec].to_numpy()

norm_c = keras.layers.Normalization()
norm_c.adapt(Xc_tr)
clf = keras.Sequential([
    keras.layers.Input((Xc_tr.shape[1],)),
    norm_c,
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
clf.compile(optimizer=keras.optimizers.Adam(0.01), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
clf.fit(Xc_tr, yc[trc], validation_split=0.2, epochs=25, batch_size=128, verbose=0)

p_nn = clf.predict(Xc_te, verbose=0)[:, 0]
logit = LogisticRegression(max_iter=1000).fit(Xc_tr, yc[trc])
auc_nn = roc_auc_score(yc[tec], p_nn)
auc_lr = roc_auc_score(yc[tec], logit.predict_proba(Xc_te)[:, 1])
print(f"test AUC: network {auc_nn:.3f} | logistic regression {auc_lr:.3f}")
print(f"test accuracy: network {((p_nn > 0.5) == (yc[tec] == 1)).mean():.3f} | majority class {max(yc[tec].mean(), 1 - yc[tec].mean()):.3f}")

# %% [markdown]
# On tabular data like this, the small network and logistic regression usually land close together. That is a real-world lesson: neural
# networks are not automatically better on rows-and-columns problems (gradient boosting often wins there). Their advantage shows on
# images, text and audio, which the following notebooks cover.

# %% [markdown]
# ## 3. When a stack is not enough: the Functional API
#
# `Sequential` can only express a single path from one input to one output. The **Functional API** treats layers as functions
# on symbolic tensors, so you can wire any directed graph: several inputs, several outputs, branches that merge, skip connections, shared layers.
#
# **Example.** Give the churn model *two* inputs: (1) the numeric features, (2) two categorical features, `contract` and
# `internet_service`, fed as integer codes into learned **embeddings** (each category gets a small trainable vector). We concatenate
# both branches and predict.

# %%
cat_cols = ["contract", "internet_service"]
codes = np.column_stack([pd.Categorical(c[col]).codes for col in cat_cols]).astype("int32")  # 0..k-1 per column
n_cats = [c[col].nunique() for col in cat_cols]
print("categories per column:", dict(zip(cat_cols, n_cats)))

num_only = c[num_c].astype("float32").to_numpy()
norm_n = keras.layers.Normalization()
norm_n.adapt(num_only[trc])

num_in = keras.Input(shape=(len(num_c),), name="numeric")
cat_in = keras.Input(shape=(2,), dtype="int32", name="categorical")

x_num = norm_n(num_in)
x_num = keras.layers.Dense(16, activation="relu")(x_num)

# one embedding per categorical column, then flatten and concatenate
emb0 = keras.layers.Flatten()(keras.layers.Embedding(n_cats[0], 2)(cat_in[:, 0:1]))
emb1 = keras.layers.Flatten()(keras.layers.Embedding(n_cats[1], 2)(cat_in[:, 1:2]))

merged = keras.layers.Concatenate()([x_num, emb0, emb1])
merged = keras.layers.Dense(16, activation="relu")(merged)
out = keras.layers.Dense(1, activation="sigmoid", name="churn_prob")(merged)

two_input = keras.Model(inputs=[num_in, cat_in], outputs=out, name="two_input_churn")
two_input.summary()

# %% [markdown]
# Each row in `summary()` lists which layers **feed** it ("Connected to"), which is how you verify the graph you meant to draw.
# Training is identical to any Keras model; inputs are passed as a list or, more safely, as a **dict keyed by input name**:

# %%
two_input.compile(optimizer=keras.optimizers.Adam(0.01), loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
inputs_tr = {"numeric": num_only[trc], "categorical": codes[trc]}
inputs_te = {"numeric": num_only[tec], "categorical": codes[tec]}
two_input.fit(inputs_tr, yc[trc], validation_split=0.2, epochs=25, batch_size=128, verbose=0)
p2 = two_input.predict(inputs_te, verbose=0)[:, 0]
print(f"test AUC: two-input model {roc_auc_score(yc[tec], p2):.3f} | one-hot Sequential {auc_nn:.3f} | logistic regression {auc_lr:.3f}")

# %% [markdown]
# *What just happened?* We built a graph with two entry points. The embeddings carry the same information as one-hot columns, so on
# three-category features do not expect a gain: embeddings pay off when a categorical column has thousands of values (product IDs, zip codes),
# where one-hot is huge and sparse. The important skill is the wiring pattern, which is identical for images + metadata, text + numbers, and so on.
#
# Two more Functional-API facts worth knowing: a `Model` is itself a layer (you can reuse a trained model as a component of a bigger one),
# and calling the *same layer object* on two tensors **shares its weights** (a siamese network).

# %% [markdown]
# ## 4. Keras 3 is multi-backend
#
# Since Keras 3 the same Keras code can run on **TensorFlow, PyTorch or JAX**. The backend is chosen when `keras` is first
# imported, by the `KERAS_BACKEND` environment variable (or the `backend` field of `~/.keras/keras.json`). This course uses the default TensorFlow backend;
# you can confirm what you are on:

# %%
print("active backend:", keras.backend.backend())

# %% [markdown]
# To run a script on PyTorch you set the variable **before** importing Keras; you cannot switch inside a running process (so we do not do it in this notebook):
#
# ```python
# import os
# os.environ["KERAS_BACKEND"] = "torch"   # or "jax" / "tensorflow"; must come before `import keras`
# import keras                              # now layers/models run on PyTorch tensors
# ```
#
# or from a shell: `KERAS_BACKEND=jax python train.py`. Code that uses only `keras.*` and `keras.ops.*` is portable. Code that calls
# `tf.*` directly (like `tf.data`, `tf.GradientTape` or `tf.function` in the previous module) is TensorFlow-specific. The PyTorch and JAX packages must be
# installed in the environment; TensorFlow is what lets Keras export a SavedModel.

# %% [markdown]
# ## Common mistakes
#
# 1. **Unscaled inputs.** Columns on different scales make training slow or unstable. Normalise (inside the model with `Normalization`).
# 2. **Wrong output layer for the loss.** Sigmoid + `binary_crossentropy` for 2 classes; softmax + `sparse_categorical_crossentropy` for k classes (integer labels);
#    no activation + `mse` for regression. Mixing these trains without error and gives bad results.
# 3. **Judging the model on training loss.** Always use validation/test data, and report the baseline next to your number.
# 4. **Fitting normalisation/imputation statistics on all rows** before splitting (data leakage). We used the training rows only.
# 5. **`Input(shape=...)` including the batch size.** The shape excludes it: `(13,)`, not `(None, 13)`.
# 6. **Reusing a trained model object** in a "fresh" experiment. Re-build (and reseed) each time or you continue from old weights.
# 7. **Expecting a network to beat gradient boosting on small tabular data.** Compare with a simple model first.

# %%
# Demonstration of mistake 2: a shape/loss mismatch fails loudly here, but a softer mismatch would not
bad = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(4, activation="softmax")])
bad.compile(loss="binary_crossentropy")
try:
    bad.fit(np.ones((8, 3), "float32"), np.arange(8) % 4, verbose=0)  # integer class labels 0..3 with a binary loss
except Exception as e:  # noqa: BLE001
    print(type(e).__name__, "-", str(e)[:100].replace("\n", " "), "...")

# %% [markdown]
# ## Interview corner
#
# **1. Sequential vs Functional vs subclassing: how do you choose?**
# <details><summary>Answer</summary>
# Sequential for a plain stack of layers with one input and one output. Functional when the graph has several inputs/outputs, branches, skip connections
# or shared layers: it stays declarative, so Keras can inspect, plot and serialise it. Subclassing <code>keras.Model</code> for dynamic Python logic in
# <code>call</code> (research, recursion), at the price of less introspection.
# </details>
#
# **2. How do you pick the last layer and loss?**
# <details><summary>Answer</summary>
# Binary: 1 unit, sigmoid, binary cross-entropy. Multi-class: k units, softmax, (sparse) categorical cross-entropy. Multi-label: k sigmoid units, binary
# cross-entropy. Regression: linear unit(s), MSE/MAE/Huber. Output activation and loss must agree (or use <code>from_logits=True</code> with a linear output).
# </details>
#
# **3. What do the parameter counts in <code>summary()</code> mean?**
# <details><summary>Answer</summary>
# For Dense: <code>inputs*units + units</code>. Trainable parameters are updated by gradients; non-trainable ones (frozen layers, normalisation statistics,
# batch-norm moving averages) are not. Checking counts is a fast way to catch a wrong input shape or an unintended frozen layer.
# </details>
#
# **4. Why an embedding for a categorical feature instead of one-hot?**
# <details><summary>Answer</summary>
# An embedding maps each category to a small learned dense vector: far fewer parameters than one-hot for high-cardinality features, and similar
# categories can end up close together. For a handful of categories one-hot is just as good.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Predict the parameter count
# Build `mini`: a `Sequential` model with an `Input` of the housing feature width (`Xh_tr.shape[1]`), a `Dense(8, relu)` and a `Dense(1)`.
# In `expected_params` store the total parameter count **computed by hand** from the formula `inputs*units + units` per Dense layer
# (as an integer), and confirm it against `mini.count_params()`. *Hint: `n_in = Xh_tr.shape[1]`.*

# %% tags=["starter"]
mini = ...
expected_params = ...

# %%
n_in = Xh_tr.shape[1]
formula = (n_in * 8 + 8) + (8 * 1 + 1)
assert expected_params == formula, f"by hand: ({n_in}*8+8) + (8*1+1) = {formula}"
assert mini.count_params() == expected_params, f"model has {mini.count_params()} parameters, your formula says {expected_params}"
assert mini.output_shape == (None, 1), "the last layer should output one value per row"
print("Exercise 1 passed:", expected_params, "parameters")

# %% tags=["solution"]
n_in = Xh_tr.shape[1]
mini = keras.Sequential([keras.layers.Input((n_in,)), keras.layers.Dense(8, activation="relu"), keras.layers.Dense(1)])
expected_params = (n_in * 8 + 8) + (8 * 1 + 1)

# %% [markdown]
# ### Exercise 2: Beat the baseline (and say so)
# Train a new Sequential regression model on the housing training data (any architecture, at most 60 epochs, use the `norm` layer defined above as
# the first layer). Store the test MAE in `my_mae` (in $1000s). It must beat the **mean predictor** (`mae_mean`). Then set `beats_linear`
# to `True` if it also beats `mae_lin`, else `False`. *Hint: `evaluate` returns `[loss, mae]` if you compile with `metrics=["mae"]`; reseed with
# `keras.utils.set_random_seed(0)` for a stable result.*

# %% tags=["starter"]
my_mae = ...
beats_linear = ...

# %%
assert my_mae < mae_mean, f"your MAE {my_mae:.1f} should beat the mean predictor {mae_mean:.1f}"
assert isinstance(beats_linear, bool) and beats_linear == (my_mae < mae_lin), "beats_linear should honestly reflect whether my_mae < mae_lin"
print(f"Exercise 2 passed (MAE {my_mae:.1f}; linear {mae_lin:.1f}; beats linear: {beats_linear})")

# %% tags=["solution"]
keras.utils.set_random_seed(0)
m2 = keras.Sequential([keras.layers.Input((Xh_tr.shape[1],)), norm, keras.layers.Dense(32, activation="relu"),
                       keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
m2.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse", metrics=["mae"])
m2.fit(Xh_tr, y_price[tr], epochs=40, batch_size=64, verbose=0)
my_mae = float(m2.evaluate(Xh_te, y_price[te], verbose=0)[1])
beats_linear = bool(my_mae < mae_lin)

# %% [markdown]
# ### Exercise 3: A two-input functional model on housing
# Build `hm`, a Functional model with **two inputs**: `"numeric"` (the 7 numeric housing columns after median imputation, standardised by a
# `Normalization` layer you adapt) and `"hood"` (the neighborhood as an integer code, 0..5, through an `Embedding(6, 3)` then `Flatten`).
# Concatenate, add `Dense(16, relu)`, and a single linear output. Compile with `mse`, train for 40 epochs, and store the test MAE ($1000s) in `hm_mae`.
# The check requires exactly two inputs named `numeric` and `hood`, and a MAE better than the mean predictor.
# *Hint: `hood_codes = pd.Categorical(h["neighborhood"]).codes`; `keras.Input(shape=(1,), dtype="int32", name="hood")`.*

# %%
hood_codes = pd.Categorical(h["neighborhood"]).codes.astype("int32").reshape(-1, 1)
num_h = h[num_cols].fillna(medians).to_numpy("float32")
print(num_h.shape, hood_codes.shape, "codes:", sorted(set(hood_codes.ravel())))

# %% tags=["starter"]
hm = ...
hm_mae = ...

# %%
assert sorted(t.name for t in hm.inputs) == ["hood", "numeric"], "inputs should be named numeric and hood"
assert hm.output_shape == (None, 1), "one linear output"
pred = hm.predict({"numeric": num_h[te], "hood": hood_codes[te]}, verbose=0)[:, 0]
assert abs(mean_absolute_error(y_price[te], pred) - hm_mae) < 0.5, "hm_mae should be the test MAE of hm"
assert hm_mae < mae_mean, f"MAE {hm_mae:.1f} should beat the mean predictor {mae_mean:.1f}"
print(f"Exercise 3 passed (MAE {hm_mae:.1f})")

# %% tags=["solution"]
keras.utils.set_random_seed(0)
nrm = keras.layers.Normalization()
nrm.adapt(num_h[tr])
in_num = keras.Input(shape=(len(num_cols),), name="numeric")
in_hood = keras.Input(shape=(1,), dtype="int32", name="hood")
z = keras.layers.Concatenate()([nrm(in_num), keras.layers.Flatten()(keras.layers.Embedding(6, 3)(in_hood))])
z = keras.layers.Dense(16, activation="relu")(z)
hm = keras.Model([in_num, in_hood], keras.layers.Dense(1)(z))
hm.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
hm.fit({"numeric": num_h[tr], "hood": hood_codes[tr]}, y_price[tr], epochs=40, batch_size=64, verbose=0)
hm_mae = float(mean_absolute_error(y_price[te], hm.predict({"numeric": num_h[te], "hood": hood_codes[te]}, verbose=0)[:, 0]))

# %% [markdown]
# ### Exercise 4 (open-ended): Which is better, and by how much?
# Using the churn split above, train the Sequential classifier and `two_input` again with 3 different seeds each, and compare mean test AUC.
# Is the difference bigger than the seed-to-seed spread? Write three sentences on what you would tell a stakeholder who asks "should we use the neural network?".

# %% [markdown]
# ## Recap
# - `Sequential` = a pipe of layers; read `summary()` for shapes and parameter counts (`inputs*units + units` for Dense).
# - Last layer + loss go together: linear+MSE (regression), sigmoid+BCE (binary), softmax+sparse CCE (multi-class).
# - Always compare with simple baselines (mean predictor, linear/logistic regression); on small tabular data they are often close.
# - The Functional API wires arbitrary graphs (multi-input/output, embeddings, shared layers); feed inputs as a dict keyed by input name.
# - Keras 3 runs on TensorFlow, PyTorch or JAX, selected by `KERAS_BACKEND` **before** import.
#
# **Go deeper:** the Keras guides "The Sequential model", "The Functional API" and "Keras 3: multi-backend" at keras.io.
#
# **Next:** [Callbacks, regularization and tuning](02_callbacks_regularization_and_tuning.ipynb)
