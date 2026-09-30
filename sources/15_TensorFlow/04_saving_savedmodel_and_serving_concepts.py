# %% [markdown]
# # Saving Models, SavedModel and Serving Concepts
#
# **Why this matters.** A model that lives only in a notebook kernel is worth nothing to the business. Someone has to load it in an API,
# a batch job or a phone app, weeks later, on a different machine, and get the *same predictions*. The two classic ways this goes wrong:
# the preprocessing was not saved with the model (training/serving skew), or the artifact cannot be loaded with the library versions in
# production. This notebook covers the formats that work in TensorFlow 2.21 / Keras 3 and how to test that a saved model is faithful.
#
# **You will learn**
# - save and reload a whole model with the `.keras` format, and look inside the file
# - save only weights, and register custom layers so they can be reloaded
# - bake preprocessing into the model so serving takes raw features
# - `model.export()` to a SavedModel, load it back, and call its serving signature
# - what TensorFlow Serving and TensorFlow Lite are for, plus a parity test you should run before shipping
#
# **Prerequisites:** [Custom training loop](03_custom_training_loop_and_tf_function.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic telecom customers, see `docs/DATASETS.md`). Everything is written to a temporary
# > directory that is deleted at the end.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json
import tempfile
import zipfile
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

keras.utils.set_random_seed(42)
TMP = tempfile.TemporaryDirectory()  # cleaned up in the last cell
WORK = Path(TMP.name)
print("TF", tf.__version__, "| Keras", keras.__version__)

# %% [markdown]
# ## 1. A small model to save
#
# Same churn features as before. One deliberate design choice: we do **not** standardise the features in pandas.
# Instead a `keras.layers.Normalization` layer, adapted on the training data, is the first layer of the model. Its mean and
# variance are stored as weights, so the saved model accepts **raw** feature values.

# %%
churn = pd.read_csv(Path("../datasets/churn.csv"))
churn["contract"] = churn["contract"].str.lower()
churn["internet_service"] = churn["internet_service"].fillna("None")
churn["total_charges"] = churn["total_charges"].fillna(churn["monthly_charges"] * churn["tenure_months"])
for c in ["paperless_billing", "autopay"]:
    churn[c] = (churn[c] == "Yes").astype("int64")
X = pd.concat(
    [churn[["tenure_months", "monthly_charges", "total_charges", "support_calls", "paperless_billing", "autopay", "senior_citizen"]],
     pd.get_dummies(churn[["contract", "internet_service"]], dtype="float32")], axis=1).astype("float32")
y = churn["churned"].to_numpy("float32")
perm = np.random.default_rng(42).permutation(len(X))
tr, te = perm[:4000], perm[4000:]
X_tr, X_te = X.iloc[tr].to_numpy(), X.iloc[te].to_numpy()  # RAW values
print(X_tr.shape, "columns:", list(X.columns)[:4], "...")

# %%
norm = keras.layers.Normalization()
norm.adapt(X_tr)  # learns per-column mean and variance from the training data only

model = keras.Sequential([
    keras.layers.Input((X_tr.shape[1],)),
    norm,
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y[tr], epochs=5, batch_size=128, verbose=0)
print("test [loss, accuracy]:", np.round(model.evaluate(X_te, y[te], verbose=0), 3))
p_ref = model.predict(X_te, verbose=0)  # reference predictions to compare every reload against

# %% [markdown]
# ## 2. The `.keras` format: one file, everything
#
# `model.save("name.keras")` stores the **architecture**, the **weights**, the compile settings and the **optimizer state** in a single
# file, so you can resume training. `keras.models.load_model` restores all of it.

# %%
model.save(WORK / "churn.keras")
restored = keras.models.load_model(WORK / "churn.keras")
p_new = restored.predict(X_te, verbose=0)
print("file size:", round((WORK / "churn.keras").stat().st_size / 1024, 1), "KiB")
print("max |difference| in predictions:", float(np.abs(p_ref - p_new).max()))

# %% [markdown]
# *What just happened?* A full round trip with identical predictions. A `.keras` file is just a zip archive; look inside it to see
# what is stored (and to find out which Keras version wrote it, which matters when debugging a load failure):

# %%
with zipfile.ZipFile(WORK / "churn.keras") as z:
    print(z.namelist())
    meta = json.loads(z.read("metadata.json"))
print(meta)

# %% [markdown]
# ### Weights only
# `save_weights` / `load_weights` store only the numbers (file name must end in `.weights.h5`). You must rebuild the same
# architecture in code first. Use it when the code is the source of truth for the architecture (research checkpoints, transfer learning).

# %%
model.save_weights(WORK / "churn.weights.h5")
twin = keras.models.clone_model(model)  # same architecture, fresh weights
twin.load_weights(WORK / "churn.weights.h5")
print("weights-only round trip identical?", np.allclose(twin.predict(X_te, verbose=0), p_ref, atol=1e-6))

# %% [markdown]
# ### Custom layers must be registered
# If your model contains your own class, Keras needs to find it by name when loading. Decorate it with
# `@keras.saving.register_keras_serializable` and implement `get_config` so constructor arguments are saved.

# %%
class Unregistered(keras.layers.Layer):
    def call(self, x):
        return x * 2.0


@keras.saving.register_keras_serializable(package="course")
class Scale(keras.layers.Layer):
    def __init__(self, factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"factor": self.factor})  # without this, factor would silently revert to the default
        return cfg


for layer_cls, name in [(Unregistered, "bad.keras"), (Scale, "good.keras")]:
    m = keras.Sequential([keras.layers.Input((3,)), layer_cls()])
    m.save(WORK / name)
    try:
        keras.models.load_model(WORK / name)
        print(name, "-> loaded fine")
    except Exception as e:  # noqa: BLE001
        print(name, "->", type(e).__name__, "(custom class not registered)")

# %% [markdown]
# ## 3. `model.export()`: SavedModel for serving
#
# `.keras` is for **Keras-to-Keras** use (it needs Python and Keras to load). For deployment to other runtimes
# (TensorFlow Serving, TFLite conversion, non-Python hosts) TensorFlow uses the **SavedModel** format: a directory with the computation
# graph and the variables, independent of your Python class definitions. In Keras 3 you create it with `model.export(path)`
# (in Keras 2 you used `model.save(dir)`; that no longer does this).

# %%
export_dir = WORK / "churn_savedmodel" / "1"  # the trailing "1" is a version number, see the serving section
model.export(export_dir)
print(sorted(p.name for p in export_dir.iterdir()))

# %% [markdown]
# `saved_model.pb` holds the graph, `variables/` the weights, `assets/` any files (like vocabularies). Load it with plain TensorFlow
# (no Keras model class involved) and inspect the **signatures**: the named entry points a server calls.

# %%
loaded = tf.saved_model.load(str(export_dir))
print("signatures:", list(loaded.signatures))
sig = loaded.signatures["serving_default"]
input_name = list(sig.structured_input_signature[1])[0]
print("input:", sig.structured_input_signature[1][input_name])
print("output:", sig.structured_outputs)

# %%
out = sig(**{input_name: tf.constant(X_te[:5])})  # signatures take keyword tensors and return a dict
key = list(out)[0]
print(np.round(out[key].numpy()[:, 0], 4))
print(np.round(p_ref[:5, 0], 4), "<- original model")

# %% [markdown]
# The exported model took **raw** feature values because normalisation lives inside it. Had we standardised in pandas, every client would
# have to reproduce that exact code and the same training mean/std: the classic *training/serving skew*.
#
# To use an exported SavedModel *inside Keras 3* wrap it in a `TFSMLayer` (inference only). Note that `load_model` refuses a SavedModel directory:

# %%
tfsm = keras.layers.TFSMLayer(str(export_dir), call_endpoint="serving_default")
res = tfsm(X_te[:5])
print("TFSMLayer output keys:", list(res) if isinstance(res, dict) else type(res).__name__)
try:
    keras.models.load_model(export_dir)
except ValueError as e:
    print("load_model on a SavedModel dir ->", str(e)[:75], "...")

# %% [markdown]
# ## 4. Serving concepts
#
# **TensorFlow Serving** is a separate high-performance server that loads SavedModels from a directory and exposes them over gRPC and
# REST. The convention is `models/<name>/<version>/saved_model.pb`: drop in a new numbered folder and the server hot-swaps to
# the highest version (and lets you roll back by removing it). The REST contract is a JSON body `{"instances": [...]}`; the reply is
# `{"predictions": [...]}`. Building the request needs no server, so you can see it:

# %%
payload = json.dumps({"signature_name": "serving_default", "instances": X_te[:2].round(2).tolist()})
print(payload[:160], "...")

# %% [markdown]
# Starting the server is a container command (needs Docker and a free port, so it is shown, not run):

# %% tags=["no-run"]
# Docker is not available in the course environment; run this on your own machine.
# docker run -p 8501:8501 \
#   -v "/path/to/churn_savedmodel:/models/churn" -e MODEL_NAME=churn tensorflow/serving
#
# import httpx
# r = httpx.post("http://localhost:8501/v1/models/churn:predict", content=payload)
# print(r.json()["predictions"])

# %% [markdown]
# Because SavedModel is not tied to Python, TF Serving can run it without your code. The other common route is your own **FastAPI**
# service that loads the model at start-up (that is the MLOps module). To mimic the TF Serving contract in-process, this function
# takes the same JSON and answers with the same shape, and can be unit-tested without a network:

# %%
def predict_rest(body: str) -> str:
    req = json.loads(body)
    batch = tf.constant(np.array(req["instances"], dtype="float32"))
    result = sig(**{input_name: batch})
    return json.dumps({"predictions": result[list(result)[0]].numpy().tolist()})


reply = json.loads(predict_rest(payload))
print("predictions:", np.round(reply["predictions"], 3).ravel())

# %% [markdown]
# **TensorFlow Lite** converts a SavedModel into a compact file for phones and embedded devices. The converter runs fine on a laptop, so
# we can look at the size. (With optional flags it can also quantise weights to 8 bits, which shrinks the file further but changes
# predictions slightly, so re-check accuracy after converting.)

# %%
tflite_bytes = tf.lite.TFLiteConverter.from_saved_model(str(export_dir)).convert()
saved_model_bytes = sum(f.stat().st_size for f in export_dir.rglob("*") if f.is_file())
print(f"SavedModel on disk: {saved_model_bytes / 1024:.1f} KiB | TFLite flatbuffer: {len(tflite_bytes) / 1024:.1f} KiB")

# %% [markdown]
# Running a TFLite model uses a small runtime. `tf.lite.Interpreter` is being deprecated in favour of the standalone LiteRT package
# (`ai_edge_litert`), which is not installed here, so this cell is shown but not run:

# %% tags=["no-run"]
# from ai_edge_litert.interpreter import Interpreter
# interp = Interpreter(model_content=tflite_bytes)
# interp.allocate_tensors()
# inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
# interp.set_tensor(inp["index"], X_te[:1])
# interp.invoke()
# print(interp.get_tensor(outp["index"]))

# %% [markdown]
# ### The parity test you should always run
# Before shipping, load the artifact **the way production will** and compare predictions with the original on real rows.
# Put it in your test suite:

# %%
def assert_parity(reference, candidate, tol=1e-5):
    diff = float(np.abs(reference - candidate).max())
    assert diff < tol, f"saved model disagrees with the original by {diff:.2e}"
    return diff


print("keras round trip diff:", assert_parity(p_ref, p_new))
print("SavedModel diff:      ", assert_parity(p_ref[:5, 0], out[key].numpy()[:, 0], 1e-4))

# %% [markdown]
# ## Common mistakes
#
# 1. **Saving without the preprocessing.** Standardisation/encoding done in pandas is lost; put it in the model (`Normalization`,
#    `StringLookup`, `TextVectorization`) or ship it as a versioned artifact next to the model.
# 2. **Using `model.save("dir")` expecting a SavedModel** (Keras 2 habit). In Keras 3 use `model.export(dir)`; `.save` wants `.keras` / `.h5`.
# 3. **Custom classes without `register_keras_serializable` + `get_config`**, so loading fails or silently uses default arguments.
# 4. **Never testing the reload.** Compare predictions from the reloaded artifact with the original on real rows.
# 5. **Unpinned library versions.** A `.keras` file records the Keras version that wrote it (see `metadata.json`); pin the same major version in production.
# 6. **Pickling Keras models** with `pickle`/`joblib`: not supported; use the formats above.
# 7. **Forgetting the version folder** for TF Serving (`models/churn/1/`): the server will not find the model.

# %% [markdown]
# ## Interview corner
#
# **1. `.keras` vs SavedModel: when do you use which?**
# <details><summary>Answer</summary>
# <code>.keras</code> is the native Keras 3 format: architecture, weights and optimizer state in one archive, reloadable as a trainable
# Keras model, but it needs Python/Keras. A SavedModel (<code>model.export</code>) is a serialised computation graph plus variables with named
# serving signatures, loadable without your code and consumable by TF Serving, TFLite conversion and other languages, but it is
# inference-oriented.
# </details>
#
# **2. What is training/serving skew and how do you reduce it?**
# <details><summary>Answer</summary>
# The model sees differently prepared inputs in production than in training (different scaling, category mapping, tokenisation, or missing-value
# handling). Reduce it by putting preprocessing inside the exported model, sharing one code path, and running parity tests on real rows.
# </details>
#
# **3. What is a serving signature?**
# <details><summary>Answer</summary>
# A named function in the SavedModel with a fixed input/output contract (tensor names, shapes, dtypes). Servers call
# <code>serving_default</code> by name, so changing it is a breaking API change for clients.
# </details>
#
# **4. How would you roll out a new model version safely with TF Serving?**
# <details><summary>Answer</summary>
# Export it to the next numbered folder, run a parity/regression test, then let the server load it (it serves the highest version by default,
# and version policies can pin or serve several). Monitor metrics and roll back by removing the folder or pinning the previous version.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Prove a save/load round trip
# Save `model` to `WORK / "ex1.keras"`, load it back into `reloaded`, and store the largest absolute prediction difference on `X_te`
# in `max_diff` (a float). *Hint: `model.save`, `keras.models.load_model`, `np.abs(a - b).max()`.*

# %% tags=["starter"]
max_diff = ...

# %%
assert (WORK / "ex1.keras").exists(), "save the model to WORK / 'ex1.keras'"
assert isinstance(max_diff, float), "max_diff should be a Python float (use float(...))"
assert max_diff < 1e-5, f"a faithful round trip has (almost) no difference, got {max_diff}"
print("Exercise 1 passed")

# %% tags=["solution"]
model.save(WORK / "ex1.keras")
reloaded = keras.models.load_model(WORK / "ex1.keras")
max_diff = float(np.abs(model.predict(X_te, verbose=0) - reloaded.predict(X_te, verbose=0)).max())

# %% [markdown]
# ### Exercise 2: Read the archive
# Without loading the model, open `WORK / "ex1.keras"` as a zip file. Store the sorted list of member names in `members` and the
# Keras version recorded in the metadata in `saved_with`. *Hint: `zipfile.ZipFile`, `namelist()`, and `json.loads` of `metadata.json`.*

# %% tags=["starter"]
members = ...
saved_with = ...

# %%
with zipfile.ZipFile(WORK / "ex1.keras") as z:
    assert members == sorted(z.namelist()), "members should be the sorted namelist of the archive"
assert saved_with == keras.__version__, f"the archive was written by Keras {keras.__version__}, not {saved_with}"
print("Exercise 2 passed:", members)

# %% tags=["solution"]
with zipfile.ZipFile(WORK / "ex1.keras") as z:
    members = sorted(z.namelist())
    saved_with = json.loads(z.read("metadata.json"))["keras_version"]

# %% [markdown]
# ### Exercise 3: Serve from the SavedModel signature
# Export `model` to `WORK / "ex3" / "1"`, load it with `tf.saved_model.load`, call the `serving_default` signature on the
# **first 8 test rows** and store the churn probabilities as a NumPy vector `served` (shape `(8,)`).
# *Hint: repeat the steps of section 3: `signatures["serving_default"]`, find the input name, pass a keyword tensor, take the first output.*

# %% tags=["starter"]
served = ...

# %%
expected = model.predict(X_te[:8], verbose=0)[:, 0]
served = np.asarray(served)
assert served.shape == (8,), f"expected shape (8,), got {served.shape}"
assert np.allclose(served, expected, atol=1e-4), "served probabilities should match model.predict"
assert ((served >= 0) & (served <= 1)).all(), "probabilities should be between 0 and 1"
print("Exercise 3 passed")

# %% tags=["solution"]
model.export(WORK / "ex3" / "1")
sig3 = tf.saved_model.load(str(WORK / "ex3" / "1")).signatures["serving_default"]
name3 = list(sig3.structured_input_signature[1])[0]
res3 = sig3(**{name3: tf.constant(X_te[:8])})
served = res3[list(res3)[0]].numpy()[:, 0]

# %% [markdown]
# ### Exercise 4 (open-ended): Skew on purpose
# Build a second model with the same architecture **without** the `Normalization` layer, train it on features standardised in pandas
# with the training mean/std, save it, and then send it *raw* rows as a careless client would. Report the accuracy on the raw vs
# correctly standardised test rows, and write two sentences on why the failure is silent (no exception) and what you would change in the
# deployment to make it impossible.

# %%
TMP.cleanup()  # remove the temporary directory (all model files)
print("cleaned up:", not WORK.exists())

# %% [markdown]
# ## Recap
# - `model.save("x.keras")` + `keras.models.load_model` is a full, faithful round trip (architecture, weights, optimizer state); `save_weights` needs the code to rebuild the model.
# - Custom classes need `@keras.saving.register_keras_serializable` and a `get_config`.
# - `model.export(dir)` writes a SavedModel with signatures for serving; load it with `tf.saved_model.load` or `keras.layers.TFSMLayer` (Keras 3 `load_model` refuses it).
# - Put preprocessing inside the model to avoid training/serving skew, and always run a parity test on real rows.
# - TF Serving hosts numbered SavedModel versions behind gRPC/REST; TFLite/LiteRT is the on-device route.
#
# **Go deeper:** the Keras guide "Save, serialize, and export models" (keras.io) and the TensorFlow guide "Using the SavedModel format"
# (tensorflow.org/guide); TensorFlow Serving documentation at tensorflow.org/tfx/guide/serving.
#
# **Next module:** [Keras: Sequential and Functional API](../16_Keras/01_sequential_and_functional_api.ipynb)
