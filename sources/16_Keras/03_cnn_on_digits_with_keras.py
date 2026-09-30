# %% [markdown]
# # A CNN on Digits with Keras: Augmentation and Error Analysis
#
# **Why this matters.** When you inherit an image classifier at work, the question is rarely "how do I build one?" It is "where does it fail,
# and would more data or a different input treatment fix that?" A convolutional network (CNN) is the default starting point for images, and
# Keras **augmentation layers** are the cheapest way to get "more data": random shifts, rotations and zooms applied during training. The
# habit to build here is to *measure* each choice against a baseline and then look at the actual mistakes instead of staring at one accuracy number.
#
# **You will learn**
# - build a small CNN with `Conv2D`, `MaxPooling2D` and `Dense`, and predict each layer's output shape by hand
# - put augmentation layers inside the model and confirm they only act during training
# - compare a CNN with a fully-connected baseline on the same split, honestly (parameters, accuracy, variation between seeds)
# - do error analysis: confusion matrix, most-confused pairs, and a look at the misclassified images
#
# **Prerequisites:** [Sequential and Functional API](01_sequential_and_functional_api.ipynb),
# [Callbacks, regularization and tuning](02_callbacks_regularization_and_tuning.ipynb).
#
# > Data used: scikit-learn's bundled `digits`: 1,797 real 8x8 grayscale handwritten digits. Tiny images keep everything fast on a CPU;
# > the ideas are identical for 224x224 photographs.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

digits = load_digits()
X = (digits.images / 16.0).astype("float32")[..., None]  # pixel values are 0..16 -> scale to 0..1; add a channel axis
y = digits.target
print("images:", X.shape, "(samples, height, width, channels) | classes:", np.unique(y))

X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=42)
print("train / validation / test:", len(X_tr), len(X_va), len(X_te))

# %% [markdown]
# Keras image layers expect **channels-last** data: `(height, width, channels)`. A grayscale image has one channel, so we added an axis of size 1.
# The `/ 16` matters: networks train badly on inputs that are 16 times larger than the scale their initial weights expect.

# %%
fig, axes = plt.subplots(2, 8, figsize=(9, 2.8))
for ax, img, label in zip(axes.ravel(), X_tr, y_tr):
    ax.imshow(img[..., 0], cmap="gray_r")
    ax.set_title(f"label {label}", fontsize=8)
    ax.axis("off")
fig.suptitle("Sixteen training digits (8x8 pixels, darker = more ink)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1. A baseline that ignores image structure
#
# **Intuition.** A fully-connected network (MLP) flattens the 8x8 grid into 64 independent numbers. It has no idea that pixel (3,4) sits next to
# pixel (3,5), so a stroke in a new position is, to it, a brand new pattern. Always train this baseline first: if it already scores well, a
# fancy architecture has little to add.

# %%
def compile_model(model, lr=2e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(model, epochs=40, patience=5):
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=64, callbacks=[stop], verbose=0)


def accuracy(model, X_, y_):
    return float((model.predict(X_, verbose=0).argmax(axis=1) == y_).mean())


keras.utils.set_random_seed(42)
mlp = compile_model(keras.Sequential([
    keras.layers.Input((8, 8, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
]))
train(mlp)
print(f"MLP: {mlp.count_params():,} parameters | validation accuracy {accuracy(mlp, X_va, y_va):.3f}")

# %% [markdown]
# ## 2. The CNN: shapes by hand
#
# **Intuition.** A `Conv2D(16, 3)` layer slides sixteen small 3x3 "stroke detectors" across the image. The same 9 weights are reused at every
# position, so the layer learns "a diagonal edge" once and finds it anywhere. `MaxPooling2D(2)` then keeps the strongest response in each 2x2
# block, halving the height and width and making the next layer less sensitive to exact position.
#
# Before running anything, predict the shapes (with the default `padding="valid"`, a 3x3 kernel trims 2 pixels from each dimension):
#
# | Layer | Output shape | Why |
# |---|---|---|
# | input | 8 x 8 x 1 | |
# | `Conv2D(16, 3)` | 6 x 6 x 16 | 8 - 3 + 1 = 6, sixteen filters |
# | `MaxPooling2D(2)` | 3 x 3 x 16 | 6 / 2 |
# | `Conv2D(32, 3)` | 1 x 1 x 32 | 3 - 3 + 1 = 1 |
# | `Flatten` | 32 | |
# | `Dense(10)` | 10 | one probability per digit |

# %%
def build_cnn(augment=False, dropout=0.25):
    layers = [keras.layers.Input((8, 8, 1))]
    if augment:  # these layers are inert at inference time; see section 3
        layers += [
            keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
            keras.layers.RandomRotation(0.03, fill_mode="constant"),
        ]
    layers += [
        keras.layers.Conv2D(16, 3, activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation="softmax"),
    ]
    return compile_model(keras.Sequential(layers))


keras.utils.set_random_seed(42)
cnn = build_cnn()
cnn.summary()

# %% [markdown]
# Check the `Output Shape` column against the table. Then count the parameters of the first convolution yourself: 16 filters x (3x3x1 weights) + 16 biases
# = 160. The weight sharing is why the CNN is small: compare its total with the MLP above.

# %%
train(cnn)
print(f"CNN: {cnn.count_params():,} parameters | validation accuracy {accuracy(cnn, X_va, y_va):.3f}")
print(f"MLP: {mlp.count_params():,} parameters | validation accuracy {accuracy(mlp, X_va, y_va):.3f}")

# %% [markdown]
# *What just happened?* The CNN reaches a similar or better accuracy with far fewer parameters. On a 360-image validation set one image is
# 0.28 percentage points, so a difference of a few images is noise. Do not claim a winner from one run; section 4 repeats the comparison over seeds.

# %% [markdown]
# ## 3. Augmentation layers
#
# **Intuition.** Real handwriting is shifted, slightly rotated and a bit bigger or smaller every time. Augmentation generates those variations
# on the fly from the images you already have, so the model cannot memorise exact pixel positions. Keras 3 provides them as layers
# (`RandomTranslation`, `RandomRotation`, `RandomZoom`, `RandomFlip`, ...) that live inside the model. Two properties to verify rather than trust:
# they randomise only when `training=True`, and **the right augmentation depends on the data** (flipping a 6 gives something that looks like a different digit's mirror image, so no flips here).

# %%
keras.utils.set_random_seed(42)
aug = keras.Sequential([keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant"), keras.layers.RandomRotation(0.03, fill_mode="constant")])
sample = X_tr[:1]
same_at_inference = np.allclose(aug(sample, training=False), sample)
differs_in_training = not np.allclose(aug(sample, training=True), sample)
print("identical to the input when training=False:", same_at_inference)
print("changed when training=True:              ", differs_in_training)

# %%
fig, axes = plt.subplots(3, 8, figsize=(9, 3.6))
for r in range(3):
    for c in range(8):
        img = X_tr[c] if r == 0 else aug(X_tr[c : c + 1], training=True).numpy()[0]
        axes[r, c].imshow(img[..., 0], cmap="gray_r", vmin=0, vmax=1)
        axes[r, c].axis("off")
    axes[r, 0].set_title("original" if r == 0 else f"augmented draw {r}", fontsize=8, loc="left")
fig.suptitle("Augmentation makes new plausible variants of the same digits")
plt.tight_layout()
plt.show()

# %% [markdown]
# On 8x8 images one pixel is 12.5% of the width, so even `0.08` (about half a pixel, with interpolation) is a visible change; we kept the ranges small. `RandomRotation(0.03)` means a fraction of a full turn, roughly +/-11 degrees. Bigger images tolerate
# proportionally bigger ranges. Now train the augmented CNN. Augmentation acts as regularisation, so it usually needs **more epochs** to converge;
# we keep the same cap of 40 epochs with early stopping.

# %%
keras.utils.set_random_seed(42)
cnn_aug = build_cnn(augment=True)
train(cnn_aug, epochs=40, patience=5)
print(f"CNN + augmentation validation accuracy: {accuracy(cnn_aug, X_va, y_va):.3f}   (plain CNN: {accuracy(cnn, X_va, y_va):.3f})")

# %% [markdown]
# ## 4. Is the gain real? Repeat over seeds
#
# One training run is one draw from a random process (initial weights, batch order, augmentation). To compare two recipes, train each several
# times and compare the **spread**, not one number. Each model below is tiny, so three seeds per recipe are affordable (an honest comparison needs more, see the exercise prompts).

# %%
def run_seeds(augment, seeds=(0, 1, 2), epochs=40):
    out = []
    for s in seeds:
        keras.utils.set_random_seed(s)
        m = build_cnn(augment=augment)
        train(m, epochs=epochs, patience=6)
        out.append(accuracy(m, X_va, y_va))
    return np.array(out)


res = {"CNN": run_seeds(False), "CNN + augmentation": run_seeds(True)}
summary = pd.DataFrame({k: {"mean val acc": v.mean(), "min": v.min(), "max": v.max()} for k, v in res.items()}).T.round(3)
summary

# %% [markdown]
# Read the `min` and `max` columns, not only the mean. If the ranges overlap heavily, the honest conclusion is "no reliable difference on this
# validation set". On data that is already centred, augmentation can cost a little accuracy (the model spends capacity on variation it will never see), so a flat or slightly negative result here is normal, not a bug. Augmentation pays off most when the training set is small relative to the variation you expect in production, and when
# deployment images are shifted or rotated compared with training. Here the digits are already centred, so the benefit is expected to be modest.
# A stronger test is to make deployment look different from training, which is exactly what we do next.

# %% [markdown]
# ### A shifted test: where augmentation should help
# Shift every validation image one pixel to the right (wrapping is avoided by filling with zeros) and score both models. The plain CNN never saw a shifted digit in training.

# %%
def shift_right(imgs, k=1):
    out = np.zeros_like(imgs)
    out[:, :, k:, :] = imgs[:, :, :-k, :]
    return out


X_va_shift = shift_right(X_va)
rows = []
for name, m in [("MLP", mlp), ("CNN", cnn), ("CNN + augmentation", cnn_aug)]:
    rows.append({"model": name, "accuracy (centred)": accuracy(m, X_va, y_va), "accuracy (shifted 1px)": accuracy(m, X_va_shift, y_va)})
pd.DataFrame(rows).set_index("model").round(3)

# %% [markdown]
# *What just happened?* We changed the deployment conditions and measured. A model's validation accuracy describes only data that looks like the
# validation set. Expect every model to lose some accuracy under the shift, and the augmented CNN to lose the least (check the table: if it does not, say so).

# %% [markdown]
# ## 5. Error analysis on the test set
#
# Accuracy hides *which* digits fail. We pick the augmented CNN, evaluate it on the test set **once**, and open up the mistakes.

# %%
proba = cnn_aug.predict(X_te, verbose=0)
pred = proba.argmax(axis=1)
print(f"test accuracy: {(pred == y_te).mean():.3f} on {len(y_te)} images -> {(pred != y_te).sum()} mistakes")

cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("predicted digit")
ax.set_ylabel("true digit")
ax.set_title("Confusion matrix on the test set (counts)")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, label="images")
plt.tight_layout()
plt.show()

# %%
off = cm.copy()
np.fill_diagonal(off, 0)
pairs = sorted(((int(off[i, j]), i, j) for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:5]
print("most frequent (count, true -> predicted):", [(n, f"{i}->{j}") for n, i, j in pairs])
per_class = cm.diagonal() / cm.sum(axis=1)
print("weakest class recall:", {int(d): round(float(per_class[d]), 2) for d in np.argsort(per_class)[:3]})

# %%
wrong = np.flatnonzero(pred != y_te)
show = wrong[np.argsort(-proba[wrong].max(axis=1))][:10]  # most confident mistakes first
fig, axes = plt.subplots(1, len(show), figsize=(1.3 * len(show), 2))
for ax, k in zip(np.atleast_1d(axes), show):
    ax.imshow(X_te[k, ..., 0], cmap="gray_r")
    ax.set_title(f"true {y_te[k]}\npred {pred[k]} ({proba[k].max():.0%})", fontsize=7)
    ax.axis("off")
fig.suptitle("Most confident mistakes (true digit, predicted digit, model confidence)", y=1.15, fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# Questions to ask of this picture: is the image ambiguous even to you (then the label noise is the ceiling, not the model)? Does the model make the
# *same* confusion repeatedly (then it is a systematic gap worth targeting with more data of that digit)? How confident is it when wrong? A model that is
# 95% sure and wrong is a bigger deployment risk than one that is 55% sure and wrong, because you cannot triage it with a confidence threshold.

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting the channel axis.** `Conv2D` wants `(n, h, w, 1)`, not `(n, h, w)`; the error message mentions "expected min_ndim=4".
# 2. **Not scaling pixels.** Raw 0-16 (or 0-255) inputs slow and destabilise training. Scale to 0-1 or put a `Rescaling` layer in the model.
# 3. **Augmenting the validation or test data.** Keras layers skip it automatically at inference, but if you augment with NumPy beforehand you contaminate the measurement. Evaluate on clean data.
# 4. **Label-changing augmentation.** Flipping digits or letters, rotating a "6" to a "9". Choose transformations that keep the label true.
# 5. **Comparing models on a single run.** The seed-to-seed spread can be as large as the difference you are trying to detect.
# 6. **Tuning on the test set while doing error analysis.** Looking at test mistakes is fine; changing the model to fix *those* mistakes and then re-scoring on the same test set is not. Use validation data for that loop.
# 7. **Using accuracy alone** when classes matter unequally. Read per-class recall and the confusion matrix.

# %%
# Demonstration of mistake 1: a missing channel axis
try:
    cnn.predict(X_te[:2, :, :, 0], verbose=0)
except ValueError as e:
    print("ValueError:", str(e).splitlines()[0][:110])

# %% [markdown]
# ## Interview corner
#
# **1. Why do CNNs have fewer parameters than a fully-connected network on images?**
# <details><summary>Answer</summary>
# Weight sharing and local connectivity: one small kernel is reused at every position, so the parameter count depends on the kernel size and the number of
# filters, not on the image size. A dense layer needs one weight per input pixel per unit.
# </details>
#
# **2. What does data augmentation do, and when can it hurt?**
# <details><summary>Answer</summary>
# It creates label-preserving variations of training examples, acting as a regulariser and encoding invariances you expect in production. It hurts when the
# transformation changes the label (flipping a 6), when it produces unrealistic inputs, or when it is so strong the model underfits.
# </details>
#
# **3. The model has 97% accuracy. What do you look at next?**
# <details><summary>Answer</summary>
# Per-class recall and the confusion matrix, the class balance, the confidence of the errors, and a sample of the misclassified inputs. Also whether the validation data
# resembles production (same lighting, resolution, shifts). Accuracy is an average that can hide one class that never works.
# </details>
#
# **4. Why can a model be less accurate on shifted images even though a convolution is "translation equivariant"?**
# <details><summary>Answer</summary>
# Convolution is equivariant (shifting the input shifts the feature map), but the final Flatten + Dense layer reads specific positions, so the whole network is only approximately
# invariant. Pooling and augmentation help but do not guarantee it; you verify it with a shifted test like the one above.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Predict the parameter count
# A `Conv2D(8, 5)` layer (8 filters, 5x5 kernels) is applied to 8x8 grayscale images. Store in `n_params` the number of parameters of that layer, computed by
# hand from the formula (no Keras call), and in `out_h` its output height (default `padding="valid"`). The check builds the real layer to verify.
# *Hint: each filter has kernel_h x kernel_w x input_channels weights plus one bias.*

# %% tags=["starter"]
n_params = ...
out_h = ...

# %%
probe = keras.Sequential([keras.layers.Input((8, 8, 1)), keras.layers.Conv2D(8, 5)])
assert n_params == probe.count_params(), f"Keras counts {probe.count_params()} parameters in this layer"
assert out_h == probe.output_shape[1], f"Keras output height is {probe.output_shape[1]}"
print("Exercise 1 passed")

# %% tags=["solution"]
n_params = 8 * (5 * 5 * 1) + 8
out_h = 8 - 5 + 1

# %% [markdown]
# ### Exercise 2: Which digit is hardest?
# Using the test-set predictions `pred` and labels `y_te` from section 5, store in `worst_digit` the digit class with the lowest **recall**
# (fraction of that digit's test images predicted correctly) and in `worst_recall` that recall. If there is a tie take the smallest digit.

# %% tags=["starter"]
worst_digit = ...
worst_recall = ...

# %%
recalls = {d: float((pred[y_te == d] == d).mean()) for d in range(10)}
exp_recall = min(recalls.values())
exp_digit = min(d for d in range(10) if recalls[d] == exp_recall)
assert worst_digit == exp_digit, f"expected digit {exp_digit}"
assert abs(worst_recall - exp_recall) < 1e-9, "recall is correct predictions / true images of that digit"
print(f"Exercise 2 passed (digit {worst_digit}, recall {worst_recall:.2f})")

# %% tags=["solution"]
recalls = {d: float((pred[y_te == d] == d).mean()) for d in range(10)}
worst_recall = min(recalls.values())
worst_digit = min(d for d in range(10) if recalls[d] == worst_recall)

# %% [markdown]
# ### Exercise 3: A stronger shift
# Write `shift_down(imgs, k)` that shifts images **down** by `k` pixels, filling with zeros (mirror the `shift_right` function). Store in `acc_down` the
# accuracy of `cnn_aug` on `X_va` shifted down by 2 pixels, and in `acc_down_plain` the accuracy of the plain `cnn` on the same images.
# Which model is more robust? Both numbers are computed, the comparison is yours to read.

# %% tags=["starter"]
def shift_down(imgs, k):
    ...


acc_down = ...
acc_down_plain = ...

# %%
manual = np.zeros_like(X_va)
manual[:, 2:, :, :] = X_va[:, :-2, :, :]
assert np.allclose(shift_down(X_va, 2), manual), "shift_down should move every image down by k rows and fill the top rows with zeros"
assert abs(acc_down - accuracy(cnn_aug, manual, y_va)) < 1e-9, "acc_down: use cnn_aug on the shifted validation images"
assert abs(acc_down_plain - accuracy(cnn, manual, y_va)) < 1e-9, "acc_down_plain: use the plain cnn"
print(f"Exercise 3 passed (augmented {acc_down:.3f} vs plain {acc_down_plain:.3f})")

# %% tags=["solution"]
def shift_down(imgs, k):
    out = np.zeros_like(imgs)
    out[:, k:, :, :] = imgs[:, :-k, :, :]
    return out


Xd = shift_down(X_va, 2)
acc_down = accuracy(cnn_aug, Xd, y_va)
acc_down_plain = accuracy(cnn, Xd, y_va)

# %% [markdown]
# ### Exercise 4: Add zoom augmentation
# Build a CNN like `build_cnn(augment=True)` but with an extra `keras.layers.RandomZoom(0.1)` placed after the rotation layer. Name it `zoom_model`, compile it with
# `compile_model`, train it with `train(zoom_model, epochs=40, patience=5)` after `keras.utils.set_random_seed(42)`, and store its validation accuracy in `zoom_acc`.
# *Hint: copy `build_cnn` and add the layer to the list.*

# %% tags=["starter"]
zoom_model = ...
zoom_acc = ...

# %%
assert any(isinstance(l, keras.layers.RandomZoom) for l in zoom_model.layers), "the model needs a RandomZoom layer"
assert abs(zoom_acc - accuracy(zoom_model, X_va, y_va)) < 1e-9, "zoom_acc must be the validation accuracy of zoom_model"
assert zoom_acc > 0.85, "a reasonable CNN should clear 85% on digits; check that you trained it"
print(f"Exercise 4 passed (validation accuracy {zoom_acc:.3f})")

# %% tags=["solution"]
keras.utils.set_random_seed(42)
zoom_model = compile_model(keras.Sequential([
    keras.layers.Input((8, 8, 1)),
    keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
    keras.layers.RandomRotation(0.03, fill_mode="constant"),
    keras.layers.RandomZoom(0.1),
    keras.layers.Conv2D(16, 3, activation="relu"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation="softmax"),
]))
train(zoom_model, epochs=40, patience=5)
zoom_acc = accuracy(zoom_model, X_va, y_va)

# %% [markdown]
# ### Exercise 5 (open-ended): Write the failure report
# Look at the most-confident-mistakes picture and the top confused pairs. In a markdown cell write three sentences as if for a teammate: which digits get confused, whether a
# human would also hesitate on those images, and one concrete change you would try next (more data of a class, different augmentation, a bigger input resolution) with how you would measure it.

# %% [markdown]
# ## Recap
# - A CNN reuses small kernels at every position: fewer parameters than an MLP and a built-in notion of locality. Predict the shapes by hand, then confirm with `summary()`.
# - Augmentation layers randomise only during training; choose transformations that keep the label true and measure the effect over several seeds.
# - Compare on the same split, report spread not a single run, and test under the conditions you expect in deployment (here: shifted images).
# - Error analysis (confusion matrix, most confused pairs, confident mistakes, the images themselves) tells you what to do next; accuracy alone does not.
#
# **Go deeper:** the Keras guides "Image classification from scratch" and "Data augmentation" (keras.io), and the `keras.layers` API reference for the preprocessing/augmentation layers.
#
# **Next:** [Text classification with Keras](04_text_classification_with_keras.ipynb)
