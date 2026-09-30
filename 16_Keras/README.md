# 16 - Keras

**Goal:** build, train, regularise and evaluate neural networks with Keras 3: the Sequential and Functional APIs, callbacks, a CNN with augmentation and error analysis,
and a text classifier compared honestly against a TF-IDF baseline. All notebooks run on CPU with the TensorFlow backend.

**Prerequisites:** `15_TensorFlow`, `13_Deep_Learning`, and scikit-learn basics from `12_Machine_Learning` (splits, metrics, baselines).

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_sequential_and_functional_api.ipynb` | housing regression and churn classifier, a two-input Functional model, `summary()`, baselines, backend selection | 2 |
| 02 | `02_callbacks_regularization_and_tuning.ipynb` | overfit on purpose, `EarlyStopping` / `ModelCheckpoint`, dropout and L2, a small honest grid search | 2 |
| 03 | `03_cnn_on_digits_with_keras.ipynb` | CNN vs MLP on digits, augmentation layers, seed-to-seed spread, a shifted test, confusion matrix and error analysis | 2 |
| 04 | `04_text_classification_with_keras.ipynb` | `TextVectorization` + embedding model vs TF-IDF + logistic regression, leak-free text split, Keras 3 multi-backend explained | 1.5 |

## You are done when you can...
- [ ] choose between `Sequential` and the Functional API, and read `model.summary()` to catch a wiring bug
- [ ] compare a network with a simple baseline before believing it
- [ ] recognise overfitting from training and validation curves and apply early stopping, dropout or L2, then verify that they helped
- [ ] select hyper-parameters on validation data, report the test score once, and explain why small differences between runs may be noise
- [ ] predict the output shape and parameter count of `Conv2D`, pooling and dense layers by hand
- [ ] use augmentation layers sensibly, and test a model under the conditions you expect in deployment
- [ ] do error analysis (confusion matrix, most-confused pairs, the actual mistakes) instead of reading only accuracy
- [ ] explain what `KERAS_BACKEND` does and what stays portable across backends

## Try next (real public resources)
- Keras developer guides and code examples at keras.io (for example "Image classification from scratch" and "Text classification from scratch")
- `keras-tuner` (a separate package, not installed here) for larger hyper-parameter searches
- scikit-learn bundled datasets and the UCI Machine Learning Repository (archive.ics.uci.edu) for practice data
- *Deep Learning with Python* by Francois Chollet (Manning)
