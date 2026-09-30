# 15 - TensorFlow

**Goal:** understand TensorFlow below the Keras layer: tensors and variables, automatic differentiation with `GradientTape`, fast input pipelines with `tf.data`,
a hand-written training loop compiled with `tf.function`, and saving models for serving. All notebooks run on CPU.

**Prerequisites:** `13_Deep_Learning` (forward pass and backpropagation), `04_NumPy`, basic Python functions and decorators.

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_tensors_variables_and_gradienttape.ipynb` | tensors vs NumPy, `tf.Variable`, gradients with `GradientTape` (and when they silently come back `None`), linear regression by hand | 1.5 |
| 02 | `02_tf_data_input_pipelines.ipynb` | `tf.data`: map, shuffle, batch, cache, prefetch; why station order matters; TFRecord basics | 1.5 |
| 03 | `03_custom_training_loop_and_tf_function.ipynb` | eager vs graph, tracing and retracing, `tf.print`, a full custom training loop | 2 |
| 04 | `04_saving_savedmodel_and_serving_concepts.ipynb` | `.keras` files, weights-only saving, preprocessing inside the model, `model.export()` and signatures, serving concepts (a few cells are display-only) | 1.5 |

## You are done when you can...
- [ ] explain how a tensor differs from a NumPy array and when a `tf.Variable` is needed
- [ ] compute a gradient with `GradientTape` and name the ways it returns `None`
- [ ] build an input pipeline with shuffle, batch and prefetch, and say why the order of the steps matters
- [ ] write a training step as a `tf.function` and explain tracing, retracing and why Python side effects run only while tracing
- [ ] save and reload a model, export a SavedModel, and describe what a serving signature is
- [ ] choose between a Keras `fit` call and a custom loop for a given problem

## Try next (real public resources)
- The TensorFlow guides at tensorflow.org/guide (for example "Introduction to gradients and automatic differentiation" and "Better performance with tf.function")
- The `tf.data` guide "Build TensorFlow input pipelines" (tensorflow.org/guide/data)
- Keras documentation and guides at keras.io (next module: `16_Keras`)
- *Deep Learning with Python* by Francois Chollet (Manning)
