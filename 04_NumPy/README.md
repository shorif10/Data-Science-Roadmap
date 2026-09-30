# 04 NumPy

**Goal.** Think in arrays: create them, select from them, combine them with broadcasting, and know the silent failure modes
(overflow, NaN, views vs copies) that cause real bugs. NumPy is the foundation under pandas, scikit-learn, PyTorch and TensorFlow.

**Prerequisites.** Python basics (`../01_Python`). A little linear algebra helps for notebook 04 (`../02_Math_for_Data_Science`).

## Notebooks (about 4 hours in total)

| # | Notebook | What you do | Time |
|---|---|---|---|
| 1 | [01_ndarray_basics](01_ndarray_basics.ipynb) | Create arrays, read `dtype`/`shape`, reshape, master `axis`, stack and split | 40 min |
| 2 | [02_indexing_slicing_masking](02_indexing_slicing_masking.ipynb) | Slice, mask, fancy-index; prove view vs copy with `np.shares_memory` | 40 min |
| 3 | [03_broadcasting_and_vectorization](03_broadcasting_and_vectorization.ipynb) | Broadcasting rules, z-scores, pairwise distances, a fair loop-vs-NumPy benchmark | 40 min |
| 4 | [04_linear_algebra_random_and_statistics](04_linear_algebra_random_and_statistics.ipynb) | `@`, `solve`, `lstsq`, `default_rng` simulations, percentiles, `np.select` | 45 min |
| 5 | [05_numpy_pitfalls_and_performance](05_numpy_pitfalls_and_performance.ipynb) | Overflow, float error, NaN, memory, in-place ops, strides | 40 min |
| 6 | [06_numpy_mini_project](06_numpy_mini_project.ipynb) | Digit images: normalise, flip, hand-written convolution, k-means from scratch | 60 min |

Run them in order from JupyterLab; each notebook uses its own folder as working directory.

## You are done when you can...

- [ ] predict the output `shape` of an operation (reductions with `axis`, `keepdims`, broadcasting) before running it
- [ ] say whether a given indexing expression returns a view or a copy, and verify it with `np.shares_memory`
- [ ] replace a Python loop over numbers with an array expression and benchmark both fairly
- [ ] standardise a feature matrix and compute pairwise distances without loops
- [ ] fit a least-squares line with `np.linalg.lstsq` and explain why `solve` beats `inv`
- [ ] generate reproducible random data with `np.random.default_rng(seed)`
- [ ] spot and fix `uint8` overflow, float comparison with `==`, and NaN-poisoned means
- [ ] implement k-means (assign, update, restarts) using only NumPy

## Try next (real, public resources)

- scikit-learn bundled datasets (`load_iris`, `load_wine`, `load_breast_cancer`, `load_diabetes`): load them as arrays and repeat the notebook 3 and 4 tricks.
- The official NumPy tutorials and user guide at numpy.org/doc/stable/user.
- UCI Machine Learning Repository (archive.ics.uci.edu): pick a numeric dataset and compute per-column statistics with NumPy only.
- Kaggle "Digit Recognizer" (MNIST): the same ideas as notebook 6 on 28x28 images.
- "100 NumPy exercises" community collection on GitHub (rougier/numpy-100) for extra practice.

Next module: [05 Pandas](../05_Pandas/01_loading_and_inspecting_data.ipynb)
