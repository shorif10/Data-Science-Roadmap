# 02 - Math for Data Science

**Goal:** build the intuition for the linear algebra, calculus and information theory that machine learning uses, code-first and visual, so that `X @ w`, gradients, PCA and cross-entropy stop being black boxes.

**Prerequisites:** Python basics and NumPy (`01_Python`, `04_NumPy`). No calculus background assumed.

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 1 | [01_vectors_and_matrices](01_vectors_and_matrices.ipynb) | dot product as similarity, norms, matrices as transformations, `X @ w` on housing data | 1 |
| 2 | [02_matrix_decompositions_eigen_svd](02_matrix_decompositions_eigen_svd.ipynb) | eigenvectors, SVD low-rank approximation, PCA preview on digits | 1.5 |
| 3 | [03_calculus_derivatives_and_gradients](03_calculus_derivatives_and_gradients.ipynb) | numerical and symbolic derivatives, gradients, chain rule, gradient checking | 1.5 |
| 4 | [04_gradient_descent_from_scratch](04_gradient_descent_from_scratch.ipynb) | fit price from area by hand, learning rates, feature scaling | 1.5 |
| 5 | [05_loss_functions_and_optimization](05_loss_functions_and_optimization.ipynb) | MSE/MAE/Huber, log-loss/hinge, convexity, momentum and Adam | 1.5 |
| 6 | [06_probability_and_information_theory_basics](06_probability_and_information_theory_basics.ipynb) | entropy, cross-entropy, KL, information gain for trees | 1.5 |

Total: about 8 hours.

## You are done when you can...

- explain the dot product and cosine similarity and pick between L1, L2 and cosine distance
- read `X @ w` and predict the shapes involved
- describe what an eigenvector and a singular value are, and what PCA computes
- verify a hand-derived gradient with a numerical gradient check
- implement gradient descent for linear regression and diagnose a diverging learning rate
- explain why MSE targets the mean, MAE the median, and why accuracy is not used as a training loss
- compute entropy, cross-entropy and information gain and connect log-loss to maximum likelihood

## Try next (real public resources)

- scikit-learn bundled datasets (`load_iris`, `load_wine`, `load_breast_cancer`): run PCA and plot explained variance
- UCI Machine Learning Repository (archive.ics.uci.edu): try gradient descent on a real regression dataset
- NumPy documentation, `numpy.linalg` reference
- PyTorch tutorial "A Gentle Introduction to torch.autograd" (pytorch.org/tutorials)

**Next module:** [03_Statistics_and_Probability](../03_Statistics_and_Probability/README.md)
