# 13 - Deep Learning (from scratch, framework-agnostic)

**Goal:** understand what a neural network computes, how it learns, and how to debug it, using only NumPy
(plus a little PyTorch for cross-checks). After this module, `nn.Module` and `model.fit()` stop being magic.

**Prerequisites:** NumPy, basic calculus and matrices (`02_Math_for_Data_Science`), and the ML module's train/validation/test ideas (`12_Machine_Learning`).

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_neurons_layers_and_the_forward_pass.ipynb` | perceptron, activations, XOR, why non-linearity matters, batched forward pass, parameter counting | 1 |
| 02 | `02_backpropagation_from_scratch.ipynb` | hand-derived gradients for an MLP, numerical gradient checking, train on digits | 1.5 |
| 03 | `03_training_dynamics_optimizers_and_regularization.ipynb` | initialisation, vanishing gradients, SGD/momentum/Adam, dropout, weight decay, early stopping, LR finder | 1.5 |
| 04 | `04_convolutions_and_cnn_intuition.ipynb` | convolution by hand, filters, pooling, output sizes, conv vs dense parameter counts | 1 |
| 05 | `05_deep_learning_debugging_checklist.ipynb` | overfit-one-batch, loss-not-decreasing playbook, shape errors, leakage, exploding loss | 1 |

## You are done when you can...
- [ ] compute the output shape and parameter count of a stack of dense or conv layers without running code
- [ ] explain why two linear layers equal one, and show XOR needs a hidden layer
- [ ] write the backward pass of a 2-layer softmax network and verify it with gradient checking
- [ ] explain vanishing gradients and pick an initialisation and activation that avoid them
- [ ] say what momentum and Adam change relative to plain SGD, and when weight decay, dropout or early stopping help
- [ ] run the overfit-one-batch test and follow a checklist when a loss will not go down
- [ ] spot data leakage (scaling before splitting, duplicates across splits) in a training script

## Try next (real public resources)
- scikit-learn bundled `load_digits`, `load_breast_cancer`, `load_wine` (no download)
- UCI Machine Learning Repository (archive.ics.uci.edu)
- Kaggle Datasets and "Getting Started" competitions (kaggle.com)
- The PyTorch tutorials (pytorch.org/tutorials) and Stanford CS231n notes (cs231n.github.io)

Next module: [14_PyTorch](../14_PyTorch/README.md)
