# 14 - PyTorch

**Goal:** turn the from-scratch understanding of module 13 into idiomatic PyTorch: tensors, autograd, `nn.Module`, `Dataset`/`DataLoader`,
CNNs, sequence models, checkpointing and transfer learning. All notebooks run on CPU in seconds to a couple of minutes.

**Prerequisites:** `13_Deep_Learning` (especially 02 backprop and 05 debugging), pandas and scikit-learn basics (`12_Machine_Learning`).

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_tensors_and_autograd.ipynb` | tensors, broadcasting, NumPy interop, autograd, `gradcheck` | 1 |
| 02 | `02_nn_module_and_training_loop.ipynb` | housing price regression, scaling, the training loop, loss curves | 1.5 |
| 03 | `03_classification_with_dataset_and_dataloader.ipynb` | churn MLP with a custom `Dataset`, compared honestly with sklearn baselines | 1.5 |
| 04 | `04_cnn_on_digits.ipynb` | conv net on digits, confusion matrix, misclassified examples | 1 |
| 05 | `05_sequence_models_rnn_lstm.ipynb` | LSTM on sensor data vs a naive baseline, leak-free windowing | 1.5 |
| 06 | `06_saving_loading_reproducibility_and_devices.ipynb` | `state_dict`, checkpoints, seeds, determinism, CPU/GPU-safe code | 1 |
| 07 | `07_transfer_learning_and_pretrained_models.ipynb` | freezing and fine-tuning; torchvision pretrained code (not runnable offline) | 1 |

## You are done when you can...
- [ ] explain `requires_grad`, `backward()`, `zero_grad()` and `no_grad()` and why gradients accumulate
- [ ] write a training loop (forward, loss, zero_grad, backward, step) plus a validation loop with `model.eval()`
- [ ] build a `Dataset` and `DataLoader` for tabular data and fit the scaler on the training split only
- [ ] choose the right loss for regression, binary and multi-class problems (and why logits go into `BCEWithLogitsLoss`)
- [ ] window a time series without leaking the future and compare a model against a naive baseline
- [ ] save and restore a model plus optimizer state, and make a run reproducible
- [ ] freeze a backbone, fine-tune a new head, and explain when to use a lower learning rate for the backbone
- [ ] say when a tabular problem is better served by gradient boosting than by a neural network

## Try next (real public resources)
- torchvision datasets such as MNIST and CIFAR-10 (downloaded once, on your own machine)
- UCI Machine Learning Repository (archive.ics.uci.edu) and Kaggle tabular competitions (kaggle.com)
- Official PyTorch tutorials and docs (pytorch.org/tutorials, pytorch.org/docs)
- scikit-learn bundled datasets for quick experiments (`load_digits`, `load_diabetes`)
