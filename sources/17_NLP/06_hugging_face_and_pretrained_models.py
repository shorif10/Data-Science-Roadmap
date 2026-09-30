# %% [markdown]
# # Hugging Face and Pretrained Models
#
# **Why this matters.** In a real NLP job you will almost never train a language model from scratch. You will download a pretrained one, decide *how* to use it (prompt it, extract features from it, or fine-tune it), and then prove
# with an honest evaluation that it beats the simple baseline you built in notebook 02. The Hugging Face `transformers` library and the Hub are the de-facto standard for that workflow. This notebook teaches the concepts and the
# exact API, and, because this course runs **offline**, it gives you a **ready-made evaluation harness** so that when you do have internet access you can plug a real model in and get a trustworthy comparison in one line.
#
# **You will learn**
# - what a tokenizer, a model, a config and a model card are, and how `pipeline`, `AutoTokenizer` and `AutoModel*` fit together
# - read tokenizer output (`input_ids`, `attention_mask`, padding, truncation) and reproduce the core of WordPiece tokenisation yourself
# - choose between **prompting**, **feature extraction** and **fine-tuning** for a given amount of data, cost and latency budget
# - evaluate *any* classifier function (scikit-learn pipeline, keyword rule, Hugging Face pipeline) on the same held-out reviews with one harness
#
# **Prerequisites:** [Attention and transformers from scratch](05_attention_and_transformers_from_scratch.ipynb), [Bag of words, TF-IDF and classifiers](02_bag_of_words_tfidf_and_classifiers.ipynb).
#
# > **Offline note.** The `transformers` package is not installed in this environment and downloading model weights needs the internet, so the cells that call Hugging Face are marked `no-run` and are shown for you to run later.
# > Everything else in the notebook (the tokenizer demo, the harness, the output-format adapter) runs offline, and the adapter is tested against a clearly labelled **mock** that imitates the library's output format.
# > Data used: `datasets/reviews.csv` (synthetic, templated; `docs/DATASETS.md`).

# %%
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

reviews = pd.read_csv(Path("../datasets/reviews.csv"))
groups = pd.factorize(reviews["review_text"])[0]
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(reviews, groups=groups))
train_df, test_df = reviews.iloc[tr].reset_index(drop=True), reviews.iloc[te].reset_index(drop=True)
LABELS = ["negative", "neutral", "positive"]
print(f"train {len(train_df)} rows | test {len(test_df)} rows ({test_df['review_text'].nunique()} distinct texts) | no text shared: {not set(train_df['review_text']) & set(test_df['review_text'])}")

# %% [markdown]
# ## 1. The pieces of the Hugging Face ecosystem
# | Piece | What it is |
# |---|---|
# | **Hub** | A website/registry hosting hundreds of thousands of models, datasets and demos, each in a versioned git repository. |
# | **Model** | The network weights (`model.safetensors`) plus a `config.json` (architecture, number of labels, label names). |
# | **Tokenizer** | The exact text-to-ids procedure the model was trained with (vocabulary, merges/pieces, special tokens). Always load the tokenizer that belongs to the model. |
# | **Model card** | The README of the repository: intended use, training data, evaluation results, limitations, licence. Read it before you trust a model. |
# | **`pipeline`** | A convenience wrapper: tokenizer + model + post-processing for a task (`text-classification`, `zero-shot-classification`, `feature-extraction`, ...). |
# | **`AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`** | Classes that read `config.json` and instantiate the right architecture for you. `AutoModel` returns hidden states; the task-specific variants add a head. |
#
# Models come with different **heads**: a base encoder such as BERT outputs one vector per token (good for features); `...ForSequenceClassification` adds a small linear layer on top to output class scores.

# %% [markdown]
# ## 2. Tokenizers: what goes into the model
# A transformer never sees strings. A tokenizer converts text to `input_ids`, adds special tokens (for BERT-style models `[CLS]` at the start and `[SEP]` at the end), pads a batch to equal length, and builds an
# **`attention_mask`** (1 = real token, 0 = padding) which is exactly the padding mask from notebook 05. Most tokenizers split words into **sub-word pieces**: BERT-style models use **WordPiece** (continuation pieces are
# written with a `##` prefix), GPT-style models use byte-level BPE (notebook 01). The real call looks like this (run it later with internet access):

# %% tags=["no-run"]
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
batch = tok(
    ["Great quality, arrived early.", "Terrible. It stopped working after a week and support never replied."],
    padding=True,  # pad to the longest text in the batch
    truncation=True,  # cut anything longer than max_length
    max_length=64,
    return_tensors="pt",  # PyTorch tensors ("tf" for TensorFlow, "np" for NumPy)
)
print(batch["input_ids"].shape, batch["attention_mask"])
print(tok.convert_ids_to_tokens(batch["input_ids"][0]))  # see the [CLS], [SEP] and ## pieces

# %% [markdown]
# Since we cannot download a vocabulary here, let us implement the heart of WordPiece ourselves on a toy vocabulary. The rule is **greedy longest match**: take the longest vocabulary entry that starts the word,
# then continue from there with `##`-prefixed pieces; if some position matches nothing, the whole word becomes `[UNK]`.

# %%
VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "great": 4, "quality": 5, "un": 6, "##believ": 7, "##able": 8, "##ably": 9,
         "support": 10, "never": 11, "replied": 12, "reply": 13, "##ed": 14, "##ing": 15, "the": 16, "broke": 17, "after": 18, "a": 19, "week": 20}


def wordpiece(word, vocab):
    pieces, start = [], 0
    while start < len(word):
        end, match = len(word), None
        while end > start:  # try the longest candidate first, then shorten
            cand = word[start:end] if start == 0 else "##" + word[start:end]
            if cand in vocab:
                match = cand
                break
            end -= 1
        if match is None:
            return ["[UNK]"]  # any unmatched stretch makes the whole word unknown
        pieces.append(match)
        start = end
    return pieces


for w in ["great", "unbelievable", "unbelievably", "replied", "replying", "xyzzy"]:
    print(f"{w:13s} -> {wordpiece(w, VOCAB)}")

# %% [markdown]
# `unbelievable` was never a vocabulary word, yet it is covered by `un + ##believ + ##able`; `replying` is `reply + ##ing`; a string made of letters the vocabulary cannot cover becomes `[UNK]`.
# Now build a padded batch with an attention mask, the same structure the library returns:

# %%
def encode_batch(texts, vocab, max_length=16):
    seqs = []
    for t in texts:
        toks = ["[CLS]"] + [p for w in re.findall(r"[a-z]+", t.lower()) for p in wordpiece(w, vocab)] + ["[SEP]"]
        seqs.append([vocab[p] for p in toks][:max_length])
    longest = max(len(s) for s in seqs)
    ids = np.array([s + [vocab["[PAD]"]] * (longest - len(s)) for s in seqs])
    mask = np.array([[1] * len(s) + [0] * (longest - len(s)) for s in seqs])
    return {"input_ids": ids, "attention_mask": mask}


enc = encode_batch(["Great quality!", "Unbelievably, support never replied"], VOCAB)
print(enc["input_ids"])
print(enc["attention_mask"])

# %% [markdown]
# *What just happened?* Two texts of different lengths became one rectangular array, and `attention_mask` records which cells are real. The model uses the mask so that padding never receives attention.
# The practical consequences: there is a **maximum length** (often 512 tokens for BERT-style models; longer text is truncated, which can silently drop the part that mattered), the same word can cost several tokens (cost and speed scale with *tokens*, not words),
# and using the **wrong tokenizer for a model** produces garbage without raising an error.

# %% [markdown]
# ## 3. Three ways to use a pretrained model
# | Approach | What you do | Needs labelled data | Cost | When it fits |
# |---|---|---|---|---|
# | **Prompt / zero-shot** | Use the model as is: a zero-shot pipeline with candidate labels, or an LLM with instructions and examples (notebooks in `21_LLMs_and_RAG`) | none to a few examples | lowest to set up; per-call cost or compute at run time | Quick start, changing label sets, no training data |
# | **Feature extraction** | Run texts through the frozen model to get embeddings, train a small classifier (logistic regression) on top | hundreds to thousands | one forward pass per text, then a cheap model | Small labelled set, need speed of iteration, semantic similarity and search |
# | **Fine-tuning** | Continue training all (or some) weights on your labelled data | thousands or more (fewer with good base models) | GPU hours, more engineering | Best accuracy on a specific task when you have data; a stable label set |
#
# A sensible order in practice: (1) baseline with TF-IDF + logistic regression, (2) zero-shot or feature extraction to see how much a pretrained model helps, (3) fine-tune only if the gap justifies the cost, and (4) check latency and
# cost against the requirement. The code below shows each route; all cells need downloads, so they are `no-run`.

# %% tags=["no-run"]
# Route 1: the one-line pipeline (downloads the model on first use, then caches it)
from transformers import pipeline

clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(clf(["Great quality, arrived early.", "Stopped working after a week."]))
# typical output: [{'label': 'POSITIVE', 'score': 0.99...}, {'label': 'NEGATIVE', 'score': 0.99...}]

# Zero-shot: no training data, you supply the labels at call time
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zs("The battery dies quickly", candidate_labels=["battery", "shipping", "price"]))

# %% tags=["no-run"]
# Route 2: the same thing with explicit Auto classes (what pipeline does internally)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
model.eval()
inputs = tok(["Great quality, arrived early."], return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits  # shape (batch, num_labels)
probs = torch.softmax(logits, dim=-1)
print({model.config.id2label[i]: float(p) for i, p in enumerate(probs[0])})

# %% tags=["no-run"]
# Route 3: feature extraction. Mean-pool the token vectors (ignoring padding) and train a small classifier on top.
import torch
from transformers import AutoModel, AutoTokenizer

name = "sentence-transformers/all-MiniLM-L6-v2"  # a compact encoder trained to produce useful sentence embeddings
tok, enc_model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)
enc_model.eval()


def embed_texts(texts, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            hidden = enc_model(**b).last_hidden_state  # (batch, tokens, dim)
        m = b["attention_mask"].unsqueeze(-1).float()  # 1 for real tokens, 0 for padding
        out.append(((hidden * m).sum(1) / m.sum(1)).numpy())  # mean over real tokens only
    return np.vstack(out)


# X_train = embed_texts(train_df["review_text"].tolist()); then LogisticRegression().fit(X_train, train_df["sentiment"])

# %% tags=["no-run"]
# Route 4: fine-tuning with the Trainer API (sketch; needs a GPU to be pleasant, and the `datasets`/`evaluate` packages are optional helpers)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.enc = tok(df["review_text"].tolist(), truncation=True, padding=True, max_length=64)
        self.labels = df["sentiment"].map({c: i for i, c in enumerate(LABELS)}).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


args = TrainingArguments(output_dir="finetune-out", num_train_epochs=2, per_device_train_batch_size=16, learning_rate=2e-5, eval_strategy="epoch", seed=42)
# note: older versions of transformers call the argument `evaluation_strategy`; check your installed version's documentation
trainer = Trainer(model=model, args=args, train_dataset=ReviewDataset(train_df), eval_dataset=ReviewDataset(test_df))
# trainer.train()   # then write the predictions into the harness below; never pick epochs or hyper-parameters on the test rows

# %% [markdown]
# **Practical notes for when you run these.**
# - **Caching and offline use.** Weights are cached (by default under `~/.cache/huggingface`, configurable with the `HF_HOME` environment variable). Set `HF_HUB_OFFLINE=1` or pass `local_files_only=True` to forbid network access once a model is cached.
# - **Reproducibility.** A model id like `org/name` tracks the repository's latest commit; pass `revision="<commit hash>"` to `from_pretrained` to pin the exact weights you evaluated.
# - **Safety.** Prefer repositories that ship `safetensors` weights, and treat `trust_remote_code=True` as running a stranger's code on your machine: only with repositories you have reviewed.
# - **Hardware.** `pipeline(..., device=0)` uses the first GPU; on CPU, small distilled models (DistilBERT, MiniLM) are the practical choice.

# %% [markdown]
# ## 4. Reading a model card before you trust a model
# The Hub page of a model contains its model card. Before using it, answer:
# 1. **What task and language was it trained for?** A classifier trained on English movie reviews is not a product-review classifier for German.
# 2. **What data?** Domain shift is the usual reason a strong published model underperforms on your data. (For instance, the sentiment model used above was tuned on short movie-review sentences with two labels: no neutral class!)
# 3. **Which metrics, on which test set?** Are they comparable to yours?
# 4. **Licence and limitations.** Commercial use allowed? Known biases or failure modes listed?
# 5. **Size and latency.** Parameters, expected speed on your hardware.
# The lesson is the one from every other notebook: the published score describes *their* data. Yours is the only number that counts, and that is what the harness below is for.

# %% [markdown]
# ## 5. A harness that evaluates any classifier function
# Design goal: a model is just a **function** `predict_fn(list_of_texts) -> list_of_labels`. The harness does everything else identically for every candidate: validates the output, computes accuracy, macro-F1, per-class precision/recall,
# the confusion matrix and the time per 1,000 texts. That makes comparisons fair, and swapping in a pretrained model later means writing one adapter function.

# %%
def evaluate_classifier(predict_fn, texts, labels, label_names=LABELS, name="model"):
    texts = list(texts)
    t0 = time.perf_counter()
    preds = list(predict_fn(texts))
    elapsed = time.perf_counter() - t0
    if len(preds) != len(texts):
        raise ValueError(f"{name}: predict_fn returned {len(preds)} predictions for {len(texts)} texts")
    unknown = sorted(set(preds) - set(label_names))
    if unknown:
        raise ValueError(f"{name}: predictions contain labels outside {label_names}: {unknown[:5]}")
    labels = list(labels)
    p, r, f, support = precision_recall_fscore_support(labels, preds, labels=label_names, zero_division=0)
    return {
        "name": name,
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", labels=label_names, zero_division=0),
        "per_class": pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": support}, index=label_names).round(3),
        "confusion": pd.DataFrame(confusion_matrix(labels, preds, labels=label_names), index=[f"true {c}" for c in label_names], columns=[f"pred {c}" for c in label_names]),
        "seconds_per_1000": 1000 * elapsed / max(len(texts), 1),
        "errors": pd.DataFrame({"text": texts, "true": labels, "predicted": preds}).query("true != predicted"),
    }


def compare(results):
    return pd.DataFrame([{"model": r["name"], "accuracy": r["accuracy"], "macro-F1": r["macro_f1"], "s / 1000 texts": r["seconds_per_1000"], "errors": len(r["errors"])} for r in results]).set_index("model").round(3)


# Tests for the harness itself: an oracle must score 1.0, a wrong-length or out-of-vocabulary answer must be rejected
lookup = dict(zip(test_df["review_text"], test_df["sentiment"]))
oracle = evaluate_classifier(lambda ts: [lookup[t] for t in ts], test_df["review_text"], test_df["sentiment"], name="oracle")
print("oracle accuracy:", oracle["accuracy"], "| macro-F1:", oracle["macro_f1"])
for bad in (lambda ts: ts[:-1], lambda ts: ["happy"] * len(ts)):
    try:
        evaluate_classifier(bad, test_df["review_text"], test_df["sentiment"])
    except ValueError as e:
        print("rejected:", str(e)[:75])

# %% [markdown]
# A harness you have tested on a known answer (an oracle scores exactly 1.0) and on wrong outputs is worth far more than one you merely trust. Now three candidates, all plain functions:

# %%
POS = {"great", "love", "excellent", "fantastic", "happy", "perfectly", "helpful", "solid", "early", "fair"}
NEG = {"terrible", "disappointed", "awful", "broke", "not", "damaged", "never", "stopped", "dies", "worth"}


def keyword_rule(texts):
    out = []
    for t in texts:
        words = re.findall(r"[a-z]+", t.lower())
        first = words[: len(words) // 2 + 1]  # the opinion is usually stated first in these reviews; an assumption about THIS data
        score = sum(w in POS for w in first) - sum(w in NEG for w in first)
        out.append("positive" if score > 0 else "negative" if score < 0 else "neutral")
    return out


tfidf_lr = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)).fit(train_df["review_text"], train_df["sentiment"])
majority = train_df["sentiment"].mode()[0]

candidates = [
    evaluate_classifier(lambda ts: [majority] * len(ts), test_df["review_text"], test_df["sentiment"], name="majority class"),
    evaluate_classifier(keyword_rule, test_df["review_text"], test_df["sentiment"], name="keyword rule (hand-written)"),
    evaluate_classifier(tfidf_lr.predict, test_df["review_text"], test_df["sentiment"], name="TF-IDF + logistic regression"),
]
compare(candidates)

# %% [markdown]
# Three things to read from this table. (1) The hand-written rule is decent on templated text, which tells you how easy the data is. (2) The TF-IDF model is essentially perfect, so **on this dataset a pretrained
# model has almost no head-room**: any honest comparison will end up "same as the baseline, slower". That is a property of the synthetic data, not of pretrained models; on real, varied text the gap can be large, which is
# why you run the harness on *your* data. (3) The time column matters: a model that is 1,000 times slower must be worth 1,000 times more to you.
#
# Look at where the rule fails, exactly as in a real error analysis:

# %%
keyword_errors = candidates[1]["errors"]
print(candidates[1]["confusion"])
keyword_errors.drop_duplicates("text").head(5)

# %% [markdown]
# ## 6. Plugging in a Hugging Face pipeline
# A Hugging Face text-classification pipeline returns a list of dictionaries such as `{"label": "POSITIVE", "score": 0.998}`. Our labels are `negative / neutral / positive`, and a two-label model like the SST-2
# sentiment model above has **no neutral class**. The common, honest workaround: treat low-confidence predictions as neutral, and *tune the threshold on validation data, not on the test set*. The adapter below does that mapping.
# We test it offline against a **mock pipeline** that imitates the output format (the mock is a keyword rule in disguise: it is not a pretrained model and proves nothing about real ones; it only proves the plumbing).

# %%
def hf_to_predict_fn(pipe, neutral_band=0.75, batch_size=32):
    """Wrap a Hugging Face text-classification pipeline as predict_fn(texts) -> labels in LABELS.
    A positive/negative prediction with confidence below `neutral_band` becomes 'neutral'."""
    def predict_fn(texts):
        labels = []
        for i in range(0, len(texts), batch_size):
            for out in pipe(texts[i : i + batch_size]):
                lab = out["label"].lower()
                if lab == "neutral":
                    labels.append("neutral")  # a 3-label model already has a neutral class (label names vary: check model.config.id2label)
                elif out["score"] < neutral_band:
                    labels.append("neutral")
                else:
                    labels.append("positive" if lab.startswith("pos") else "negative")
        return labels

    return predict_fn


def mock_pipeline(texts):
    """Imitates the output format of transformers.pipeline('text-classification'). NOT a real model."""
    results = []
    for t in texts:
        words = re.findall(r"[a-z]+", t.lower())
        margin = sum(w in POS for w in words) - sum(w in NEG for w in words)
        results.append({"label": "POSITIVE" if margin >= 0 else "NEGATIVE", "score": min(0.5 + 0.2 * abs(margin), 0.99)})
    return results


print(mock_pipeline(["Great quality and perfectly fine", "Not worth it"]))
mock_result = evaluate_classifier(hf_to_predict_fn(mock_pipeline), test_df["review_text"], test_df["sentiment"], name="mock HF-format pipeline")
compare([candidates[2], mock_result])

# %% [markdown]
# The adapter handled the format and the missing neutral class end to end. With internet access and the `transformers` package, the real call is exactly one more line (shown below, `no-run`), and the result lands in the
# same table. **Do pick `neutral_band` on a validation split** carved from the training data (for example by re-using `GroupShuffleSplit` on `train_df`): if you choose it by looking at test accuracy, you have trained on the test set.

# %% tags=["no-run"]
from transformers import pipeline

hf_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=128)
hf_result = evaluate_classifier(hf_to_predict_fn(hf_pipe, neutral_band=0.75), test_df["review_text"], test_df["sentiment"], name="DistilBERT SST-2 (zero-shot, neutral via threshold)")
print(compare(candidates + [hf_result]))
print(hf_result["confusion"])
hf_result["errors"].head(10)
# Questions to ask when it runs: Does it beat TF-IDF + LR on this data? By how much, and is the gap larger than the noise
# of ~200 distinct test texts? How much slower is it per 1000 texts? Is the threshold tuned on validation data only?

# %% [markdown]
# ## Common mistakes
#
# 1. **Using the wrong tokenizer for a model** (or a different `max_length` at serving). Load both from the same repository id and revision.
# 2. **Silent truncation.** Texts longer than the model's maximum are cut. Count how many of your texts exceed `max_length` before you trust a score.
# 3. **Comparing a pretrained model with a baseline on different splits or different data**. Use one harness and one held-out set.
# 4. **Tuning thresholds or hyper-parameters on the test set.** Carve out validation data.
# 5. **Ignoring the label mapping.** A model with labels `LABEL_0/LABEL_1` or a different label order silently gives a wrong evaluation. Read `model.config.id2label` and test the adapter on a mock first, as above.
# 6. **Skipping the model card**, then discovering the training data, language or licence does not fit.
# 7. **Forgetting `model.eval()` and `torch.no_grad()`** when running inference yourself, which enables dropout and wastes memory.
# 8. **Padding side and masks ignored when pooling**: averaging token vectors including padding pollutes the embedding (the `attention_mask` weighting in Route 3 avoids this).

# %%
# Demonstration of mistake 5: the same predictions, evaluated with a swapped label mapping, look terrible
oracle_fn = lambda ts: [lookup[t] for t in ts]
swap = {"positive": "negative", "negative": "positive", "neutral": "neutral"}
wrong = evaluate_classifier(lambda ts: [swap[p] for p in oracle_fn(ts)], test_df["review_text"], test_df["sentiment"], name="oracle with swapped labels")
print(f"correct predictions, wrong label mapping -> accuracy {wrong['accuracy']:.2f} (a perfect model looks broken)")

# %% [markdown]
# ## Interview corner
#
# **1. Fine-tuning versus feature extraction versus prompting: how do you choose?**
# <details><summary>Answer</summary>
# By labelled data, budget and requirements. With little or no labelled data: zero-shot/prompting. With a few hundred to a few thousand labels and a need for cheap iteration: frozen embeddings + a simple classifier.
# With enough data and a stable task where the last points of accuracy matter: fine-tune. Always measure against a TF-IDF + logistic-regression baseline, and consider latency and cost per prediction.
# </details>
#
# **2. What are `input_ids` and `attention_mask`?**
# <details><summary>Answer</summary>
# <code>input_ids</code> are the integer ids of the tokens (including special tokens like [CLS]/[SEP]); <code>attention_mask</code> is 1 for real tokens and 0 for padding, so attention ignores padded positions when a batch is padded to equal length.
# </details>
#
# **3. A published model reports 95% on its benchmark but only 80% on your data. Why?**
# <details><summary>Answer</summary>
# Domain and label shift (different vocabulary, length, style, label definitions), different preprocessing or truncation, benchmark-specific tuning, and sometimes test-set contamination of the benchmark. Evaluate on a held-out sample of your own data with your own labels before committing.
# </details>
#
# **4. What do you check before putting a Hugging Face model into production?**
# <details><summary>Answer</summary>
# Licence and intended use from the model card, pinned revision and safetensors weights, evaluation on representative data (including subgroups and long inputs), latency and memory on target hardware, behaviour on out-of-distribution input,
# and a monitoring plan for drift.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Greedy WordPiece
# Write `my_wordpiece(word, vocab)`: greedy longest-match-first over `vocab` (a set or dict of pieces; continuation pieces start with `##`), returning the list of pieces, or `["[UNK]"]` if the word cannot be fully covered.
# Use the toy `VOCAB` above. (You may not call the notebook's `wordpiece`.)

# %% tags=["starter"]
def my_wordpiece(word, vocab):
    ...


# %%
_cases = {"great": ["great"], "unbelievable": ["un", "##believ", "##able"], "unbelievably": ["un", "##believ", "##ably"], "replying": ["reply", "##ing"], "repliedly": ["[UNK]"], "qqq": ["[UNK]"]}
for _w, _exp in _cases.items():
    assert my_wordpiece(_w, VOCAB) == _exp, f"{_w!r}: expected {_exp}, got {my_wordpiece(_w, VOCAB)}"
print("Exercise 1 passed")

# %% tags=["solution"]
def my_wordpiece(word, vocab):
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        piece = None
        while end > start:
            cand = word[start:end] if start == 0 else "##" + word[start:end]
            if cand in vocab:
                piece = cand
                break
            end -= 1
        if piece is None:
            return ["[UNK]"]
        pieces.append(piece)
        start = end
    return pieces


# %% [markdown]
# ### Exercise 2: Padding and the attention mask
# Write `pad_batch(seqs, pad_id=0)` that takes a list of integer lists and returns `(ids, mask)` as NumPy arrays: `ids` padded on the right with `pad_id` to the longest length, `mask` with 1 for real positions and 0 for padding.
# Then store in `n_real_tokens` the total number of real tokens in the batch (from the mask) for `[[5, 6, 7], [8], [9, 10]]`.

# %% tags=["starter"]
def pad_batch(seqs, pad_id=0):
    ...


n_real_tokens = ...

# %%
_ids, _mask = pad_batch([[5, 6, 7], [8], [9, 10]])
assert _ids.tolist() == [[5, 6, 7], [8, 0, 0], [9, 10, 0]], f"ids wrong: {_ids.tolist()}"
assert _mask.tolist() == [[1, 1, 1], [1, 0, 0], [1, 1, 0]], f"mask wrong: {_mask.tolist()}"
assert pad_batch([[1, 2]], pad_id=99)[0].tolist() == [[1, 2]] and pad_batch([[1], [2, 3]], pad_id=99)[0].tolist() == [[1, 99], [2, 3]], "custom pad_id must be used"
assert n_real_tokens == 6, "3 + 1 + 2 real tokens"
print("Exercise 2 passed")

# %% tags=["solution"]
def pad_batch(seqs, pad_id=0):
    longest = max(len(s) for s in seqs)
    ids = np.array([list(s) + [pad_id] * (longest - len(s)) for s in seqs])
    mask = np.array([[1] * len(s) + [0] * (longest - len(s)) for s in seqs])
    return ids, mask


n_real_tokens = int(pad_batch([[5, 6, 7], [8], [9, 10]])[1].sum())

# %% [markdown]
# ### Exercise 3: Map a model's output to our labels
# Write `map_outputs(outputs, band)` for a list of Hugging Face-style dictionaries (`{"label": "POSITIVE" | "NEGATIVE", "score": float}`): return `"positive"` or `"negative"` when `score >= band`, and `"neutral"` when `score < band`.
# Then use it on the outputs of `mock_pipeline` for the test reviews with `band=0.75` and store the resulting **accuracy against the true sentiment** in `mock_acc`.

# %% tags=["starter"]
def map_outputs(outputs, band):
    ...


mock_acc = ...

# %%
_o = [{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.95}, {"label": "POSITIVE", "score": 0.6}, {"label": "NEGATIVE", "score": 0.75}]
assert map_outputs(_o, 0.75) == ["positive", "negative", "neutral", "negative"], f"got {map_outputs(_o, 0.75)}"
_exp = accuracy_score(test_df["sentiment"], hf_to_predict_fn(mock_pipeline, neutral_band=0.75)(test_df["review_text"].tolist()))
assert abs(mock_acc - _exp) < 1e-9, f"expected {_exp:.4f}"
print(f"Exercise 3 passed (mock accuracy {mock_acc:.3f})")

# %% tags=["solution"]
def map_outputs(outputs, band):
    return ["neutral" if o["score"] < band else ("positive" if o["label"].lower().startswith("pos") else "negative") for o in outputs]


mock_acc = float(accuracy_score(test_df["sentiment"], map_outputs(mock_pipeline(test_df["review_text"].tolist()), 0.75)))

# %% [markdown]
# ### Exercise 4: Evaluate your own classifier in the harness
# Train a `make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC())` on `train_df` (use `from sklearn.svm import LinearSVC`), wrap it as a predict function and evaluate it with `evaluate_classifier` on the test rows.
# Store the result dict in `svm_result`, and store in `worst_class` the class with the **lowest F1** in its `per_class` table (first one in `LABELS` order on ties).

# %% tags=["starter"]
svm_result = ...
worst_class = ...

# %%
from sklearn.svm import LinearSVC

_m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC()).fit(train_df["review_text"], train_df["sentiment"])
_pred = _m.predict(test_df["review_text"])
assert abs(svm_result["accuracy"] - accuracy_score(test_df["sentiment"], _pred)) < 1e-9, "accuracy must match an independent computation"
assert abs(svm_result["macro_f1"] - f1_score(test_df["sentiment"], _pred, average="macro")) < 1e-9, "macro-F1 must match an independent computation"
_f1 = f1_score(test_df["sentiment"], _pred, average=None, labels=LABELS)
assert worst_class == LABELS[int(np.argmin(_f1))], f"expected {LABELS[int(np.argmin(_f1))]}"
print(f"Exercise 4 passed (accuracy {svm_result['accuracy']:.3f}, weakest class: {worst_class})")

# %% tags=["solution"]
from sklearn.svm import LinearSVC

_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC()).fit(train_df["review_text"], train_df["sentiment"])
svm_result = evaluate_classifier(_model.predict, test_df["review_text"], test_df["sentiment"], name="TF-IDF (1,2) + LinearSVC")
worst_class = svm_result["per_class"]["f1"].idxmin()

# %% [markdown]
# ### Exercise 5 (open-ended): Audit a model card
# When you have internet access, choose one sentiment or text-classification model from the Hugging Face Hub and answer in a markdown cell the five questions of section 4 (task and language, training data, metrics and test set,
# licence and limitations, size and latency). Then write what you expect its accuracy on `reviews.csv` to be relative to the TF-IDF baseline *before* you run the harness, and compare afterwards. Being wrong here teaches more than being right.

# %% [markdown]
# ## Recap
# - A pretrained model = tokenizer + weights + config + model card. Load the tokenizer and model from the same repository (and pin a revision); `pipeline` is the quick path, `AutoTokenizer` + `AutoModel*` the explicit one.
# - Tokenizers produce `input_ids` and an `attention_mask`; WordPiece/BPE split rare words into pieces, there is a maximum length, and truncation is silent.
# - Choose between prompting (no labels), feature extraction (few labels, cheap) and fine-tuning (most accuracy, most cost), and always measure against a TF-IDF + logistic regression baseline and on latency.
# - Treat a classifier as `predict_fn(texts) -> labels` and evaluate every candidate with one tested harness on one held-out set; adapters translate model-specific outputs (label names, missing neutral class) into your label space.
# - Published scores describe someone else's data: read the model card, then run your own evaluation. This dataset is templated, so expect little head-room here and more on real text.
#
# **Go deeper:** the Hugging Face documentation (huggingface.co/docs): the *Transformers* quick tour, the "Pipelines" and "Fine-tune a pretrained model" guides, and the free Hugging Face NLP/LLM course; the scikit-learn documentation for `precision_recall_fscore_support`.
#
# **Next module:** [LLMs and RAG: how LLMs work](../21_LLMs_and_RAG/01_how_llms_work.ipynb)
