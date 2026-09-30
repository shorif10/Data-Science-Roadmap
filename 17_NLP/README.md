# 17 - Natural Language Processing

**Goal:** go from raw text to transformers with nothing hidden: preprocessing and tokenisation (including a BPE you write), bag-of-words and TF-IDF classifiers,
word embeddings built from co-occurrence counts, recurrent models, attention and a transformer block in NumPy (checked against PyTorch), and the workflow for using
pretrained Hugging Face models with an honest evaluation harness.

**Prerequisites:** `12_Machine_Learning` (splits, metrics, baselines), `14_PyTorch` (for notebook 04), `16_Keras/04` as a companion, and basic linear algebra from `02_Math_for_Data_Science`.

**Data note:** `datasets/reviews.csv` is synthetic and templated (only a few hundred distinct texts), so sentiment scores are optimistic. The notebooks say so, split by distinct text
to avoid leakage, and use computable ceilings and constructed tests to keep the comparisons meaningful. Notebook 06 needs the `transformers` package and internet access for its
Hugging Face cells, which are display-only here; the evaluation harness in it runs offline.

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_text_preprocessing_and_tokenization.ipynb` | normalisation, regex tokenisers, toy stemmer and lemmatiser, `CountVectorizer` analysers, implement BPE | 2 |
| 02 | `02_bag_of_words_tfidf_and_classifiers.ipynb` | TF-IDF by hand, logistic regression / SVM / naive Bayes, learning curves, the ceiling of a noisy target, error analysis | 2 |
| 03 | `03_word_embeddings_from_scratch.ipynb` | co-occurrence, PPMI, truncated SVD, cosine neighbours, a downstream test, limits | 2 |
| 04 | `04_sequence_models_for_text.ipynb` | PyTorch GRU/LSTM with packing vs TF-IDF, an order-sensitive test where unigrams fail | 2 |
| 05 | `05_attention_and_transformers_from_scratch.ipynb` | scaled dot-product and multi-head attention, masks, positional encoding, a transformer block, checked against `nn.MultiheadAttention` | 2.5 |
| 06 | `06_hugging_face_and_pretrained_models.ipynb` | tokenizers, `pipeline` / `Auto*` usage, prompt vs features vs fine-tune, a tested evaluation harness | 1.5 |

## You are done when you can...
- [ ] list what each normalisation step removes, and choose preprocessing for a task instead of copying a recipe
- [ ] explain stemming vs lemmatisation, and why sub-word tokenisers (BPE, WordPiece) handle unseen words
- [ ] compute TF-IDF by hand, build a leak-free text `Pipeline`, and read a linear model's weights for suspicious features
- [ ] split text data so duplicates cannot leak, and compare models against a majority baseline and a computable ceiling
- [ ] explain PPMI and embeddings, and name their limits (antonyms, rare words, one vector per word)
- [ ] pack or mask padding in a recurrent model, and say when TF-IDF (or n-grams) is already enough
- [ ] write scaled dot-product attention, explain the `1/sqrt(d)` factor, and prove a causal mask blocks the future with a perturbation test
- [ ] decide between prompting, feature extraction and fine-tuning, and evaluate any classifier function on your own held-out data

## Try next (real public resources)
- Jurafsky and Martin, *Speech and Language Processing* (free draft online)
- The Hugging Face documentation and free courses (huggingface.co/docs, huggingface.co/learn)
- Vaswani et al. (2017), "Attention Is All You Need", and Jay Alammar's "The Illustrated Transformer"
- Tunstall, von Werra and Wolf, *Natural Language Processing with Transformers* (O'Reilly)
- Real text data to practise on (download on your own machine): IMDB movie reviews, 20 Newsgroups (`sklearn.datasets.fetch_20newsgroups`), Kaggle text competitions
