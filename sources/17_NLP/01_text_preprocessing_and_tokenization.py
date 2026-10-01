# %% [markdown]
# # Text Preprocessing and Tokenization
#
# **Why this matters.** Every NLP system, from a spam filter to a large language model, starts by chopping text into pieces and turning those pieces into numbers.
# The choices made here (lowercase or not, split on spaces or on sub-words, keep the numbers or drop them) quietly decide what the model can and cannot learn. A surprising number of
# production bugs are tokenisation bugs: the training pipeline normalised text one way and the serving code another, or an unseen word became an unknown token that the model never handled.
#
# **You will learn**
# - normalise text (case, accents, whitespace) and know what each step destroys
# - tokenise with regular expressions and see where `str.split()` fails
# - explain stemming versus lemmatisation by implementing tiny versions of each (no downloads needed)
# - read the `CountVectorizer` analyser options: word n-grams, character n-grams, `min_df`, stop words
# - implement a toy Byte-Pair Encoding (BPE) tokeniser, the idea behind the sub-word tokenisers of modern language models
#
# **Prerequisites:** [Python basics](../01_Python/), [pandas: loading data](../05_Pandas/01_loading_and_inspecting_data.ipynb).
#
# > Data used: `datasets/reviews.csv` (4,000 synthetic product reviews, `docs/DATASETS.md`) plus short hand-written strings that contain the awkward cases real text has.

# %%
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

reviews = pd.read_csv(Path("../datasets/reviews.csv"))
messy = [
    "  Don't  PANIC!!  It's only 3.5 stars... ",
    "Café “deluxe” was great — the naïve waiter said \"très bien\".",
    "Email me at jane.doe@example.com or visit https://example.com/help?id=42 today",
    "Shipped in 2-3 days; cost $19.99 (20% off) :) #blessed",
]
for s in messy:
    print(repr(s))

# %% [markdown]
# ## 1. Normalisation: make equivalent strings identical
# **Intuition.** "Café", "cafe" and "CAFÉ" mean the same thing to a human, but to a computer they are three different strings. Normalisation maps them to one form so the model does not
# have to learn the same fact three times. Each step is a trade: it **reduces vocabulary** but also **throws information away** (for a sentiment model, "GREAT!!!" carries more emotion than "great").

# %%
def normalise(text, strip_accents=True):
    text = unicodedata.normalize("NFKC", text)  # e.g. compatibility characters, full-width letters
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-")  # typographic quotes/dashes
    text = text.casefold()  # like lower(), but also handles cases such as German sharp-s
    if strip_accents:
        decomposed = unicodedata.normalize("NFKD", text)  # split e + accent into two code points
        text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()  # collapse runs of whitespace


for s in messy[:2]:
    print(normalise(s))

# %% [markdown]
# *What just happened?* We applied four independent steps: Unicode normalisation, typographic-character replacement, case folding and accent removal, then whitespace cleanup.
# Accent stripping is **language dependent**: fine for an English product-review model, harmful for Spanish or French where accents distinguish words ("si" / "sí").

# %% [markdown]
# ## 2. Tokenisation: where `split()` is not enough
# A **token** is the unit the model sees. The simplest tokeniser is `str.split()`. Watch what it does to punctuation:

# %%
text = normalise(messy[0])
print("split():", text.split())

# %% [markdown]
# `"panic!!"` and `"panic"` would become different tokens, and `"stars..."` likewise. A regular expression can separate words from punctuation.
# `\w+` matches a run of letters/digits/underscore; `[^\w\s]` matches one punctuation character. Keeping apostrophes inside words (`don't`) needs an explicit rule.

# %%
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[a-z]+(?:'[a-z]+)*|[^\w\s]")  # numbers | words with inner apostrophes | one punctuation mark


def tokenize(text):
    return TOKEN_RE.findall(normalise(text))


for s in messy:
    print(tokenize(s))

# %% [markdown]
# Notice what is **good** (contractions and decimals like `3.5` and `19.99` survive) and what is **bad**: the e-mail address and the URL are shattered into pieces, and `20%` becomes two tokens.
# No single tokeniser is right for every task. For search you might want the URL whole; for sentiment you might replace it with a placeholder token such as `<url>`.

# %%
def tokenize_with_placeholders(text):
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", " <email> ", text)
    tokens = TOKEN_RE.findall(normalise(text))
    # the placeholders were lowercased and split by normalise/TOKEN_RE: glue "< url >" back together
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == "<" and i + 2 < len(tokens) and tokens[i + 2] == ">":
            out.append(f"<{tokens[i + 1]}>")
            i += 3
        else:
            out.append(tokens[i])
            i += 1
    return out


print(tokenize_with_placeholders(messy[2]))

# %% [markdown]
# ### Stop words
# **Stop words** are very frequent function words ("the", "and", "of"). Dropping them shrinks the data, which helps bag-of-words models on topic tasks. But they can carry meaning:
# "not" is often on stop-word lists, and removing it flips the meaning of "not good". Let us see what is frequent in the reviews first instead of assuming.

# %%
tok_reviews = [tokenize(t) for t in reviews["review_text"]]
counts = Counter(tok for toks in tok_reviews for tok in toks)
print("tokens:", sum(counts.values()), "| distinct:", len(counts))
print(counts.most_common(12))

# %%
tiny_stop = {"and", "the", "it", "a", "is", "was", "but", "also", "although", "for", "not"}
negatives = reviews.loc[reviews["sentiment"] == "negative", "review_text"]
has_not = negatives.str.contains(r"\bnot\b", case=False)
print(f"{has_not.mean():.0%} of negative reviews contain the word 'not'; a stop-list holding 'not' deletes that signal")
print("example:", tokenize(negatives.iloc[2]), "->", [t for t in tokenize(negatives.iloc[2]) if t not in tiny_stop])

# %% [markdown]
# ## 3. Stemming versus lemmatisation
# Both reduce word variants ("connect", "connected", "connecting") to one form so they share a vocabulary slot.
# - A **stemmer** chops suffixes with rules. Fast, language-specific, and blunt: the result need not be a real word.
# - A **lemmatiser** maps a word to its dictionary form (the *lemma*) using a vocabulary and often the word's part of speech. Slower, more accurate.
#
# Libraries give you good versions (NLTK's Porter/Snowball stemmers and WordNet lemmatiser, spaCy's lemmatiser); some need data downloads, which this course avoids.
# To understand the trade-off we build a miniature of each.

# %%
def toy_stem(word):
    for suf in ("ingly", "edly", "ing", "ed", "ly", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)]
            if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":  # running -> runn -> run
                stem = stem[:-1]
            return stem
    return word


words = ["connected", "connecting", "connects", "running", "runs", "happily", "news", "caring", "car", "cars", "better", "was", "mice"]
print({w: toy_stem(w) for w in words})

# %% [markdown]
# Read the output critically. **Good:** `connected/connecting/connects` collapse to `connect`; `running` and `runs` agree. **Over-stemming** (different words merged): `caring` becomes
# `car`, which collides with `car`, and `news` becomes `new`. **Under-stemming** (same lemma not merged): `better` (lemma *good*), `was` (lemma *be*) and `mice` (lemma *mouse*) are untouched.
# Fixing the last group needs knowledge that rules cannot provide. A lemmatiser uses a lookup:

# %%
LEMMAS = {"better": "good", "was": "be", "were": "be", "is": "be", "mice": "mouse", "children": "child", "running": "run", "ran": "run", "caring": "care"}


def toy_lemma(word):
    if word in LEMMAS:
        return LEMMAS[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


print({w: toy_lemma(w) for w in words})

# %% [markdown]
# *What just happened?* The lookup table handles irregular forms that no suffix rule can, and `caring -> care` stays distinct from `car`. The cost is coverage: a real lemmatiser ships
# with a dictionary of ~100,000+ forms (plus part-of-speech tagging so that "saw" the noun stays "saw" and "saw" the verb becomes "see"). **When to use which:** for classic bag-of-words
# classifiers a stemmer is often good enough and very cheap; for anything where the output is shown to people or fed to a dictionary, lemmatise. **For modern neural models, neither**:
# sub-word tokenisers (section 5) and pretrained embeddings make this step mostly unnecessary, and stemming can hurt by discarding information the model could use.

# %% [markdown]
# ## 4. What `CountVectorizer` actually does
# scikit-learn's `CountVectorizer` bundles normalise + tokenise + build vocabulary + count. Its **analyser** is a function from a string to a list of features. Ask it for the analyser and call it directly:

# %%
sentence = "Great quality, but the battery life isn't great!"
word_an = CountVectorizer().build_analyzer()
print("word analyser:   ", word_an(sentence))

# %% [markdown]
# The default lowercases, applies the pattern `(?u)\b\w\w+\b` (runs of 2+ word characters), so `isn't` becomes `isn` and single letters vanish. The options that matter:

# %%
for name, kw in {
    "bigrams (1,2)": dict(ngram_range=(1, 2)),
    "english stop words": dict(stop_words="english"),
    "char_wb 3-grams": dict(analyzer="char_wb", ngram_range=(3, 3)),
}.items():
    out = CountVectorizer(**kw).build_analyzer()(sentence)
    print(f"{name:20s} {len(out):3d} features, e.g. {out[:7]}")

# %% [markdown]
# - **Word n-grams** keep a little local order: the bigram `"not good"` is a feature, where two unigrams lose the link. The price is a bigger vocabulary.
# - **Stop words** (`stop_words="english"` ships inside scikit-learn, no download): notice it also removes "not"-like words, see the warning above.
# - **Character n-grams** (`char_wb` pads each word with spaces and slides a window inside it) are robust to typos and unseen word forms ("greatt" still shares most n-grams with "great") and work on languages without spaces, at the cost of many more features.
#
# Now the full document-term matrix on the reviews. `min_df=5` drops words that occur in fewer than 5 documents.

# %%
cv = CountVectorizer(min_df=5)
dtm = cv.fit_transform(reviews["review_text"])
density = dtm.nnz / (dtm.shape[0] * dtm.shape[1])
print("matrix shape (documents, vocabulary):", dtm.shape, f"| {density:.1%} of cells are non-zero")
print("five most frequent words:", [(w, int(c)) for c, w in sorted(zip(dtm.sum(axis=0).A1, cv.get_feature_names_out()), reverse=True)[:5]])

# %% [markdown]
# Most cells are zero, which is why these matrices are stored as **sparse** matrices (only the non-zero entries). Two rules you will meet again in every NLP project:
# fit the vectoriser on **training text only** (vocabulary from test data is leakage), and use the *same fitted vectoriser* at serving time.
#
# ## 5. Sub-word tokenisation and Byte-Pair Encoding (BPE)
# **Problem.** Word-level vocabularies meet unseen words ("unbelievably", a typo, a product name) and must map them to a useless `<unk>`. Character-level tokens never fail, but the sequences get very long.
# **Sub-word** tokenisation is the compromise: frequent words stay whole, rare words are split into frequent pieces.
#
# **BPE** learns the pieces from data with a greedy loop: start with single characters, repeatedly find the **most frequent adjacent pair** of symbols in the corpus, merge it into a new symbol, and repeat.
# Below is a complete implementation in about twenty lines. Words are stored with an end-of-word marker `</w>` so that "er" inside a word differs from "er" at the end.

# %%
def word_freqs(tokens):
    return Counter(tokens)


def pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(pair, vocab):
    merged = {}
    for symbols, freq in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = freq
    return merged


def learn_bpe(tokens, num_merges):
    vocab = {tuple(w) + ("</w>",): f for w, f in word_freqs(tokens).items()}
    merges = []
    for _ in range(num_merges):
        pairs = pair_counts(vocab)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # most frequent; ties broken alphabetically for determinism
        merges.append(best)
        vocab = merge_pair(best, vocab)
    return merges


def bpe_encode(word, merges):
    symbols = tuple(word) + ("</w>",)
    for pair in merges:  # apply merges in the order they were learned
        symbols = tuple(merge_pair(pair, {symbols: 1}).popitem()[0])
    return list(symbols)


toy_corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest".split()
toy_merges = learn_bpe(toy_corpus, 8)
print("merges learned in order:")
for i, m in enumerate(toy_merges, 1):
    print(f"  {i}. {m[0]!r} + {m[1]!r} -> {m[0] + m[1]!r}")

# %%
for w in ["lowest", "newer", "low"]:
    print(f"{w:8s} -> {bpe_encode(w, toy_merges)}")

# %% [markdown]
# *What just happened?* `lowest` never appeared in the corpus, yet it is encoded from pieces that did (`low` + `est</w>`). That is the whole point: no unknown words, a bounded vocabulary
# (number of merges + characters), and common words become a single token. Real tokenisers (GPT-style byte-level BPE, WordPiece, SentencePiece) follow the same principle with bigger corpora and
# tens of thousands of merges; byte-level variants start from bytes so that *any* string, emoji included, can be encoded.
#
# Now train it on the review words and look at how it splits a word it has not seen:

# %%
review_tokens = [t for toks in tok_reviews for t in toks if t.isalpha()]
seen_words = sorted(set(review_tokens))
for n_merges in (30, 150, 400):
    m = learn_bpe(review_tokens, n_merges)
    whole = sum(len([p for p in bpe_encode(w, m) if p != "</w>"]) == 1 for w in seen_words)  # exactly one piece (a trailing lone </w> marker does not count)
    print(f"{n_merges:3d} merges: {whole} of {len(seen_words)} distinct review words are a single piece")
merges = learn_bpe(review_tokens, 150)  # try 400 and watch "great" become one piece
for w in ["great", "greatness", "unbelievably", "packaging", "packagings"]:
    print(f"{w:13s} -> {bpe_encode(w, merges)}")

# %% [markdown]
# With few merges most words are still several pieces; with more merges the frequent words become single tokens. `greatness`, which never occurs, is encoded from the pieces of `great` plus leftovers,
# and `unbelievably` (not in the corpus at all) falls back to short pieces and characters. Only about 85 distinct words occur in these templated reviews, so the learned pieces reflect the templates.
# Real tokenisers are trained on billions of words, so their pieces reflect real morphology.

# %% [markdown]
# ## Common mistakes
#
# 1. **Normalising differently in training and serving.** Keep one function (or one fitted object) and import it in both places. Persist the fitted vectoriser with the model.
# 2. **Fitting the vocabulary on the whole dataset** before the train/test split.
# 3. **Removing stop words blindly.** "not", "no", "never" are on many lists and carry the sentiment.
# 4. **Stemming for neural models.** It throws information away that an embedding could have used; keep it for cheap bag-of-words baselines.
# 5. **Stripping accents for non-English text**, which merges distinct words.
# 6. **Using a default token pattern and forgetting what it drops**: `CountVectorizer` ignores one-letter tokens and splits `isn't` into `isn`. Print `build_analyzer()(text)` once to see.
# 7. **Reading counts of a `str.split()` vocabulary as if it were clean**: `"good"`, `"good."` and `"Good"` are three entries.

# %%
# Demonstration of mistake 6
print(CountVectorizer().build_analyzer()("I'm not a fan of it, 5/5 would NOT buy"))

# %% [markdown]
# ## Interview corner
#
# **1. Stemming or lemmatisation: what is the difference and when do you use each?**
# <details><summary>Answer</summary>
# Stemming chops affixes by rules (fast, crude, may produce non-words and merge unrelated words); lemmatisation returns the dictionary form using a vocabulary and often the part of speech (slower, more
# accurate). Stemming suits cheap search or bag-of-words baselines; lemmatisation when the output is read by people. Neural models with sub-word tokenisers usually need neither.
# </details>
#
# **2. Why do modern language models use sub-word tokenisation instead of words?**
# <details><summary>Answer</summary>
# Word vocabularies are huge and still meet unseen words; character sequences are too long. Sub-words keep the vocabulary bounded (tens of thousands), represent any word as known pieces, and keep common words
# as single tokens. BPE, WordPiece and SentencePiece are the common algorithms.
# </details>
#
# **3. You are building a sentiment classifier. Which preprocessing steps would you be careful with?**
# <details><summary>Answer</summary>
# Stop-word removal (negations), lowercasing (emphasis like "GREAT!!!"), punctuation removal (exclamation marks, emoticons), and stemming. Check each step on validation data rather than applying a standard recipe.
# </details>
#
# **4. Why fit the vectoriser on training data only?**
# <details><summary>Answer</summary>
# The vocabulary and IDF weights are learned parameters. Fitting them on test text leaks information and makes evaluation optimistic. At serving time, words not in the training vocabulary are simply ignored
# (or split into sub-words with BPE), which the validation set should already simulate.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A tokeniser for support tickets
# Write `ticket_tokens(text)` that lowercases the text and returns a list of tokens where: words keep inner apostrophes (`can't`), order IDs like `ab-1234` (letters, a hyphen, digits) stay as **one** token,
# and all other punctuation is dropped (not returned). *Hint: one `re.findall` with an alternation; put the order-ID pattern first.*

# %% tags=["starter"]
def ticket_tokens(text):
    ...


# %%
cases = {
    "Can't open order AB-1234!!": ["can't", "open", "order", "ab-1234"],
    "Refund for ab-99, please... thanks": ["refund", "for", "ab-99", "please", "thanks"],
    "It's broken (again)": ["it's", "broken", "again"],
}
for text_in, expected in cases.items():
    got = ticket_tokens(text_in)
    assert got == expected, f"{text_in!r}: expected {expected}, got {got}"
print("Exercise 1 passed")

# %% tags=["solution"]
def ticket_tokens(text):
    return re.findall(r"[a-z]+-\d+|[a-z]+(?:'[a-z]+)*|\d+", text.lower())


# %% [markdown]
# ### Exercise 2: How concentrated is the vocabulary?
# Using `CountVectorizer()` with default settings on `reviews["review_text"]`, store in `vocab_size` the number of distinct features and in `top10_share` the fraction of all word occurrences
# that belong to the 10 most frequent words (a number between 0 and 1). The check recomputes both with `collections.Counter` and the same default token pattern.

# %% tags=["starter"]
vocab_size = ...
top10_share = ...

# %%
pattern = re.compile(r"(?u)\b\w\w+\b")
ref = Counter(tok for t in reviews["review_text"] for tok in pattern.findall(t.lower()))
assert vocab_size == len(ref), f"expected {len(ref)} distinct words"
assert abs(top10_share - sum(c for _, c in ref.most_common(10)) / sum(ref.values())) < 1e-9, "share = occurrences of the 10 most frequent words / all occurrences"
print(f"Exercise 2 passed (vocabulary {vocab_size}; top-10 words cover {top10_share:.1%} of all tokens)")

# %% tags=["solution"]
cv2 = CountVectorizer()
m2 = cv2.fit_transform(reviews["review_text"])
totals = m2.sum(axis=0).A1
vocab_size = len(cv2.get_feature_names_out())
top10_share = float(sum(sorted(totals, reverse=True)[:10]) / totals.sum())

# %% [markdown]
# ### Exercise 3: Reproduce the character n-gram analyser
# Write `char_wb_ngrams(word, n)` that returns the character n-grams of a single word the way `CountVectorizer(analyzer="char_wb")` does: pad the lowercase word with one space on each side, then slide
# a window of size `n`. The check compares with scikit-learn on several words. *Hint: `padded = f" {word.lower()} "`.*

# %% tags=["starter"]
def char_wb_ngrams(word, n):
    ...


# %%
for w in ["great", "greatt", "a", "unbelievable"]:
    for n in (2, 3):
        expected = CountVectorizer(analyzer="char_wb", ngram_range=(n, n)).build_analyzer()(w)
        assert char_wb_ngrams(w, n) == expected, f"word {w!r}, n={n}: expected {expected}, got {char_wb_ngrams(w, n)}"
shared = set(char_wb_ngrams("great", 3)) & set(char_wb_ngrams("greatt", 3))
print(f"Exercise 3 passed ('great' and the typo 'greatt' share {len(shared)} of {len(set(char_wb_ngrams('great', 3)))} trigrams)")

# %% tags=["solution"]
def char_wb_ngrams(word, n):
    padded = f" {word.lower()} "
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


# %% [markdown]
# ### Exercise 4: Verify the first BPE merge
# For the corpus `["banana", "bandana", "bananas", "band"]` (each word once), store in `first_merge` the pair that `learn_bpe(corpus, 1)` returns, and in `first_count` the number of times that pair occurs
# in the corpus **counting the end-of-word marker as a symbol**. Then store in `encoded` the BPE encoding of `"bandanas"` after 6 merges learned on the same corpus.
# The check recounts adjacent pairs independently and verifies that joining your pieces gives back the word.

# %% tags=["starter"]
corpus = ["banana", "bandana", "bananas", "band"]
first_merge = ...
first_count = ...
encoded = ...

# %%
manual = Counter()
for w in corpus:
    syms = list(w) + ["</w>"]
    for a, b in zip(syms, syms[1:]):
        manual[(a, b)] += 1
top = max(manual.items(), key=lambda kv: (kv[1], kv[0]))
assert tuple(first_merge) == top[0], f"most frequent pair is {top[0]}"
assert first_count == top[1], f"that pair occurs {top[1]} times"
assert "".join(encoded).replace("</w>", "") == "bandanas", "joining the pieces must give back the word"
assert len(encoded) < len("bandanas") + 1, "after 6 merges the word should be segmented into fewer pieces than characters"
print("Exercise 4 passed:", first_merge, first_count, encoded)

# %% tags=["solution"]
corpus = ["banana", "bandana", "bananas", "band"]
first_merge = learn_bpe(corpus, 1)[0]
pc = pair_counts({tuple(w) + ("</w>",): 1 for w in corpus})
first_count = pc[first_merge]
encoded = bpe_encode("bandanas", learn_bpe(corpus, 6))

# %% [markdown]
# ### Exercise 5 (open-ended): Choose a preprocessing recipe
# You must build a classifier that flags *angry* support tickets. List five preprocessing decisions (for example: lowercase or not, keep exclamation marks, keep numbers, stop words, stemming).
# For each one write whether you would apply it and **how you would test the choice on validation data** rather than assume. No code required; two lines per decision.

# %% [markdown]
# ## Recap
# - Normalise to merge equivalent strings, knowing that every step loses some information (case, accents, punctuation).
# - `str.split()` leaves punctuation attached; regexes give control, but every rule is a decision about your task (URLs, numbers, apostrophes).
# - Stemmers chop suffixes (fast, can over- or under-stem); lemmatisers use a dictionary (accurate, needs resources). Neural models with sub-word tokens rarely need either.
# - `CountVectorizer(...).build_analyzer()` shows exactly what features a text becomes; word n-grams keep local order, char n-grams tolerate typos.
# - BPE repeatedly merges the most frequent adjacent pair; unseen words are encoded from known pieces, which is why modern models use sub-word tokenisers.
#
# **Go deeper:** the scikit-learn user guide "Working with text data" and the `CountVectorizer` API reference; the Hugging Face course/documentation chapter on tokenizers (huggingface.co/docs); the original BPE-for-NLP paper by Sennrich, Haddow and Birch (2016), "Neural Machine Translation of Rare Words with Subword Units".
#
# **Next:** [Bag of words, TF-IDF and classifiers](02_bag_of_words_tfidf_and_classifiers.ipynb)
