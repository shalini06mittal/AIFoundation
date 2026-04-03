# spaCy & NLTK in Python — A Step-by-Step Guide

> A comprehensive, hands-on guide to Natural Language Processing (NLP) using two of Python's most powerful libraries: **NLTK** and **spaCy**.

---

## Table of Contents

### Part 1 — NLTK (Natural Language Toolkit)
1. [Introduction to NLTK](#1-introduction-to-nltk)
2. [Installation & Setup](#2-installation--setup)
3. [Tokenization](#3-tokenization)
   - [Word Tokenization](#word-tokenization)
   - [Sentence Tokenization](#sentence-tokenization)
4. [Stop Words Removal](#4-stop-words-removal)
5. [Stemming](#5-stemming)
6. [Lemmatization with NLTK](#6-lemmatization-with-nltk)
7. [Part-of-Speech (POS) Tagging](#7-part-of-speech-pos-tagging)
8. [Named Entity Recognition with NLTK](#8-named-entity-recognition-with-nltk)
9. [Chunking](#9-chunking)
10. [Frequency Distribution](#10-frequency-distribution)
11. [Concordance and Collocations](#11-concordance-and-collocations)

### Part 2 — spaCy
12. [Introduction to spaCy](#12-introduction-to-spacy)
13. [Installation & Loading Models](#13-installation--loading-models)
14. [The Doc Object](#14-the-doc-object)
15. [Tokenization with spaCy](#15-tokenization-with-spacy)
16. [Linguistic Annotations](#16-linguistic-annotations)
    - [POS Tagging](#pos-tagging)
    - [Dependency Parsing](#dependency-parsing)
17. [Lemmatization with spaCy](#17-lemmatization-with-spacy)
18. [Named Entity Recognition with spaCy](#18-named-entity-recognition-with-spacy)
19. [Noun Chunks](#19-noun-chunks)
20. [Sentence Segmentation](#20-sentence-segmentation)
21. [Similarity and Word Vectors](#21-similarity-and-word-vectors)
22. [Custom Pipelines](#22-custom-pipelines)
23. [Rule-Based Matching](#23-rule-based-matching)

### Part 3 — Comparison & Practical Examples
24. [NLTK vs spaCy — When to Use Which](#24-nltk-vs-spacy--when-to-use-which)
25. [Practical Examples](#25-practical-examples)
26. [Quick Reference Cheat Sheet](#26-quick-reference-cheat-sheet)

---

## Part 1 — NLTK (Natural Language Toolkit)

---

## 1. Introduction to NLTK

**NLTK** (Natural Language Toolkit) is one of the oldest and most comprehensive Python libraries for NLP. It provides:

- Text tokenization and normalization
- POS tagging and parsing
- Stemming and lemmatization
- Corpora and lexical resources (WordNet, stopwords, etc.)
- Classification and sentiment tools

NLTK is **research and education oriented** — excellent for understanding NLP fundamentals.

---

## 2. Installation & Setup

```bash
# Install NLTK
pip install nltk
```

```python
import nltk

# Download all commonly used datasets at once
nltk.download('all')

# OR download only what you need (recommended)
nltk.download('punkt')           # Tokenizer models
nltk.download('punkt_tab')       # Updated tokenizer tables
nltk.download('stopwords')       # Stop word lists
nltk.download('wordnet')         # Lemmatizer (WordNet)
nltk.download('averaged_perceptron_tagger')   # POS tagger
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')  # Named entity chunker
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')           # Word corpus
```

---

## 3. Tokenization

Tokenization is the process of splitting text into smaller units — words or sentences.

### Word Tokenization

```python
from nltk.tokenize import word_tokenize

text = "Hello! My name is Alice. I love Natural Language Processing."

tokens = word_tokenize(text)
print(tokens)
# ['Hello', '!', 'My', 'name', 'is', 'Alice', '.', 'I', 'love',
#  'Natural', 'Language', 'Processing', '.']

print(f"Total tokens: {len(tokens)}")
# Total tokens: 13
```

### Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize

text = "Hello! My name is Alice. I love NLP. It's fascinating."

sentences = sent_tokenize(text)
for i, sent in enumerate(sentences, 1):
    print(f"Sentence {i}: {sent}")

# Sentence 1: Hello!
# Sentence 2: My name is Alice.
# Sentence 3: I love NLP.
# Sentence 4: It's fascinating.
```

### Other Tokenizers

```python
from nltk.tokenize import TweetTokenizer, MWETokenizer

# Tweet-aware tokenizer (handles hashtags and mentions)
tweet_tokenizer = TweetTokenizer()
tweet = "I love #NLP! It's amazing 😊 @alice check this out"
print(tweet_tokenizer.tokenize(tweet))
# ['I', 'love', '#NLP', '!', "It's", 'amazing', '😊', '@alice', 'check', 'this', 'out']

# Multi-word expression tokenizer
mwe = MWETokenizer([("New", "York"), ("Natural", "Language", "Processing")])
text = "I visited New York for a Natural Language Processing conference"
print(mwe.tokenize(text.split()))
# ['I', 'visited', 'New_York', 'for', 'a', 'Natural_Language_Processing', 'conference']
```

---

## 4. Stop Words Removal

Stop words are common words (like "the", "is", "in") that carry little meaning and are often removed in NLP pipelines.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is a simple example showing stop words removal in NLTK"

# Load English stop words
stop_words = set(stopwords.words('english'))
print(f"Total English stop words: {len(stop_words)}")
# Total English stop words: 179

tokens = word_tokenize(text.lower())
filtered = [word for word in tokens if word not in stop_words and word.isalpha()]

print("Original tokens:", tokens)
# ['this', 'is', 'a', 'simple', 'example', 'showing', 'stop', 'words', 'removal', 'in', 'nltk']

print("Filtered tokens:", filtered)
# ['simple', 'example', 'showing', 'stop', 'words', 'removal', 'nltk']
```

```python
# Available languages for stop words
print(stopwords.fileids())
# ['arabic', 'azerbaijani', 'basque', 'bengali', 'catalan', 'chinese',
#  'danish', 'dutch', 'english', 'finnish', 'french', 'german', ...]

# Spanish stop words example
spanish_stops = set(stopwords.words('spanish'))
print(list(spanish_stops)[:10])
```

---

## 5. Stemming

Stemming reduces a word to its **root form** by chopping off suffixes. It's fast but can produce non-real words.

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

words = ["running", "flies", "happily", "studies", "connection", "beautiful"]

# Porter Stemmer (most commonly used)
porter = PorterStemmer()
print("Porter Stemmer:")
for word in words:
    print(f"  {word:<15} → {porter.stem(word)}")
# running         → run
# flies           → fli
# happily         → happili
# studies         → studi
# connection      → connect
# beautiful       → beauti

print()

# Snowball Stemmer (improved Porter, supports multiple languages)
snowball = SnowballStemmer("english")
print("Snowball Stemmer:")
for word in words:
    print(f"  {word:<15} → {snowball.stem(word)}")

print()

# Lancaster Stemmer (most aggressive)
lancaster = LancasterStemmer()
print("Lancaster Stemmer:")
for word in words:
    print(f"  {word:<15} → {lancaster.stem(word)}")
```

> **Note:** Stemming does NOT guarantee a real dictionary word. Use **Lemmatization** when you need actual valid words.

---

## 6. Lemmatization with NLTK

Lemmatization reduces a word to its **dictionary base form** (lemma) using vocabulary and morphological analysis.

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Basic lemmatization (default POS = noun)
words = ["running", "flies", "better", "studies", "geese", "mice"]

print("Default (noun) lemmatization:")
for word in words:
    print(f"  {word:<12} → {lemmatizer.lemmatize(word)}")
# running      → running   (needs pos='v' to work correctly)
# flies        → fly
# better       → better
# geese        → goose
# mice         → mouse

print()

# Specifying Part-of-Speech for better results
# pos: 'n'=noun, 'v'=verb, 'a'=adjective, 'r'=adverb
print("With correct POS tags:")
examples = [
    ("running", "v"),
    ("better",  "a"),
    ("flies",   "v"),
    ("studies", "v"),
    ("happily", "r"),
]
for word, pos in examples:
    print(f"  {word:<12} (pos={pos}) → {lemmatizer.lemmatize(word, pos=pos)}")
# running      (pos=v) → run
# better       (pos=a) → good
# flies        (pos=v) → fly
# studies      (pos=v) → study
# happily      (pos=r) → happily
```

---

## 7. Part-of-Speech (POS) Tagging

POS tagging assigns a grammatical label (noun, verb, adjective, etc.) to each token.

```python
import nltk
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)

# Tag each word
tagged = nltk.pos_tag(tokens)
print(tagged)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'),
#  ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]
```

```python
# Understanding POS tags
pos_meanings = {
    "CC":  "Coordinating conjunction",
    "DT":  "Determiner",
    "IN":  "Preposition or subordinating conjunction",
    "JJ":  "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "NN":  "Noun, singular",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "PRP": "Personal pronoun",
    "RB":  "Adverb",
    "VB":  "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund/present participle",
    "VBN": "Verb, past participle",
    "VBZ": "Verb, 3rd person singular present",
}

for word, tag in tagged:
    meaning = pos_meanings.get(tag, "Other")
    print(f"  {word:<10} {tag:<6} — {meaning}")
```

---

## 8. Named Entity Recognition with NLTK

NER identifies and classifies real-world entities like people, organizations, and locations.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk import ne_chunk

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

tokens  = word_tokenize(text)
tagged  = nltk.pos_tag(tokens)
chunked = ne_chunk(tagged)

print(chunked)
# (S
#   (ORGANIZATION Apple/NNP Inc./NNP)
#   was/VBD
#   founded/VBN
#   by/IN
#   (PERSON Steve/NNP Jobs/NNP)
#   in/IN
#   (GPE Cupertino/NNP)
#   ,/,
#   (GPE California/NNP)
#   ./.)
```

```python
# Extract entities programmatically
def extract_entities(text):
    tokens  = word_tokenize(text)
    tagged  = nltk.pos_tag(tokens)
    chunked = ne_chunk(tagged)

    entities = []
    for subtree in chunked:
        if hasattr(subtree, 'label'):
            entity_name = " ".join(word for word, tag in subtree.leaves())
            entities.append((entity_name, subtree.label()))
    return entities

text = "Elon Musk founded SpaceX in Hawthorne, California in 2002."
for entity, label in extract_entities(text):
    print(f"  {entity:<20} [{label}]")
# Elon Musk            [PERSON]
# SpaceX               [ORGANIZATION]
# Hawthorne            [GPE]
# California           [GPE]
```

---

## 9. Chunking

Chunking (shallow parsing) groups tokens into meaningful chunks like noun phrases (NP) and verb phrases (VP) using grammar rules.

```python
import nltk
from nltk.tokenize import word_tokenize

text = "The big red car drove quickly down the empty street"
tokens = word_tokenize(text)
tagged = nltk.pos_tag(tokens)

# Define a grammar for Noun Phrases
grammar = r"""
    NP: {<DT>?<JJ>*<NN>}    # Optional determiner, 0+ adjectives, noun
    VP: {<VB.*><RB>?}        # Verb optionally followed by adverb
"""

parser = nltk.RegexpParser(grammar)
tree   = parser.parse(tagged)
print(tree)
# (S
#   (NP The/DT big/JJ red/JJ car/NN)
#   (VP drove/VBD quickly/RB)
#   down/IN
#   (NP the/DT empty/JJ street/NN))
```

---

## 10. Frequency Distribution

`FreqDist` counts how often each word appears in a text.

```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = """
Python is an amazing programming language. Python is used for web development,
data science, machine learning, and artificial intelligence. Python developers
love Python because Python is simple and readable.
"""

tokens    = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
clean     = [w for w in tokens if w.isalpha() and w not in stop_words]

fdist = FreqDist(clean)

print("Most common words:")
for word, count in fdist.most_common(5):
    print(f"  {word:<15} {count} times")
# python          5 times
# amazing         1 times
# programming     1 times

# Total unique words
print(f"\nUnique words: {len(fdist)}")

# Frequency of a specific word
print(f"'python' appears: {fdist['python']} times")
```

---

## 11. Concordance and Collocations

```python
from nltk.text import Text
from nltk.tokenize import word_tokenize

text_str = """
Natural language processing is a subfield of linguistics and computer science.
Natural language understanding involves comprehension of natural human language.
Processing natural language requires both linguistic and computational methods.
"""

tokens    = word_tokenize(text_str.lower())
nltk_text = Text(tokens)

# Concordance — shows context of a word
print("Concordance for 'natural':")
nltk_text.concordance("natural", width=60, lines=5)
# Displaying 3 of 3 matches:
#  processing is a subfield of linguistics and computer science
#  understanding involves comprehension of natural human language
#  requires both linguistic and computational methods

# Collocations — frequently co-occurring word pairs
print("\nCollocations:")
nltk_text.collocations()
# natural language; language processing
```

---

---

## Part 2 — spaCy

---

## 12. Introduction to spaCy

**spaCy** is a modern, industrial-strength NLP library designed for **production use**. It provides:

- Extremely fast tokenization and parsing
- Pre-trained statistical models for many languages
- Built-in Named Entity Recognition, POS tagging, and dependency parsing
- Word vectors and similarity
- Easy pipeline customization

spaCy is **production oriented** — optimized for speed and real-world applications.

---

## 13. Installation & Loading Models

```bash
# Install spaCy
pip install spacy

# Download English model (small — fast, less accurate)
python -m spacy download en_core_web_sm

# Download English model (medium — balanced)
python -m spacy download en_core_web_md

# Download English model (large — most accurate, includes word vectors)
python -m spacy download en_core_web_lg
```

```python
import spacy

# Load a model
nlp = spacy.load("en_core_web_sm")

print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'senter', 'ner', 'attribute_ruler', 'lemmatizer']

# Quick check
doc = nlp("spaCy is awesome!")
for token in doc:
    print(token.text, token.pos_)
# spaCy PROPN
# is    AUX
# awesome ADJ
# !     PUNCT
```

---

## 14. The Doc Object

When you call `nlp(text)`, spaCy returns a **`Doc`** object — the central container for all NLP data.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a U.K. startup for $1 billion.")

# Doc behaves like a list of tokens
print(f"Number of tokens: {len(doc)}")  # 14

# Access individual tokens
token = doc[0]
print(f"Token text:    {token.text}")      # Apple
print(f"Token index:   {token.i}")         # 0
print(f"Is alpha:      {token.is_alpha}")  # True
print(f"Is digit:      {token.is_digit}")  # False
print(f"Is stop word:  {token.is_stop}")   # False

# Slice a Doc to get a Span
span = doc[0:3]
print(f"Span: {span.text}")  # Apple is looking
```

---

## 15. Tokenization with spaCy

spaCy tokenizes text using **rules and exceptions** — no model required for this step.

```python
import spacy

nlp  = spacy.load("en_core_web_sm")
doc  = nlp("Don't hesitate to e-mail us at hello@example.com!")

print("Tokens:")
for token in doc:
    print(f"  {token.text!r:<20} is_punct={token.is_punct}  is_space={token.is_space}")

# 'Do'                 is_punct=False  is_space=False
# "n't"                is_punct=False  is_space=False
# 'hesitate'           is_punct=False  is_space=False
# 'to'                 is_punct=False  is_space=False
# 'e-mail'             is_punct=False  is_space=False
# 'us'                 is_punct=False  is_space=False
# 'at'                 is_punct=False  is_space=False
# 'hello@example.com'  is_punct=False  is_space=False
# '!'                  is_punct=True   is_space=False
```

```python
# Useful token attributes
print(f"\n{'Token':<20} {'is_alpha':<10} {'is_digit':<10} {'like_email':<12} {'like_url'}")
print("-" * 60)
for token in doc:
    print(f"{token.text:<20} {str(token.is_alpha):<10} {str(token.is_digit):<10} "
          f"{str(token.like_email):<12} {token.like_url}")
```

---

## 16. Linguistic Annotations

### POS Tagging

spaCy provides both coarse-grained (`pos_`) and fine-grained (`tag_`) POS labels.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog")

print(f"{'Token':<12} {'POS':<8} {'Tag':<8} {'Explanation'}")
print("-" * 55)
for token in doc:
    print(f"{token.text:<12} {token.pos_:<8} {token.tag_:<8} {spacy.explain(token.tag_)}")

# Token        POS      Tag      Explanation
# The          DET      DT       determiner
# quick        ADJ      JJ       adjective (base)
# brown        ADJ      JJ       adjective (base)
# fox          NOUN     NN       noun, singular or mass
# jumps        VERB     VBZ      verb, 3rd person singular present
# over         ADP      IN       conjunction, subordinating or preposition
# the          DET      DT       determiner
# lazy         ADJ      JJ       adjective (base)
# dog          NOUN     NN       noun, singular or mass
```

### Dependency Parsing

Dependency parsing shows **grammatical relationships** between tokens.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat on the mat")

print(f"{'Token':<10} {'Dep':<12} {'Head':<10} {'Children'}")
print("-" * 55)
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<10} {token.dep_:<12} {token.head.text:<10} {children}")

# Token      Dep          Head       Children
# The        det          cat        []
# cat        nsubj        sat        ['The']
# sat        ROOT         sat        ['cat', 'on', '.']
# on         prep         sat        ['mat']
# the        det          mat        []
# mat        pobj         on         ['the']
```

```python
# Navigate the dependency tree
root = [token for token in doc if token.dep_ == "ROOT"][0]
print(f"\nRoot verb: {root.text}")
print(f"Subject:   {[t.text for t in root.lefts if t.dep_ == 'nsubj']}")
print(f"Object:    {[t.text for t in root.rights if t.dep_ == 'pobj']}")
```

---

## 17. Lemmatization with spaCy

spaCy's lemmatization uses language models — it's context-aware and produces valid dictionary words.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The geese were flying over the swimming children")

print(f"{'Token':<15} {'Lemma'}")
print("-" * 30)
for token in doc:
    print(f"{token.text:<15} {token.lemma_}")

# Token           Lemma
# The             the
# geese           goose
# were            be
# flying          fly
# over            over
# the             the
# swimming        swim
# children        child
```

---

## 18. Named Entity Recognition with spaCy

spaCy's NER is fast, accurate, and identifies a wide range of entity types.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Tesla was founded by Elon Musk in 2003 and is headquartered in Austin, Texas.")

print("Named Entities:")
print(f"{'Entity':<25} {'Label':<12} {'Explanation'}")
print("-" * 60)
for ent in doc.ents:
    print(f"{ent.text:<25} {ent.label_:<12} {spacy.explain(ent.label_)}")

# Entity                    Label        Explanation
# Tesla                     ORG          Companies, agencies, institutions
# Elon Musk                 PERSON       People, including fictional
# 2003                      DATE         Absolute or relative dates or periods
# Austin                    GPE          Countries, cities, states
# Texas                     GPE          Countries, cities, states
```

```python
# Access entity positions in the original text
for ent in doc.ents:
    print(f"  [{ent.start_char}:{ent.end_char}] {ent.text!r} → {ent.label_}")

# [0:5]   'Tesla' → ORG
# [19:28] 'Elon Musk' → PERSON
# [32:36] '2003' → DATE
# [58:64] 'Austin' → GPE
# [66:71] 'Texas' → GPE
```

```python
# Common spaCy NER labels
ner_labels = {
    "PERSON":   "People, including fictional",
    "ORG":      "Companies, agencies, institutions",
    "GPE":      "Geopolitical entity (countries, cities)",
    "LOC":      "Non-GPE locations (mountains, rivers)",
    "PRODUCT":  "Objects, vehicles, foods",
    "DATE":     "Absolute or relative dates",
    "TIME":     "Times smaller than a day",
    "MONEY":    "Monetary values, including unit",
    "PERCENT":  "Percentage (including %)",
    "NORP":     "Nationalities, religious, political groups",
    "EVENT":    "Named hurricanes, battles, wars, sports events",
    "LAW":      "Named documents made into laws",
    "LANGUAGE": "Any named language",
}

for label, desc in ner_labels.items():
    print(f"  {label:<12} — {desc}")
```

---

## 19. Noun Chunks

Noun chunks are base noun phrases — "flat" spans of text with a noun as the head.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The autonomous electric vehicle drove through the busy city streets.")

print("Noun Chunks:")
for chunk in doc.noun_chunks:
    print(f"  Text:       {chunk.text}")
    print(f"  Root:       {chunk.root.text}")
    print(f"  Root dep:   {chunk.root.dep_}")
    print(f"  Root head:  {chunk.root.head.text}")
    print()

# Text:       The autonomous electric vehicle
# Root:       vehicle
# Root dep:   nsubj
# Root head:  drove
#
# Text:       the busy city streets
# Root:       streets
# Root dep:   pobj
# Root head:  through
```

---

## 20. Sentence Segmentation

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is the first sentence. Here is the second one! And a third?")

print("Sentences:")
for i, sent in enumerate(doc.sents, 1):
    print(f"  {i}: {sent.text.strip()}")
    print(f"     Start token index: {sent.start}, End: {sent.end}")

# 1: This is the first sentence.
#    Start token index: 0, End: 6
# 2: Here is the second one!
#    Start token index: 6, End: 12
# 3: And a third?
#    Start token index: 12, End: 15
```

---

## 21. Similarity and Word Vectors

> **Requires `en_core_web_md` or `en_core_web_lg`** (medium/large models include word vectors).

```python
import spacy

nlp = spacy.load("en_core_web_md")  # Use md or lg for vectors

# Word similarity
word1 = nlp("cat")
word2 = nlp("dog")
word3 = nlp("car")

print(f"cat  ↔ dog : {word1.similarity(word2):.4f}")  # ~0.8016
print(f"cat  ↔ car : {word1.similarity(word3):.4f}")  # ~0.2083

# Sentence similarity
sent1 = nlp("I love playing football")
sent2 = nlp("I enjoy watching soccer")
sent3 = nlp("The stock market crashed today")

print(f"\nSentence similarity:")
print(f"  Sent1 ↔ Sent2 : {sent1.similarity(sent2):.4f}")  # High
print(f"  Sent1 ↔ Sent3 : {sent1.similarity(sent3):.4f}")  # Low
```

```python
# Access raw word vectors
token = nlp("banana")
print(f"Vector shape: {token.vector.shape}")  # (300,)
print(f"First 5 values: {token.vector[:5]}")
print(f"Has vector: {token.has_vector}")
```

---

## 22. Custom Pipelines

spaCy processes text through a **pipeline** of components. You can add custom ones.

```python
import spacy
from spacy.language import Language

nlp = spacy.load("en_core_web_sm")

# View the current pipeline
print("Default pipeline:", nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'senter', 'ner', 'attribute_ruler', 'lemmatizer']

# --- Define a custom pipeline component ---
@Language.component("custom_word_counter")
def word_counter(doc):
    """Adds a custom extension attribute: word count."""
    word_count = sum(1 for token in doc if token.is_alpha)
    doc._.word_count = word_count  # stored in doc extension
    return doc

# Register a custom attribute on Doc
if not spacy.tokens.Doc.has_extension("word_count"):
    spacy.tokens.Doc.set_extension("word_count", default=0)

# Add the component to the pipeline
nlp.add_pipe("custom_word_counter", last=True)
print("\nUpdated pipeline:", nlp.pipe_names)

# Test it
doc = nlp("This sentence has exactly five words.")
print(f"Word count: {doc._.word_count}")  # 6
```

```python
# Disable pipeline components for speed when you don't need them
nlp = spacy.load("en_core_web_sm")

# Only run tokenizer (fastest)
with nlp.select_pipes(disable=["tagger", "parser", "ner"]):
    doc = nlp("Just tokenizing this text quickly")
    for token in doc:
        print(token.text, end=" ")
```

---

## 23. Rule-Based Matching

spaCy's `Matcher` lets you find tokens based on **linguistic patterns** — more powerful than regex.

```python
import spacy
from spacy.matcher import Matcher

nlp     = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Pattern: an adjective followed by a noun
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}]
matcher.add("ADJ_NOUN", [pattern])

doc     = nlp("The quick brown fox jumps over the lazy dog on a cold winter morning.")
matches = matcher(doc)

print("Adjective-Noun pairs found:")
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"  {span.text}")
# quick fox        (approximate — depends on parse)
# lazy dog
# cold morning
# winter morning
```

```python
from spacy.matcher import PhraseMatcher

# PhraseMatcher: match exact phrases efficiently
nlp           = spacy.load("en_core_web_sm")
phrase_matcher = PhraseMatcher(nlp.vocab)

phrases   = ["machine learning", "deep learning", "natural language processing"]
patterns  = [nlp.make_doc(text) for text in phrases]
phrase_matcher.add("ML_TERMS", patterns)

doc     = nlp("I am studying natural language processing and deep learning techniques.")
matches = phrase_matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(f"  Matched: {span.text!r}")
# Matched: 'natural language processing'
# Matched: 'deep learning'
```

---

---

## Part 3 — Comparison & Practical Examples

---

## 24. NLTK vs spaCy — When to Use Which

| Feature                    | NLTK                              | spaCy                               |
|----------------------------|-----------------------------------|-------------------------------------|
| **Purpose**                | Education & research              | Production & industry               |
| **Speed**                  | Slower                            | Very fast (Cython-based)            |
| **Ease of use**            | Verbose, more manual steps        | Clean API, less boilerplate         |
| **Tokenization**           | Rule-based, configurable          | Rule + exception based, very accurate|
| **POS Tagging**            | Good                              | Excellent, context-aware            |
| **NER**                    | Basic (rule-based)                | Advanced (neural models)            |
| **Lemmatization**          | Needs manual POS input            | Automatic, context-aware            |
| **Word Vectors**           | Via Gensim integration            | Built-in (md/lg models)             |
| **Dependency Parsing**     | Basic chunking                    | Full dependency tree                |
| **Custom Pipelines**       | Limited                           | First-class support                 |
| **Language Support**       | Many corpora/languages            | 60+ languages with trained models   |
| **Best For**               | Learning NLP, research, prototyping| Real apps, APIs, data pipelines     |

---

## 25. Practical Examples

### Text Preprocessing Pipeline (NLTK)

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_nltk(text):
    """Full NLTK preprocessing pipeline."""
    # 1. Lowercase
    text = text.lower()

    # 2. Tokenize
    tokens = word_tokenize(text)

    # 3. Remove punctuation and stop words
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # 4. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

    return tokens

text = "The cats were running quickly through the beautiful gardens."
result = preprocess_nltk(text)
print(result)
# ['cat', 'run', 'quickly', 'beautiful', 'garden']
```

---

### Text Preprocessing Pipeline (spaCy)

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_spacy(text):
    """Full spaCy preprocessing pipeline."""
    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha          # alphabetic only
        and not token.is_stop      # remove stop words
        and not token.is_punct     # remove punctuation
    ]
    return tokens

text = "The cats were running quickly through the beautiful gardens."
result = preprocess_spacy(text)
print(result)
# ['cat', 'run', 'quickly', 'beautiful', 'garden']
```

---

### Extract Key Information from Text (spaCy)

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_info(text):
    doc = nlp(text)
    return {
        "sentences":   [sent.text.strip() for sent in doc.sents],
        "entities":    [(ent.text, ent.label_) for ent in doc.ents],
        "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
        "verbs":       [token.lemma_ for token in doc if token.pos_ == "VERB"],
    }

text = """
Elon Musk announced that SpaceX will launch a new rocket in June 2025.
The mission will carry astronauts to the International Space Station.
"""

info = extract_info(text)

print("📌 Sentences:")
for s in info["sentences"]: print(f"   {s}")

print("\n🏷️  Entities:")
for text, label in info["entities"]: print(f"   {text:<25} [{label}]")

print("\n📦 Noun Chunks:")
for chunk in info["noun_chunks"]: print(f"   {chunk}")

print("\n⚡ Verbs (lemmatized):")
print("  ", info["verbs"])
```

---

### Simple Keyword Extractor (NLTK + spaCy combined)

```python
import spacy
from nltk import FreqDist
from nltk.corpus import stopwords

nlp        = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def extract_keywords(text, top_n=10):
    """Extract top N keywords using spaCy + NLTK FreqDist."""
    doc = nlp(text.lower())

    keywords = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.pos_ in ("NOUN", "PROPN", "ADJ", "VERB")
        and token.lemma_ not in stop_words
        and len(token.lemma_) > 2
    ]

    fdist = FreqDist(keywords)
    return fdist.most_common(top_n)

text = """
Machine learning is a subset of artificial intelligence. It enables computers
to learn from data without being explicitly programmed. Deep learning uses
neural networks with many layers to process data and make predictions.
Machine learning algorithms are used in image recognition, speech processing,
and natural language understanding.
"""

print("Top Keywords:")
for word, count in extract_keywords(text):
    print(f"  {word:<20} → {count} occurrences")
```

---

### Sentiment-Ready Text Cleaner

```python
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def clean_for_sentiment(text):
    """Clean text while preserving sentiment-relevant words."""
    # Remove URLs and mentions
    text = re.sub(r"http\S+|@\w+", "", text)

    doc = nlp(text)

    tokens = [
        token.text.lower()
        for token in doc
        if not token.is_punct
        and not token.is_space
        and not token.like_url
        and not token.is_digit
    ]

    return " ".join(tokens)

reviews = [
    "Amazing product! @user check this out: https://example.com 😍",
    "Terrible experience. Not recommended at all!!!",
    "Pretty good, but could be better. 7/10",
]

for review in reviews:
    cleaned = clean_for_sentiment(review)
    print(f"Original: {review}")
    print(f"Cleaned:  {cleaned}")
    print()
```

---

## 26. Quick Reference Cheat Sheet

```
══════════════════════════════════════════════════════
  NLTK QUICK REFERENCE
══════════════════════════════════════════════════════

SETUP
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('averaged_perceptron_tagger')

TOKENIZATION
  from nltk.tokenize import word_tokenize, sent_tokenize
  word_tokenize(text)         → list of word tokens
  sent_tokenize(text)         → list of sentence strings

STOP WORDS
  from nltk.corpus import stopwords
  stopwords.words('english')  → set of stop words

STEMMING
  from nltk.stem import PorterStemmer
  PorterStemmer().stem(word)  → stemmed word (may not be real word)

LEMMATIZATION
  from nltk.stem import WordNetLemmatizer
  WordNetLemmatizer().lemmatize(word, pos='v')  → lemma

POS TAGGING
  import nltk
  nltk.pos_tag(tokens)        → [(word, tag), ...]

NAMED ENTITY RECOGNITION
  nltk.ne_chunk(tagged)       → tree with NE labels

FREQUENCY DISTRIBUTION
  from nltk import FreqDist
  FreqDist(tokens).most_common(n) → [(word, count), ...]


══════════════════════════════════════════════════════
  spaCy QUICK REFERENCE
══════════════════════════════════════════════════════

SETUP
  import spacy
  nlp = spacy.load("en_core_web_sm")   # sm / md / lg
  doc = nlp("Your text here.")

TOKEN ATTRIBUTES
  token.text          → original text
  token.lemma_        → base form
  token.pos_          → coarse POS (NOUN, VERB, ADJ…)
  token.tag_          → fine-grained POS (NN, VBZ…)
  token.dep_          → syntactic dependency
  token.head          → syntactic head token
  token.is_stop       → is stop word?
  token.is_alpha      → is alphabetic?
  token.is_punct      → is punctuation?
  token.ent_type_     → entity type (if any)

NAMED ENTITIES
  doc.ents            → tuple of Span objects
  ent.text            → entity string
  ent.label_          → entity type (PERSON, ORG…)
  spacy.explain(label)→ human-readable description

NOUN CHUNKS
  doc.noun_chunks     → iterator of Span objects

SENTENCES
  doc.sents           → iterator of Span objects

SIMILARITY (md/lg models only)
  doc1.similarity(doc2) → float 0–1

MATCHER
  from spacy.matcher import Matcher, PhraseMatcher
  Matcher(vocab)      → token-based pattern matching
  PhraseMatcher(vocab)→ exact phrase matching

PIPELINE
  nlp.pipe_names      → list active components
  nlp.add_pipe(name)  → add component
  nlp.select_pipes(disable=[…]) → context manager


══════════════════════════════════════════════════════
  POS TAG REFERENCE (Universal)
══════════════════════════════════════════════════════
  ADJ    Adjective              (big, red, fast)
  ADP    Adposition             (in, on, at)
  ADV    Adverb                 (quickly, very)
  AUX    Auxiliary verb         (is, has, will)
  CCONJ  Coordinating conj.     (and, or, but)
  DET    Determiner             (the, a, an)
  INTJ   Interjection           (oh, wow)
  NOUN   Noun                   (dog, city)
  NUM    Numeral                (1, one, IV)
  PART   Particle               (not, 's, up)
  PRON   Pronoun                (I, he, they)
  PROPN  Proper noun            (London, Alice)
  PUNCT  Punctuation            (. , ! ?)
  SCONJ  Subordinating conj.    (if, while, that)
  SYM    Symbol                 ($, %, +)
  VERB   Verb                   (run, is, go)
  X      Other                  (foreign words)
```

---

*Happy NLP coding! 🐍🧠*
