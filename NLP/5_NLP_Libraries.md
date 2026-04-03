# LAB DOCUMENT

**Module 6**

# NLP Libraries

**NLTK • spaCy • Gensim • fastText**

*A comprehensive hands-on guide covering installation, core features, API usage, practical examples, and comparative analysis*

---

## Table of Contents

1. [Introduction to NLP Libraries](#61-introduction-to-nlp-libraries)
2. [NLTK - Natural Language Toolkit](#62-nltk--natural-language-toolkit)
   - [2.1 Installation & Setup](#621-installation--setup)
   - [2.2 Core Features](#622-core-features)
     - [A. Tokenization](#a-tokenization)
     - [B. Stop Word Removal](#b-stop-word-removal)
     - [C. Stemming & Lemmatization](#c-stemming--lemmatization)
     - [D. Part-of-Speech (POS) Tagging](#d-part-of-speech-pos-tagging)
     - [E. Named Entity Recognition (NER)](#e-named-entity-recognition-ner)
     - [F. Frequency Distribution](#f-frequency-distribution)
     - [G. Sentiment Analysis with VADER](#g-sentiment-analysis-with-vader)
   - [2.3 NLTK Key Points Summary](#623-nltk-key-points-summary)
3. [spaCy - Industrial-Strength NLP](#63-spacy--industrial-strength-nlp)
   - [3.1 Installation & Model Download](#631-installation--model-download)
   - [3.2 The spaCy Pipeline](#632-the-spacy-pipeline)
   - [3.3 Core Features](#633-core-features)
     - [A. Basic NLP Processing](#a-basic-nlp-processing)
     - [B. Dependency Parsing](#b-dependency-parsing)
     - [C. Word Vectors & Similarity](#c-word-vectors--similarity)
     - [D. Custom Pipeline Component](#d-custom-pipeline-component)
     - [E. Matcher - Rule-Based Matching](#e-matcher--rule-based-matching)
     - [F. Training a Custom NER Model](#f-training-a-custom-ner-model)
   - [3.4 spaCy Key Points Summary](#634-spacy-key-points-summary)
4. [Gensim - Topic Modeling & Document Similarity](#64-gensim--topic-modeling--document-similarity)
   - [4.1 Installation](#641-installation)
   - [4.2 Core Features](#642-core-features)
     - [A. Word2Vec - Word Embeddings](#a-word2vec--word-embeddings)
     - [B. LDA - Latent Dirichlet Allocation (Topic Modeling)](#b-lda--latent-dirichlet-allocation-topic-modeling)
     - [C. Doc2Vec - Document Embeddings](#c-doc2vec--document-embeddings)
     - [D. TF-IDF Model](#d-tf-idf-model)
     - [E. Loading Pre-trained Word2Vec (Google News)](#e-loading-pre-trained-word2vec-google-news)
   - [4.3 Gensim Key Points Summary](#643-gensim-key-points-summary)
5. [fastText - Word Embeddings & Text Classification](#65-fasttext--word-embeddings--text-classification)
   - [5.1 Installation](#651-installation)
   - [5.2 Core Features](#652-core-features)
     - [A. Word Embeddings with Subword Information](#a-word-embeddings-with-subword-information)
     - [B. Text Classification](#b-text-classification)
     - [C. FastText via Gensim](#c-fasttext-via-gensim)
     - [D. Loading Pre-trained fastText Vectors](#d-loading-pre-trained-fasttext-vectors)
   - [5.3 fastText Key Points Summary](#653-fasttext-key-points-summary)
6. [Comparative Analysis](#66-comparative-analysis)
   - [6.1 Feature Comparison Table](#661-feature-comparison-table)
   - [6.2 When to Use Which Library](#662-when-to-use-which-library)
   - [6.3 End-to-End Pipeline Example](#663-end-to-end-pipeline-example)
7. [Lab Exercises](#67-lab-exercises)
8. [Summary](#68-summary)

---

| **NLTK**                      | **spaCy**                | **Gensim**               | **fastText**             |
|-------------------------------|--------------------------|--------------------------|--------------------------|
| *Educational & Research*      | *Production NLP*         | *Topic Modeling*         | *Word Embeddings*        |

---

## 6.1 Introduction to NLP Libraries

Natural Language Processing (NLP) libraries provide ready-made tools for processing, analyzing, and understanding human language using Python. These libraries abstract complex linguistic computations and allow developers and data scientists to build powerful text-processing pipelines with minimal effort. In this lab, we explore the four most widely-used NLP libraries: **NLTK**, **spaCy**, **Gensim**, and **fastText**.

| **Library**  | **Primary Use Case / Strength**                                                      |
|--------------|--------------------------------------------------------------------------------------|
| **NLTK**     | Academic research, education, linguistic analysis, classical NLP methods             |
| **spaCy**    | Industrial-grade production NLP, fast pipeline processing, modern ML models          |
| **Gensim**   | Unsupervised topic modeling, Word2Vec, document similarity, large-scale corpora      |
| **fastText** | Subword-level word embeddings, text classification, multilingual NLP, rare words     |

---

## 6.2 NLTK - Natural Language Toolkit

NLTK (Natural Language Toolkit) is one of the oldest and most comprehensive Python libraries for NLP. Developed at the University of Pennsylvania in 2001, NLTK is primarily designed for teaching, research, and prototyping. It includes over 50 corpora and lexical resources, and provides tools for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

### 6.2.1 Installation & Setup

```bash
# Install NLTK
pip install nltk
```

```python
# Download required corpora and models
import nltk

nltk.download('punkt')                      # Tokenizer models
nltk.download('stopwords')                  # Stop words
nltk.download('averaged_perceptron_tagger') # POS tagger
nltk.download('wordnet')                    # WordNet lexical database
nltk.download('vader_lexicon')              # Sentiment analysis
nltk.download('maxent_ne_chunker')          # Named entity chunker
nltk.download('words')                      # Word lists
```

### 6.2.2 Core Features

#### A. Tokenization

Tokenization splits raw text into individual tokens (words or sentences). NLTK provides both word-level and sentence-level tokenizers.

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

text = 'NLTK is a leading platform for building Python programs to work with NLP.'

# Word tokenization
words = word_tokenize(text)
print('Words:', words)
# Output: ['NLTK', 'is', 'a', 'leading', 'platform', ...]

# Sentence tokenization
sentences = sent_tokenize('Hello World. How are you? I am fine!')
print('Sentences:', sentences)
# Output: ['Hello World.', 'How are you?', 'I am fine!']
```

#### B. Stop Word Removal

Stop words are common words (e.g., 'the', 'is', 'in') that carry little meaningful information and are often removed during preprocessing.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
text = 'This is a sample sentence, and this is the second one.'
tokens = word_tokenize(text)
filtered = [w for w in tokens if w.lower() not in stop_words]

print('Filtered:', filtered)
# Output: ['sample', 'sentence', ',', 'second', '.']
```

#### C. Stemming & Lemmatization

Stemming reduces words to their root form (e.g., 'running' → 'run'), while lemmatization finds the base form using vocabulary and grammar context (e.g., 'better' → 'good').

```python
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer

# Stemming
ps = PorterStemmer()
words = ['running', 'flies', 'happily', 'studying']
stemmed = [ps.stem(w) for w in words]
print('Stemmed:', stemmed)  # ['run', 'fli', 'happili', 'studi']

# Lemmatization
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('running', pos='v'))  # run
print(lemmatizer.lemmatize('better', pos='a'))   # good
print(lemmatizer.lemmatize('dogs'))              # dog
```

#### D. Part-of-Speech (POS) Tagging

POS tagging identifies grammatical roles of words: nouns (NN), verbs (VB), adjectives (JJ), etc.

```python
import nltk
from nltk.tokenize import word_tokenize

text = 'The quick brown fox jumps over the lazy dog'
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

print(pos_tags)
# [('The','DT'),('quick','JJ'),('brown','JJ'),('fox','NN'),
#  ('jumps','VBZ'),('over','IN'),('the','DT'),('lazy','JJ'),('dog','NN')]
```

#### E. Named Entity Recognition (NER)

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag

sentence = 'Barack Obama was born in Honolulu, Hawaii.'
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)

print(entities)
# Tree: (S (PERSON Barack/NNP Obama/NNP) was/VBD born/VBN
#       in/IN (GPE Honolulu/NNP) ,/, (GPE Hawaii/NNP) ./.)
```

#### F. Frequency Distribution

```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize

text = 'NLP is fun. NLP is interesting. NLTK makes NLP easy.'
tokens = word_tokenize(text.lower())
fdist = FreqDist(tokens)

print(fdist.most_common(5))
# [('.', 3), ('nlp', 3), ('is', 2), ('fun', 1), ('interesting', 1)]

fdist.plot(10, cumulative=False)  # Visualize top 10 words
```

#### G. Sentiment Analysis with VADER

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
text = 'NLTK is absolutely fantastic and very easy to use!'
scores = sia.polarity_scores(text)

print(scores)
# {'neg': 0.0, 'neu': 0.328, 'pos': 0.672, 'compound': 0.8658}

if scores['compound'] >= 0.05:
    print('Positive')
elif scores['compound'] <= -0.05:
    print('Negative')
else:
    print('Neutral')
```

### 6.2.3 NLTK Key Points Summary

| **Feature**          | **Description**                                                    |
|----------------------|--------------------------------------------------------------------|
| **License**          | Apache License 2.0 (open source)                                   |
| **Language Support** | English (primary), 50+ corpora for other languages                 |
| **Best For**         | Education, research, classical NLP pipelines                       |
| **Speed**            | Slow for large-scale production use                                |
| **Key Modules**      | tokenize, stem, corpus, tag, chunk, parse, sentiment               |
| **Corpora**          | WordNet, Brown, Reuters, Gutenberg, Twitter, and 50+ more          |

---

## 6.3 spaCy - Industrial-Strength NLP

spaCy is a modern, high-performance NLP library developed by Explosion AI. Unlike NLTK, spaCy is built for production use: it is significantly faster, uses pre-trained statistical models, and supports modern deep learning backends. spaCy processes text as a pipeline and stores all results in a single `Doc` object, making it memory-efficient and easy to extend.

### 6.3.1 Installation & Model Download

```bash
# Install spaCy
pip install spacy

# Download pre-trained English model (small)
python -m spacy download en_core_web_sm

# Medium model (better accuracy)
python -m spacy download en_core_web_md

# Large model with word vectors
python -m spacy download en_core_web_lg

# Transformer-based model (best accuracy, needs GPU recommended)
pip install spacy[transformers]
python -m spacy download en_core_web_trf
```

### 6.3.2 The spaCy Pipeline

spaCy processes text through a series of components (pipeline stages). Each component annotates the `Doc` object in sequence.

| **Component**         | **Name**       | **Function**                                          |
|-----------------------|----------------|-------------------------------------------------------|
| **Tokenizer**         | tokenizer      | Splits text into tokens (words, punctuation)          |
| **Tagger**            | tagger         | Assigns part-of-speech tags                           |
| **DependencyParser**  | parser         | Determines syntactic dependency structure             |
| **NER**               | ner            | Detects named entities (Person, Org, GPE...)          |
| **Lemmatizer**        | lemmatizer     | Finds base form of each token                         |
| **Morphologizer**     | morphologizer  | Predicts morphological features                       |
| **SentenceRecognizer**| senter         | Identifies sentence boundaries                        |

### 6.3.3 Core Features

#### A. Basic NLP Processing

```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')
text = 'Apple is looking at buying U.K. startup for $1 billion.'
doc = nlp(text)

# Tokens
for token in doc:
    print(f'{token.text:<15} POS: {token.pos_:<8} Lemma: {token.lemma_}')

# Named Entities
for ent in doc.ents:
    print(f'Entity: {ent.text:<20} Label: {ent.label_}')
    # Apple → ORG, U.K. → GPE, $1 billion → MONEY
```

#### B. Dependency Parsing

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp('The cat sat on the mat.')

for token in doc:
    print(f'{token.text:<10} dep: {token.dep_:<12} head: {token.head.text}')
    # cat → nsubj (subject of 'sat')
    # sat → ROOT (root of sentence)
    # mat → pobj (object of preposition 'on')
```

#### C. Word Vectors & Similarity

```python
import spacy

nlp = spacy.load('en_core_web_md')  # needs md or lg for vectors

doc1 = nlp('I like fast food')
doc2 = nlp('I love eating pizza')
print(f'Doc similarity: {doc1.similarity(doc2):.4f}')  # ~0.78

# Token-level similarity
king = nlp('king')[0]
queen = nlp('queen')[0]
print(f'king <-> queen: {king.similarity(queen):.4f}')  # ~0.75
```

#### D. Custom Pipeline Component

```python
import spacy
from spacy.language import Language

@Language.component('custom_component')
def my_component(doc):
    print(f'Processing doc of length {len(doc)}')
    return doc

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('custom_component', first=True)

doc = nlp('Hello spaCy!')
print(nlp.pipe_names)  # ['custom_component', 'tok2vec', 'tagger', ...]
```

#### E. Matcher - Rule-Based Matching

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Pattern: 'iPhone' followed by optional digit
pattern = [{'LOWER': 'iphone'}, {'IS_DIGIT': True, 'OP': '?'}]
matcher.add('IPHONE', [pattern])

doc = nlp('I bought iPhone 12 and also an iphone.')
matches = matcher(doc)

for match_id, start, end in matches:
    print('Match:', doc[start:end].text)  # 'iPhone 12', 'iphone'
```

#### F. Training a Custom NER Model

```python
import spacy
from spacy.training import Example

# Training data format
TRAIN_DATA = [
    ('Tesla released Model X.', {'entities': [(0, 5, 'ORG'), (14, 21, 'PRODUCT')]}),
    ('Elon Musk founded SpaceX.', {'entities': [(0, 9, 'PERSON'), (18, 24, 'ORG')]}),
]

nlp = spacy.blank('en')
ner = nlp.add_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

optimizer = nlp.begin_training()

for itn in range(20):
    for text, annotation in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotation)
        nlp.update([example], sgd=optimizer)

nlp.to_disk('./custom_ner_model')
```

### 6.3.4 spaCy Key Points Summary

| **Feature**              | **Detail**                                                       |
|--------------------------|------------------------------------------------------------------|
| **Speed**                | ~10x faster than NLTK on equivalent tasks                        |
| **Pipeline**             | Modular, composable, supports custom components                  |
| **Models Available**     | sm (12MB), md (43MB), lg (741MB), trf (transformer-based)        |
| **Supported Languages**  | 70+ languages with trained pipelines                             |
| **Integrations**         | Transformers (BERT/GPT), scikit-learn, PyTorch, TensorFlow       |
| **Key Objects**          | Doc, Token, Span, Matcher, PhraseMatcher, EntityRuler            |

---

## 6.4 Gensim - Topic Modeling & Document Similarity

Gensim is an open-source Python library specializing in unsupervised topic modeling and natural language processing. It is designed to handle large text corpora efficiently using data streaming and online (incremental) learning algorithms. Gensim is the de-facto library for Word2Vec, Doc2Vec, LSA, and LDA topic models.

### 6.4.1 Installation

```bash
pip install gensim

# Optional: faster training on large corpora
pip install gensim[distributed]
```

```python
import gensim
print(gensim.__version__)  # e.g., 4.3.2
```

### 6.4.2 Core Features

#### A. Word2Vec - Word Embeddings

Word2Vec trains shallow neural networks to produce dense vector representations of words. Semantically similar words are close in vector space.

```python
from gensim.models import Word2Vec

# Training corpus: list of tokenized sentences
sentences = [
    ['machine', 'learning', 'is', 'amazing'],
    ['deep', 'learning', 'and', 'neural', 'networks'],
    ['natural', 'language', 'processing', 'with', 'python'],
    ['word', 'embeddings', 'capture', 'semantic', 'meaning'],
]

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,     # Dimensionality of word vectors
    window=5,            # Context window size
    min_count=1,         # Ignore words with freq < min_count
    workers=4,           # Parallel threads
    sg=0,                # 0=CBOW, 1=Skip-gram
    epochs=10
)

# Vocabulary
print(list(model.wv.key_to_index.keys())[:5])

# Word vector
vector = model.wv['learning']
print('Vector shape:', vector.shape)  # (100,)

# Most similar words
similar = model.wv.most_similar('learning', topn=3)
print(similar)

# Save and load
model.save('word2vec.model')
loaded_model = Word2Vec.load('word2vec.model')
```

#### B. LDA - Latent Dirichlet Allocation (Topic Modeling)

LDA is a generative probabilistic model that discovers latent topics in a document corpus. Each document is represented as a mixture of topics.

```python
from gensim import corpora, models
from gensim.utils import simple_preprocess

# Sample documents
documents = [
    'Machine learning algorithms improve with data',
    'Deep learning uses artificial neural networks',
    'Python is popular for data science and AI',
    'NLP processes human language text data',
    'Topic modeling discovers hidden themes in text',
]

# Preprocessing
texts = [simple_preprocess(doc) for doc in documents]

# Build dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA model
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,        # Number of topics
    passes=15,           # Training iterations
    random_state=42
)

# Print topics
for idx, topic in lda_model.print_topics(num_words=4):
    print(f'Topic {idx}: {topic}')

# Get topic distribution for a new document
new_doc = 'deep learning for text classification'
bow = dictionary.doc2bow(simple_preprocess(new_doc))
topic_dist = lda_model[bow]
print('Topic distribution:', topic_dist)
```

#### C. Doc2Vec - Document Embeddings

Doc2Vec extends Word2Vec to produce embeddings for entire documents (paragraphs). Useful for document similarity, classification, and clustering.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [
    'NLP is great for text analysis',
    'Machine learning transforms AI',
    'Python is powerful for data science',
]

# Tag each document
tagged_docs = [TaggedDocument(words=doc.split(), tags=[i])
               for i, doc in enumerate(documents)]

# Train Doc2Vec
model = Doc2Vec(tagged_docs, vector_size=50, min_count=1, epochs=40)

# Infer vector for a new document
new_doc = 'AI and deep learning for NLP'.split()
vec = model.infer_vector(new_doc)
print('Doc vector shape:', vec.shape)  # (50,)

# Find most similar documents
similar = model.dv.most_similar(0, topn=2)
print('Most similar to doc 0:', similar)
```

#### D. TF-IDF Model

```python
from gensim import corpora, models, similarities

corpus = [[(0,1),(1,1)], [(1,1),(2,1)], [(0,1),(2,1),(3,1)]]

# Train TF-IDF
tfidf = models.TfidfModel(corpus)
transformed = tfidf[corpus[0]]
print('TF-IDF transformed:', transformed)

# Document similarity index
index = similarities.MatrixSimilarity(tfidf[corpus])
query = [(0, 1), (2, 1)]
sims = index[tfidf[query]]
print('Similarities:', list(enumerate(sims)))
```

#### E. Loading Pre-trained Word2Vec (Google News)

```python
import gensim.downloader as api

# Download pre-trained model (1.6GB)
word_vectors = api.load('word2vec-google-news-300')

# Semantic analogy: king - man + woman = ?
result = word_vectors.most_similar(
    positive=['king', 'woman'], negative=['man'], topn=1
)
print(result)  # [('queen', 0.7118)]

# Vocabulary size
print(len(word_vectors))  # 3,000,000 words
```

### 6.4.3 Gensim Key Points Summary

| **Feature**           | **Detail**                                                      |
|-----------------------|-----------------------------------------------------------------|
| **Primary Models**    | Word2Vec, Doc2Vec, FastText, LDA, LSA, TF-IDF, HDP              |
| **Memory Efficiency** | Streaming API -- processes data without loading all into RAM    |
| **Pre-trained Models**| GloVe, Google News Word2Vec, FastText Wikipedia via downloader  |
| **Best For**          | Topic modeling, document similarity, semantic search, embeddings|
| **Training Mode**     | Online (incremental) learning -- suitable for large corpora     |
| **Visualization**     | pyLDAvis compatible for interactive topic visualization         |

---

## 6.5 fastText - Word Embeddings & Text Classification

fastText is an open-source library developed by Facebook AI Research (FAIR) in 2016. It extends Word2Vec by representing each word as a bag of character n-grams (subwords). This allows fastText to handle out-of-vocabulary (OOV) words, rare words, and morphologically rich languages significantly better than standard word embedding approaches. fastText also provides state-of-the-art text classification capabilities.

> **ℹ Key Advantage of fastText**
>
> By using character n-grams, fastText can generate embeddings for ANY word, even if it was never seen during training. For example, the word 'unhappiness' can be represented using the n-grams of 'unhappy', 'happy', etc.

### 6.5.1 Installation

```bash
# Install via pip
pip install fasttext

# Or install via conda
conda install -c conda-forge fasttext

# Python binding (Gensim includes FastText)
pip install gensim
```

```python
import fasttext
print(fasttext.__version__)
```

### 6.5.2 Core Features

#### A. Word Embeddings with Subword Information

```python
import fasttext

# Prepare a text file for training (one sentence per line)
# Save as 'corpus.txt'
with open('corpus.txt', 'w') as f:
    f.write('NLP is a fascinating field of study\n')
    f.write('Machine learning enables intelligent systems\n')
    f.write('Deep neural networks power modern AI applications\n')

# Train FastText word embedding model
model = fasttext.train_unsupervised(
    'corpus.txt',
    model='skipgram',    # or 'cbow'
    dim=100,             # vector dimensions
    ws=5,                # window size
    minCount=1,          # min word count
    minn=3,              # min n-gram length
    maxn=6,              # max n-gram length
    epoch=10
)

# Word vector
vec = model.get_word_vector('learning')
print('Vector shape:', vec.shape)  # (100,)

# OOV: vector for unseen word (subword magic!)
oov_vec = model.get_word_vector('unknownword123')
print('OOV vector shape:', oov_vec.shape)  # (100,) --- still works!

# Nearest neighbors
neighbors = model.get_nearest_neighbors('deep')
print(neighbors)

model.save_model('fasttext_model.bin')
```

#### B. Text Classification

fastText's text classification is extremely fast and achieves competitive accuracy with deep learning classifiers. Training data must be in a specific format with labels prefixed by `__label__`.

```python
# Training data format (train.txt):
# __label__positive This movie was absolutely wonderful!
# __label__negative The film was boring and disappointing.
# __label__positive Great acting and beautiful cinematography.
# __label__negative Terrible plot and bad dialogue.

import fasttext

# Train classifier
classifier = fasttext.train_supervised(
    'train.txt',
    epoch=25,
    lr=0.5,             # learning rate
    wordNgrams=2,       # use bigrams
    dim=100,
    loss='softmax'      # or 'ns', 'hs', 'ova'
)

# Evaluate on test set
result = classifier.test('test.txt')
print(f'Precision: {result[1]:.4f}')
print(f'Recall: {result[2]:.4f}')

# Predict
label, prob = classifier.predict('This product is amazing!')
print(f'Predicted: {label[0]} with probability {prob[0]:.4f}')
# Predicted: __label__positive with probability 0.9821

# Multi-label prediction (top 3)
labels, probs = classifier.predict('Great story and amazing actors', k=3)
print(list(zip(labels, probs)))
```

#### C. FastText via Gensim

```python
from gensim.models import FastText

sentences = [
    ['natural', 'language', 'processing'],
    ['machine', 'learning', 'algorithms'],
    ['deep', 'learning', 'neural', 'networks'],
]

# Train FastText model through Gensim
ft_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=3,
    min_count=1,
    min_n=3,            # min char n-gram
    max_n=6,            # max char n-gram
    epochs=10
)

# Get word vector
print(ft_model.wv['learning'].shape)     # (100,)

# OOV handling
print(ft_model.wv['learninggg'].shape)   # (100,) -- works via subwords

# Word similarity
print(ft_model.wv.most_similar('deep', topn=3))
```

#### D. Loading Pre-trained fastText Vectors

```python
import fasttext
import fasttext.util

# Download pre-trained English model (trained on Common Crawl + Wikipedia)
fasttext.util.download_model('en', if_exists='ignore')  # cc.en.300.bin

ft = fasttext.load_model('cc.en.300.bin')

# Reduce dimensions (optional, for memory efficiency)
fasttext.util.reduce_model(ft, 100)

vec = ft.get_word_vector('technology')
print('Vector shape:', vec.shape)  # (100,)

# Available languages: 157 languages supported
# Hindi: cc.hi.300.bin, French: cc.fr.300.bin, etc.
```

### 6.5.3 fastText Key Points Summary

| **Feature**                  | **Detail**                                                    |
|------------------------------|---------------------------------------------------------------|
| **Subword Modeling**         | Char n-grams (e.g., minn=3, maxn=6) enable OOV word handling |
| **Classification Speed**     | ~10 million sentences/sec on a standard CPU                   |
| **Pre-trained Models**       | 157 languages trained on Wikipedia + Common Crawl (300-dim)   |
| **Best For**                 | Text classification, rare word handling, morphologically rich languages |
| **Training Mode**            | Supervised (classification) and unsupervised (embeddings)     |
| **n-gram for Classification**| wordNgrams=2 gives strong accuracy boost over unigrams        |

---

## 6.6 Comparative Analysis

### 6.6.1 Feature Comparison Table

| **Criterion**            | **NLTK**              | **spaCy**                    | **Gensim**              | **fastText**                  |
|--------------------------|-----------------------|------------------------------|-------------------------|-------------------------------|
| **Primary Focus**        | Education & Research  | Production NLP               | Topic Modeling          | Word Embeddings & Classification |
| **Speed**                | Slow                  | Very Fast                    | Fast (streaming)        | Extremely Fast                |
| **Ease of Use**          | Moderate              | High                         | Moderate                | High                          |
| **Pre-trained Models**   | Limited               | Excellent                    | Good                    | Excellent (157 langs)         |
| **Neural Network Support** | Limited             | Full (via spacy-transformers)| Word2Vec/Doc2Vec        | Built-in                      |
| **OOV Handling**         | No                    | No                           | No (standard)           | Yes (subword n-grams)         |
| **Topic Modeling**       | No                    | No                           | Yes (LDA, LSA)          | No                            |
| **Text Classification**  | Basic                 | Yes (TextCategorizer)        | No                      | Yes (very fast)               |
| **Language Support**     | ~50 corpora           | 70+ pipelines                | Any (train own)         | 157 languages                 |
| **Production Use**       | Not recommended       | Yes                          | Yes                     | Yes                           |

### 6.6.2 When to Use Which Library

| **Use Case**                                   | **Recommended Library**            |
|------------------------------------------------|------------------------------------|
| **Learning NLP fundamentals / coursework**     | NLTK                               |
| **Building production NLP APIs or pipelines**  | spaCy                              |
| **Text preprocessing (tokenize, stem, POS tag)** | NLTK or spaCy                    |
| **Named Entity Recognition at scale**          | spaCy                              |
| **Topic modeling (LDA, LSA)**                  | Gensim                             |
| **Document similarity / semantic search**      | Gensim (TF-IDF, Doc2Vec)           |
| **Word embeddings (Word2Vec)**                 | Gensim or spaCy (md/lg)            |
| **Handling rare or OOV words**                 | fastText                           |
| **Text classification (fast training)**        | fastText                           |
| **Multilingual NLP (157 languages)**           | fastText                           |
| **Transformer-based NLP (BERT, GPT)**          | spaCy + spacy-transformers         |
| **Large-scale corpus processing**              | Gensim (streaming)                 |

### 6.6.3 End-to-End Pipeline Example

The following example demonstrates how to combine NLTK (preprocessing), spaCy (NER), and Gensim (TF-IDF) in a unified text analysis pipeline:

```python
import nltk
import spacy
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 1. NLTK Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t.isalpha() and t not in stop_words]
    return tokens

docs = ['NLP enables computers to understand human language.',
        'Machine learning models process text efficiently.',
        'Deep learning transforms natural language tasks.']

processed = [preprocess(doc) for doc in docs]

# 2. Gensim TF-IDF
dictionary = corpora.Dictionary(processed)
corpus = [dictionary.doc2bow(doc) for doc in processed]
tfidf = models.TfidfModel(corpus)

# 3. spaCy NER
nlp = spacy.load('en_core_web_sm')

for doc in docs:
    spacy_doc = nlp(doc)
    entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
    print(f'Entities in: "{doc[:40]}..." → {entities}')
```

---

## 6.7 Lab Exercises

### Exercise 1: NLTK Text Analysis Pipeline

Build a complete preprocessing pipeline using NLTK that performs: tokenization, stop word removal, stemming, POS tagging, and frequency distribution on any paragraph of text.

> **ℹ Expected Output**
>
> A cleaned token list, POS-tagged pairs, a frequency distribution object, and a bar chart of the top-10 most frequent words.

### Exercise 2: spaCy NER and Dependency Visualization

Process a news article using spaCy. Extract and categorize all named entities. Visualize the dependency parse tree for the first 3 sentences using `displacy.render()`. Identify and print all subject-verb-object (SVO) triplets.

### Exercise 3: Gensim Word2Vec Training

Train a Word2Vec model on the 20 Newsgroups dataset (available via `sklearn.datasets`). Find the top-5 most similar words for: 'computer', 'science', 'government'. Demonstrate the `king - man + woman = queen` analogy on your trained model.

### Exercise 4: fastText Sentiment Classifier

Using the IMDB movie review dataset, create a fastText classifier. Train with `wordNgrams=2`, `epoch=25`, `lr=0.5`. Evaluate precision/recall on the test set. Compare performance with and without subword information (`minn=0` vs `minn=3`).

### Exercise 5: LDA Topic Modeling

Apply Gensim's LDA to the Reuters Corpus from NLTK (available via `nltk.corpus.reuters`). Discover 10 latent topics. Visualize using `pyLDAvis`. For each topic, identify and label the dominant theme based on the top words.

---

## 6.8 Summary

In this lab, we have explored four of the most powerful NLP libraries in the Python ecosystem. Each library occupies a distinct niche and excels in different scenarios:

- **NLTK** remains the go-to library for learning NLP concepts, academic research, and classical linguistic analysis. Its extensive collection of corpora, lexical databases (WordNet), and interpretable algorithms make it ideal for beginners and researchers exploring traditional NLP methods.

- **spaCy** is the industry standard for production-grade NLP. Its speed, modern pre-trained models, modular pipeline architecture, and seamless integration with transformer models (BERT, RoBERTa) make it the preferred choice for building scalable NLP applications.

- **Gensim** specializes in unsupervised learning at scale. Its memory-efficient streaming APIs, powerful topic modeling algorithms (LDA, LSA, HDP), and state-of-the-art word embedding implementations (Word2Vec, Doc2Vec) make it invaluable for large corpus analysis, document similarity, and semantic information retrieval.

- **fastText** brings breakthrough performance in word embedding and text classification through its innovative subword (character n-gram) representation. Its ability to handle out-of-vocabulary words, support for 157 languages, and blazing-fast classification speed make it particularly powerful for multilingual NLP and low-resource language settings.

A mature NLP practitioner combines these libraries strategically: using NLTK for educational prototyping, spaCy for production pipelines, Gensim for topic discovery and semantic search, and fastText when classification speed or multilingual coverage is paramount.

### Quick Decision Guide

| **If you need...**                           | **Use**                               |
|----------------------------------------------|---------------------------------------|
| **To learn NLP from scratch**                | NLTK                                  |
| **Fast, production-ready pipelines**         | spaCy                                 |
| **Topic modeling on large corpora**          | Gensim                                |
| **Fast text classification**                 | fastText                              |
| **Semantic similarity between documents**    | Gensim (Doc2Vec / TF-IDF)             |
| **Handling words not in training vocab**     | fastText                              |
| **Pre-trained multilingual models**          | fastText (157 langs) or spaCy (70+)   |
| **Transformer integration (BERT)**           | spaCy + spacy-transformers            |

---

*End of Lab 6 - NLP Libraries*
