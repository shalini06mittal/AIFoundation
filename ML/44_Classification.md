# MACHINE LEARNING
## A Complete Beginner-to-Advanced Guide

---

# Chapter 4: Classification Algorithms

*Continuing from Chapters 1 – 3:*

*Ch.1 — What is ML? | Ch.2 — How Machines Learn | Ch.3 — Types of ML*

---

## Table of Contents

- [Chapter Overview](#chapter-overview)
- [4.1 k-Nearest Neighbors (kNN)](#41-k-nearest-neighbors-knn)
  - [The Core Idea — An Everyday Analogy](#the-core-idea--an-everyday-analogy)
  - [How Distance is Measured](#how-distance-is-measured)
  - [The kNN Algorithm — Step by Step](#the-knn-algorithm--step-by-step)
  - [Choosing K — The Make-or-Break Parameter](#choosing-k--the-make-or-break-parameter)
  - [Strengths and Weaknesses](#strengths-and-weaknesses-of-knn)
  - [Implementation in Python](#implementation-in-python-knn)
- [4.2 Naive Bayes](#42-naive-bayes)
  - [The Core Idea — Probability from Evidence](#the-core-idea--probability-from-evidence)
  - [Bayes' Theorem — The Foundation](#bayes-theorem--the-foundation)
  - [The "Naive" Assumption](#the-naive-assumption)
  - [Types of Naive Bayes](#types-of-naive-bayes)
  - [Worked Example — Email Spam Detection](#worked-example--email-spam-detection)
  - [Strengths and Weaknesses](#strengths-and-weaknesses-of-naive-bayes)
  - [Implementation in Python](#implementation-in-python-naive-bayes)
- [4.3 Decision Trees](#43-decision-trees)
  - [The Core Idea — Ask Questions and Split](#the-core-idea--ask-questions-and-split)
  - [How Trees Split — Information Gain & Gini Impurity](#how-trees-split--information-gain--gini-impurity)
  - [Building a Tree — The Complete Algorithm](#building-a-tree--the-complete-algorithm)
  - [Overfitting and Pruning](#overfitting-and-pruning)
  - [Strengths and Weaknesses](#strengths-and-weaknesses-of-decision-trees)
  - [Implementation in Python](#implementation-in-python-decision-trees)
- [4.4 Comparing the Three Algorithms](#44-comparing-the-three-algorithms)
  - [Side-by-Side Comparison](#side-by-side-comparison)
  - [Decision Guide — Which Algorithm to Pick?](#decision-guide--which-algorithm-to-pick)
  - [Same Dataset, Three Algorithms — Results Compared](#same-dataset-three-algorithms--results-compared)
- [Chapter 4 Summary](#chapter-4-summary)

---

## Chapter Overview

In Chapter 3 we saw that Supervised Learning includes **Classification** — predicting which category a new example belongs to. This chapter goes deep on **three foundational classification algorithms**, each built on a completely different philosophy of intelligence.

We start from pure intuition, build up to the mathematics, trace every step through worked examples, and finish with a practical guide to knowing which algorithm to pick.

> ### 🗺️ Three Algorithms, Three Philosophies
>
> **kNN** — 'Show me your neighbours and I will tell you who you are.'  
> Classify by similarity. No training. Memorise everything.
>
> **Naive Bayes** — 'What is the probability of each class, given this evidence?'  
> Classify by probability theory. Fast. Works on text.
>
> **Decision Tree** — 'Ask the most useful question first, then split.'  
> Classify by divide-and-conquer. Human-readable rules.

---

## 4.1 k-Nearest Neighbors (kNN)

> ### 📍 k-Nearest Neighbors (kNN)
> *Lazy Learning — Classify by similarity to known examples*

### The Core Idea — An Everyday Analogy

You move to a new city and want to know whether a neighbourhood is safe. You do not study criminology — you just ask the **five families living closest to you**. If four of them say it is safe and one says it is not, you conclude it is probably safe.

That is **kNN**. To classify a new, unknown data point, look at the **K training examples most similar to it**, and take a **majority vote**.

```
┌─────────────────────────────────────────────────────────────┐
│ 🏘️ kNN — The Neighbourhood Vote                            │
├─────────────────────────────────────────────────────────────┤
│ Training data: fruit classification                          │
│                                                              │
│ Features: Weight (grams) and Colour score (0 = green, 10 = red)│
│                                                              │
│ Known fruits in training set:                                │
│   Apple A1:  Weight=180g, Colour=9  → Label: APPLE          │
│   Apple A2:  Weight=165g, Colour=8  → Label: APPLE          │
│   Apple A3:  Weight=175g, Colour=9  → Label: APPLE          │
│   Orange O1: Weight=210g, Colour=5  → Label: ORANGE         │
│   Orange O2: Weight=225g, Colour=4  → Label: ORANGE         │
│   Lemon L1:  Weight=120g, Colour=2  → Label: LEMON          │
│                                                              │
│ NEW unknown fruit: Weight=170g, Colour=8                     │
│ Question: What is it?                                        │
│                                                              │
│ k = 3: Find the 3 nearest neighbours...                     │
│   A1 (distance = 10.05) → APPLE                             │
│   A2 (distance = 5.39)  → APPLE ← closest                   │
│   A3 (distance = 7.07)  → APPLE                             │
│                                                              │
│ Vote: APPLE=3, ORANGE=0, LEMON=0                            │
│ Prediction: APPLE (unanimous!)                              │
└─────────────────────────────────────────────────────────────┘
```

### How Distance is Measured

The core of kNN is measuring **similarity** — or equivalently, **distance**. There are several distance metrics to choose from:

```
┌─────────────────────────────────────────────────────────────┐
│ 📐 Distance Metrics in kNN                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ EUCLIDEAN DISTANCE (most common — 'straight line distance') │
│   d(P, Q) = sqrt( (x2-x1)² + (y2-y1)² + ... )              │
│                                                              │
│   Example: P = (Weight=170, Colour=8)                       │
│            Q = (Weight=165, Colour=8)                       │
│   d = sqrt( (170-165)² + (8-8)² )                          │
│     = sqrt( 25 + 0 ) = sqrt(25) = 5.0                      │
│                                                              │
│ MANHATTAN DISTANCE ('city block distance')                  │
│   d(P, Q) = |x2-x1| + |y2-y1| + ...                        │
│                                                              │
│   Example: d = |170-165| + |8-8| = 5 + 0 = 5.0             │
│                                                              │
│ MINKOWSKI DISTANCE (general form)                           │
│   p=2 gives Euclidean, p=1 gives Manhattan                  │
│   d(P, Q) = ( |x2-x1|^p + |y2-y1|^p )^(1/p)                │
│                                                              │
│ COSINE SIMILARITY (for text/high-dimensional data)          │
│   Measures the angle between two vectors, not absolute distance│
│   similarity = (A · B) / (||A|| × ||B||)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

> ### ⚠️ Feature Scaling is Critical for kNN
>
> Without scaling, features with larger ranges dominate the distance calculation.
>
> **Example:**
> - Feature 1: House size (500–5,000 sq ft)
> - Feature 2: Number of bedrooms (1–5)
>
> Without scaling, size differences (e.g., 1,000) dwarf bedroom differences (e.g., 2), making bedrooms irrelevant.
>
> **Solution:** Scale all features to the same range (e.g., 0–1 or mean=0, std=1).

```python
from sklearn.preprocessing import StandardScaler

# Feature scaling before kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now all features have mean=0, std=1
```

### The kNN Algorithm — Step by Step

Here is the complete kNN classification process:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔄 The kNN Algorithm (Classification)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ INPUT:                                                       │
│   • Training data: X_train (features), y_train (labels)     │
│   • New data point: x_new                                   │
│   • Hyperparameter: k (number of neighbours)                │
│                                                              │
│ ALGORITHM:                                                   │
│                                                              │
│ 1. STORE TRAINING DATA                                       │
│    → No computation at this stage (lazy learning)           │
│    → Just memorise all training examples                    │
│                                                              │
│ 2. COMPUTE DISTANCES                                         │
│    → For each training point x_i in X_train:                │
│        Calculate distance(x_new, x_i)                       │
│    → You now have N distances (N = training set size)       │
│                                                              │
│ 3. FIND K NEAREST NEIGHBOURS                                 │
│    → Sort distances from smallest to largest                │
│    → Select the K training points with smallest distances   │
│                                                              │
│ 4. MAJORITY VOTE                                             │
│    → Count the class labels of the K nearest neighbours     │
│    → Predict the class with the most votes                  │
│                                                              │
│ OUTPUT:                                                      │
│   • Predicted class label for x_new                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Worked Example:**

```python
# Training data (already stored in memory)
training_data = [
    ([180, 9], 'Apple'),   # (weight, colour) → label
    ([165, 8], 'Apple'),
    ([175, 9], 'Apple'),
    ([210, 5], 'Orange'),
    ([225, 4], 'Orange'),
    ([120, 2], 'Lemon')
]

# New data point to classify
x_new = [170, 8]
k = 3

# Step 1: Compute all distances (using Euclidean)
distances = []
for features, label in training_data:
    dist = sqrt((170-features[0])**2 + (8-features[1])**2)
    distances.append((dist, label))

# Result:
# [(10.05, 'Apple'), (5.39, 'Apple'), (7.07, 'Apple'),
#  (40.31, 'Orange'), (55.36, 'Orange'), (51.48, 'Lemon')]

# Step 2: Sort by distance and select K=3 nearest
distances.sort()  # Sort by distance
nearest_k = distances[:3]  # Take first 3

# Result: [(5.39, 'Apple'), (7.07, 'Apple'), (10.05, 'Apple')]

# Step 3: Majority vote
votes = {'Apple': 3, 'Orange': 0, 'Lemon': 0}
prediction = 'Apple'  # Winner!
```

### Choosing K — The Make-or-Break Parameter

The value of **K** dramatically affects model behaviour:

```
┌─────────────────────────────────────────────────────────────┐
│ 🎚️ The Effect of K on kNN Performance                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  K=1 (Too Small)                                             │
│   • Uses only the single nearest neighbour                  │
│   • Very sensitive to noise and outliers                    │
│   • OVERFITS — memorises training noise                     │
│   • Decision boundary: jagged, complex                      │
│                                                              │
│  K=5 to K=10 (Just Right)                                    │
│   • Balances local patterns with noise resistance           │
│   • GENERALISES well to new data                            │
│   • Decision boundary: smooth, robust                       │
│                                                              │
│  K=N (Too Large, where N = total training samples)          │
│   • Uses all training data for every prediction             │
│   • Always predicts the majority class                      │
│   • UNDERFITS — ignores local patterns                      │
│   • Decision boundary: overly simple                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Visual Comparison:**

```
K=1 (Overfitting)         K=5 (Good)              K=100 (Underfitting)
Decision boundary:        Decision boundary:       Decision boundary:

    ○ ● ○                     ○ ● ○                   ○ ● ○
  ● ○ ● ○ ●               ● ○ ● ○ ●               ● ○ ● ○ ●
○ ● ○ ● ○ ●             ○ ● ○ ● ○ ●             ○ ● ○ ● ○ ●
  /‾\/‾\/‾\                 /‾‾‾‾‾‾‾\               ────────────
 Very jagged              Smooth curve            Straight line
 (memorises noise)        (captures pattern)      (ignores pattern)
```

**How to Choose K:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Try different K values
k_values = range(1, 31)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Use cross-validation to get reliable estimate
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    scores.append(cv_scores.mean())

# Find best K
best_k = k_values[np.argmax(scores)]
print(f"Best K: {best_k}")

# Plot K vs accuracy to visualise
import matplotlib.pyplot as plt
plt.plot(k_values, scores)
plt.xlabel('K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding Optimal K')
plt.show()
```

**Rules of Thumb:**

| Dataset Size | Recommended K Range |
|--------------|---------------------|
| < 100 samples | k = 3 to 5 |
| 100–1,000 samples | k = 5 to 10 |
| 1,000–10,000 samples | k = 10 to 20 |
| > 10,000 samples | k = 20 to 50 |

> **Pro Tip:** Use an **odd K** for binary classification to avoid ties.

### Strengths and Weaknesses of kNN

| ✅ Strengths | ❌ Weaknesses |
|-------------|--------------|
| **Simple to understand and implement** — no complex math | **Computationally expensive at prediction** — must scan all training data |
| **No training phase** — just stores data | **Memory intensive** — stores entire training set |
| **Naturally handles multi-class problems** — just count votes | **Sensitive to irrelevant features** — all features affect distance |
| **Non-parametric** — makes no assumptions about data distribution | **Requires feature scaling** — large-range features dominate |
| **Adapts to complex decision boundaries** — can fit any shape | **Poor on high-dimensional data** — curse of dimensionality |
| **Works well on small datasets** — no overfitting on tiny data | **Sensitive to outliers and noise** — especially with small K |

**When to Use kNN:**

```
✅ USE kNN when:
  • Dataset is small (< 10,000 samples)
  • Features are low-dimensional (< 20 features)
  • Decision boundaries are irregular/complex
  • You need a quick baseline model
  • You want zero training time

❌ AVOID kNN when:
  • Dataset is large (> 100,000 samples) → too slow
  • Features are high-dimensional (> 100) → curse of dimensionality
  • You need fast real-time predictions → use Decision Tree or Naive Bayes
  • Features have very different scales → must scale first
```

### Implementation in Python (kNN)

**Complete Example: Iris Flower Classification**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Feature scaling (CRITICAL for kNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train kNN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,           # K=5 neighbours
    weights='uniform',       # All neighbours equally weighted
    metric='euclidean'       # Distance metric
)
knn.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = knn.predict(X_test_scaled)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Expected Output:**

```
Accuracy: 0.978

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        19
  versicolor       1.00      0.92      0.96        13
   virginica       0.93      1.00      0.96        13

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.97        45
weighted avg       0.98      0.98      0.98        45

Confusion Matrix:
[[19  0  0]
 [ 0 12  1]
 [ 0  0 13]]
```

**Advanced kNN Options:**

```python
# Weighted kNN — closer neighbours have more influence
knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'  # Weight by 1/distance
)

# Different distance metrics
knn_manhattan = KNeighborsClassifier(
    n_neighbors=5,
    metric='manhattan'  # L1 distance instead of L2
)

# Custom distance function
def custom_distance(x, y):
    return np.sum(np.abs(x - y))  # Manhattan distance

knn_custom = KNeighborsClassifier(
    n_neighbors=5,
    metric=custom_distance
)

# For very large datasets, use approximate nearest neighbours
from sklearn.neighbors import NearestNeighbors
# Use algorithms like 'ball_tree' or 'kd_tree' for speed
knn_fast = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree'  # Faster for low-dimensional data
)
```

---

## 4.2 Naive Bayes

> ### 🎲 Naive Bayes
> *Probabilistic Learning — Classify by calculating probabilities*

### The Core Idea — Probability from Evidence

You receive an email with the words "Congratulations! You have won!" Should you trust it? Most people instantly think "spam" — not because they memorised spam patterns, but because they intuitively calculate: *Given these words, what is the probability this is spam versus legitimate?*

That is **Naive Bayes**. It classifies by computing: **P(Class | Evidence)** — the probability of each class given the observed features.

```
┌─────────────────────────────────────────────────────────────┐
│ 📧 Email Classification Example                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Email: "Congratulations! Free prize money!"                 │
│                                                              │
│ Question: Spam or Legitimate?                               │
│                                                              │
│ Naive Bayes computes:                                       │
│   P(Spam | "congratulations", "free", "prize", "money")     │
│   P(Legit | "congratulations", "free", "prize", "money")    │
│                                                              │
│ Result:                                                      │
│   P(Spam)  = 0.95  ← Much higher!                           │
│   P(Legit) = 0.05                                            │
│                                                              │
│ Prediction: SPAM (because 0.95 > 0.05)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Bayes' Theorem — The Foundation

Naive Bayes is built entirely on **Bayes' Theorem** from probability theory:

```
┌─────────────────────────────────────────────────────────────┐
│ 📐 Bayes' Theorem                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│         P(Class | Features) = P(Features | Class) × P(Class)│
│                               ─────────────────────────────  │
│                                    P(Features)               │
│                                                              │
│ Where:                                                       │
│   P(Class | Features) = Posterior probability               │
│                         (what we want to find)              │
│                                                              │
│   P(Features | Class) = Likelihood                          │
│                         (how often these features appear    │
│                          in this class)                     │
│                                                              │
│   P(Class)            = Prior probability                   │
│                         (how common is this class overall)  │
│                                                              │
│   P(Features)         = Evidence                            │
│                         (normalisation constant — we can    │
│                          ignore it for classification)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**For Classification:**

We compare probabilities for each class and pick the highest:

```python
# We want:
class_prediction = argmax[ P(Class | Features) ]

# By Bayes' Theorem:
class_prediction = argmax[ P(Features | Class) × P(Class) / P(Features) ]

# P(Features) is the same for all classes, so we can ignore it:
class_prediction = argmax[ P(Features | Class) × P(Class) ]
```

### The "Naive" Assumption

The algorithm is called "**Naive**" because it makes a simplifying assumption: **all features are independent given the class**.

```
┌─────────────────────────────────────────────────────────────┐
│ 🤔 The Naive Independence Assumption                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ WITHOUT the naive assumption:                               │
│   P(word1, word2, word3 | Spam)                             │
│     = P(word1 | word2, word3, Spam) ×                       │
│       P(word2 | word3, Spam) ×                              │
│       P(word3 | Spam) × P(Spam)                             │
│   → VERY COMPLEX — too many conditional probabilities!      │
│                                                              │
│ WITH the naive assumption (features are independent):       │
│   P(word1, word2, word3 | Spam)                             │
│     = P(word1 | Spam) ×                                     │
│       P(word2 | Spam) ×                                     │
│       P(word3 | Spam) × P(Spam)                             │
│   → SIMPLE — just multiply individual probabilities!        │
│                                                              │
│ Formula:                                                     │
│   P(Features | Class) = ∏ P(feature_i | Class)              │
│                         i=1..n                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Why This Works (Despite Being "Naive"):**

In reality, features are often **not** independent. For example, in emails:
- "Free" and "prize" often appear together
- "Meeting" and "agenda" are correlated

But surprisingly, **Naive Bayes works well anyway** because:
1. We only care about **relative probabilities** (Spam vs Legit), not absolute values
2. The independence assumption affects all classes equally, so rankings stay correct
3. It works especially well when features are **weakly correlated** or **high-dimensional** (like text)

### Types of Naive Bayes

There are **three main variants**, depending on the type of features:

| Variant | Feature Type | Use Cases | Distribution Assumption |
|---------|--------------|-----------|------------------------|
| **Gaussian Naive Bayes** | Continuous (real numbers) | Sensor data, measurements, numerical features | Features follow a normal (Gaussian) distribution |
| **Multinomial Naive Bayes** | Discrete counts (non-negative integers) | Text classification, word counts, document topics | Features are word frequencies or counts |
| **Bernoulli Naive Bayes** | Binary (0/1, True/False) | Presence/absence of words, binary features | Features are binary indicators |

#### Gaussian Naive Bayes

For **continuous features** (e.g., height, weight, temperature):

```
P(feature | Class) = (1 / √(2πσ²)) × exp(-(x - μ)² / 2σ²)

Where:
  μ = mean of feature values for this class
  σ² = variance of feature values for this class
```

**Example:**

```python
from sklearn.naive_bayes import GaussianNB

# Features: height (cm), weight (kg)
# Classes: Male, Female

model = GaussianNB()
model.fit(X_train, y_train)

# For prediction, it calculates:
# P(Male | height=175, weight=70) 
#   = P(height=175 | Male) × P(weight=70 | Male) × P(Male)
# Where P(height=175 | Male) uses Gaussian distribution with
#   μ_male = mean height of males in training data
#   σ_male = std dev of male heights
```

#### Multinomial Naive Bayes

For **word counts** or **frequency data** (e.g., text classification):

```
P(word | Class) = (count of word in Class + α) / (total words in Class + α × vocabulary size)

Where α is a smoothing parameter (default α=1, called Laplace smoothing)
```

**Example:**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Convert text to word count vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(emails_train)

# Train
model = MultinomialNB(alpha=1.0)  # Laplace smoothing
model.fit(X_train_counts, y_train)
```

#### Bernoulli Naive Bayes

For **binary features** (word present/absent, not counts):

```
P(word=1 | Class) = (count of documents with word in Class + α) / (total documents in Class + 2α)
P(word=0 | Class) = 1 - P(word=1 | Class)
```

**Example:**

```python
from sklearn.naive_bayes import BernoulliNB

# Features: [has_word_"free", has_word_"prize", ...]
# Each feature is 0 or 1

model = BernoulliNB(alpha=1.0)
model.fit(X_train_binary, y_train)
```

### Worked Example — Email Spam Detection

Let's work through a complete example step-by-step.

**Training Data:**

| Email | Words | Label |
|-------|-------|-------|
| 1 | "Free prize money" | Spam |
| 2 | "Congratulations winner" | Spam |
| 3 | "Meeting agenda attached" | Legit |
| 4 | "Your report is ready" | Legit |

**Test Email:** "Congratulations! Free money!"

**Step 1: Calculate Prior Probabilities**

```
P(Spam)  = 2/4 = 0.5
P(Legit) = 2/4 = 0.5
```

**Step 2: Count Word Frequencies**

Word counts in each class:

| Word | Count in Spam | Count in Legit | Total Spam Words | Total Legit Words |
|------|---------------|----------------|------------------|-------------------|
| "free" | 1 | 0 | 5 | 6 |
| "prize" | 1 | 0 | | |
| "money" | 1 | 0 | | |
| "congratulations" | 1 | 0 | | |
| "winner" | 1 | 0 | | |
| "meeting" | 0 | 1 | | |
| "agenda" | 0 | 1 | | |
| "attached" | 0 | 1 | | |
| "your" | 0 | 1 | | |
| "report" | 0 | 1 | | |
| "ready" | 0 | 1 | | |

**Step 3: Calculate Likelihoods (with Laplace Smoothing)**

```
Vocabulary size = 11 unique words
α = 1 (Laplace smoothing)

P(word | Spam) = (count in Spam + 1) / (total Spam words + 11)
               = (count in Spam + 1) / (5 + 11)
               = (count in Spam + 1) / 16

P(word | Legit) = (count in Legit + 1) / (total Legit words + 11)
                = (count in Legit + 1) / (6 + 11)
                = (count in Legit + 1) / 17
```

For the test email words:

| Word | P(word \| Spam) | P(word \| Legit) |
|------|----------------|-----------------|
| "congratulations" | (1+1)/16 = 0.125 | (0+1)/17 = 0.059 |
| "free" | (1+1)/16 = 0.125 | (0+1)/17 = 0.059 |
| "money" | (1+1)/16 = 0.125 | (0+1)/17 = 0.059 |

**Step 4: Calculate Posterior Probabilities**

```
P(Spam | test email) ∝ P(Spam) × P("congratulations" | Spam) 
                                × P("free" | Spam) 
                                × P("money" | Spam)
                     = 0.5 × 0.125 × 0.125 × 0.125
                     = 0.000977

P(Legit | test email) ∝ P(Legit) × P("congratulations" | Legit) 
                                  × P("free" | Legit) 
                                  × P("money" | Legit)
                      = 0.5 × 0.059 × 0.059 × 0.059
                      = 0.000103
```

**Step 5: Predict**

```
P(Spam | test) = 0.000977  ← Higher!
P(Legit | test) = 0.000103

Prediction: SPAM ✅
```

> **Note:** We don't need to normalize by P(Features) because we only care which class has the higher probability.

### Strengths and Weaknesses of Naive Bayes

| ✅ Strengths | ❌ Weaknesses |
|-------------|--------------|
| **Extremely fast training and prediction** — just count and multiply | **Naive assumption rarely holds** — features are often correlated |
| **Works brilliantly on text data** — spam detection, sentiment analysis, topic classification | **Zero-frequency problem** — needs smoothing to handle unseen words |
| **Requires little training data** — can work with small datasets | **Poor probability estimates** — good for ranking, not calibrated probabilities |
| **Handles high-dimensional data well** — scales to thousands of features | **Cannot learn feature interactions** — treats all features independently |
| **Naturally multi-class** — extends to any number of classes | **Sensitive to irrelevant features** — all features affect the product |
| **Interpretable** — you can see which words/features drive decisions | **Assumes feature independence** — can't capture "free AND prize together" patterns |

**When to Use Naive Bayes:**

```
✅ USE Naive Bayes when:
  • Working with TEXT data (emails, reviews, documents)
  • Need FAST training and prediction
  • Features are high-dimensional and sparse (e.g., bag-of-words)
  • You have LIMITED training data
  • Features are weakly correlated or independent
  • You want an interpretable baseline

❌ AVOID Naive Bayes when:
  • Features have strong correlations (use Decision Trees or Logistic Regression)
  • You need well-calibrated probabilities (use Logistic Regression)
  • Features have complex interactions (use Neural Networks or Ensemble methods)
```

### Implementation in Python (Naive Bayes)

**Complete Example: SMS Spam Detection**

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load text data (example: 20 newsgroups dataset)
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

X = newsgroups.data  # Text documents
y = newsgroups.target  # Category labels

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Convert text to numerical features (word counts)
vectorizer = CountVectorizer(
    max_features=5000,  # Use top 5000 most common words
    stop_words='english'  # Remove common words like "the", "is", etc.
)
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 4. Train Multinomial Naive Bayes
nb = MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing
nb.fit(X_train_counts, y_train)

# 5. Make predictions
y_pred = nb.predict(X_test_counts)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Using TF-IDF Instead of Raw Counts:**

```python
# TF-IDF gives better results than raw counts for text
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)  # Use unigrams and bigrams
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
```

**All Three Naive Bayes Variants:**

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian NB — for continuous features
gnb = GaussianNB()
gnb.fit(X_train_continuous, y_train)

# Multinomial NB — for word counts
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)

# Bernoulli NB — for binary features (word present/absent)
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_binary, y_train)

# Get probability predictions (not just labels)
proba = mnb.predict_proba(X_test_counts)
# proba[i][j] = P(class j | test example i)
```

**Inspecting Learned Probabilities:**

```python
# See which words are most indicative of each class
import numpy as np

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# For each class, find top 10 most important words
for i, category in enumerate(newsgroups.target_names):
    # Log probabilities learned by the model
    log_probs = nb.feature_log_prob_[i]
    
    # Get indices of top 10 highest probabilities
    top_10_indices = np.argsort(log_probs)[-10:]
    
    print(f"\nTop words for {category}:")
    for idx in reversed(top_10_indices):
        print(f"  {feature_names[idx]}: {np.exp(log_probs[idx]):.4f}")
```

---

## 4.3 Decision Trees

> ### 🌳 Decision Trees
> *Divide and Conquer — Classify by asking questions*

### The Core Idea — Ask Questions and Split

Imagine you are playing **20 Questions** to guess an animal. You do not start with "Is it a giraffe?" — you ask broad questions first: "Is it a mammal?" Then narrow down: "Does it live in water?" Each answer splits the possibilities in half until you isolate the answer.

**Decision Trees work exactly this way.** They learn which questions (features) are most useful, then recursively split the data until each group is pure (or close enough).

```
┌─────────────────────────────────────────────────────────────┐
│ 🎮 Decision Tree — Playing 20 Questions                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                   Is it a mammal?                           │
│                   /              \                          │
│                 YES              NO                         │
│                 /                  \                        │
│         Lives in water?        Has feathers?                │
│           /        \              /        \                │
│         YES        NO           YES        NO               │
│         /            \          /            \              │
│      Whale         Has spots?  Bird       Reptile          │
│                    /        \                               │
│                  YES        NO                              │
│                  /            \                             │
│              Leopard         Lion                           │
│                                                              │
│ Each question splits the data into purer groups.            │
│ Leaves = final predictions.                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Real ML Example: Loan Approval**

```
                  Credit Score > 700?
                 /                  \
              YES                   NO
              /                       \
      Annual Income > $50k?      Approved = NO
         /            \
       YES            NO
       /                \
  Approved = YES   Loan Amount < $10k?
                      /           \
                    YES           NO
                    /               \
              Approved = YES   Approved = NO
```

### How Trees Split — Information Gain & Gini Impurity

The key question: **Which feature should we split on first?** We want the feature that **best separates the classes** — that gives us the most information.

There are two main measures:

#### 1. Gini Impurity

**Gini Impurity** measures how "mixed" a group is:

```
Gini = 1 - Σ (p_i)²
       i=1..C

Where:
  C = number of classes
  p_i = proportion of class i in the group

Interpretation:
  Gini = 0   → Perfectly pure (all same class)
  Gini = 0.5 → Maximum impurity (50/50 split for 2 classes)
```

**Example:**

```python
# Group 1: [Apple, Apple, Apple, Apple] → 4 apples, 0 oranges
p_apple = 4/4 = 1.0
p_orange = 0/4 = 0.0
Gini = 1 - (1.0² + 0.0²) = 1 - 1.0 = 0.0  ✅ Perfectly pure!

# Group 2: [Apple, Apple, Orange, Orange] → 2 apples, 2 oranges
p_apple = 2/4 = 0.5
p_orange = 2/4 = 0.5
Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5  ❌ Maximum impurity

# Group 3: [Apple, Apple, Apple, Orange] → 3 apples, 1 orange
p_apple = 3/4 = 0.75
p_orange = 1/4 = 0.25
Gini = 1 - (0.75² + 0.25²) = 1 - 0.625 = 0.375  ⚠️ Some impurity
```

**Gini Gain from a Split:**

```
Gini_Gain = Gini_parent - Σ (n_child / n_parent) × Gini_child

We want to MAXIMIZE Gini Gain (or equivalently, MINIMIZE weighted Gini of children)
```

#### 2. Information Gain (Entropy-based)

**Entropy** measures uncertainty or randomness:

```
Entropy = - Σ p_i × log₂(p_i)
          i=1..C

Where:
  p_i = proportion of class i

Interpretation:
  Entropy = 0     → Perfectly pure (no uncertainty)
  Entropy = 1     → Maximum uncertainty (50/50 for 2 classes)
  Entropy = log₂(C) → Maximum for C classes
```

**Example:**

```python
# Pure group: [Apple, Apple, Apple, Apple]
p_apple = 1.0, p_orange = 0.0
Entropy = -(1.0 × log₂(1.0) + 0.0 × log₂(0.0))
        = 0  ✅ No uncertainty!

# 50/50 split: [Apple, Apple, Orange, Orange]
p_apple = 0.5, p_orange = 0.5
Entropy = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
        = -(0.5 × -1 + 0.5 × -1)
        = 1.0  ❌ Maximum uncertainty for 2 classes

# 75/25 split: [Apple, Apple, Apple, Orange]
p_apple = 0.75, p_orange = 0.25
Entropy = -(0.75 × log₂(0.75) + 0.25 × log₂(0.25))
        = -(0.75 × -0.415 + 0.25 × -2.0)
        = 0.811  ⚠️ Some uncertainty
```

**Information Gain from a Split:**

```
Information_Gain = Entropy_parent - Σ (n_child / n_parent) × Entropy_child

We want to MAXIMIZE Information Gain
```

**Which to Use?**

| Metric | Calculation Speed | Results |
|--------|-------------------|---------|
| **Gini Impurity** | Faster (no logarithm) | Slightly favours larger partitions |
| **Information Gain** | Slower (requires log) | Slightly favours balanced partitions |

> In practice, **both give very similar trees**. Scikit-learn uses Gini by default because it is faster.

### Building a Tree — The Complete Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│ 🌲 Decision Tree Construction Algorithm (CART)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ FUNCTION BuildTree(data, features):                         │
│                                                              │
│   1. STOPPING CONDITIONS (return a leaf):                   │
│      • If all examples have same label → return that label  │
│      • If no features left → return majority label          │
│      • If max_depth reached → return majority label         │
│      • If node has < min_samples_split → return majority    │
│                                                              │
│   2. FIND BEST SPLIT:                                        │
│      For each feature f:                                    │
│        For each possible split point:                       │
│          Calculate Gini gain (or Info gain)                 │
│      Select feature and split with highest gain             │
│                                                              │
│   3. SPLIT THE DATA:                                         │
│      left_data  = examples where feature <= threshold       │
│      right_data = examples where feature > threshold        │
│                                                              │
│   4. RECURSIVE CALLS:                                        │
│      left_subtree  = BuildTree(left_data, features)         │
│      right_subtree = BuildTree(right_data, features)        │
│                                                              │
│   5. RETURN DECISION NODE:                                   │
│      return Node(feature, threshold, left_subtree, right_subtree)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Worked Example:**

Dataset: Predict if a customer will buy based on age and income

| Age | Income | Buys |
|-----|--------|------|
| 25 | 40k | No |
| 30 | 60k | No |
| 35 | 80k | Yes |
| 40 | 85k | Yes |
| 45 | 90k | Yes |
| 50 | 70k | No |

**Step 1: Calculate Gini for root node**

```
Total: 3 Yes, 3 No
p_yes = 3/6 = 0.5
p_no = 3/6 = 0.5
Gini_root = 1 - (0.5² + 0.5²) = 0.5
```

**Step 2: Try splitting on Age**

Try split at Age=37.5:
- Left: Age ≤ 37.5 → [No, No, Yes] → Gini = 1 - (1/3² + 2/3²) = 0.444
- Right: Age > 37.5 → [Yes, Yes, No] → Gini = 1 - (2/3² + 1/3²) = 0.444

Weighted Gini = (3/6) × 0.444 + (3/6) × 0.444 = 0.444
Gini Gain = 0.5 - 0.444 = 0.056

**Step 3: Try splitting on Income**

Try split at Income=65k:
- Left: Income ≤ 65k → [No, No, No] → Gini = 0.0 ✅ Pure!
- Right: Income > 65k → [Yes, Yes, Yes] → Gini = 0.0 ✅ Pure!

Weighted Gini = (3/6) × 0.0 + (3/6) × 0.0 = 0.0
Gini Gain = 0.5 - 0.0 = 0.5 🎯 Best split!

**Step 4: Build the tree**

```
              Income ≤ 65k?
               /           \
             YES           NO
             /               \
         Buys = NO        Buys = YES

Done! Both leaves are pure, no further splits needed.
```

### Overfitting and Pruning

**The Overfitting Problem:**

Without limits, a decision tree will keep splitting until every leaf has exactly one training example — perfect training accuracy, terrible test accuracy.

```
┌─────────────────────────────────────────────────────────────┐
│ 🌳 Overfitting in Decision Trees                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ SMALL TREE (Underfitting)      LARGE TREE (Overfitting)     │
│                                                              │
│     Income > 60k?               Income > 60k?               │
│      /        \                  /           \              │
│    No         Yes               /             \             │
│                              Age > 30?      Income > 85k?   │
│ Too simple!                   /    \          /      \      │
│ Misses patterns            No    Yes       Yes      No     │
│                                  /  \       /  \     / \    │
│                          Owns car? ...   ...  ... ... ...  │
│                           /    \                            │
│                         No    Yes                           │
│                               / \                           │
│                       Employed?  ...                        │
│                                                              │
│                         Memorises noise!                    │
│                         100% train, 65% test                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Solutions:**

#### 1. Pre-Pruning (Early Stopping)

Stop growing the tree early using constraints:

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=5,              # Maximum tree depth
    min_samples_split=20,     # Minimum samples to split a node
    min_samples_leaf=10,      # Minimum samples in a leaf
    max_features=0.5,         # Use only 50% of features per split
    max_leaf_nodes=20         # Maximum number of leaf nodes
)
```

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `max_depth` | Limits tree depth | 3–10 for interpretability, 10–30 for accuracy |
| `min_samples_split` | Won't split nodes with fewer samples | 2–20 |
| `min_samples_leaf` | Leaves must have this many samples | 1–10 |
| `max_features` | Randomly select this many features per split | `sqrt(n)` for classification, `n/3` for regression |

#### 2. Post-Pruning (Cost Complexity Pruning)

Grow a full tree, then prune back branches that do not improve validation performance:

```python
from sklearn.tree import DecisionTreeClassifier

# 1. Grow a full tree
tree_full = DecisionTreeClassifier()
tree_full.fit(X_train, y_train)

# 2. Get pruning path (different alpha values)
path = tree_full.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# 3. Train trees with different pruning strengths
trees = []
for alpha in alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

# 4. Select best alpha using validation set
val_scores = [tree.score(X_val, y_val) for tree in trees]
best_alpha = alphas[np.argmax(val_scores)]

# 5. Train final model
final_tree = DecisionTreeClassifier(ccp_alpha=best_alpha)
final_tree.fit(X_train, y_train)
```

### Strengths and Weaknesses of Decision Trees

| ✅ Strengths | ❌ Weaknesses |
|-------------|--------------|
| **Fully interpretable** — you can visualise and explain every decision | **Prone to overfitting** — especially on noisy data |
| **Handles mixed data types** — numeric, categorical, ordinal all together | **Unstable** — small data changes can drastically change the tree |
| **No feature scaling needed** — works with raw features | **Biased toward features with many values** — prefers high-cardinality features |
| **Captures non-linear relationships** — automatically finds complex patterns | **Cannot extrapolate** — predictions limited to training data range |
| **Fast prediction** — O(log n) or O(depth) | **Greedy algorithm** — locally optimal splits may miss globally best tree |
| **Feature importance** — tells you which features matter most | **Poor on linear relationships** — decision boundaries are axis-aligned |

**When to Use Decision Trees:**

```
✅ USE Decision Trees when:
  • You need INTERPRETABLE models for stakeholders/regulators
  • Features have mixed types (numeric + categorical)
  • You want to understand WHICH features drive decisions
  • Decision boundaries are complex and non-linear
  • You have imbalanced classes (can weight samples)

❌ AVOID single Decision Trees when:
  • You need maximum accuracy (use Random Forest or Gradient Boosting instead)
  • Data is very high-dimensional and sparse (use Naive Bayes or Linear models)
  • You have small datasets with noise (trees overfit easily)
  
✅ PREFER ensembles (Random Forest, XGBoost) for production — they fix the instability and overfitting issues while keeping the benefits.
```

### Implementation in Python (Decision Trees)

**Complete Example: Iris Classification**

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load data
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train Decision Tree
tree = DecisionTreeClassifier(
    criterion='gini',       # or 'entropy' for information gain
    max_depth=3,            # Limit depth to prevent overfitting
    min_samples_split=10,   # Minimum samples to split
    min_samples_leaf=5,     # Minimum samples in leaf
    random_state=42
)
tree.fit(X_train, y_train)

# 4. Predict
y_pred = tree.predict(X_test)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Visualise the tree
plt.figure(figsize=(20, 10))
plot_tree(tree, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,           # Color nodes by majority class
          rounded=True,          # Rounded boxes
          fontsize=10)
plt.title("Decision Tree for Iris Classification")
plt.show()

# 7. Feature importance
feature_importance = tree.feature_importances_
for name, importance in zip(iris.feature_names, feature_importance):
    print(f"{name}: {importance:.3f}")
```

**Visualising the Tree as Text:**

```python
from sklearn.tree import export_text

tree_rules = export_text(tree, feature_names=iris.feature_names)
print(tree_rules)
```

**Output Example:**

```
|--- petal width (cm) <= 0.80
|   |--- class: setosa
|--- petal width (cm) >  0.80
|   |--- petal width (cm) <= 1.75
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: versicolor
|   |   |--- petal length (cm) >  4.95
|   |   |   |--- class: virginica
|   |--- petal width (cm) >  1.75
|   |   |--- class: virginica
```

**Getting Prediction Paths:**

```python
# For a specific sample, show which path the tree took
sample = X_test[0:1]
prediction = tree.predict(sample)

# Get the decision path
node_indicator = tree.decision_path(sample)
leaf_id = tree.apply(sample)

print(f"Prediction: {iris.target_names[prediction[0]]}")
print(f"Decision path (nodes visited): {node_indicator.indices}")
print(f"Final leaf node: {leaf_id[0]}")
```

**Exporting Tree for External Use:**

```python
from sklearn.tree import export_graphviz
import graphviz

# Export to DOT format
dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

# Render with graphviz (requires graphviz installation)
graph = graphviz.Source(dot_data)
graph.render("iris_tree")  # Saves as iris_tree.pdf
```

---

## 4.4 Comparing the Three Algorithms

### Side-by-Side Comparison

| Aspect | kNN | Naive Bayes | Decision Tree |
|--------|-----|-------------|---------------|
| **Learning paradigm** | Lazy (instance-based) | Probabilistic (Bayes) | Divide and conquer |
| **Training time** | None (just stores data) | Very fast | Fast to moderate |
| **Prediction time** | Slow (scans all data) | Very fast | Fast (O(depth)) |
| **Memory usage** | High (stores all training data) | Low (stores probabilities) | Low (stores tree structure) |
| **Interpretability** | Low (hard to explain neighbours) | Medium (can see probabilities) | High (visual tree, rules) |
| **Feature scaling** | Required | Not needed | Not needed |
| **Handles missing data** | No (needs imputation) | Yes (can ignore missing features) | Yes (can use surrogate splits) |
| **Feature types** | Numeric primarily | Mixed (varies by variant) | Mixed (numeric + categorical) |
| **Overfitting risk** | Low k → overfits | Low (regularised by assumption) | High (requires pruning) |
| **Best data type** | Numeric, low-dimensional | Text, high-dimensional | Mixed types, tabular |
| **Typical accuracy** | Good on local patterns | Strong on text tasks | Good, great in ensembles |

### Decision Guide — Which Algorithm to Pick?

```
┌─────────────────────────────────────────────────────────────┐
│ 🧭 Choosing Between kNN, Naive Bayes, and Decision Tree     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ START: What is your primary concern?                        │
│   │                                                          │
│   ├── Is your data TEXT (words, documents, messages)?       │
│   │   → NAIVE BAYES (built for this — fast, accurate on text)│
│   │                                                          │
│   ├── Do you need HUMAN-READABLE rules you can explain to   │
│   │   stakeholders?                                          │
│   │   → DECISION TREE (fully interpretable, auditable rules)│
│   │                                                          │
│   ├── Is your dataset SMALL (< 1,000 examples) and          │
│   │   LOW-DIMENSIONAL?                                       │
│   │   → kNN (no training needed, adapts perfectly to small data)│
│   │                                                          │
│   ├── Do you need VERY FAST predictions at scale?           │
│   │   → NAIVE BAYES or DECISION TREE (O(f) or O(depth) prediction)│
│   │   → NOT kNN (must scan all training data for each prediction)│
│   │                                                          │
│   ├── Are features INDEPENDENT or nearly so?                │
│   │   → NAIVE BAYES (that is its assumption — and it thrives there)│
│   │                                                          │
│   └── None of the above / you want the best accuracy?       │
│       → Try RANDOM FOREST or GRADIENT BOOSTING               │
│         (ensembles of Decision Trees — best on structured/tabular data)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Same Dataset, Three Algorithms — Results Compared

To make the comparison concrete, here is how each algorithm performs on the **UCI Breast Cancer Wisconsin dataset** (569 samples, 30 features, binary classification):

| Algorithm | Accuracy | Precision | Recall | Notes |
|-----------|----------|-----------|--------|-------|
| **kNN (k=5)** | 94.7% | 94.1% | 95.3% | Sensitive to feature scaling |
| **Naive Bayes (Gaussian)** | 93.0% | 92.0% | 94.2% | Features not perfectly Gaussian |
| **Decision Tree (max_depth=5)** | 92.1% | 90.8% | 93.5% | Improved with pruning |
| **Decision Tree (unpruned)** | 89.3% | 87.9% | 90.8% | Overfitting visible on test set |
| **Random Forest (100 trees)** | 97.2% | 96.8% | 97.6% | Best accuracy — ensemble wins |

> ### 💡 Key Takeaway from the Comparison
>
> No single algorithm wins all tasks. The best approach is always to **try multiple algorithms**, evaluate them properly on held-out test data, and select based on your specific constraints — accuracy, speed, interpretability, and data type.
>
> In practice: start with **Naive Bayes** or **Decision Tree** as a fast baseline, then move to **Random Forest** or **Gradient Boosting** for best accuracy.

---

## Chapter 4 Summary

Here is what this chapter covered — from basic intuition to advanced implementation:

| Topic | Key Takeaway |
|-------|--------------|
| **4.1 kNN** | Lazy learning: stores all training data, classifies by majority vote of K nearest neighbours. No training computation. Slow at prediction. Requires feature scaling. Best for small, low-dimensional, locally-structured data. |
| **4.2 Naive Bayes** | Probabilistic learning: applies Bayes' theorem assuming feature independence. Blazing fast. Works brilliantly on text (spam, sentiment, topics). Variants: Gaussian (continuous), Multinomial (word counts), Bernoulli (binary). |
| **4.3 Decision Trees** | Divide and conquer: recursively splits data using the most informative feature at each step (minimising Gini or maximising Information Gain). Fully interpretable rules. Prone to overfitting — use pruning or ensemble methods. |
| **4.4 Comparison** | Each algorithm has a home territory: kNN for local patterns, Naive Bayes for text and high-dimensional sparse data, Decision Trees for interpretable tabular data. For best accuracy, extend trees into Random Forests or Gradient Boosting. |

---

> ### 🚀 What is Coming Next
>
> **Chapter 5: More Classification Algorithms — SVMs, Logistic Regression & Ensembles**
>
> - Support Vector Machines — maximum margin classifiers
> - Logistic Regression — probabilistic linear classification
> - Random Forests — bagging trees for robust accuracy
> - Gradient Boosting (XGBoost, LightGBM) — the competition winners
> - Neural Networks for classification — a gentle introduction

---

*End of Chapter 4*
