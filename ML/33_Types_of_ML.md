# MACHINE LEARNING
### *A Complete Beginner-to-Advanced Guide*

## Chapter 3: Types of Machine Learning

*Continuing from Chapters 1 & 2:*

- *Ch. 1 — What is ML? | Origins & Evolution | Real-world Use Cases*
- *Ch. 2 — Training vs Testing | Abstraction | Overfitting | Model Assessment*

---

## Table of Contents

- [Chapter Overview](#chapter-overview)
- [3.1 Supervised Learning — Learning with a Teacher](#31-supervised-learning)
  - [3.1.1 Classification — Predicting Categories](#311-classification)
  - [3.1.2 Regression — Predicting Numbers](#312-regression)
  - [Real-World Supervised Learning Applications](#real-world-supervised-learning-applications)
- [3.2 Unsupervised Learning — Finding Hidden Structure](#32-unsupervised-learning)
  - [3.2.1 Clustering — Finding Natural Groups](#321-clustering)
  - [3.2.2 Dimensionality Reduction — Compressing Information](#322-dimensionality-reduction)
  - [3.2.3 Anomaly Detection — Finding the Unusual](#323-anomaly-detection)
  - [3.2.4 Association Rule Learning — Co-occurrence Patterns](#324-association-rule-learning)
  - [Real-World Unsupervised Learning Applications](#real-world-unsupervised-learning-applications)
- [3.3 Reinforcement Learning — Learning by Reward](#33-reinforcement-learning)
- [3.4 How to Choose the Right Type](#34-how-to-choose-the-right-type)
- [Chapter 3 Summary](#chapter-3-summary)

---

## Chapter Overview

We have covered what ML is and how a model learns. Now we answer a fundamental question: what kinds of problems can ML solve — and which flavour of ML applies to each?

There are three core paradigms of machine learning. Each reflects a fundamentally different relationship between data, feedback, and learning. By the end of this chapter, you will know exactly which paradigm to reach for — and why.

---

### 🗺️ The Big Map of Machine Learning

| Paradigm | Description | Analogy |
|---|---|---|
| **Supervised Learning** | You provide labelled examples. The model learns to predict the label for new data. | A student learning from a textbook with an answer key. |
| **Unsupervised Learning** | You provide data with NO labels. The model discovers hidden structure on its own. | A student grouping similar textbook pages without being told the topics. |
| **Reinforcement Learning** | An agent learns by taking actions and receiving rewards or penalties. | Training a dog with treats — good behaviour gets rewarded. |

---

## 3.1 Supervised Learning

*The most widely used paradigm — learning from labelled examples*

### What is Supervised Learning?

In supervised learning, every training example comes with a correct answer — the label. The model learns a mapping from inputs (features) to outputs (labels). Once trained, it can predict the label for examples it has never seen.

```
👩‍🏫 SUPERVISED LEARNING — THE LABELLED CLASSROOM

TRAINING PHASE:

  Input (Features)              Label (Correct Answer)
  ─────────────────             ──────────────────────
  Email text + metadata    →    Spam / Not Spam
  X-ray image              →    Cancer / No Cancer
  House size + location    →    Price: Rs. 85 Lakhs
  Customer purchase hist   →    Will churn: Yes / No

PREDICTION PHASE:

  New Input → [Trained Model] → Predicted Label
  (no answer key needed)
```

### The Two Flavours of Supervised Learning

Supervised learning splits into two major types depending on what kind of output you want to predict:

| Type | Goal | Examples |
|---|---|---|
| **Classification** | Predict a CATEGORY (a discrete class label). | Is this email spam or not? Is this tumour malignant or benign? Which digit is this handwritten number? |
| **Regression** | Predict a NUMBER (a continuous value). | What will this house sell for? How many units will we sell next month? What is a patient's blood pressure likely to be? |

---

### 3.1.1 Classification

*Predicting Categories*

Classification is about sorting inputs into buckets. The simplest case is **binary classification** (two buckets: yes/no). When there are more than two possible labels, it is called **multi-class classification**.

```
🚦 BINARY vs MULTI-CLASS CLASSIFICATION

BINARY CLASSIFICATION (2 classes):
  Input: Email          → Output: Spam | Not Spam
  Input: Loan app       → Output: Approve | Reject
  Input: Tumour scan    → Output: Malignant | Benign

MULTI-CLASS CLASSIFICATION (3+ classes):
  Input: Photo of animal   → Output: Cat | Dog | Bird | Fish
  Input: Handwritten digit → Output: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
  Input: Product review    → Output: 1-star | 2-star | 3-star | 4-star | 5-star

MULTI-LABEL CLASSIFICATION (multiple labels at once):
  Input: News article → Output: Politics AND Economy AND International
  Input: Movie        → Output: Action AND Thriller AND Sci-Fi
```

#### Common Classification Algorithms

| Algorithm | How It Works & When to Use |
|---|---|
| **Logistic Regression** | Despite the name, this is a classifier! Outputs a probability between 0 and 1, then applies a threshold. Simple, fast, and interpretable. Great starting point for any binary problem. |
| **Decision Tree** | Builds a flowchart of if/then rules. Easy to understand and visualise. Prone to overfitting on its own — usually better used in ensembles. |
| **Random Forest** | Builds many decision trees and votes on the answer. Much more robust than a single tree. Handles messy data well. One of the most reliable algorithms in practice. |
| **Support Vector Machine (SVM)** | Finds the boundary that maximally separates two classes. Works well in high-dimensional spaces (many features). The kernel trick lets it handle non-linear boundaries. |
| **k-Nearest Neighbours (kNN)** | Classifies by finding the K most similar training examples and taking a vote. No training needed — but slow at prediction time on large datasets. |
| **Neural Networks** | Can learn extremely complex boundaries. The backbone of deep learning for images, text, and audio. Needs lots of data and compute to shine. |
| **Naive Bayes** | Uses probability theory (Bayes theorem). Very fast. Works surprisingly well for text classification (spam filters, sentiment analysis) despite simplistic assumptions. |

#### Deep Dive: How a Decision Tree Classifies

Let us walk through exactly what happens when a Decision Tree classifies whether a loan should be approved:

```
🌳 DECISION TREE: LOAN APPROVAL

                    [Credit Score > 700?]
                          /        \
                        YES         NO
                         |           |
          [Income > Rs.50k/mo?]   [Existing Debt < 20%?]
               /       \               /        \
             YES         NO          YES          NO
              |           |           |            |
  [Loan < 3x annual inc?] REJECT   REVIEW       REJECT
         /      \
       YES       NO
        |         |
    APPROVE    REJECT

Each question = a feature the tree learned was important from training data.
To classify a new applicant, just follow the branches from top to bottom.
```

---

### 3.1.2 Regression

*Predicting Numbers*

Regression predicts a continuous numeric value. Instead of asking 'which category?', it asks 'how much?' or 'how many?'. The model learns a mathematical relationship between the input features and a numerical output.

```
📈 REGRESSION: HOUSE PRICE PREDICTION

Simple Linear Regression (one feature):

Price (Rs. Lakhs)
200 |          *  *
    |        *  *
150 |      *  *
    |    *  *
100 |  *  *  ← Best-fit line learned during training
    |*  *
 50 |*  *
    └──────────────────────────────────────
    500   1000   1500   2000   2500  sq ft

New house = 1,800 sq ft → Model predicts Rs. 138 Lakhs
(reads off the line at x = 1800)
```

In practice, multiple input features are used simultaneously — this is called **Multiple Linear Regression**:

```
📐 MULTIPLE LINEAR REGRESSION FORMULA

Price = (w1 × Size) + (w2 × Bedrooms) + (w3 × Age) + (w4 × Location) + b

Example weights learned during training:
  w1 =  0.085   (each sq ft adds Rs. 0.085 Lakh)
  w2 =  3.20    (each bedroom adds Rs. 3.2 Lakhs)
  w3 = -0.40    (each year of age reduces price by Rs. 0.4 Lakh)
  w4 = 12.50    (premium location score multiplier)
  b  = -8.00    (base intercept)

For a house: 1,200 sq ft | 3 bed | 5 yrs old | location = 7

  Price = (0.085×1200) + (3.20×3) + (-0.40×5) + (12.50×7) + (-8.00)
        = 102 + 9.6 - 2 + 87.5 - 8
        = Rs. 189.1 Lakhs
```

#### Common Regression Algorithms

| Algorithm | Description & Best Use |
|---|---|
| **Linear Regression** | Fits a straight line (or hyperplane in multiple dimensions). Simple, fast, interpretable. Best when the relationship is roughly linear. |
| **Polynomial Regression** | Fits a curved line by adding polynomial features (x, x², x³). More flexible than linear, but can overfit easily. |
| **Ridge / Lasso Regression** | Linear regression with regularisation. Ridge (L2) shrinks all weights. Lasso (L1) can zero out irrelevant features entirely (automatic feature selection). |
| **Decision Tree Regressor** | Instead of voting on a class, the leaf nodes output an average value. Can capture non-linear relationships without assuming any shape. |
| **Random Forest Regressor** | Averages the predictions of many regression trees. Robust and accurate. Handles outliers and missing data better than single models. |
| **Gradient Boosting (XGBoost, LightGBM)** | Builds trees sequentially, each one correcting the errors of the last. Often the top performer on structured/tabular data in competitions. |
| **Neural Network Regressor** | Can fit arbitrarily complex functions. Shines for unstructured data (images, audio). Needs more data and tuning than tree-based methods. |

---

### Real-World Supervised Learning Applications

| Domain | Problem | Output Type |
|---|---|---|
| Healthcare | Classify X-rays / MRI scans | Disease present or absent |
| Finance | Predict stock return | Expected % change in price |
| E-commerce | Recommend product | Will customer buy? (Yes/No) |
| Spam Detection | Classify emails | Spam / Not Spam |
| Credit Scoring | Predict default risk | Probability of non-payment |
| Weather | Predict tomorrow's temperature | Temperature in degrees |
| HR / Recruitment | Screen CVs | Shortlist / Reject |
| Self-driving Cars | Detect pedestrians in camera feed | Person present / absent |

> 💡 **When to Use Supervised Learning**
>
> Use supervised learning when you have a clear outcome you want to predict AND you have historical data where that outcome is already known. The quality and quantity of your labels directly determines how good your model can be.

---

## 3.2 Unsupervised Learning

*Discovering hidden patterns in data — no labels required*

### What is Unsupervised Learning?

In unsupervised learning, the training data has **NO labels**. There is no answer key. The model must find structure, patterns, and organisation entirely on its own.

Think of it like this: you are handed a box of 10,000 unlabelled photographs and asked to organise them. You would naturally group similar ones together — beach photos, family events, food shots — without anyone telling you the categories upfront. That is exactly what unsupervised learning does.

```
🔍 UNSUPERVISED vs SUPERVISED — THE KEY DIFFERENCE

  SUPERVISED:   Data + Labels → Model learns to predict labels

  UNSUPERVISED: Data (no labels) → Model discovers structure

You cannot be 'wrong' in unsupervised learning in the usual sense.
Instead, you evaluate whether the discovered structure is useful,
meaningful, or actionable for your specific goal.
```

### The Major Unsupervised Learning Tasks

| Task | What It Does |
|---|---|
| **Clustering** | Group similar data points together. Points in the same group (cluster) are more alike than points in different groups. No predefined groups — the model finds them. |
| **Dimensionality Reduction** | Compress high-dimensional data into fewer dimensions while preserving as much important information as possible. Makes visualisation and downstream modelling easier. |
| **Anomaly Detection** | Find data points that are unusual or do not fit any normal pattern. Critical for fraud detection, equipment failure prediction, and quality control. |
| **Association Rule Learning** | Discover rules about which items tend to co-occur. Classic example: 'customers who buy bread also often buy butter and milk.' Used in market basket analysis. |
| **Generative Modelling** | Learn the underlying distribution of data and generate new, realistic examples. Examples: generating new faces, writing text, creating music. |

---

### 3.2.1 Clustering

*Finding Natural Groups*

Clustering is the most common unsupervised learning task. The goal is to partition data into groups (clusters) such that data points within a cluster are similar to each other and different from points in other clusters.

```
🫧 CLUSTERING EXAMPLE: CUSTOMER SEGMENTATION

Raw data: 5,000 customers with age, income, spend history
No labels given. Model must find natural groups.

  BEFORE CLUSTERING          AFTER CLUSTERING
  ─────────────────          ────────────────
  Income                     Income
  | * * * * *                | [A] [A]
  |* * * * * * *             |[A] [A] [A]
  | * * * * * * *            | [A] [B] [B]
  | * * * * * * *            | [C] [B] [B]
  | * * * * *                |[C] [C]
  └────────────              └────────────
        Age                        Age

  Cluster A: Young, high income    → Premium digital customers
  Cluster B: Middle-aged, medium   → Mainstream family buyers
  Cluster C: Older, lower income   → Value-seeking segment
```

#### k-Means Clustering — Step by Step

k-Means is the most popular clustering algorithm. Here is exactly how it works:

```
⚙️ k-MEANS ALGORITHM (k = 3 clusters)

Step 1: Choose k = 3 (you decide how many clusters to look for)
        Place 3 centroids (cluster centres) at random positions

Step 2: ASSIGNMENT
        Assign each data point to the nearest centroid
        (measured by Euclidean distance)

Step 3: UPDATE
        Move each centroid to the average position of its assigned points

Step 4: Repeat Steps 2 and 3 until centroids stop moving
        (convergence — usually 10–100 iterations)

Step 5: Final clusters = the stable groupings

Example — Distance calculation:
  Point P    = (Age=35, Income=60)
  Centroid A = (Age=30, Income=65)

  Distance = sqrt((35-30)² + (60-65)²)
           = sqrt(25 + 25)
           = 7.07
```

> ⚠️ **Choosing k — The Elbow Method**
>
> k-Means requires you to specify k upfront. Use the Elbow Method: plot the total within-cluster variance for k = 1, 2, 3, … and look for the 'elbow' — the point where adding more clusters stops dramatically reducing variance. That k is usually optimal.

#### Other Clustering Algorithms

| Algorithm | Key Properties |
|---|---|
| **DBSCAN** | Groups points based on density — no need to specify k upfront. Can find clusters of arbitrary shape and automatically marks outliers as noise. Great for geographic data. |
| **Hierarchical Clustering** | Builds a tree (dendrogram) of nested clusters. You can cut the tree at any level to get any number of clusters. Useful when you want to understand cluster hierarchy. |
| **Gaussian Mixture Models (GMM)** | Assumes data comes from a mix of Gaussian distributions. Softer than k-Means — a point can partially belong to multiple clusters. Better handles oval-shaped clusters. |
| **Mean-Shift** | Slides a window across the data and shifts it toward the densest region. Automatically finds k. Slower than k-Means but more flexible. |

---

### 3.2.2 Dimensionality Reduction

*Compressing Information*

Real-world datasets often have hundreds or thousands of features (dimensions). Many are redundant or correlated. Dimensionality reduction compresses this into a smaller set of features while retaining the most important information.

```
🗜️ WHY REDUCE DIMENSIONS?

Problem 1 — The Curse of Dimensionality:
  As dimensions increase, data becomes increasingly sparse. Distance
  metrics lose meaning. Models need exponentially more data to work well.

Problem 2 — Visualisation:
  Humans can only see 2D or 3D. To visualise 100-dimensional data,
  we must compress it to 2 or 3 dimensions first.

Problem 3 — Computational cost:
  Fewer features = faster training and less memory.

Problem 4 — Noise removal:
  Low-variance dimensions are often noise. Removing them can improve accuracy.
```

#### PCA — Principal Component Analysis

PCA is the most widely used dimensionality reduction technique. It finds the directions of maximum variance in the data and projects the data onto those directions:

```
📉 PCA: COMPRESSING 3D DATA TO 2D

ORIGINAL DATA: 3 features (Height, Weight, Age)
Each person = a point in 3D space

PCA finds the 2 directions that capture the most spread:

  PC1 (1st Principal Component):
      = 0.6 × Height + 0.7 × Weight - 0.1 × Age
      Captures 78% of total variance

  PC2 (2nd Principal Component):
      = -0.3 × Height + 0.2 × Weight + 0.9 × Age
      Captures 18% of total variance

  Total variance retained with 2 components = 96%

  We compressed 3D → 2D with only 4% information loss!

  Original features: Height, Weight, Age  (3 numbers per person)
  After PCA:         PC1, PC2             (2 numbers per person) — 33% smaller!
```

#### Dimensionality Reduction Methods

| Method | Properties & Best Use |
|---|---|
| **PCA** | Linear. Fast. Standard choice for numeric data. Preserves global structure (variance). |
| **t-SNE** | Non-linear. Excellent for visualisation. Preserves local neighbourhoods. Slow on large data. Do not use for anything other than 2D/3D visualisation. |
| **UMAP** | Non-linear. Faster than t-SNE. Preserves both local and some global structure. Increasingly popular for biological data (genomics, single-cell RNA). |
| **Autoencoders** | Neural network that learns to compress data through a bottleneck layer and reconstruct it. Powerful for complex data (images, audio). Learnable compression. |

---

### 3.2.3 Anomaly Detection

*Finding the Unusual*

Anomaly detection (also called outlier detection) identifies data points that deviate significantly from the normal pattern. The model first learns what 'normal' looks like — then flags anything that does not fit.

```
🚨 ANOMALY DETECTION: CREDIT CARD FRAUD

Normal transaction pattern (learned from millions of transactions):
  - Amount:   Rs. 200 – Rs. 5,000
  - Location: Home city or nearby
  - Time:     Daytime hours
  - Merchant: Regular categories (groceries, fuel, restaurants)

ANOMALOUS transaction (flagged for review):
  - Amount:   Rs. 85,000       ← Unusually large
  - Location: Different country ← Never seen before
  - Time:     3:47 AM          ← Outside normal pattern
  - Merchant: Jewellery store  ← Uncommon category

  Anomaly Score: 0.97 (very high) → FLAGGED FOR FRAUD REVIEW
```

| Domain | Application |
|---|---|
| Credit Card | Flag suspicious transactions in real time |
| Manufacturing | Detect defective products from sensor readings |
| IT Security | Identify unusual network traffic or login patterns |
| Healthcare | Flag abnormal patient vitals before they become critical |
| Predictive Maintenance | Detect early signs of machine failure from vibration/temperature data |

---

### 3.2.4 Association Rule Learning

*Co-occurrence Patterns*

Association rule learning finds relationships between variables in large datasets. It is most famous for **market basket analysis** — understanding which products customers tend to buy together.

```
🛒 ASSOCIATION RULES: SUPERMARKET BASKET ANALYSIS

Transaction data (10,000 receipts analysed):

  Rule 1: {Bread, Butter} → {Milk}
    Support:    42%  (42% of all baskets contain all three items)
    Confidence: 87%  (when bread & butter bought, milk 87% of the time)
    Lift:       2.3  (buying bread+butter makes milk 2.3× more likely)

  Rule 2: {Nappies} → {Beer}
    Support: 18% | Confidence: 73% | Lift: 3.1
    (Famous real-world finding: new fathers buy beer while buying nappies!)

  Rule 3: {Phone Case} → {Screen Protector}
    Support: 61% | Confidence: 91% | Lift: 4.2
    → Strong rule: almost always bought together

Business action:   Place these items near each other in the store.
E-commerce action: 'Customers who bought this also bought...'
```

> 📊 **Key Metrics for Association Rules**
>
> - **Support** = how often items appear together overall.
> - **Confidence** = given item A, how often does item B appear.
> - **Lift** = how much more likely B is given A, compared to B on its own.
> - Lift > 1 means a genuine positive association; Lift = 1 means no relationship.

---

### Real-World Unsupervised Learning Applications

| Domain | Technique | Application |
|---|---|---|
| Customer Analytics | Clustering | Segment customers by behaviour for targeted campaigns |
| Genomics | Dimensionality Reduction | Visualise gene expression patterns across thousands of genes |
| Retail | Association Rules | Power 'frequently bought together' recommendations |
| Cybersecurity | Anomaly Detection | Flag unusual login activity or data exfiltration |
| Image Compression | Autoencoders | Compress and reconstruct images with minimal loss |
| Document Organisation | Clustering | Automatically group news articles by topic |
| Finance | Anomaly Detection | Flag unusual trading activity for insider trading investigations |

> 💡 **When to Use Unsupervised Learning**
>
> Use unsupervised learning when you have NO labels and want to explore, organise, or compress your data. It is also powerful as a pre-processing step before supervised learning — use clustering or PCA to create better features for a downstream classifier.

---

## 3.3 Reinforcement Learning

*A quick introduction — learning through reward and exploration*

### What is Reinforcement Learning?

Reinforcement Learning (RL) is fundamentally different from supervised and unsupervised learning. There are no labelled examples and no fixed dataset. Instead, an **agent** learns by interacting with an **environment** — taking actions and receiving feedback in the form of rewards or penalties.

```
🐕 THE TRAINING A DOG ANALOGY

  The AGENT       is the dog.
  The ENVIRONMENT is the room and everything in it.
  The STATE       is what the dog currently perceives (where it is, what it sees).
  The ACTION      is what the dog does (sit, stay, fetch, bark).
  The REWARD      is the treat (+1) or the scolding (-1).

  Over time, the dog learns a POLICY:
  'When my owner says SIT, sitting gets me a treat. So I should SIT.'

  The goal: maximise total reward over time — not just the next immediate treat.
```

### The RL Loop — How an Agent Learns

```
🔄 THE REINFORCEMENT LEARNING CYCLE

  ┌─────────────────────┐
  │     ENVIRONMENT     │
  │  (the world/game)   │
  └──────────┬──────────┘
             │
  State (what the agent observes)
  Reward (did that action help?)
             │
  ┌──────────▼──────────┐
  │        AGENT        │
  │  (learns a policy)  │
  └──────────┬──────────┘
             │
  Action (what the agent does)
             │
  ┌──────────▼──────────┐
  │     ENVIRONMENT     │
  │ transitions to new  │
  │ state based on the  │
  │    agent's action   │
  └─────────────────────┘
```

### Core RL Concepts

| Concept | What It Means |
|---|---|
| **Agent** | The learner / decision-maker. In a chess game, this is the chess player AI. |
| **Environment** | Everything outside the agent that it interacts with. In chess: the board, the opponent. |
| **State (s)** | The current situation as observed by the agent. In chess: the current configuration of all pieces. |
| **Action (a)** | A choice the agent can make. In chess: moving a piece to a valid square. |
| **Reward (r)** | Feedback signal. Positive = good outcome, negative = bad. In chess: +1 for winning, -1 for losing, 0 for ongoing play. |
| **Policy (π)** | The strategy: given a state, which action to take. This is what the agent is trying to learn. |
| **Value Function (V)** | Estimates the total future reward expected from a given state. Helps the agent plan ahead rather than just grabbing immediate rewards. |
| **Exploration vs Exploitation** | The fundamental tension: try new actions (explore) to discover better strategies, or stick with what works (exploit) to maximise current rewards. |

### Famous RL Achievements

```
🏆 WHAT REINFORCEMENT LEARNING HAS ACCOMPLISHED

2013 — DeepMind's DQN
  Learned to play 49 Atari games directly from raw pixel input.
  Reached superhuman performance on many games (e.g. Breakout, Pong).
  No game-specific programming — just reward signals from the score.

2016 — AlphaGo (DeepMind)
  Defeated the world Go champion Lee Sedol 4–1.
  Go has more possible positions than atoms in the universe.
  Previously thought to require human intuition — impossible to brute-force.

2019 — AlphaStar (DeepMind)
  Reached Grandmaster level in StarCraft II — a real-time strategy game
  requiring long-term planning, resource management, and uncertainty.

2022 — ChatGPT / RLHF
  Reinforcement Learning from Human Feedback (RLHF) used to align
  large language models with human preferences — making them safer,
  more helpful, and more conversational.

Robotics: RL agents learning to walk, grasp objects, assemble parts.
Energy:   DeepMind RL reduced Google data centre cooling energy by 40%.
```

> ⚠️ **When is RL the Right Choice?**
>
> RL is powerful but expensive and complex. Use it when:
> 1. You have a sequential decision-making problem.
> 2. There is a clear reward signal.
> 3. Supervised learning is not feasible because you cannot label every correct action in advance.
> 4. You can afford to simulate or interact with the environment many times.
>
> For most business problems, supervised or unsupervised learning is simpler and more practical.

---

## 3.4 How to Choose the Right Type

*A practical decision framework for real problems*

Given a new ML problem, how do you decide which paradigm to use? Work through these questions in order:

```
🧭 THE ML TYPE SELECTION FLOWCHART

START: What does your data look like?
│
├── Do you have LABELLED data (known correct answers)?
│     │
│     ├── YES → SUPERVISED LEARNING
│     │           │
│     │           ├── Output is a CATEGORY? → Classification
│     │           └── Output is a NUMBER?   → Regression
│     │
│     └── NO → UNSUPERVISED LEARNING
│                 │
│                 ├── Want to GROUP similar items?     → Clustering
│                 ├── Want to COMPRESS the data?       → Dim. Reduction
│                 ├── Want to find UNUSUAL points?     → Anomaly Detection
│                 └── Want ITEM CO-OCCURRENCE rules?   → Association Rules
│
└── Is it a SEQUENTIAL DECISION problem with a reward signal?
          → REINFORCEMENT LEARNING
```

### Side-by-Side Comparison

| | **Supervised** | **Unsupervised** | **Reinforcement** |
|---|---|---|---|
| **Data needed** | Labelled data required | No labels needed | Reward signal needed |
| **Task type** | Classification / Regression | Clustering / Compression / Anomaly | Sequential decision-making |
| **Output** | Clear target output | Discover hidden structure | Agent learns a policy |
| **Example** | Spam filter, price prediction | Customer segmentation, PCA | Game playing, robotics |
| **Evaluation** | Easier (accuracy, RMSE) | Harder (subjective) | Hard (long-horizon reward) |
| **Industry use** | Most common | Very common for exploration | Specialised, high-compute |

### A Note on Semi-Supervised & Self-Supervised Learning

Two important modern paradigms sit between supervised and unsupervised learning:

```
🔬 BEYOND THE THREE CORE TYPES

SEMI-SUPERVISED LEARNING
  You have a SMALL amount of labelled data + a LARGE amount of unlabelled data.
  The model uses the unlabelled data to improve its representation before
  fine-tuning on the labelled examples.
  Example: Training a medical image classifier with 500 labelled scans
  and 50,000 unlabelled scans.

SELF-SUPERVISED LEARNING
  The model creates its OWN labels from the raw data.
  Example: Mask 15% of words in a sentence → train the model to predict
  the masked words. This is how BERT and GPT were trained.
  Now powers most of modern NLP and computer vision.

TRANSFER LEARNING
  Take a model trained on a large dataset (e.g. ImageNet, the internet),
  then fine-tune it on a small specific dataset.
  Dramatically reduces the data and compute needed for your task.
```

---

## Chapter 3 Summary

| Topic | Key Takeaway |
|---|---|
| **3.1 Supervised Learning** | The model learns from labelled data. Two types: Classification (predicting categories) and Regression (predicting numbers). Powers the vast majority of real-world ML applications. |
| **3.2 Unsupervised Learning** | No labels — the model discovers structure. Includes Clustering (grouping), Dimensionality Reduction (compressing), Anomaly Detection (finding outliers), and Association Rules (co-occurrence patterns). |
| **3.3 Reinforcement Learning** | An agent learns a policy by interacting with an environment and maximising cumulative rewards. Powers game AI, robotics, and the alignment of large language models. |
| **3.4 Choosing the Right Type** | Start with the data: labelled data → supervised. No labels → unsupervised. Sequential decision + reward → reinforcement. Semi-supervised and self-supervised bridge the gaps. |

---

## 🚀 What is Coming Next

**Chapter 4: Machine Learning Algorithms — Deep Dive**

- Linear models (regression, logistic regression, SVMs)
- Tree-based models (Decision Trees, Random Forests, Gradient Boosting)
- Introduction to Neural Networks and Deep Learning
- When to use which algorithm — a practical cheat sheet
- Hands-on worked examples for each algorithm type
