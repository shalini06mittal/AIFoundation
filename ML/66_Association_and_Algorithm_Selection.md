# MACHINE LEARNING
### *A Complete Beginner-to-Advanced Guide*

## Chapter 6: Association & Choosing the Right Algorithm

*Continuing from Chapters 1 – 5:*
*Ch.1 Intro | Ch.2 How ML Works | Ch.3 Types of ML | Ch.4 Classification | Ch.5 Clustering*

---

## Table of Contents

**PART A — Association Rules & Market Basket Analysis**
- [6.1 What is Association Rule Learning?](#61-what-is-association-rule-learning)
- [6.2 Key Metrics: Support, Confidence, Lift, Conviction](#62-key-metrics--support-confidence-lift-conviction)
- [6.3 The Apriori Algorithm — Step by Step](#63-the-apriori-algorithm--step-by-step)
- [6.4 The FP-Growth Algorithm — Faster & Scalable](#64-the-fp-growth-algorithm--faster--scalable)
- [6.5 Real-World Applications Across Industries](#65-real-world-applications-across-industries)

**PART B — Choosing the Right Algorithm**
- [6.6 A Framework for Algorithm Selection](#66-a-framework-for-algorithm-selection)
- [6.7 Matching Problem Type to Algorithm](#67-matching-problem-type-to-algorithm)
- [6.8 Thinking About Your Input Data](#68-thinking-about-your-input-data)
- [6.9 Feature Types: Numerical vs Categorical](#69-feature-types-numerical-vs-categorical)
- [6.10 The Master Algorithm Cheat Sheet](#610-the-master-algorithm-cheat-sheet)
- [Chapter 6 Summary](#chapter-6-summary)

---

# PART A — Association Rules
*Market Basket Analysis & Finding What Goes Together*

---

## 6.1 What is Association Rule Learning?

*Discovering which items, events, or behaviours tend to occur together*

### The Everyday Intuition

Walk into any Big Bazaar or D-Mart store. You notice that diapers and baby wipes are placed next to each other. Bread, butter, and jam share a shelf. Chips and cold drinks are co-located at the entrance. These arrangements are not accidental — they are the result of analysing millions of purchase transactions to discover which products are frequently bought together.

Association Rule Learning is the branch of ML that discovers exactly these kinds of co-occurrence patterns. Given a massive database of transactions, it automatically finds rules of the form:

> ### 📦 The Structure of an Association Rule
>
> ```
> IF {Antecedent} THEN {Consequent}
> ```
>
> **Example rules discovered from supermarket data:**
>
> ```
> IF {Bread, Butter}        THEN {Milk}
> IF {Phone Case}           THEN {Screen Protector}
> IF {Nappies}              THEN {Beer}          ← Famous real-world finding from US supermarket data!
> IF {IPL Jersey, Cricket Bat} THEN {Cricket Ball}  ← From Myntra/Decathlon data during cricket season
> ```
>
> *'Fathers buying nappies also often buy beer.'*

---

### Where Does It Fit in the ML Map?

| Category | Approach | Examples |
|---|---|---|
| Supervised Learning | Has labelled data — predicts a target variable | Spam detection, price prediction, image classification |
| Unsupervised Learning | No labels — discovers hidden structure | Clustering (Ch.5), Association Rules (this chapter) |
| Association Rules | No labels — discovers co-occurrence relationships | Market basket, cross-sell, sequential pattern mining |

> ### 💡 Association vs Clustering
>
> **Clustering** (Ch.5) groups similar DATA POINTS together (e.g. similar customers).
>
> **Association Rules** find relationships between ITEMS within transactions (e.g. co-purchased products).
>
> Both are unsupervised — neither requires labelled data.

---

### Real-World Motivation — Scale of the Problem

Amazon India has over 500 million products. Flipkart has over 150 million. A supermarket like Reliance Fresh carries 40,000+ SKUs. The number of possible item pairs is astronomically large:

> ### 📊 Scale of the Search Space
>
> ```
> Supermarket with 40,000 items:
>
> Possible item pairs:    40,000 × 39,999 / 2  = ~800 million pairs
> Possible item triples:  40,000^3 / 6         = ~10 trillion triples
>
> Naive approach: check every possible combination
> → Even at 1 billion checks/second: 10,000 seconds for triples alone
> → Completely infeasible for large itemsets
>
> We need SMART algorithms that prune the search space.
> That is what Apriori and FP-Growth do.
> ```

---

## 6.2 Key Metrics — Support, Confidence, Lift, Conviction

*The four numbers that define how strong an association rule really is*

### The Dataset We Will Use Throughout

A simple grocery dataset of 10 transactions. Each row is one customer's basket:

| Transaction ID | Items Purchased |
|---|---|
| T01 | Bread, Milk, Butter |
| T02 | Bread, Milk |
| T03 | Bread, Butter, Jam |
| T04 | Milk, Butter, Eggs |
| T05 | Bread, Milk, Butter, Eggs |
| T06 | Bread, Jam |
| T07 | Milk, Eggs |
| T08 | Bread, Milk, Jam |
| T09 | Butter, Eggs, Milk |
| T10 | Bread, Milk, Butter |

---

### Metric 1 — Support

Support tells you how popular an item or itemset is across all transactions. It is the fraction of transactions that contain the item(s).

> ### 📐 Support — Formula and Examples
>
> ```
> Support(X) = (Number of transactions containing X)
>              ─────────────────────────────────────
>              (Total number of transactions)
> ```
>
> **From our 10-transaction dataset:**
> ```
> Support(Bread)         = 7/10 = 0.70 (70%)
> Support(Milk)          = 8/10 = 0.80 (80%)
> Support(Butter)        = 5/10 = 0.50 (50%)
> Support(Bread, Milk)   = 5/10 = 0.50 (50%)  [T01, T02, T05, T08, T10]
> Support(Bread, Butter) = 4/10 = 0.40 (40%)  [T01, T03, T05, T10]
> Support(Milk, Butter)  = 4/10 = 0.40 (40%)  [T01, T04, T05, T10]
> ```
>
> **Minimum Support Threshold (`min_sup`):**
> You set a threshold below which rules are too rare to be useful.
> Common choice: 5%–20% depending on domain.
> If `min_sup = 0.40` (40%), only keep itemsets with support >= 0.40.

> ### 💡 Why Support Matters
>
> - **Low support** = the rule applies to very few customers — might not be actionable.
> - **High support** = the rule is widespread — definitely worth acting on.
> - Setting `min_sup` prunes rare patterns early, making the search tractable.

---

### Metric 2 — Confidence

Confidence measures the reliability of the rule. Given that the antecedent (left side) was purchased, how often was the consequent (right side) also purchased?

> ### 📐 Confidence — Formula and Examples
>
> ```
> Confidence(X → Y) = Support(X ∪ Y)
>                     ──────────────
>                     Support(X)
> ```
>
> *In plain English: 'Of all baskets containing X, what fraction also contains Y?'*
>
> **Examples from our dataset:**
> ```
> Confidence(Bread → Milk):
>   = Support(Bread, Milk) / Support(Bread)
>   = 0.50 / 0.70 = 0.714 (71.4%)
>   → 71.4% of customers who bought Bread also bought Milk
>
> Confidence(Milk → Bread):
>   = Support(Milk, Bread) / Support(Milk)
>   = 0.50 / 0.80 = 0.625 (62.5%)
>   → Note: Confidence is NOT symmetric! X→Y ≠ Y→X
>
> Confidence(Butter → Milk):
>   = Support(Butter, Milk) / Support(Butter)
>   = 0.40 / 0.50 = 0.80 (80%)
>   → Strong rule! 80% of Butter buyers also buy Milk.
> ```

> ### ⚠️ The Confidence Trap — Why Confidence Alone Misleads
>
> Consider: `Confidence(Milk → Bread) = 62.5%`
>
> This sounds impressive! But Bread has a base rate of **70%** in our data. A customer who buys NOTHING is still 70% likely to buy Bread!
>
> So seeing Milk actually **DECREASES** the probability of Bread (70% → 62.5%).
>
> High confidence can be misleading when the consequent is very popular. This is why we need **Lift**.

---

### Metric 3 — Lift

Lift corrects for baseline popularity. It measures how much MORE likely Y is to be purchased when X is purchased, compared to Y's overall purchase rate. Lift is the gold standard metric for rule quality.

> ### 📐 Lift — Formula, Examples, and Interpretation
>
> ```
> Lift(X → Y) = Confidence(X → Y)      Support(X ∪ Y)
>               ─────────────────  =  ──────────────────────
>               Support(Y)             Support(X) × Support(Y)
> ```
>
> | Lift Value | Meaning |
> |---|---|
> | Lift > 1.0 | Positive association — X and Y co-occur MORE than by chance |
> | Lift = 1.0 | Independence — knowing X tells you nothing about Y |
> | Lift < 1.0 | Negative association — X and Y co-occur LESS than by chance |
>
> **Examples from our dataset:**
> ```
> Lift(Bread → Milk):
>   = Confidence(Bread→Milk) / Support(Milk)
>   = 0.714 / 0.80 = 0.893
>   → Lift < 1! Bread actually makes Milk LESS likely. Negative association.
>
> Lift(Butter → Milk):
>   = Confidence(Butter→Milk) / Support(Milk)
>   = 0.80 / 0.80 = 1.0
>   → Lift = 1.0 exactly. Butter and Milk are independent!
>
> Lift(Butter → Eggs):
>   Support(Butter, Eggs) = 2/10 = 0.20  [T04, T09]
>   Confidence(Butter→Eggs) = 0.20 / 0.50 = 0.40
>   Support(Eggs) = 3/10 = 0.30
>   Lift = 0.40 / 0.30 = 1.33
>   → Positive association! Buying Butter makes Eggs 33% more likely.
> ```

---

### Metric 4 — Conviction

Conviction measures how often the rule would be wrong if the items were statistically independent. Unlike Lift, it is directional and captures the degree to which the antecedent and consequent are NOT independent.

> ### 📐 Conviction — Formula and Interpretation
>
> ```
> Conviction(X → Y) =  1 - Support(Y)
>                      ──────────────────────
>                      1 - Confidence(X → Y)
> ```
>
> | Conviction | Meaning |
> |---|---|
> | = 1.0 | X and Y are independent (same as Lift = 1.0) |
> | > 1.0 | Positive dependence — rule is better than chance |
> | = ∞ | Perfect rule — X always implies Y (Confidence = 1.0) |
> | < 1.0 | Negative dependence — rule is worse than chance |
>
> Conviction is sensitive to the **DIRECTION** of the rule and handles the case of Confidence = 1 more elegantly than Lift.
>
> **Example:**
> ```
> Conviction(Butter → Eggs):
>   = (1 - 0.30) / (1 - 0.40)
>   = 0.70 / 0.60 = 1.17
>   → Rule is 17% better than random chance would predict.
> ```

---

### Metrics Summary

| Metric | Question It Answers | What It Tells You | Rule of Thumb |
|---|---|---|---|
| **Support** | What fraction of all transactions contain this itemset? | How common/popular is the pattern? | Keep if above `min_sup` threshold |
| **Confidence** | Given X is bought, how often is Y also bought? | How reliable is the rule? | Keep if above `min_conf` threshold |
| **Lift** | How much more likely is Y given X vs Y alone? | Is the relationship real or just popular? | Look for Lift > 1.2 minimum |
| **Conviction** | How much does X increase dependency on Y? | Directional strength of the implication | Higher is better; 1.0 = no association |

---

## 6.3 The Apriori Algorithm

*The classic algorithm for mining frequent itemsets efficiently*

### The Key Insight — The Apriori Property

> ### 🔑 The Apriori Property
>
> **If an itemset is infrequent (below `min_sup`), then ALL its supersets are also infrequent.**
>
> ```
> If {Bread, Butter, Jam} is infrequent →
>   {Bread, Butter, Jam, Milk} MUST also be infrequent
>   {Bread, Butter, Jam, Eggs} MUST also be infrequent
>   → We can PRUNE these supersets without ever calculating their support!
>
> Conversely: every SUBSET of a frequent itemset must also be frequent.
>   If {Bread, Milk, Butter} is frequent →
>   {Bread, Milk}, {Bread, Butter}, {Milk, Butter} must all be frequent too.
> ```
>
> This allows Apriori to eliminate huge portions of the search space early, making what was computationally infeasible suddenly tractable.

---

### The Apriori Algorithm — Step by Step

Using our 10-transaction dataset with `min_sup = 0.40` (at least 4 out of 10 transactions):

> ### 🔄 Apriori — Full Worked Example
>
> **Step 1 — SCAN 1: Count individual item frequencies (1-itemsets)**
> ```
> Bread: 7    Milk: 8    Butter: 5    Jam: 3    Eggs: 3
>
> With min_sup = 0.40 (need >= 4):
>   ✅ Bread(7), Milk(8), Butter(5) → PASS
>   ❌ Jam(3), Eggs(3) → PRUNE. Never consider again.
> ```
>
> **Step 2 — GENERATE: Create candidate 2-itemsets from frequent 1-itemsets**
> ```
> Combine all pairs of {Bread, Milk, Butter}:
>   {Bread, Milk},  {Bread, Butter},  {Milk, Butter}
> ```
>
> **Step 3 — SCAN 2: Count 2-itemset frequencies**
> ```
> {Bread, Milk}   = 5 transactions (T01,T02,T05,T08,T10) → Support=0.50 ✅ PASS
> {Bread, Butter} = 4 transactions (T01,T03,T05,T10)     → Support=0.40 ✅ PASS
> {Milk, Butter}  = 4 transactions (T01,T04,T05,T10)     → Support=0.40 ✅ PASS
> All three 2-itemsets are frequent.
> ```
>
> **Step 4 — GENERATE: Create candidate 3-itemsets from frequent 2-itemsets**
> ```
> Only one possible combination: {Bread, Milk, Butter}
> Rule: all subsets must be frequent (they are — verified in Step 3) ✅
> ```
>
> **Step 5 — SCAN 3: Count 3-itemset frequencies**
> ```
> {Bread, Milk, Butter} = 3 transactions (T01, T05, T10) → Support=0.30
> Support 0.30 < 0.40 min_sup threshold. ❌ PRUNE. Stop here.
> ```
>
> **Step 6 — GENERATE RULES: From every frequent itemset, generate all possible rules**
> ```
> From {Bread, Milk}:   Bread→Milk   (conf=0.71),  Milk→Bread   (conf=0.63)
> From {Bread, Butter}: Bread→Butter (conf=0.57),  Butter→Bread (conf=0.80)
> From {Milk, Butter}:  Milk→Butter  (conf=0.50),  Butter→Milk  (conf=0.80)
>
> Apply min_confidence threshold (e.g. 0.70) to filter weak rules.
> ```

> ### 🏆 Final Rules After Applying `min_conf = 0.70`
>
> ```
> Rule                  Support   Confidence   Lift    Verdict
> ─────────────────────  ───────  ──────────   ────    ─────────────────────────────
> Bread  → Milk          0.50     0.714        0.89    BORDERLINE (Lift < 1)
> Butter → Bread         0.40     0.800        1.14    ✅ GOOD
> Butter → Milk          0.40     0.800        1.00    WEAK (Lift = 1, independent)
>
> Applying Lift > 1.0 filter:
> Only STRONG rule: Butter → Bread (Lift = 1.14)
> ```
>
> **Business insight:** Customers who buy butter are 14% more likely than average to also buy bread. Stock them near each other.

---

### Apriori — Complexity and Limitations

| Property | Value | Implication |
|---|---|---|
| **Time Complexity** | O(2^n) in worst case | Still practical with good `min_sup` pruning on typical data |
| **Space Complexity** | O(n × transactions) | Stores all candidate itemsets — can be large |
| **Multiple DB Scans** | One scan per itemset size | Slow on disk-based databases; better on in-memory data |
| **Candidate Generation** | Generates many false candidates | Many candidates computed but pruned — wasteful CPU |
| **Scalability** | Struggles on millions of items | FP-Growth (next section) is better for large-scale data |

---

## 6.4 The FP-Growth Algorithm

*Mining frequent patterns without candidate generation — the scalable solution*

### Why FP-Growth Was Invented

Apriori's main bottleneck is candidate generation and multiple database scans. For a supermarket with 40,000 items, even after pruning, millions of candidate itemsets can be generated. FP-Growth solves this by compressing the entire database into a compact tree structure and mining it directly.

> ### ⚡ FP-Growth vs Apriori — Core Difference
>
> **APRIORI approach:**
> ```
> 1. Scan database multiple times
> 2. Generate candidate itemsets explicitly
> 3. Count support of each candidate
> 4. Prune based on Apriori property
>
> Problem: Huge number of candidates; slow with many DB scans
> ```
>
> **FP-GROWTH approach:**
> ```
> 1. Scan database TWICE only
> 2. Build a compressed FP-Tree in memory
> 3. Mine the FP-Tree recursively — no candidate generation!
>
> Result: Typically 10–100x faster than Apriori on real datasets
> ```

---

### Building the FP-Tree — Visual Walkthrough

Using our same dataset (`min_sup = 0.40`, so Bread, Milk, Butter are frequent):

> ### 🌳 FP-Tree Construction — Step by Step
>
> **Step 1: Find frequent items (same as Apriori Scan 1)**
> ```
> Order by frequency: Milk(8), Bread(7), Butter(5)
> ```
>
> **Step 2: Re-order each transaction by frequency rank, keeping only frequent items:**
> ```
> T01: Milk, Bread, Butter   (original: Bread, Milk, Butter)
> T02: Milk, Bread           (original: Bread, Milk)
> T03: Bread, Butter         (Jam removed — infrequent)
> T04: Milk, Butter          (Eggs removed — infrequent)
> T05: Milk, Bread, Butter
> T06: Bread                 (Jam removed)
> T07: Milk                  (Eggs removed)
> T08: Milk, Bread           (Jam removed)
> T09: Milk, Butter          (Eggs removed)
> T10: Milk, Bread, Butter
> ```
>
> **Step 3: Insert each transaction into the FP-Tree**
> ```
>                [root]
>                   |
>            [Milk: count=8]
>              /          \
>       [Bread: 5]      [Butter: 2]
>        /      \
>  [Butter: 3]  [end]
>
>   +
>
>  [Bread: 2]  ← Bread-only paths
>   [Butter: 1]
> ```
> The tree compresses 10 transactions into a compact structure.
> Shared prefixes are merged — Milk appears only once at the top.

> ### 🚀 FP-Growth Performance on Real Data
>
> - **Walmart** (200+ million transactions): FP-Growth mines rules in minutes; Apriori takes days.
> - **Amazon** product catalogue (500M+ items): Only FP-Growth is computationally feasible.
> - In practice: `scikit-learn` and `Spark MLlib` both implement FP-Growth for production use.

---

## 6.5 Real-World Applications

*How association rules power recommendations, operations, and strategy across industries*

### Application 1 — E-Commerce: Amazon's 'Frequently Bought Together'

> ### 🛒 Amazon India — Cross-Sell Engine
>
> - **Dataset:** 500 million product transactions per month
> - **Algorithm:** FP-Growth with `min_sup = 0.001` (0.1% of transactions)
>
> **Sample discovered rules (Electronics category):**
> ```
> {Samsung Galaxy S24} → {Mobile Cover, Screen Protector, USB-C Cable}
>   Support: 0.0034   Confidence: 0.89   Lift: 4.2
>   Action: Bundle widget on product page; co-display in 'Frequently Bought'
>
> {MacBook Pro M3} → {USB-C Hub, Mouse, Laptop Bag}
>   Support: 0.0018   Confidence: 0.76   Lift: 5.8
>   Action: 'Customers who bought this also bought' recommendation row
>
> {DSLR Camera} → {SD Card, Camera Bag, Extra Battery}
>   Support: 0.0042   Confidence: 0.84   Lift: 6.1
>   Action: Bundle offer at 12% discount — boosts average order value
> ```
>
> **Business Impact:**
> - Amazon reports 35% of total revenue comes from recommendation engines.
> - Cross-sell rules contribute an estimated 15% of that revenue.
> - Average order value increases 20–27% when bundle rules are triggered.

---

### Application 2 — Retail Banking: Cross-Sell at HDFC Bank

> ### 🏦 HDFC Bank — Financial Product Association Mining
>
> - **Dataset:** 40 million customer transactions and product holdings
> - **Goal:** Find which products are naturally bought together
>
> **Discovered rules:**
> ```
> {Savings Account, Salary Account} → {Fixed Deposit}
>   Confidence: 0.61   Lift: 2.8
>   Segment: Salaried professionals aged 28–40
>   Action: Auto-trigger FD offer at month-end when salary credited
>
> {Home Loan} → {Home Insurance, Term Life Insurance}
>   Confidence: 0.73   Lift: 4.1
>   Action: Bundle insurance offer with home loan disbursement call
>
> {Credit Card} → {Personal Loan within 12 months}
>   Confidence: 0.38   Lift: 1.9 (Negative lift if credit score drops)
>   Action: Flag for credit monitoring; proactive debt management offer
>
> {SIP in Equity Fund} → {SIP in Debt Fund within 6 months}
>   Confidence: 0.44   Lift: 3.2
>   Action: Suggest portfolio diversification — improves AUM retention
> ```
>
> **Result:** 23% increase in product per customer ratio over 18 months.

---

### Application 3 — Healthcare: Clinical Pathway Mining

> ### 🏥 Apollo Hospitals — Diagnosis Co-Occurrence Mining
>
> - **Dataset:** 2.3 million patient records over 10 years
> - **'Transaction'** = set of diagnoses for one patient
> - **Goal:** Find which conditions tend to co-occur (comorbidities)
>
> **Discovered rules:**
> ```
> {Type 2 Diabetes} → {Hypertension}
>   Confidence: 0.74   Lift: 2.1
>   Clinical insight: 74% of T2D patients develop hypertension.
>   Action: Mandatory BP screening at every diabetes follow-up
>
> {Hypertension, High Cholesterol} → {Coronary Artery Disease}
>   Confidence: 0.58   Lift: 3.4
>   Action: Proactive cardiac screening protocol for this risk group
>
> {Hip Replacement Surgery} → {Deep Vein Thrombosis within 30 days}
>   Confidence: 0.31   Lift: 4.7
>   Action: Mandatory anticoagulant protocol post hip surgery
>
> {COPD, Sleep Apnea} → {Pulmonary Hypertension}
>   Confidence: 0.42   Lift: 5.2
>   Action: Routine pulmonary function tests for COPD + Sleep Apnea patients
> ```
>
> **Impact:** 18% reduction in preventable hospital readmissions.

---

### Application 4 — Media & Entertainment: Streaming at Hotstar

> ### 🎬 Disney+ Hotstar — Content Co-Viewing Mining
>
> - **Dataset:** 500 million viewing sessions
> - **'Transaction'** = set of content watched by one user in a 30-day window
> - **Goal:** Which content is consumed together (co-viewing patterns)
>
> **Discovered rules:**
> ```
> {IPL Matches} → {Cricket documentaries, Anand Kumar Story}
>   Confidence: 0.67   Lift: 3.8
>   Action: Surface cricket docs in 'Watch Next' during IPL season
>
> {KBC (Kaun Banega Crorepati)} → {Taarak Mehta Ka Ooltah Chashmah}
>   Confidence: 0.54   Lift: 2.9   (Family evening viewing pattern)
>   Action: Auto-play TMKOC after KBC episode; promotes family plan upgrades
>
> {Marvel movies} → {What If...?, Loki, WandaVision}
>   Confidence: 0.81   Lift: 4.6
>   Action: MCU content roadmap widget; priority notification on new releases
>
> Sequential pattern: {Movie} → {Sequel within 7 days}
>   Used to schedule sequel notifications exactly when engagement is highest
> ```
>
> **Result:** 34% increase in content consumption per subscriber. Subscriber churn reduced by 12% via proactive 'next watch' nudges.

---

### Application 5 — Supply Chain: Inventory & Store Layout

> ### 🏪 Big Bazaar / Reliance Smart — Store Layout Optimisation
>
> - **Dataset:** 8 million POS transactions across 300 stores
>
> **Discovered rules with high Lift:**
> ```
> {Atta (wheat flour)} → {Ghee, Oil}              Lift: 3.1
>   Action: Place ghee and oil in the same aisle as atta.
>   Result: +18% ghee sales, reduced customer search time
>
> {Baby Diapers} → {Baby Wipes, Baby Powder}       Lift: 4.8
>   Action: Co-locate entire baby care category; dedicated section.
>   Result: +31% basket size for customers entering baby section
>
> {Maggi Noodles} → {Tomato Ketchup, Chilli Sauce} Lift: 2.7
>   Action: Cross-display with condiments; bundle promotions during school terms
>
> Negative association found:
> {Premium Organic Products} → NOT {Value Pack Products}  Lift: 0.3
>   Action: Do NOT co-display premium organic and budget value packs.
>   Placing them together cannibalises premium margins.
>
> Seasonal rules (Diwali, Holi, wedding season):
> {Gift Hamper} → {Premium Sweets, Dry Fruits}     Lift: 6.2 (festive season only)
> ```

---

# PART B — Choosing the Right Algorithm
*A complete framework for matching your problem to the right ML approach*

---

## 6.6 A Framework for Algorithm Selection

*The five questions every data scientist asks before writing a single line of code*

### Why Algorithm Choice Matters

Beginners often ask: *'Which algorithm is the best?'* The honest answer is: there is no universally best algorithm (this is the **No Free Lunch Theorem**). Every algorithm makes assumptions, and the best algorithm for your problem depends on the nature of your data, your goals, and your constraints.

> ### 🎯 The Five Questions to Ask Before Choosing an Algorithm
>
> ```
> Question 1: WHAT TYPE OF OUTPUT do you need?
>   A category? A number? Groups? Rules? A sequence?
>
> Question 2: DO YOU HAVE LABELS?
>   Labelled data (correct answers) → Supervised Learning
>   No labels                       → Unsupervised Learning
>
> Question 3: WHAT TYPE OF FEATURES do you have?
>   Numerical (continuous/discrete)? Categorical? Text? Image? Time series?
>
> Question 4: HOW MUCH DATA do you have?
>   < 1,000 samples:        simple models, cross-validation
>   1,000 – 100,000:        most algorithms work
>   > 100,000:              scalability becomes critical
>
> Question 5: WHAT ARE YOUR CONSTRAINTS?
>   Interpretability needed? Prediction speed? Training time? Memory?
> ```

---

### The No Free Lunch Theorem — Why No Algorithm 'Wins'

> ### 📊 No Free Lunch — What It Means in Practice
>
> The No Free Lunch (NFL) Theorem (Wolpert & Macready, 1997) states:
> *'Averaged over ALL possible problems, every algorithm performs equally well.'*
>
> **Practical implication:**
> - An algorithm that works brilliantly on images may fail on tabular data
> - A neural network that beats everything on text may lose to a Decision Tree on a small, structured business dataset
> - There is NO algorithm you can blindly apply to every problem
>
> **What this means for YOU:**
> ```
> Step 1: Start with a simple baseline (Logistic Regression, Naive Bayes)
> Step 2: Try 2–3 more complex algorithms
> Step 3: Evaluate all on held-out test data
> Step 4: Choose based on your specific performance + constraint tradeoffs
> Step 5: Invest in feature engineering — often beats switching algorithms
> ```

---

## 6.7 Matching Problem Type to Algorithm

*A complete decision guide from problem statement to algorithm shortlist*

### The Master Decision Flowchart

> ### 🗺️ Algorithm Selection — Full Decision Tree
>
> ```
> STEP 1: Do you have LABELLED DATA (known correct answers)?
> │
> ├── YES → SUPERVISED LEARNING
> │     │
> │     ├── Output is a CATEGORY (discrete class)?
> │     │   → CLASSIFICATION
> │     │     Algorithms: Logistic Regression, kNN, Naive Bayes,
> │     │                 Decision Tree, Random Forest, SVM, Neural Network
> │     │
> │     └── Output is a NUMBER (continuous value)?
> │         → REGRESSION
> │           Algorithms: Linear Regression, Ridge/Lasso, Decision Tree,
> │                       Random Forest, XGBoost, Neural Network
> │
> └── NO → UNSUPERVISED LEARNING
>       │
>       ├── Want to GROUP similar items?
>       │   → CLUSTERING
>       │     Algorithms: K-Means, DBSCAN, Hierarchical, GMM
>       │
>       ├── Want to find CO-OCCURRENCE rules?
>       │   → ASSOCIATION RULES
>       │     Algorithms: Apriori, FP-Growth, Eclat
>       │
>       ├── Want to COMPRESS high-dimensional data?
>       │   → DIMENSIONALITY REDUCTION
>       │     Algorithms: PCA, t-SNE, UMAP, Autoencoders
>       │
>       └── Want to find UNUSUAL / ANOMALOUS points?
>           → ANOMALY DETECTION
>             Algorithms: Isolation Forest, One-Class SVM, Autoencoders
> ```

---

### Classification Algorithm Selection

Once you have determined you need classification, the right algorithm depends on several sub-factors:

> **Is your data TEXT or very high-dimensional sparse data?**
> ```
> YES → Use NAIVE BAYES (Multinomial or Bernoulli NB)
>        Fast, accurate, handles thousands of features natively
>
> NO  → Continue to next question
> ```

> **Do you need HUMAN-READABLE rules that can be explained to stakeholders?**
> ```
> YES → Use DECISION TREE or LOGISTIC REGRESSION
>        (Most interpretable models)
>
> NO  → Continue to next question
> ```

> **Is your dataset SMALL (< 1,000 samples) and LOW-DIMENSIONAL (< 20 features)?**
> ```
> YES → Use kNN or SVM
>        (No training needed; handles small data well)
>
> NO  → Continue to next question
> ```

> **Do you have STRUCTURED/TABULAR data and need the BEST ACCURACY?**
> ```
> YES → Use RANDOM FOREST or GRADIENT BOOSTING (XGBoost, LightGBM, CatBoost)
>
> NO  → Use NEURAL NETWORK
>        (Unstructured: images, audio, text sequences)
> ```

---

### Regression Algorithm Selection

| Problem Characteristics | Recommended Algorithm | Real Example |
|---|---|---|
| Linear relationship, few features, interpretability key | Linear Regression | Salary prediction from years experience |
| Linear but many features, risk of overfitting | Ridge / Lasso Regression | Predicting house prices with 50+ features |
| Non-linear relationship, tabular data, accuracy key | Random Forest Regressor | Sales forecasting, demand prediction |
| Competition / highest accuracy on tabular data | XGBoost / LightGBM | Kaggle competitions, financial modelling |
| Time series with temporal patterns | LSTM / ARIMA / Prophet | Stock prices, weather, energy demand |
| Image or audio data | CNN-based Neural Network | Medical imaging measurement regression |

---

### Worked Examples — Problem to Algorithm Matching

| Problem | Data | Task Type | Algorithm | Why |
|---|---|---|---|---|
| Predict whether a Zomato customer will reorder within 7 days | Labelled: Yes/No reorder history | Binary Classification | XGBoost / Random Forest | High accuracy on tabular data; handles mixed feature types |
| Estimate next month's sales for 5,000 SKUs | Historical sales numbers | Regression (time series) | LightGBM + lag features / Prophet | Captures non-linear trends and seasonality |
| Group 10 million Swiggy customers for targeted campaigns | No labels | Clustering | K-Means (k=5–8) with Mini-Batch | Scalable; interpretable cluster centres |
| Find which menu items are ordered together on Swiggy | No labels | Association Rules | FP-Growth (`min_sup=0.02`) | Efficient on millions of transactions |
| Detect fraudulent UPI transactions in real time | Some labelled fraud cases | Anomaly Detection + Classification | Isolation Forest + XGBoost | Handles rare fraud events (class imbalance) |
| Classify customer support tickets by category | Labelled ticket categories | Multi-class Text Classification | BERT fine-tuned / Naive Bayes | BERT for high accuracy; NB for speed |

---

## 6.8 Thinking About Your Input Data

*Data volume, quality, and structure are as important as algorithm choice*

### Data Size — How Much Data Do You Have?

| Sample Size | Scale | Best Algorithms | Key Considerations | Example |
|---|---|---|---|---|
| < 100 samples | Tiny | Logistic Regression, SVM, kNN | Avoid deep learning. Cross-validation essential. Consider data augmentation. | Medical trial with 50 patients |
| 100 – 1,000 | Small | Random Forest, SVM, Naive Bayes | Regularisation critical. Cross-validation. Simple neural nets possible. | Customer churn for a small startup |
| 1,000 – 10,000 | Medium | Random Forest, XGBoost, Shallow NN | Most classical algorithms work well. Grid search for hyperparameters. | Hospital patient records |
| 10,000 – 1,000,000 | Large | XGBoost, LightGBM, Deep Learning | Scalability starts to matter. Mini-batch training. Feature engineering rewarding. | E-commerce transactions |
| > 1,000,000 | Very Large | LightGBM, Neural Nets, Spark ML | Distributed computing may be needed. Mini-batch essential. Use GPU. | Social media, clickstream data |

---

### Data Dimensionality — How Many Features?

> ### 📐 Feature Count and Algorithm Suitability
>
> **1 – 10 Features (Low Dimensional):**
> ```
> Any algorithm works. Visualisation easy. Try all options.
> Example: Predict loan default from age, income, credit score.
> ```
>
> **10 – 100 Features (Medium Dimensional):**
> ```
> Most algorithms suitable. Feature selection / importance analysis helpful.
> Decision Trees, Random Forest, Logistic Regression all work well.
> Example: Customer churn with 50 behavioural and demographic features.
> ```
>
> **100 – 10,000 Features (High Dimensional):**
> ```
> kNN suffers (curse of dimensionality — distances lose meaning).
> Linear models still work well if regularised (Lasso for selection).
> Tree-based models robust to irrelevant features.
> Consider PCA / feature selection first.
> Example: Gene expression data (30,000 genes per patient).
> ```
>
> **> 10,000 Features (Very High Dimensional):**
> ```
> Naive Bayes and Linear SVM shine (designed for sparse, high-dim data).
> Deep learning with embedding layers.
> Example: Text classification (vocabulary = 50,000+ words).
>
> RULE: High dimensions + few samples = high overfitting risk.
>       Always regularise. Consider dimensionality reduction first.
> ```

---

### Data Quality — What Does Your Data Look Like?

| Issue | What It Means | Algorithm Impact | How to Handle |
|---|---|---|---|
| **Missing Values** | Some data points have empty fields | kNN, SVM struggle; Random Forest, XGBoost handle natively | Imputation strategies: mean/median/mode, model-based, or treat as a feature |
| **Class Imbalance** | 99% Not Fraud, 1% Fraud | Accuracy is misleading; use Precision/Recall/F1/AUC | SMOTE oversampling, `class_weight='balanced'`, or threshold tuning |
| **Outliers** | A few extreme data points | Linear models and kNN sensitive; Tree-based models robust | Cap/Winsorise, log transform, or use robust scalers |
| **Noise** | Random errors in labels or features | kNN (k=1) very sensitive; ensembles more robust | Increase k in kNN; more trees in Random Forest; regularise |
| **Duplicate Records** | Same transaction recorded twice | All algorithms biased by duplicates — never ignore this | Deduplication as mandatory preprocessing step |

---

## 6.9 Feature Types: Numerical vs Categorical

*Understanding your data types is the foundation of good modelling*

### The Taxonomy of Feature Types

> ### 🗂️ Complete Feature Type Taxonomy
>
> **NUMERICAL FEATURES**
> ```
> ├── Continuous: Can take any real value in a range
> │     Examples: Height (1.72m), Temperature (36.8°C), Income (Rs.85,000)
> │     ML Treatment: Use directly. Scale before distance-based models.
> │
> └── Discrete: Can only take integer/countable values
>       Examples: Age (25 years), No. of children (2), Purchase count (14)
>       ML Treatment: Often treat like continuous, but check distribution.
> ```
>
> **CATEGORICAL FEATURES**
> ```
> ├── Nominal: Categories with NO natural order
> │     Examples: City (Mumbai/Delhi/Bangalore), Gender, Colour
> │     ML Treatment: One-Hot Encoding or Embedding
> │
> └── Ordinal: Categories WITH a natural order
>       Examples: Education (High School < Grad < Post-Grad)
>                 Rating (1-star < 2-star < 3-star < 4-star < 5-star)
>                 T-shirt size (S < M < L < XL < XXL)
>       ML Treatment: Label Encoding (0,1,2,3...) or Ordinal Encoding
> ```
>
> **SPECIAL TYPES**
> ```
> Text:       Bag-of-Words, TF-IDF, or Embeddings (Word2Vec, BERT)
> Images:     Pixel values, CNN feature maps
> Dates/Times: Extract: year, month, day, weekday, hour, is_weekend
> Binary:     0/1 — treat as numerical or as categorical (both work)
> ```

---

### Encoding Categorical Features — The Details

#### One-Hot Encoding — For Nominal Categories

> ### 🔢 One-Hot Encoding Example
>
> **Original feature:** City (Mumbai, Delhi, Bangalore, Hyderabad)
>
> **After One-Hot Encoding:**
> ```
> Customer   City_Mumbai   City_Delhi   City_Bangalore   City_Hyderabad
> ────────   ───────────   ──────────   ──────────────   ──────────────
> Ravi            1             0              0                0
> Priya           0             1              0                0
> Anjali          0             0              1                0
> Kiran           0             0              0                1
> ```
>
> **Problem — High Cardinality:**
> If 'City' has 500 possible values, OHE creates 500 new columns!
>
> **Solution for high-cardinality: Target Encoding or Embedding**
> ```
> Target Encoding: Replace city name with the mean of the target
>   variable for that city (e.g. mean income for Mumbai customers)
>
> Pro: Single column.
> Con: Potential data leakage — use with care.
> ```

#### Label Encoding — For Ordinal Categories

> ### 🔢 Label / Ordinal Encoding Example
>
> **Original feature:** Education Level (ordinal — order matters)
>
> ```
> Education Level    Label Encoded    Ordinal Encoded
> ───────────────    ─────────────    ───────────────
> High School              0                1
> Diploma                  1                2
> Bachelor's               2                3
> Master's                 3                4
> PhD                      4                5
> ```
>
> Why ordinal encoding preserves order:
> `PhD (5) > Master's (4) > Bachelor's (3)` is mathematically meaningful.
> Using One-Hot on ordinal data LOSES this order information.
>
> **⚠️ NEVER use Label Encoding on NOMINAL data!**
> ```
> City: Mumbai=0, Delhi=1, Bangalore=2
> → This implies Bangalore > Delhi > Mumbai, which is meaningless
>   and will confuse your model.
> ```

---

### Which Algorithm Handles Which Feature Type?

| Algorithm | Numerical | Categorical | Scaling | Notes |
|---|---|---|---|---|
| **Logistic Regression** | ✅ YES | Encode first | Scale needed | Fast, interpretable. Must encode categoricals. Sensitive to scale. |
| **kNN** | YES (scale first) | Encode first | **Scale REQUIRED** | Distance-based — must scale all features. Struggles with many cats. |
| **Naive Bayes** | Gaussian NB | Bernoulli/Multinom | No scaling needed | Different NB variants for different feature types. |
| **Decision Tree** | ✅ YES | ✅ YES (no encoding needed) | No scaling needed | Handles mixed types natively. The most flexible for raw data. |
| **Random Forest** | ✅ YES | YES (label encode) | No scaling needed | Extremely robust to mixed types. One of the best default choices. |
| **XGBoost / LightGBM** | ✅ YES | YES (native in LGB) | No scaling needed | LightGBM has native categorical support. Best general performer. |
| **SVM** | YES (scale first) | Encode first | **Scale REQUIRED** | Very sensitive to scaling. Encode high-cardinality cats carefully. |
| **Neural Network** | YES (normalise) | Embedding layer | **Normalise REQUIRED** | Learns its own representations. Embedding layer handles categories. |

---

### The Pre-Processing Checklist — Before Running Any Algorithm

> ### ✅ Pre-Processing Checklist
>
> **Step 1 — Understand your data:**
> ```
> What type is each feature? (numeric / categorical / text / date?)
> How many missing values? Any clear outliers or data entry errors?
> ```
>
> **Step 2 — Handle missing values:**
> ```
> Impute: mean/median for numeric, mode for categorical
> Or use algorithms that handle NaN natively (Random Forest, XGBoost)
> ```
>
> **Step 3 — Encode categorical features:**
> ```
> Nominal  → One-Hot Encoding (or Target Encoding if > 10 categories)
> Ordinal  → Label/Ordinal Encoding (preserving the order)
> ```
>
> **Step 4 — Scale numerical features:**
> ```
> Distance-based models (kNN, SVM, Neural Net) → Z-score or Min-Max
> Tree-based models (Decision Tree, Random Forest, XGBoost) → No scaling needed
> ```
>
> **Step 5 — Handle class imbalance (if classification):**
> ```
> Check class distribution with value_counts()
> If imbalanced (< 20% minority): apply SMOTE, class_weight='balanced',
>   or threshold tuning
> ```
>
> **Step 6 — Split your data:**
> ```
> Train (70–80%) / Validation (10–15%) / Test (10–15%)
> Use stratified split for classification to preserve class distribution
>   in each split
> ```

---

## 6.10 The Master Algorithm Cheat Sheet

*Your go-to quick reference for any ML problem*

### Complete Algorithm Comparison Table

| Algorithm | Task | Type | Feature Types | Interpretability | Complexity | Speed | Data Size |
|---|---|---|---|---|---|---|---|
| **Linear Regression** | Regression | Supervised | Numeric | Very High | Low | Very Fast | Any |
| **Logistic Regression** | Classification | Supervised | Numeric + Encoded Cat. | Very High | Low | Very Fast | Any |
| **k-Nearest Neighbors** | Both | Supervised | Numeric (scaled) | Low | Low-Med | Slow predict | Small |
| **Naive Bayes** | Classification | Supervised | Text / Categorical | High | Low | Very Fast | Any |
| **Decision Tree** | Both | Supervised | Mixed (any) | Very High | Medium | Fast | Any |
| **Random Forest** | Both | Supervised | Mixed (any) | Medium | Medium-High | Moderate | Any |
| **SVM** | Both | Supervised | Numeric (scaled) | Low | Med-High | Slow on large | Small-Med |
| **XGBoost / LightGBM** | Both | Supervised | Mixed (any) | Medium | High | Fast-Mod | Any |
| **Neural Network** | Both | Supervised | Any (with encoding) | Low | Very High | Variable | Large |
| **K-Means** | Clustering | Unsupervised | Numeric (scaled) | High | Low | Very Fast | Large |
| **DBSCAN** | Clustering | Unsupervised | Numeric | Medium | Medium | Moderate | Medium |
| **Apriori / FP-Growth** | Association | Unsupervised | Transactional | Medium | Low | FP: Fast | Large |
| **PCA** | Dim. Reduction | Unsupervised | Numeric | Medium | Low | Fast | Any |

---

### Quick-Pick Guide — By Industry / Domain

| Industry | Common ML Problems | Recommended Starting Point |
|---|---|---|
| Banking & Finance | Credit scoring, fraud detection, customer lifetime value | XGBoost / LightGBM + Logistic Regression baseline |
| Healthcare | Disease prediction, patient risk stratification, readmission | Random Forest, Gradient Boosting, interpretable models |
| E-Commerce | Recommendation, cross-sell, demand forecasting, churn | Association Rules, XGBoost, Neural Collaborative Filtering |
| Telecom | Churn prediction, network anomaly, usage forecasting | Random Forest, LSTM for time series, Isolation Forest |
| Insurance | Claim prediction, fraud, pricing, risk segmentation | Gradient Boosting, Logistic Regression (regulatory interpretability) |
| HR & Recruitment | Attrition risk, candidate screening, salary benchmarking | Random Forest, Logistic Regression, Decision Tree |
| Manufacturing | Predictive maintenance, quality control, supply chain | LSTM for time series, Isolation Forest, SVM for defects |
| Media & Entertainment | Content recommendation, churn, viewership prediction | Association Rules, Collaborative Filtering, Deep Learning |
| Real Estate | Price prediction, demand forecasting, neighbourhood scoring | XGBoost, Linear Regression, K-Means for segments |
| Agriculture | Crop yield prediction, disease detection, pricing forecast | Random Forest, CNN for image-based disease detection |

---

## Chapter 6 Summary

| Topic | Key Takeaway |
|---|---|
| **6.1–6.5 Association Rules** | Association Rule Learning discovers co-occurrence patterns in transaction data without labels. Key metrics are Support (how common), Confidence (how reliable), Lift (genuine association above baseline), and Conviction (directional strength). Apriori uses the anti-monotonicity property to prune efficiently; FP-Growth uses a tree structure for 10–100x speedup. Real applications: Amazon cross-sell, HDFC product bundling, Apollo comorbidity discovery, Hotstar content sequencing, Big Bazaar store layout. |
| **6.6 Algorithm Selection Framework** | Always ask 5 questions first: What output do I need? Do I have labels? What features do I have? How much data? What constraints? The No Free Lunch theorem means no single algorithm wins everything — always start with a simple baseline and compare. |
| **6.7 Matching Problem to Algorithm** | Labels + Category output = Classification. Labels + Numeric output = Regression. No labels + grouping = Clustering. No labels + co-occurrence = Association Rules. Specific algorithm depends on data type, size, and interpretability requirements. |
| **6.8 Input Data Considerations** | Data volume guides algorithm complexity. Missing values, class imbalance, outliers, and noise each require specific handling strategies. High dimensionality favours linear models and tree ensembles over distance-based methods. |
| **6.9 Feature Types** | Numerical features (continuous/discrete): scale before distance-based models. Categorical features: One-Hot Encoding for nominal, Label/Ordinal Encoding for ordinal. Tree-based models handle mixed types natively. Distance-based and linear models require encoding and scaling. |
| **6.10 Master Cheat Sheet** | Quick reference covering all major algorithms by task type, feature compatibility, interpretability, complexity, speed, and scale. Domain-specific quick-pick table for 10 industries. |

---

> ### 🚀 You Have Completed the Core ML Curriculum!
>
> You have now covered the complete arc of Machine Learning:
>
> ```
> Ch.1 What is ML?  |  Ch.2 How Machines Learn  |  Ch.3 Types of ML
> Ch.4 Classification Algorithms  |  Ch.5 Clustering
> Ch.6 Association Rules + Choosing the Right Algorithm
> ```
>
> **Suggested Next Steps:**
> - **Practice:** Work through real datasets on [kaggle.com](https://www.kaggle.com)
> - **Code:** Implement each algorithm in Python using `scikit-learn`
> - **Projects:** Build end-to-end projects with data from your own domain
> - **Advanced:** Explore Neural Networks, Deep Learning, and NLP
