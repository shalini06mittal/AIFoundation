# Machine Learning Types
## Real-World Case Studies & Practice Exercises

---

## Table of Contents

- [Introduction](#introduction)
- [1. Supervised Learning](#1-supervised-learning)
  - [Case Study 1.1: Netflix Content Recommendation Quality Prediction](#case-study-11-netflix-content-recommendation-quality-prediction)
  - [Case Study 1.2: Zillow Home Price Prediction](#case-study-12-zillow-home-price-prediction)
- [2. Unsupervised Learning](#2-unsupervised-learning)
  - [Case Study 2.1: Spotify User Segmentation](#case-study-21-spotify-user-segmentation)
  - [Case Study 2.2: Credit Card Fraud Detection](#case-study-22-credit-card-fraud-detection-anomaly-detection)
- [3. Reinforcement Learning](#3-reinforcement-learning)
  - [Case Study 3.1: Google Data Center Cooling Optimization](#case-study-31-google-data-center-cooling-optimization)
  - [Case Study 3.2: Amazon Warehouse Robot Navigation](#case-study-32-amazon-warehouse-robot-navigation)
- [4. Semi-Supervised Learning](#4-semi-supervised-learning)
  - [Case Study 4.1: Medical Image Analysis for Cancer Detection](#case-study-41-medical-image-analysis-for-cancer-detection)
- [Practice Exercises](#practice-exercises)
- [Answer Key](#answer-key)
- [Quick Decision Guide](#quick-decision-guide)
- [Key Takeaways](#key-takeaways)

---

## Introduction

This document provides comprehensive real-world examples and case studies across all major types of machine learning. Each section includes detailed scenarios, data characteristics, solutions, and hands-on exercises for participants to practice categorizing ML problems.

---

## 1. Supervised Learning

**Definition:** Learning from labeled data where each input has a known output. The algorithm learns to map inputs to outputs.

### Case Study 1.1: Netflix Content Recommendation Quality Prediction

#### Scenario

Netflix wants to predict whether a user will rate a movie 4-5 stars (positive) or 1-3 stars (negative) based on their viewing history and preferences.

#### Data Available

- User demographics: age, location, subscription type
- Viewing history: genres watched, completion rate, binge-watching patterns
- Movie features: genre, director, cast, release year, runtime
- Historical ratings: labeled data showing actual ratings users gave
- Time of day viewing patterns

#### Why Supervised Learning?

- Clear labeled data: historical ratings provide ground truth
- Prediction goal: want to predict a specific outcome (rating category)
- Classification task: categorizing into positive or negative rating

#### Solution Approach

**Algorithm:** Random Forest or Gradient Boosting (XGBoost)

**Why this choice:** Handles mixed data types, captures non-linear patterns, provides feature importance

**Success metric:** Accuracy, Precision, Recall, F1-score

#### Sample Data Structure

| User Age | Genre Pref | Completion | Movie Genre | Director | Rating |
| --- | --- | --- | --- | --- | --- |
| 28 | Sci-Fi | 95% | Sci-Fi | Nolan | **5 stars** |
| 45 | Drama | 40% | Action | Bay | **2 stars** |

---

### Case Study 1.2: Zillow Home Price Prediction

#### Scenario

Zillow wants to predict the exact selling price of homes based on property characteristics and market conditions.

#### Data Available

- Property features: square footage, bedrooms, bathrooms, lot size
- Location: zip code, school district ratings, crime rates
- Property age and condition: year built, recent renovations
- Market data: recent comparable sales, days on market
- Historical sale prices: labeled data showing actual selling prices

#### Why Supervised Learning?

**Type:** Regression (predicting continuous numerical value)

- Historical prices provide labeled training data
- Clear input-output relationship to learn
- Goal is to predict specific numerical values

#### Solution Approach

**Algorithm:** XGBoost Regression or Neural Networks

**Success metric:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

---

## 2. Unsupervised Learning

**Definition:** Finding hidden patterns and structures in unlabeled data without predefined categories or outcomes.

### Case Study 2.1: Spotify User Segmentation

#### Scenario

Spotify wants to discover natural groupings of users based on their listening behavior to create personalized marketing campaigns, but they do not have predefined user categories.

#### Data Available

- Listening patterns: genres, artists, song length preferences
- Time of day listening habits
- Playlist creation behavior
- Skip rates and replay rates
- Social sharing behavior
- **NO labels for user types**

#### Why Unsupervised Learning?

- No predefined categories of users
- Want to discover natural groupings in the data
- No labeled training data available

#### Solution Approach

**Algorithm:** K-Means Clustering or DBSCAN

**Discovered segments might include:**

- Workout Warriors: Listen to high-energy music during morning hours
- Focus Seekers: Prefer instrumental and lo-fi during work hours
- Party Playlist Curators: Create and share social playlists
- Deep Listeners: Complete albums, minimal skipping

**Success metric:** Silhouette score, business value of identified segments

---

### Case Study 2.2: Credit Card Fraud Detection (Anomaly Detection)

#### Scenario

American Express wants to detect unusual transaction patterns that might indicate fraud. Most transactions are normal, and fraud patterns constantly evolve, making it hard to maintain labeled fraud examples.

#### Data Available

- Transaction amount, location, merchant category
- Time and day of transaction
- Historical spending patterns per user
- Device and IP information
- Mostly unlabeled normal transactions

#### Why Unsupervised Learning?

- Fraud is rare, making labeled data scarce and imbalanced
- Fraud patterns change constantly, labels become outdated
- Need to detect new, unknown fraud patterns

#### Solution Approach

**Algorithm:** Isolation Forest, Autoencoders, or One-Class SVM

**Success metric:** Precision at catching fraud, false positive rate

---

## 3. Reinforcement Learning

**Definition:** Learning optimal behavior through trial and error, receiving rewards or penalties for actions taken in an environment.

### Case Study 3.1: Google Data Center Cooling Optimization

#### Scenario

Google wants to optimize cooling systems in data centers to minimize energy consumption while maintaining safe operating temperatures. The system must learn to adjust cooling based on changing server loads, external temperatures, and complex interactions.

#### Environment & Actions

**State:** Current temperatures, server loads, outside weather, cooling equipment status

**Actions:** Adjust fan speeds, change cooling water flow, modify air distribution

**Rewards:** Negative reward for energy used, large penalty for overheating

#### Why Reinforcement Learning?

- No dataset of optimal cooling decisions exists
- System must learn through interaction with the environment
- Actions have delayed consequences
- Need to balance competing objectives (efficiency vs safety)

#### Solution Approach

**Algorithm:** Deep Q-Network (DQN) or Policy Gradient methods

**Result:** Google achieved 40% reduction in cooling energy costs

---

### Case Study 3.2: Amazon Warehouse Robot Navigation

#### Scenario

Amazon needs robots to navigate warehouse floors efficiently, avoiding obstacles, other robots, and optimizing paths to pick up and deliver packages.

#### Environment & Actions

**State:** Robot position, destination, obstacles, other robot locations, battery level

**Actions:** Move forward, turn left/right, pick up package, charge battery

**Rewards:** Positive reward for successful delivery, penalty for collisions, small negative reward for each time step (to encourage efficiency)

#### Why Reinforcement Learning?

- Environment is dynamic and unpredictable
- Must learn through trial and error
- Sequential decision-making problem
- No labeled dataset of optimal paths

#### Solution Approach

**Algorithm:** Multi-Agent Reinforcement Learning

---

## 4. Semi-Supervised Learning

**Definition:** Using a small amount of labeled data combined with a large amount of unlabeled data for training.

### Case Study 4.1: Medical Image Analysis for Cancer Detection

#### Scenario

A hospital has 50,000 chest X-rays, but only 500 have been labeled by expert radiologists as showing cancer or being normal. Labeling requires expensive expert time.

#### Data Available

- 500 labeled X-rays (cancer/normal) - expensive expert labels
- 49,500 unlabeled X-rays - available but not labeled

#### Why Semi-Supervised Learning?

- Labeling is expensive and requires rare expert knowledge
- Large amount of unlabeled data available
- Purely supervised learning on 500 samples would not perform well
- Unlabeled data can help learn better feature representations

#### Solution Approach

**Algorithm:** Self-training, Co-training, or Pseudo-labeling with deep neural networks

---

## Practice Exercises

For each scenario below, identify which type of machine learning is most appropriate and explain why.

### Exercise 1: YouTube Video Recommendation

**Scenario:** YouTube wants to recommend the next video a user should watch. They have data on billions of user-video interactions including which videos users watched, liked, commented on, and how long they watched each video.

**Your task:** Identify the ML type and justify your answer.

---

### Exercise 2: Retail Customer Shopping Patterns

**Scenario:** A retail chain has transaction data from millions of customers showing what products they buy together. They want to discover natural groupings of products that are frequently purchased together to optimize store layout.

**Data:** Transaction history, product categories, purchase amounts - NO predefined product groupings

**Your task:** Identify the ML type and justify your answer.

---

### Exercise 3: Stock Trading Bot

**Scenario:** A hedge fund wants to build a trading algorithm that learns optimal buy/sell decisions. The algorithm should learn from market feedback (profit/loss) and adapt to changing market conditions.

**Environment:** Stock prices, market indicators, portfolio state, feedback is profit or loss from trades

**Your task:** Identify the ML type and justify your answer.

---

### Exercise 4: Email Spam Classifier

**Scenario:** Gmail wants to classify incoming emails as spam or not spam. They have millions of historical emails that users have marked as spam or not spam.

**Data:** Email content, sender information, historical user markings (spam/not spam labels)

**Your task:** Identify the ML type and justify your answer.

---

### Exercise 5: Manufacturing Defect Detection

**Scenario:** A manufacturing plant produces thousands of products daily. Defects are very rare (0.1% of products). They have images of products but only a few hundred labeled defects because defects are so rare.

**Data:** 100,000 product images, 200 labeled as defective, 99,800 unlabeled (mostly normal)

**Your task:** Identify the ML type and justify your answer.

---

### Exercise 6: Smart Thermostat Learning

**Scenario:** A smart thermostat needs to learn a homeowner's temperature preferences throughout the day and automatically adjust heating/cooling to maximize comfort while minimizing energy use. It learns from user adjustments.

**Environment:** Current temperature, time of day, outside weather, user manual adjustments provide feedback

**Your task:** Identify the ML type and justify your answer.

---

### Exercise 7: News Article Topic Discovery

**Scenario:** A news aggregator has collected 100,000 news articles and wants to automatically discover what topics they cover to organize them, but they don't have predefined topic categories.

**Data:** Article text content, NO topic labels

**Your task:** Identify the ML type and justify your answer.

---

### Exercise 8: Language Translation

**Scenario:** Google Translate wants to build a system that translates English sentences to French. They have millions of English-French sentence pairs from books and websites.

**Data:** Paired sentences: English sentence → corresponding French translation

**Your task:** Identify the ML type and justify your answer.

---

## Answer Key

### Exercise 1: Supervised Learning (Classification/Ranking)

**Why:** User interactions provide implicit labels (watched = relevant, skipped = not relevant). This is a prediction task with historical labeled data.

---

### Exercise 2: Unsupervised Learning (Clustering/Association)

**Why:** No predefined product groupings. Want to discover natural patterns in purchasing behavior. This is market basket analysis.

---

### Exercise 3: Reinforcement Learning

**Why:** Sequential decision-making with delayed feedback (profit/loss). The bot learns optimal strategy through trial and error. Actions affect future states.

---

### Exercise 4: Supervised Learning (Classification)

**Why:** Clear labeled data from user markings. Binary classification task (spam vs not spam). Predicting category based on features.

---

### Exercise 5: Unsupervised (Anomaly Detection) OR Semi-Supervised

**Why:** Could use unsupervised anomaly detection (defects are anomalies) OR semi-supervised learning to leverage both labeled defects and unlabeled normal products. The extreme rarity makes this a special case.

---

### Exercise 6: Reinforcement Learning

**Why:** Learns from user feedback (manual adjustments). Sequential decisions with rewards (comfort) and penalties (energy cost, discomfort). Trial and error learning.

---

### Exercise 7: Unsupervised Learning (Topic Modeling/Clustering)

**Why:** No topic labels. Want to discover hidden topic structure. Classic use case for techniques like LDA (Latent Dirichlet Allocation) or clustering.

---

### Exercise 8: Supervised Learning (Sequence-to-Sequence)

**Why:** Paired training data provides clear input-output examples. This is supervised learning even though the output is a sequence. Each English sentence has its known French translation.

---

## Quick Decision Guide

Use this table to quickly identify the ML type for any new problem:

| Question | Answer | ML Type |
| --- | --- | --- |
| Do you have labels/targets? | YES | **Supervised Learning** |
| Do you have labels/targets? | NO | Continue to next question... |
| Is it sequential decision-making with feedback? | YES | **Reinforcement Learning** |
| Want to find patterns/groups? | YES | **Unsupervised Learning** |
| Have few labels + many unlabeled? | YES | **Semi-Supervised Learning** |

---

## Key Takeaways

1. **Supervised Learning:** Use when you have labeled data and want to predict outcomes

2. **Unsupervised Learning:** Use when you want to discover patterns without labels

3. **Reinforcement Learning:** Use for sequential decision-making with feedback

4. **Semi-Supervised Learning:** Use when labeling is expensive but you have lots of unlabeled data

---

**The key is always: What data do you have, and what do you want to achieve?**
