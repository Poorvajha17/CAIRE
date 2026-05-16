# CAIRE – Cart Abandonment Insights & Recovery Engine

## Overview

CAIRE is an intelligent e-commerce analytics and recovery system that tracks customer shopping behavior, identifies cart abandonment patterns, segments customers, and triggers personalized recovery strategies.

The project combines:

* Customer Behavior Analytics
* Feature Engineering
* Apriori Recommendation System
* Customer Segmentation
* Personalized Recovery Strategies
* Streamlit-based Shopping Simulation

---

# Key Features

* Real-time shopping session simulation

* Customer interaction tracking

* Apriori-based product recommendations

* Rule-based customer segmentation

* Personalized recovery actions

* Feature preprocessing and transformation

* Session data storage for ML prediction

---

# Customer Segments

The system classifies users into:

* High-Value Loyalists
* At-Risk Converters
* Engaged Researchers
* Price-Sensitive Shoppers
* Casual Browsers

---

# Apriori Recommendation System

Apriori Association Rule Mining is used to identify frequently purchased product combinations.

These recommendations are used during cart recovery to improve cross-selling and conversion chances.

---

# Recovery Strategy Engine

Based on customer segment and abandonment behavior, the system triggers personalized actions such as:

* Discounts
* Loyalty rewards
* Product recommendations
* Stock alerts
* Demo videos
* Re-engagement campaigns

---

# Project Workflow

1. User starts shopping session
2. User interactions are tracked
3. Features are engineered and preprocessed
4. Customer is segmented
5. Recovery strategies are triggered
6. Apriori generates complementary recommendations
7. Session data is stored for analytics and prediction

---

# Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Apriori Algorithm

---

# Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# Project Structure

```bash
CAIRE/
├── app.py
├── apriori.py
├── data/
├── analytics_data/
├── test_data/
└── README.md
```
