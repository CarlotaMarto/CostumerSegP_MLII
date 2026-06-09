# Customer Segmentation — Machine Learning II

**Machine Learning II Class · Data Science Degree · NOVA IMS**

**Authors:** Carlota Marto (20241729) · Francisca Teixeira (20241702) · Pedro Gouveia (20231657)

---

## Overview

This project applies unsupervised machine learning to segment a retail customer base into meaningful groups based on demographic characteristics, spending behaviour, and purchasing patterns. The goal is to uncover natural customer groupings — without relying on predefined labels — and translate those segments into actionable, targeted marketing strategies.

## Interactive Report & Demo

👉 **[Launch the Streamlit App](https://costumersegappcm-pcv5zjyel6ref4vshysqtr.streamlit.app/)**

The app is the primary deliverable of this project. It presents the full segmentation pipeline interactively, including cluster profiles, exploratory analysis, and targeted promotion suggestions per segment.

---

## Datasets

Two datasets were used:

**`customer_info`** — Customer demographics and lifetime spend behaviour, including fields such as age, gender, household composition, location, loyalty card number, lifetime spend across 10+ product categories, and shopping patterns (typical hour, promotion usage, distinct stores visited).

**`customer_basket`** — 100,000 randomly sampled shopping baskets, each containing a `customer_id`, `invoice_id`, and a list of purchased products. Used for association rule mining to inform targeted promotions.

---

## Project Structure

```
├── preprocessing/
│   ├── 00_data_analysis.ipynb          # Initial data exploration
│   ├── 01_eda_preprocessing.ipynb      # Feature engineering & cleaning
│   ├── 02_eda_geographic.ipynb         # Geographic analysis
│   └── utils_eda.py                    # EDA utility functions
│
├── clustering/
│   ├── 03_clustering.ipynb             # Clustering models & selection
│   ├── 04_cluster_characterization.ipynb  # Segment profiling
│   ├── 05_association_rules.ipynb      # Market basket analysis
│   ├── utils_clustering.py             # Clustering utilities
│   ├── utils_cluster_characterization.py
│   └── utils_association_rules.py
│
├── web_app/                            # Streamlit application
│   ├── app.py
│   └── ...
│
└── datasets/                           # Input data files
```

---

## Methodology

1. **Exploratory Data Analysis & Pre-Processing** — handling missing values, feature engineering (e.g. age from birthdate, total lifetime spend, promotion sensitivity), and geographic exploration.
2. **Customer Segmentation & Clustering** — evaluation of multiple unsupervised algorithms; final segments selected based on interpretability and clustering quality metrics.
3. **Cluster Characterisation** — profiling each segment across demographic and behavioural dimensions.
4. **Targeted Promotion** — association rule mining on `customer_basket` to derive segment-specific campaign recommendations.

---

## How to Run Locally

```bash
# Install dependencies
pip install -r web_app/requirements.txt

# Launch the app
streamlit run web_app/app.py
```

Or access the hosted version directly: [costumersegappcm-pcv5zjyel6ref4vshysqtr.streamlit.app](https://costumersegappcm-pcv5zjyel6ref4vshysqtr.streamlit.app/)
