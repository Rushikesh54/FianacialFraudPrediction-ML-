# 💸 Production-Grade Financial Fraud Detection Dashboard

An end-to-end Machine Learning web application designed to ingest, process, and analyze financial transaction fraud. This platform serves live predictive insights, visualizes global financial risk patterns, and evaluates model performance bottlenecks for production-level decision-making.

Built using **Python, Streamlit, XGBoost, and Parquet**.

---

## 🎯 Key Capabilities
* **Global Executive Metrics:** High-fidelity KPIs tracking real-time fraudulent transaction frequencies, total chargeback risks, and potential financial losses mitigated.
* **Granular Risk Visualization:** Behavioral trend analysis tracking transaction vectors over time, high-risk merchant categories, and device entry methods (*Chip vs. Swipe*).
* **Investigation Desk:** An interactive, highly filterable transactional ledger allowing risk analysts to drill down into specific anomalous events.
* **Model Validation Studio:** A dedicated diagnostics suite providing real-time model evaluation tools, rendering live Confusion Matrices and ROC curves to monitor precision-recall trade-offs.

---

## 🏗️ Technical Architecture & Optimization
Handling **13+ million rows of data** requires production-level performance awareness. The following engineering principles were implemented:
* **Storage Optimization:** Leveraged compressed columnar `.parquet` storage instead of standard `.csv` files, reducing memory usage and optimizing chunked data ingestion.
* **Memory Management:** Implemented caching mechanisms within Streamlit (`st.cache_data`) via a dedicated `utils.py` script to ensure lightning-fast dashboard interactions and prevent UI lag during complex filter updates.
* **Class Imbalance Engineering:** Addressed extreme fraud-to-legitimate class ratios utilizing algorithmic adjustments within Scikit-learn and XGBoost to prioritize high model recall while maintaining low false-positive overhead.

---

## 📂 Data & Model Setup

Because the dataset encompasses **over 13 million records**, large binaries are excluded from version control to maintain repo cleanliness.

### 1. Data Placement
1. Download the transaction dataset from Kaggle: [Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)
2. Create a `/data` folder in the project root directory.
3. Place your optimized `data_full_engineered.parquet` file inside it:
   ```text
   data/data_full_engineered.parquet
