# 💸 Financial Fraud Detection Dashboard

An interactive dashboard for analyzing financial transaction fraud, visualizing risk patterns, and evaluating machine learning model performance.

## 📋 Features
- **Global Metrics**: Real-time KPIs for fraudulent transactions and potential financial loss.
- **Visualizations**:
  - Fraud trends over time.
  - High-risk merchant and category analysis.
  - Entry method (Chip vs. Swipe) risk comparison.
- **Investigation Desk**: Filterable view of specific fraudulent transactions.
- **Model Performance (Beta)**: Evaluate XGBoost/Random Forest models with ROC curves and Confusion Matrices.

## 🛠️ Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

## 🚀 Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Data Setup
Because the dataset is large, it is **not included** in this repository.

1.  Download the dataset from Kaggle:
    [Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)
2.  Create a folder named `data` in the project root.
3.  Place the `data_full_engineered.parquet` (or optimized parquet file) inside the `data/` folder.
    - *Expected Path:* `data/data_full_engineered.parquet`

## 🤖 Model Setup
The trained models are also excluded from the repo to save space. You can regenerate them:

1.  Run the training scripts (found in users workspace or notebooks):
    ```bash
    python evaluate_models.py
    ```
    *This will generate `.pkl` files in the `models/` directory.*

## 🏃‍♂️ Running the Dashboard

1.  **Start the Streamlit App**
    ```bash
    streamlit run app.py
    ```
2.  access the dashboard in your browser at `http://localhost:8501`.

## 📁 Project Structure
- `app.py`: Main dashboard entry point.
- `utils.py`: Shared utilities for efficient data loading.
- `pages/`: Contains the "Model Performance" sub-page.
- `models/`: Directory for trained model binaries (local only).
- `data/`: Directory for dataset files (local only).
