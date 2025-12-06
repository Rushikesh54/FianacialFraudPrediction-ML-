import streamlit as st
import pandas as pd
import os

# Use cache_resource to store the actual object in memory (Shared pointer)
# This avoids the overhead of copying/serializing the DataFrame for every page load.
@st.cache_resource
def load_shared_data():
    parquet_path = "data/data_full_engineered.parquet"
    if not os.path.exists(parquet_path):
        return pd.DataFrame(), {}
    
    # print("Loading shared data (Global)...")
    df = pd.read_parquet(parquet_path)
    
    # --- PRE-COMPUTE / CLEANING (Run once) ---
    
    # Helper to find cols
    def find_col(keywords, columns):
        for col in columns:
            if any(k in col.lower() for k in keywords):
                return col
        return None

    cols = df.columns
    fraud_col = find_col(['is_fraud', 'fraud', 'target', 'class'], cols)
    cat_col = find_col(['category', 'mcc', 'type'], cols)
    time_col = find_col(['year', 'date', 'time', 'trans_date'], cols)
    amt_col = find_col(['amount', 'price', 'value', 'total'], cols)
    chip_col = find_col(['chip', 'use_chip', 'method', 'type'], cols)
    merch_col = find_col(['merchant', 'store', 'mcc', 'name'], cols)
    
    # Clean Amount
    if amt_col:
        # Check if already numeric
        if not pd.api.types.is_numeric_dtype(df[amt_col]):
             df['__clean_amt'] = pd.to_numeric(
                df[amt_col].astype(str).str.replace(r'[$,]', '', regex=True), 
                errors='coerce'
            ).fillna(0)
        else:
             df['__clean_amt'] = df[amt_col].fillna(0)
    
    # Clean Time
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df['_daily_date'] = df[time_col].dt.date
        df['_hour'] = df[time_col].dt.hour
        
    col_map = {
        'fraud': fraud_col,
        'cat': cat_col,
        'time': time_col,
        'amt': amt_col,
        'chip': chip_col,
        'merch': merch_col
    }
    
    return df, col_map
