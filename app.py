import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Financial Fraud Dashboard",
    page_icon="💸",
    layout="wide"
)

st.title("💸 Financial Fraud Analysis Dashboard")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("ℹ️ About this Dashboard")
    st.markdown("""
    **Goal:** Detect patterns in fraudulent transactions.
    
    **New Features:**
    - 💰 **Total Potential Loss**: Track financial impact.
    - 🏪 **Merchant Watchlist**: See where fraud happens.
    - 🕵️ **Investigation Desk**: Drill down into alerts.
    
    **Dataset Columns:**
    - `Amount`: Transaction value in USD.
    - `Category`: Type of expense.
    - `Time`: Transaction timestamp.
    - `Chip`: Card entry method.
    - `Fraud`: 1 = Fraud, 0 = Normal.
    """)
    st.info("💡 Tip: Double-click on chart legends to isolate trace lines.")

import utils

# ... (Config)

# 2. Data Loading (Shared)
# We rely on utils.py to handle the heavy lifting.
with st.spinner("Loading data... (Shared Cache)"):
    df, col_map = utils.load_shared_data()

# Check if data loaded successfully
if not df.empty:
    # Unpack columns
    fraud_col = col_map['fraud']
    cat_col = col_map['cat']
    time_col = col_map['time']
    amt_col = col_map['amt']
    chip_col = col_map['chip']
    merch_col = col_map['merch']

    # --- SECTION A: METRICS ---
    st.markdown("### 📊 Global Metrics")
    st.markdown("High-level financial overview.")
    
    if fraud_col:
        total_tx = len(df)
        fraud_tx = df[df[fraud_col] == 1].shape[0]
        fraud_rate = (fraud_tx / total_tx) * 100
        
        # New Financial KPI
        total_loss = 0.0
        if '__clean_amt' in df.columns:
            total_loss = df[df[fraud_col] == 1]['__clean_amt'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total_tx:,}", help="Total raw transactions in the dataset.")
        c2.metric("Fraud Cases", f"{fraud_tx:,}", help="Number of verified fraud cases.")
        c3.metric("Fraud Rate", f"{fraud_rate:.3f}%", delta_color="inverse", help="Proportion of fraud.")
        c4.metric("Potential Loss", f"${total_loss:,.0f}", delta_color="inverse", help="Sum of amount for all fraud transactions.")
    else:
        st.warning("⚠️ Could not detect a 'fraud' column. Metrics unavailable.")

    st.divider()

    # --- SECTION B: BASIC VISUALIZATIONS ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Fraud by Category")
        if fraud_col and cat_col:
            # Groupby is fast enough on clean data, or could be cached further if needed
            data_cat = df.groupby(cat_col)[fraud_col].sum().reset_index()
            data_cat = data_cat.sort_values(by=fraud_col, ascending=False).head(10)
            
            fig_cat = px.bar(data_cat, x=cat_col, y=fraud_col, 
                             color=fraud_col, title="Top 10 High-Risk Categories",
                             color_continuous_scale="Reds",
                             labels={cat_col: "Transaction Category", fraud_col: "Fraud Count"})
            st.plotly_chart(fig_cat)
        else:
            st.info("Category chart unavailable.")

    with col_chart2:
        st.subheader("Fraud Trends")
        if fraud_col and time_col:
            try:
                # Time cleaning already done!
                data_time = df.groupby('_daily_date')[fraud_col].sum().reset_index()
                
                fig_time = px.line(data_time, x='_daily_date', y=fraud_col, 
                                   title="Fraud Volume Over Time (Daily)", markers=True,
                                   labels={'_daily_date': "Date", fraud_col: "Fraud Transactions"})
                st.plotly_chart(fig_time)
            except Exception as e:
                st.warning(f"Could not parse time column: {e}")
        else:
            st.info("Time series chart unavailable.")
    
    st.divider()

    # --- SECTION C: MERCHANT WATCHLIST (NEW) ---
    st.markdown("### 🏪 Merchant Watchlist")
    if merch_col and fraud_col:
        c_merch1, c_merch2 = st.columns([2, 1])
        with c_merch1:
            st.write(f"Top 10 Risky Merchants (by Fraud Volume)")
            # Top 10 Merchants by fraud count
            top_merchants = df[df[fraud_col]==1][merch_col].value_counts().head(10).reset_index()
            top_merchants.columns = [merch_col, 'Fraud Count']
            
            fig_merch = px.bar(top_merchants, x='Fraud Count', y=merch_col, orientation='h',
                               title="Top 10 High-Risk Merchants", color='Fraud Count', color_continuous_scale='Reds',
                               labels={merch_col: "Merchant Name", 'Fraud Count': "Number of Fraud Cases"})
            fig_merch.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_merch)
        
        with c_merch2:
            st.write("Risk Table")
            st.dataframe(top_merchants, hide_index=True)
    else:
        st.info("Merchant information not found. (Check column names)")

    st.divider()

    # --- SECTION D: DEEP DIVE (Amount & Chip) ---
    col_deep1, col_deep2 = st.columns(2)

    with col_deep1:
        st.subheader("💰 Transaction Amounts")
        if amt_col and fraud_col:
            st.write("Distribution of Transaction Amounts (Fraud vs Normal)")
            try:
                # Use clean amt calculated at top
                fraud_amts = df[df[fraud_col]==1]['__clean_amt']
                normal_amts = df[df[fraud_col]==0]['__clean_amt']
                
                avg_fraud = fraud_amts.mean() if not fraud_amts.empty else 0.0
                avg_normal = normal_amts.mean() if not normal_amts.empty else 0.0
                
                c1, c2 = st.columns(2)
                c1.metric("Avg Fraud Amount", f"${avg_fraud:,.2f}")
                c2.metric("Avg Normal Amount", f"${avg_normal:,.2f}")

                sample_size = min(10000, len(df))
                sample_df = df.sample(sample_size, random_state=42)
                fig_box = px.box(sample_df, x=fraud_col, y='__clean_amt', 
                                 color=fraud_col, title="Amount Distribution (Sampled)",
                                 labels={'__clean_amt': 'Amount ($)', fraud_col: 'Is Fraud?'})
                st.plotly_chart(fig_box)
            except Exception as e:
                st.warning(f"Error calculating amounts: {e}")
        else:
            st.info("Amount information not found.")

    with col_deep2:
        st.subheader("💳 Chip / Entry Method Analysis")
        if chip_col and fraud_col:
            chip_data = df.groupby(chip_col)[fraud_col].mean().reset_index()
            fig_chip = px.bar(chip_data, x=chip_col, y=fraud_col, 
                              title="Fraud Rate by Entry Method",
                              color=fraud_col, color_continuous_scale="Reds",
                              labels={chip_col: "Entry Method (Chip/Swipe)", fraud_col: "Fraud Probability (0-1)"})
            st.plotly_chart(fig_chip)
            
    # --- SECTION E: HOURLY ANALYSIS ---
    st.divider()
    st.subheader("🕐 Fraud by Hour of Day")
    if '_hour' in df.columns:
        hourly_fraud = df[df[fraud_col]==1].groupby('_hour').size().reset_index(name='count')
        
        fig_hour = px.bar(hourly_fraud, x='_hour', y='count', 
                          title="Peak Fraud Hours (24h format)",
                          labels={'_hour': 'Hour of Day', 'count': 'Fraud Transactions'})
        fig_hour.update_xaxes(range=[-0.5, 23.5])
        st.plotly_chart(fig_hour)
    else:
        st.warning("Could not extract hour from time column.")

    # --- SECTION F: INVESTIGATION DESK (NEW) ---
    st.divider()
    st.subheader("🕵️ Investigation Desk")
    st.markdown("Filter and inspect flagged transactions.")
    
    if fraud_col:
        # Show only fraud transactions
        fraud_df = df[df[fraud_col] == 1].copy()
        
        # Add friendly date string for display
        if time_col and time_col in fraud_df.columns:
             fraud_df['Date_Str'] = fraud_df[time_col].dt.strftime('%Y-%m-%d %H:%M')
        
        # Columns to display - ensure unique
        raw_disp_cols = [time_col, amt_col, cat_col, merch_col, chip_col]
        # remove None and duplicates while preserving order
        disp_cols = list(dict.fromkeys([c for c in raw_disp_cols if c]))
        
        st.dataframe(fraud_df[disp_cols].head(200), use_container_width=True)
    
    # --- RAW DATA EXPANDER ---
    with st.expander("📂 View Raw Data Sample"):
        st.dataframe(df.head(100))

else:
    st.error("❌ Data failed to load. Please check 'data/' directory.")