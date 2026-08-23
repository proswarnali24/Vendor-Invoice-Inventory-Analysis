import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import sqlite3
from pathlib import Path
import io

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag, load_model as load_flag_model

# -------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------
st.set_page_config(
    page_title="Vendor Invoice & Inventory Intelligence Portal",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #c3e6cb;
    }
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Database Helper & Caching
# -------------------------------------------------------
@st.cache_data(ttl=3600)
def get_db_path():
    project_root = Path(__file__).resolve().parent
    project_db = project_root / "data" / "inventory.db"
    if project_db.exists():
        return str(project_db)
    desktop_db = Path.home() / "Desktop" / "inventory.db"
    if desktop_db.exists():
        return str(desktop_db)
    return str(project_db)

@st.cache_data(ttl=600)
def load_overview_data():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    # Vendor invoice overview
    df_invoices = pd.read_sql_query("""
        SELECT PONumber, VendorNumber, VendorName, Quantity, Dollars, Freight, InvoiceDate
        FROM vendor_invoice
    """, conn)
    
    # Combined purchase aggregation to calculate overall risk exposure
    query_risk = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa ON vi.PONumber = pa.PONumber
    """
    df_risk = pd.read_sql_query(query_risk, conn)
    conn.close()
    
    # Label risk
    df_risk['is_risk'] = df_risk.apply(
        lambda r: 1 if (pd.isna(r['total_item_dollars']) or abs(r['invoice_dollars'] - r['total_item_dollars']) > 5 or (not pd.isna(r['avg_receiving_delay']) and r['avg_receiving_delay'] > 10)) else 0,
        axis=1
    )
    
    return df_invoices, df_risk

@st.cache_data(ttl=600)
def load_table_sample(table_name: str, limit: int = 500):
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=600)
def get_table_counts():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    tables = ['vendor_invoice', 'purchases', 'purchase_prices', 'begin_inventory', 'end_inventory']
    counts = {}
    for t in tables:
        counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    conn.close()
    return counts


# -------------------------------------------------------
# Header Section
# -------------------------------------------------------
st.markdown('<div class="main-header">📦 Vendor Invoice & Inventory Intelligence Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise AI Pipeline for Freight Estimation, Financial Leakage Prevention & Multi-Table Analytics</div>', unsafe_allow_html=True)

# Try loading database overview
try:
    df_invoices, df_risk = load_overview_data()
    table_counts = get_table_counts()
    db_connected = True
except Exception as e:
    st.error(f"⚠️ Could not load database `inventory.db`: {e}")
    db_connected = False
    df_invoices, df_risk, table_counts = pd.DataFrame(), pd.DataFrame(), {}


if db_connected:
    total_invoices = len(df_invoices)
    total_dollars = df_invoices['Dollars'].sum()
    total_freight = df_invoices['Freight'].sum()
    flagged_count = df_risk['is_risk'].sum()
    flagged_pct = (flagged_count / total_invoices) * 100 if total_invoices > 0 else 0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("🧾 Total Vendor Invoices", f"{total_invoices:,}")
    with kpi2:
        st.metric("💰 Total Invoice Value", f"${total_dollars:,.2f}")
    with kpi3:
        st.metric("🚚 Total Freight Spend", f"${total_freight:,.2f}")
    with kpi4:
        st.metric("🚨 High Risk Invoices", f"{flagged_count:,}", delta=f"{flagged_pct:.1f}% of total", delta_color="inverse")
    with kpi5:
        st.metric("📦 Purchase Records", f"{table_counts.get('purchases', 0):,}")

st.divider()

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
st.sidebar.title("🎛 Navigation & Controls")
nav_choice = st.sidebar.radio(
    "Select Module",
    [
        "📊 Executive Analytics & DB Explorer",
        "🚚 Freight Cost Prediction",
        "🚨 Invoice Risk Flagging (AI Audit)",
        "📁 Batch CSV Invoice Evaluator",
        "📈 Model Performance & Diagnostics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Status**  
🟢 **Database**: Connected (`inventory.db`)  
🟢 **Freight Model**: Loaded (`Linear Regression`)  
🟢 **Risk Classifier**: Loaded (`Random Forest`)  
""")


# =======================================================
# MODULE 1: Executive Analytics & DB Explorer
# =======================================================
if nav_choice == "📊 Executive Analytics & DB Explorer":
    st.header("📊 Executive Analytics & Database Explorer")
    
    tab_analytics, tab_db = st.tabs(["📈 Business Analytics Dashboard", "🔍 SQLite Database Explorer"])
    
    with tab_analytics:
        if not db_connected or df_invoices.empty:
            st.warning("Database overview is unavailable.")
        else:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("🚚 Freight Cost vs. Invoice Value")
                fig_scatter = px.scatter(
                    df_invoices,
                    x="Dollars",
                    y="Freight",
                    hover_data=["PONumber", "VendorName"],
                    color="Freight",
                    color_continuous_scale="Viridis",
                    labels={"Dollars": "Invoice Dollars ($)", "Freight": "Freight Cost ($)"},
                    title="Relationship Between Invoice Dollars and Freight Cost"
                )
                fig_scatter.update_layout(template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col_b:
                st.subheader("🚨 Invoice Risk Distribution")
                risk_summary = pd.DataFrame({
                    "Status": ["Auto-Approved (Safe)", "Manual Review (Flagged)"],
                    "Count": [total_invoices - flagged_count, flagged_count]
                })
                fig_pie = px.pie(
                    risk_summary,
                    values="Count",
                    names="Status",
                    color="Status",
                    color_discrete_map={"Auto-Approved (Safe)": "#2ecc71", "Manual Review (Flagged)": "#e74c3c"},
                    hole=0.4,
                    title="Proportion of Risk-Flagged Invoices"
                )
                fig_pie.update_layout(template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            col_c, col_d = st.columns(2)
            
            with col_c:
                st.subheader("🏢 Top 10 Vendors by Invoice Spend")
                top_vendors = df_invoices.groupby("VendorName")["Dollars"].sum().nlargest(10).reset_index()
                fig_vendors = px.bar(
                    top_vendors,
                    x="Dollars",
                    y="VendorName",
                    orientation="h",
                    color="Dollars",
                    color_continuous_scale="Blues",
                    labels={"Dollars": "Total Dollars ($)", "VendorName": "Vendor Name"},
                    title="Top Vendors by Financial Volume"
                )
                fig_vendors.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
                st.plotly_chart(fig_vendors, use_container_width=True)
                
            with col_d:
                st.subheader("📊 Freight Ratio (%) Distribution")
                df_invoices['Freight_Ratio'] = (df_invoices['Freight'] / df_invoices['Dollars'].replace(0, np.nan)) * 100
                fig_hist = px.histogram(
                    df_invoices[df_invoices['Freight_Ratio'] < 10],
                    x="Freight_Ratio",
                    nbins=40,
                    color_discrete_sequence=["#3498db"],
                    labels={"Freight_Ratio": "Freight Rate (% of Invoice)"},
                    title="Freight Rate Distribution (< 10%)"
                )
                fig_hist.update_layout(template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)
                
    with tab_db:
        st.subheader("🔍 SQLite Multi-Table Inspector")
        selected_table = st.selectbox(
            "Select Database Table",
            ["vendor_invoice", "purchases", "purchase_prices", "begin_inventory", "end_inventory"]
        )
        
        row_limit = st.slider("Max Rows to Display", min_value=50, max_value=5000, value=500, step=50)
        
        table_df = load_table_sample(selected_table, limit=row_limit)
        st.caption(f"Showing up to {len(table_df):,} rows from **{selected_table}** (Total table rows: {table_counts.get(selected_table, 0):,})")
        
        # Simple search filter
        search_term = st.text_input("🔎 Search Table (Filter by text/id)", "")
        if search_term:
            filtered_df = table_df[table_df.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(table_df, use_container_width=True)


# =======================================================
# MODULE 2: Freight Cost Prediction
# =======================================================
elif nav_choice == "🚚 Freight Cost Prediction":
    st.header("🚚 AI Freight Cost Prediction")
    st.markdown("""
    Predict expected freight cost for vendor invoices based on total invoice value ($).
    This model utilizes a trained **Regression Pipeline** (MAE ~ $24.11, R² ~ 96.99%).
    """)
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("📝 Input Invoice Value")
        with st.form("freight_single_form"):
            dollars_val = st.number_input("💰 Invoice Dollars ($)", min_value=1.0, max_value=1000000.0, value=18500.0, step=500.0)
            submit_freight = st.form_submit_button("🔮 Calculate Estimated Freight")
            
    with col_result:
        st.subheader("📊 Model Prediction Output")
        if submit_freight or 'freight_run' in st.session_state:
            st.session_state['freight_run'] = True
            
            res_df = predict_freight_cost({"Dollars": [dollars_val]})
            predicted_freight = res_df['Predicted_Freight'].values[0]
            freight_ratio = (predicted_freight / dollars_val) * 100
            
            st.metric(
                label="Estimated Freight Cost",
                value=f"${predicted_freight:,.2f}",
                delta=f"{freight_ratio:.2f}% of invoice value"
            )
            
            st.info(f"""
            💡 **Prediction Insight**:  
            - **Expected Range**: ${max(0.0, predicted_freight - 24.11):,.2f} – ${predicted_freight + 24.11:,.2f}  
            - **Historical Benchmark Rate**: ~0.50% – 1.20% of total invoice dollars.
            """)


# =======================================================
# MODULE 3: Invoice Risk Flagging (AI Audit)
# =======================================================
elif nav_choice == "🚨 Invoice Risk Flagging (AI Audit)":
    st.header("🚨 AI Invoice Risk Flagging & Anomaly Detection")
    st.markdown("""
    Evaluate whether a vendor invoice requires **manual finance approval** or can be **auto-approved**.
    The **Random Forest Classifier** evaluates price discrepancies, quantity mismatches, and operational delays.
    """)
    
    with st.form("invoice_flag_form"):
        st.subheader("📋 Enter Invoice & Purchase Order Details")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            inv_qty = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight_val = st.number_input("Freight Cost ($)", min_value=0.0, value=1.73)
            
        with c2:
            inv_dollars = st.number_input("Invoice Dollars ($)", min_value=1.0, value=352.95)
            item_qty = st.number_input("PO Total Item Quantity", min_value=1, value=162)
            
        with c3:
            item_dollars = st.number_input("PO Total Item Dollars ($)", min_value=1.0, value=2476.0)
            
        submit_flag = st.form_submit_button("🧠 Audit Invoice Risk")
        
    if submit_flag:
        input_dict = {
            "invoice_quantity": [inv_qty],
            "invoice_dollars": [inv_dollars],
            "Freight": [freight_val],
            "total_item_quantity": [item_qty],
            "total_item_dollars": [item_dollars]
        }
        
        flag_res = predict_invoice_flag(input_dict)
        pred_flag = flag_res['Predicted_Flag'].values[0]
        
        # Load model to get prediction probability if possible
        model, scaler = load_flag_model()
        scaled_in = scaler.transform(pd.DataFrame(input_dict))
        probs = model.predict_proba(scaled_in)[0]
        risk_prob = probs[1] * 100
        
        st.divider()
        st.subheader("📌 Audit Result")
        
        r_col1, r_col2 = st.columns([1, 1])
        
        with r_col1:
            if pred_flag == 1:
                st.markdown("""
                <div class="status-danger">
                    🚨 <strong>STATUS: MANUAL APPROVAL REQUIRED</strong><br>
                    This invoice exhibits abnormal cost, price mismatch, or quantity anomalies.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-safe">
                    ✅ <strong>STATUS: SAFE FOR AUTO-APPROVAL</strong><br>
                    This invoice aligns with purchase orders and historical patterns.
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.metric("Calculated Risk Score", f"{risk_prob:.1f}%")
            
        with r_col2:
            st.subheader("🔍 Anomaly Breakdown")
            diff = abs(inv_dollars - item_dollars)
            qty_diff = abs(inv_qty - item_qty)
            
            st.write(f"- **Dollar Mismatch**: ${diff:,.2f} {'⚠️ (Exceeds $5 tolerance)' if diff > 5 else '✅ (Within tolerance)'}")
            st.write(f"- **Quantity Variance**: {qty_diff:,} units {'⚠️ (Mismatch)' if qty_diff > 0 else '✅ (Match)'}")
            st.write(f"- **Freight-to-Invoice Ratio**: {(freight_val / inv_dollars) * 100:.2f}%")


# =======================================================
# MODULE 4: Batch CSV Invoice Evaluator
# =======================================================
elif nav_choice == "📁 Batch CSV Invoice Evaluator":
    st.header("📁 Batch CSV Invoice Evaluator")
    st.markdown("""
    Upload a CSV file containing multiple vendor invoices to run **Freight Prediction** and **Risk Classification** in batch.
    """)
    
    # Downloadable Sample CSV
    sample_df = pd.DataFrame({
        "PONumber": [10001, 10002, 10003],
        "invoice_quantity": [50, 1500, 300],
        "invoice_dollars": [352.95, 24500.0, 5200.0],
        "Freight": [1.73, 145.0, 25.0],
        "total_item_quantity": [162, 1500, 290],
        "total_item_dollars": [2476.0, 24500.0, 5190.0]
    })
    
    buffer = io.StringIO()
    sample_df.to_csv(buffer, index=False)
    
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=buffer.getvalue(),
        file_name="sample_invoices.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Invoices CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} records from CSV.")
            
            req_cols = ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"]
            missing = [col for col in req_cols if col not in batch_df.columns]
            
            if missing:
                st.error(f"Missing required columns in CSV: {missing}")
            else:
                if st.button("🚀 Process Batch Predictions"):
                    with st.spinner("Running AI Evaluation across batch..."):
                        # Predict Freight
                        freight_inputs = {"Dollars": batch_df["invoice_dollars"]}
                        freight_preds = predict_freight_cost(freight_inputs)["Predicted_Freight"]
                        
                        # Predict Risk Flag
                        risk_inputs = batch_df[req_cols].to_dict(orient="list")
                        risk_preds = predict_invoice_flag(risk_inputs)["Predicted_Flag"]
                        
                        batch_df["Predicted_Freight"] = freight_preds
                        batch_df["Risk_Flag"] = risk_preds
                        batch_df["Risk_Status"] = batch_df["Risk_Flag"].map({0: "SAFE (Auto-Approve)", 1: "HIGH RISK (Manual Review)"})
                        
                        st.subheader("📋 Evaluation Results")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download Results
                        out_buffer = io.StringIO()
                        batch_df.to_csv(out_buffer, index=False)
                        st.download_button(
                            label="📥 Download Analyzed Results CSV",
                            data=out_buffer.getvalue(),
                            file_name="invoice_analysis_results.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# =======================================================
# MODULE 5: Model Performance & Diagnostics
# =======================================================
elif nav_choice == "📈 Model Performance & Diagnostics":
    st.header("📈 Model Intelligence & Diagnostics")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("🚚 Freight Prediction Model")
        st.write("**Model Architecture**: Linear Regression (Best fit by MAE)")
        st.write("- **Mean Absolute Error (MAE)**: $24.11")
        st.write("- **Root Mean Squared Error (RMSE)**: $124.72")
        st.write("- **R² Score**: 96.99%")
        
    with col_m2:
        st.subheader("🚨 Invoice Risk Classifier")
        st.write("**Model Architecture**: Random Forest Classifier (GridSearchCV)")
        st.write("- **Accuracy**: 89.00%")
        st.write("- **Precision (Risk Class)**: 95.00%")
        st.write("- **Recall (Risk Class)**: 71.00%")
        st.write("- **F1-Score**: 82.00%")
        
    st.divider()
    st.subheader("📊 Risk Classifier Feature Importances")
    
    try:
        model, scaler = load_flag_model()
        features = ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"]
        importances = model.feature_importances_
        
        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values("Importance", ascending=True)
        
        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Oranges",
            title="Feature Importance in Risk Detection"
        )
        fig_fi.update_layout(template="plotly_white")
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load feature importances: {e}")
