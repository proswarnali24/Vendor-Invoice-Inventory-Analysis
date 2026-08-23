# 📦 Vendor Invoice & Inventory Intelligence System
### **AI-Driven Freight Cost Prediction, Invoice Risk Flagging & Financial Leakage Analytics Portal**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.62-red.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-6.9-purple.svg)
![SQLite](https://img.shields.io/badge/SQLite-3.0-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Executive Summary

The **Vendor Invoice & Inventory Intelligence System** is an end-to-end machine learning and analytics application designed to streamline corporate finance operations, improve landed-cost forecasting, and eliminate financial leakage in vendor procurement.

By mining a relational database of **2.37+ million purchase records** and **5,543 vendor invoices**, this system delivers dual AI prediction pipelines alongside an interactive **Streamlit Intelligence Portal**:

1. **🚚 Freight Cost Prediction (Regression)**: Predicts expected freight cost per invoice with an **R² score of 96.99%** and **MAE of $24.11**.
2. **🚨 Invoice Risk Flagging (Classification)**: Identifies high-risk invoices requiring manual finance approval with **89.00% Accuracy** and **95.00% Precision**.
3. **📊 Enterprise Analytics & DB Explorer**: Provides interactive multi-table SQLite exploration, executive KPI monitoring, and bulk CSV invoice auditing.

---

## 🎯 Business Impact & Objectives

### 1. Freight Cost Estimation & Landed Cost Optimization
* **Problem**: Freight is a variable landed-cost component that is often inaccurately estimated, distorting profit margins.
* **Solution**: AI regression model forecasts expected freight expense instantly from invoice dollar values, enabling accurate margin calculations and better carrier negotiation.

### 2. Automated Financial Leakage & Fraud Detection
* **Problem**: Finance teams manually review thousands of invoices, causing bottlenecks while missing subtle price discrepancies and delay anomalies.
* **Solution**: AI classification model automatically audits incoming invoices, auto-approving safe invoices while flagging anomalous invoices for human review.

### 3. Multi-Table Supply Chain Visibility
* **Problem**: Disconnected inventory, purchase order, and invoice records obscure root causes of variance.
* **Solution**: Unified SQL data aggregation connects purchase orders, vendor invoices, price catalogs, and inventory snapshots.

---

## 📂 Data Architecture & Database Schema

The core dataset is stored in an optimized SQLite database (`inventory.db`) with **424+ MB** of structured relational records across 5 primary tables:

| Table Name | Description | Key Attributes | Row Count |
| :--- | :--- | :--- | :--- |
| `vendor_invoice` | Invoice financial & timing data | `PONumber`, `VendorNumber`, `Quantity`, `Dollars`, `Freight`, `InvoiceDate`, `PayDate` | 5,543 |
| `purchases` | Item-level purchase transactions | `PONumber`, `Brand`, `Quantity`, `Dollars`, `PODate`, `ReceivingDate` | 2,372,474 |
| `purchase_prices` | Reference purchase catalog prices | `Brand`, `PurchasePrice`, `Volume` | 12,261 |
| `begin_inventory` | Opening inventory snapshots | `Store`, `Brand`, `onHand`, `Price` | 206,529 |
| `end_inventory` | Closing inventory snapshots | `Store`, `Brand`, `onHand`, `Price` | 224,489 |

---

## 🤖 Machine Learning Architecture

### 1. Freight Cost Prediction (Regression)
- **Feature Matrix**: `Invoice Dollars`
- **Target Variable**: `Freight` ($)
- **Evaluated Models**: Linear Regression, Decision Tree Regressor, Random Forest Regressor
- **Selected Model**: **Linear Regression** (Lowest MAE, highest stability)
- **Performance Metrics**:
  - **Mean Absolute Error (MAE)**: `$24.11`
  - **Root Mean Squared Error (RMSE)**: `$124.72`
  - **R² Score**: `96.99%`

### 2. Invoice Risk Flagging (Classification)
- **Feature Matrix**: `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars`
- **Target Variable**: `flag_invoice` (`0` = Safe / Auto-Approve, `1` = Risk / Manual Approval Required)
- **Tuning Strategy**: `GridSearchCV` with 5-Fold Cross-Validation & `StandardScaler` feature normalization
- **Selected Model**: **Random Forest Classifier**
- **Performance Metrics**:
  - **Accuracy**: `89.00%`
  - **Precision (Risk Class)**: `95.00%`
  - **Recall (Risk Class)**: `71.00%`
  - **F1-Score**: `0.88`

---

## 🖥 Streamlit Analytics Portal Features

The interactive portal ([`app.py`](file:///Users/sornalisen/Desktop/Inventory-Invoice-Analytics/app.py)) is organized into 5 specialized modules:

1. **📊 Executive Analytics & DB Explorer**:
   - **Executive KPIs**: Real-time summary metrics for invoice volume, total dollar value, total freight spend, and high-risk count.
   - **Interactive Visualizations**: Scatter plot of Freight vs. Invoice Dollars, Risk distribution donut chart, Top 10 vendors bar chart, and freight ratio distribution histogram.
   - **SQLite Multi-Table Explorer**: Searchable and pageable data viewer for all 5 database tables.
2. **🚚 Freight Cost Prediction**:
   - Interactive single-invoice freight calculator with expected ranges and historical benchmark rates.
3. **🚨 Invoice Risk Flagging (AI Audit)**:
   - Real-time invoice risk evaluator outputting status badges (`SAFE` vs `HIGH RISK`), risk probability percentage, and detailed variance breakdown.
4. **📁 Batch CSV Invoice Evaluator**:
   - Bulk uploader to process multiple invoices simultaneously, complete with a downloadable sample template and one-click exported CSV analysis.
5. **📈 Model Diagnostics**:
   - Feature importances chart displaying the impact of dollar discrepancies and freight ratios on risk classification.

---

## 📁 Repository Structure

```bash
Vendor-Invoice-Inventory-Analysis/
│
├── data/
│   └── inventory.db                     # Relational SQLite database (ignored in git due to size)
│
├── freight_cost_prediction/
│   ├── __init__.py
│   ├── data_preprocessing.py            # SQLite data extraction & feature preparation
│   ├── modeling_evaluation.py           # Regression model training & metrics evaluation
│   └── train.py                         # Training pipeline for Freight Prediction model
│
├── invoice_flagging/
│   ├── __init__.py
│   ├── data_preprocessing.py            # SQL aggregations & risk label generation logic
│   ├── modeling_evaluation.py           # GridSearchCV hyperparameter tuning & evaluation
│   └── train.py                         # Training pipeline for Invoice Risk Classifier
│
├── inference/
│   ├── __init__.py
│   ├── predict_freight.py               # Inference module for freight prediction
│   └── predict_invoice_flag.py          # Inference module for invoice risk evaluation
│
├── models/
│   ├── predict_freight_model.pkl        # Serialized Linear Regression model
│   ├── predict_flag_invoice.pkl         # Serialized Random Forest Classifier model
│   └── scaler.pkl                       # Serialized StandardScaler object
│
├── notebooks/
│   ├── Invoice Flagging.ipynb           # Exploratory Data Analysis & Classifier research
│   └── Predicting Freight Cost.ipynb    # Exploratory Data Analysis & Regressor research
│
├── app.py                               # Multi-module Streamlit Analytics Web Application
├── run_portal.py                        # Single-command environment checker & launcher
├── .gitignore                           # Git ignore rules for database and cache files
└── README.md                            # Professional project documentation
```

---

## 🚀 Quick Start & How to Run

### Prerequisites
- Python 3.9+ (Python 3.13 recommended)
- Dependencies: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `joblib`

### 1. Clone the Repository
```bash
git clone https://github.com/proswarnali24/Vendor-Invoice-Inventory-Analysis.git
cd Vendor-Invoice-Inventory-Analysis
```

### 2. Install Required Dependencies
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

### 3. Option A: Run via Launcher Script (Recommended)
The launcher script verifies database connectivity, ensures model `.pkl` files are present, and starts Streamlit automatically:
```bash
python3 run_portal.py
```

### 4. Option B: Run Streamlit Directly
```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

---

## 👤 Author & Contact

**Sornali Sen**  
Data Analyst & AI Developer  
🔗 **GitHub**: [@proswarnali24](https://github.com/proswarnali24)  
🔗 **Repository**: [Vendor-Invoice-Inventory-Analysis](https://github.com/proswarnali24/Vendor-Invoice-Inventory-Analysis.git)
