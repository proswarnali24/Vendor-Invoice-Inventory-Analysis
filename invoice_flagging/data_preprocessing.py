import sqlite3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def get_default_db_path():
    project_root = Path(__file__).resolve().parent.parent
    project_db = project_root / "data" / "inventory.db"
    if project_db.exists():
        return str(project_db)
    desktop_db = Path.home() / "Desktop" / "inventory.db"
    if desktop_db.exists():
        return str(desktop_db)
    return str(project_db)

def load_invoice_data(db_path=None):
    if db_path is None or not Path(db_path).exists():
        db_path = get_default_db_path()
    conn = sqlite3.connect(db_path)

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
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
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    if pd.isna(row["total_item_dollars"]) or abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1
    if not pd.isna(row["avg_receiving_delay"]) and row["avg_receiving_delay"] > 10:
        return 1
    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def split_data(df, features, target):
    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )

def scale_features(X_train, X_test, scaler_path=None):
    if scaler_path is None:
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        scaler_path = models_dir / "scaler.pkl"
    else:
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled
