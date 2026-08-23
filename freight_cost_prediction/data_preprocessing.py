import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd


from pathlib import Path

def get_default_db_path():
    project_root = Path(__file__).resolve().parent.parent
    project_db = project_root / "data" / "inventory.db"
    if project_db.exists():
        return str(project_db)
    desktop_db = Path.home() / "Desktop" / "inventory.db"
    if desktop_db.exists():
        return str(desktop_db)
    return str(project_db)

def load_vendor_invoice_data(db_path: str = None):
    """
    Load vendor invoice data from SQLite database.
    """
    if db_path is None or not Path(db_path).exists():
        db_path = get_default_db_path()
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable.
    """
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )