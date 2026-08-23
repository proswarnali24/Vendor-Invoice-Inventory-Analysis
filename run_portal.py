#!/usr/bin/env python3
"""
Launcher script for Vendor Invoice & Inventory Intelligence Portal.
Verifies database connection, checks trained models, and starts Streamlit.
"""

import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    print("=" * 60)
    print("🚀 Starting Vendor Invoice & Inventory Intelligence Portal")
    print("=" * 60)

    # 1. Database Check
    db_path = project_root / "data" / "inventory.db"
    desktop_db = Path.home() / "Desktop" / "inventory.db"
    
    if db_path.exists():
        print(f"✅ Found database: {db_path}")
    elif desktop_db.exists():
        print(f"✅ Found database on Desktop: {desktop_db}")
    else:
        print("❌ Error: inventory.db not found!")
        sys.exit(1)

    # 2. Models Check
    freight_model = project_root / "models" / "predict_freight_model.pkl"
    flag_model = project_root / "models" / "predict_flag_invoice.pkl"

    if not freight_model.exists():
        print("⏳ Training Freight Cost Prediction model...")
        subprocess.run([sys.executable, str(project_root / "freight_cost_prediction" / "train.py")], check=True)
    else:
        print(f"✅ Freight model ready: {freight_model.name}")

    if not flag_model.exists():
        print("⏳ Training Invoice Risk Flagging model...")
        subprocess.run([sys.executable, str(project_root / "invoice_flagging" / "train.py")], check=True)
    else:
        print(f"✅ Risk classifier model ready: {flag_model.name}")

    # 3. Launch Streamlit
    print("\n🌐 Launching Streamlit Portal...")
    app_path = project_root / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)], cwd=str(project_root))

if __name__ == "__main__":
    main()
