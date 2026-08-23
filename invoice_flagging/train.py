import sys
from pathlib import Path
import joblib

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modeling_evaluation import train_random_forest, evaluate_classifier

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"
    
    
def main():
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, models_dir / 'scaler.pkl'
    )
    
    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )
    
    # Save best model
    model_path = models_dir / 'predict_flag_invoice.pkl'
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\nBest classifier saved to {model_path}")

if __name__ == "__main__":
    main()
