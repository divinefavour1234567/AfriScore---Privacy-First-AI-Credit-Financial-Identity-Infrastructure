"""
AfriScore - Credit Scoring Model Training
Uses XGBoost for performance + SHAP for explainability.
Trains on synthetic alt-data and saves model + metadata.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
import pathlib
warnings.filterwarnings("ignore")

# Paths relative to project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_users.csv"
MODEL_PATH = PROJECT_ROOT / "backend" / "afri_score_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "backend" / "feature_names.txt"

def prepare_features(df):
    """Select and engineer features for the model."""
    feature_cols = [
        "age",
        "mobile_txn_count_6m",
        "avg_txn_amount",
        "txn_regularity_score",
        "utility_consistency",
        "has_electricity_payment",
        "education_level",
        "skills_completed",
        "savings_rate_proxy",
        "app_engagement_score",
        "is_smallholder",
        "farm_size_proxy",
        "yield_risk_score"
    ]
    
    X = df[feature_cols].copy()
    y = df["creditworthy_label"]
    
    # Simple feature engineering
    X["txn_volume_proxy"] = X["mobile_txn_count_6m"] * X["avg_txn_amount"] / 10000
    X["engagement_x_savings"] = X["app_engagement_score"] * X["savings_rate_proxy"]
    
    final_features = feature_cols + ["txn_volume_proxy", "engagement_x_savings"]
    return X[final_features], y, final_features

def train_and_save():
    print("Loading synthetic data...")
    df = pd.read_csv(DATA_PATH)
    
    X, y, feature_names = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}...")
    
    # XGBoost classifier (good balance of performance + speed for hackathon)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=5,
        learning_rate=0.1,
        n_estimators=150,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # SHAP explainer (for demo explanations)
    print("\nCreating SHAP explainer (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    
    # Save everything
    joblib.dump({
        "model": model,
        "explainer": explainer,
        "feature_names": feature_names
    }, MODEL_PATH)
    
    with open(FEATURES_PATH, "w") as f:
        f.write("\n".join(feature_names))
    
    print(f"\nModel + Explainer saved to: {MODEL_PATH}")
    print(f"Feature list saved to: {FEATURES_PATH}")
    
    # Quick feature importance
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\nTop 8 Most Important Features:")
    print(importance.head(8).to_string(index=False))
    
    return model, explainer, feature_names

if __name__ == "__main__":
    train_and_save()