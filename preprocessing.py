# preprocessing.py
# Handles all data cleaning, encoding, and scaling
# Used by both train_model.py and prediction.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Paths for saving encoders and scaler
ENCODER_PATH = "encoders.pkl"
SCALER_PATH  = "scaler.pkl"

# Categorical columns that need label encoding
CATEGORICAL_COLS = ["product_name", "category", "season", "quality_grade", "farmer_location"]

# Numeric feature columns used for training
FEATURE_COLS = [
    "mandi_price", "views", "units_sold", "stock_quantity",
    "demand_score", "product_name_enc", "category_enc",
    "season_enc", "quality_grade_enc", "farmer_location_enc",
    "price_per_view", "demand_stock_ratio", "mandi_demand_interaction"
]

TARGET_COL = "final_price"


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with appropriate strategies per column type."""
    df = df.copy()

    # Numeric columns → fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns → fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True, encoders: dict = None) -> tuple:
    """
    Label encode all categorical columns.
    fit=True  → fit new encoders (training)
    fit=False → use existing encoders (prediction)
    Returns (encoded_df, encoders_dict)
    """
    df = df.copy()

    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        # Use existing encoders — handle unseen labels safely
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[f"{col}_enc"] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0]
                    if x in le.classes_ else 0
                )

    return df, encoders


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones to improve model accuracy.
    These capture the dynamic pricing logic:
    - views ↑ stock ↓ → price ↑
    - demand ↓ → price ↓
    """
    df = df.copy()

    # Price per view — higher engagement → justify higher price
    df["price_per_view"] = df["mandi_price"] / (df["views"] + 1)

    # Demand to stock ratio — scarcity signal
    df["demand_stock_ratio"] = df["demand_score"] / (df["stock_quantity"] + 1)

    # Mandi price × demand — premium market interaction
    df["mandi_demand_interaction"] = df["mandi_price"] * df["demand_score"]

    return df


def scale_features(X: pd.DataFrame, fit: bool = True, scaler=None):
    """
    StandardScaler on numeric features.
    fit=True  → fit and transform (training)
    fit=False → transform only (prediction)
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def preprocess_for_training(df: pd.DataFrame):
    """Full preprocessing pipeline for training."""
    df = handle_missing_values(df)
    df = add_engineered_features(df)
    df, encoders = encode_categoricals(df, fit=True)

    # Select feature columns that exist
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols]
    y = df[TARGET_COL]

    X_scaled, scaler = scale_features(X, fit=True)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feat_cols)

    # Save encoders and scaler for prediction use
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(feat_cols, "feature_cols.pkl")

    return X_scaled_df, y, encoders, scaler, feat_cols


def preprocess_for_prediction(input_dict: dict) -> np.ndarray:
    """
    Preprocess a single prediction input.
    Loads saved encoders and scaler automatically.
    """
    # Load saved artifacts
    encoders  = joblib.load(ENCODER_PATH)
    scaler    = joblib.load(SCALER_PATH)
    feat_cols = joblib.load("feature_cols.pkl")

    df = pd.DataFrame([input_dict])
    df = handle_missing_values(df)
    df = add_engineered_features(df)
    df, _ = encode_categoricals(df, fit=False, encoders=encoders)

    # Fill any missing feature columns with 0
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feat_cols]
    X_scaled, _ = scale_features(X, fit=False, scaler=scaler)

    return X_scaled
