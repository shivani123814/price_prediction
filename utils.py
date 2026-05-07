# utils.py
# Shared helper functions used across the app

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

HISTORY_PATH = "prediction_history.csv"
HISTORY_COLS = [
    "timestamp", "product_name", "category", "mandi_price",
    "views", "units_sold", "stock_quantity", "season",
    "quality_grade", "demand_score", "farmer_location",
    "predicted_price", "status"
]


def get_current_season() -> str:
    """Return current Indian agricultural season based on month."""
    month = datetime.now().month
    if month in [3, 4, 5]:       return "Summer"
    elif month in [6, 7, 8, 9]:  return "Monsoon"
    elif month in [10, 11]:      return "Autumn"
    else:                        return "Winter"


def save_prediction_to_history(input_data: dict, result: dict):
    """Append a prediction to the history CSV."""
    row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "product_name":    input_data.get("product_name"),
        "category":        input_data.get("category"),
        "mandi_price":     input_data.get("mandi_price"),
        "views":           input_data.get("views"),
        "units_sold":      input_data.get("units_sold"),
        "stock_quantity":  input_data.get("stock_quantity"),
        "season":          input_data.get("season"),
        "quality_grade":   input_data.get("quality_grade"),
        "demand_score":    input_data.get("demand_score"),
        "farmer_location": input_data.get("farmer_location"),
        "predicted_price": result.get("final_price"),
        "status":          result.get("status"),
    }

    new_row = pd.DataFrame([row])

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history.to_csv(HISTORY_PATH, index=False)


def load_prediction_history() -> pd.DataFrame:
    """Load prediction history — returns empty DataFrame if none exists."""
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=HISTORY_COLS)


def load_metrics() -> dict:
    """Load saved training metrics."""
    if os.path.exists("metrics.pkl"):
        return joblib.load("metrics.pkl")
    return None


def is_model_ready() -> bool:
    """Check if model and all required artifacts exist."""
    required = ["model.pkl", "encoders.pkl", "scaler.pkl", "feature_cols.pkl"]
    return all(os.path.exists(f) for f in required)


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the dataset."""
    return {
        "total_rows":    len(df),
        "categories":    df["category"].nunique() if "category" in df.columns else 0,
        "products":      df["product_name"].nunique() if "product_name" in df.columns else 0,
        "avg_price":     df["final_price"].mean() if "final_price" in df.columns else 0,
        "max_price":     df["final_price"].max()  if "final_price" in df.columns else 0,
        "min_price":     df["final_price"].min()  if "final_price" in df.columns else 0,
        "locations":     df["farmer_location"].nunique() if "farmer_location" in df.columns else 0,
    }


# ── Dropdown options for UI ────────────────────────────────────

PRODUCTS = [
    "Tomato", "Onion", "Potato", "Spinach", "Coriander",
    "Mango", "Banana", "Grapes", "Papaya", "Pomegranate", "Lemon",
    "Wheat", "Rice",
    "Garlic", "Ginger",
    "Cauliflower", "Brinjal", "Okra", "Peas", "Bitter Gourd",
]

CATEGORIES = ["Vegetable", "Fruit", "Grain", "Leafy", "Spice"]

SEASONS = ["Summer", "Monsoon", "Autumn", "Winter"]

QUALITY_GRADES = ["A", "B", "C"]

LOCATIONS = ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Rewa", "Sagar", "Dewas"]

CATEGORY_MAP = {
    "Tomato":"Vegetable","Onion":"Vegetable","Potato":"Vegetable",
    "Spinach":"Leafy","Coriander":"Leafy",
    "Mango":"Fruit","Banana":"Fruit","Grapes":"Fruit",
    "Papaya":"Fruit","Pomegranate":"Fruit","Lemon":"Fruit",
    "Wheat":"Grain","Rice":"Grain",
    "Garlic":"Spice","Ginger":"Spice",
    "Cauliflower":"Vegetable","Brinjal":"Vegetable","Okra":"Vegetable",
    "Peas":"Vegetable","Bitter Gourd":"Vegetable",
}
