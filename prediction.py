# prediction.py
# Loads trained model and returns dynamic price prediction
# Applies business logic on top of ML output

import joblib
import numpy as np
import os
from preprocessing import preprocess_for_prediction

MODEL_PATH = "model.pkl"


def load_model():
    """Load model — returns None if not trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def apply_dynamic_pricing_rules(
    base_price:     float,
    views:          int,
    stock_quantity: int,
    demand_score:   float,
    units_sold:     int,
    mandi_price:    float,
) -> dict:
    """
    Apply business rules on top of ML base price.

    Rules:
    1. High views + low stock   → surge premium
    2. Low demand score         → reduce price
    3. High units_sold relative → demand confirmed, slight premium
    4. Far from mandi price     → cap adjustment
    """
    adjusted = base_price
    adjustments = []

    # Rule 1: Views surge — many people looking, stock running low
    view_stock_ratio = views / max(stock_quantity, 1)
    if view_stock_ratio > 2.5:
        surge = min(view_stock_ratio * 0.04, 0.18)   # max +18%
        adjusted *= (1 + surge)
        adjustments.append(f"📈 High demand surge: +{surge*100:.1f}%")

    # Rule 2: Low demand → discount
    if demand_score < 4.5:
        discount = (4.5 - demand_score) * 0.025      # up to -11%
        adjusted *= (1 - discount)
        adjustments.append(f"📉 Low demand discount: -{discount*100:.1f}%")

    # Rule 3: Strong sales history → small premium
    if units_sold > 100:
        premium = min((units_sold - 100) * 0.0003, 0.08)
        adjusted *= (1 + premium)
        adjustments.append(f"✅ Proven seller premium: +{premium*100:.1f}%")

    # Rule 4: Price cap — never go below mandi or above 3× mandi
    floor = mandi_price * 0.90
    ceil  = mandi_price * 3.0
    if adjusted < floor:
        adjusted = floor
        adjustments.append(f"🔒 Floor applied (90% of mandi price)")
    elif adjusted > ceil:
        adjusted = ceil
        adjustments.append(f"🔒 Ceiling applied (3× mandi price)")

    return {
        "adjusted_price": round(adjusted, 2),
        "adjustments":    adjustments,
    }


def predict_price(input_data: dict) -> dict:
    """
    Full prediction pipeline.
    Returns predicted price + dynamic pricing adjustments.

    input_data keys:
        product_name, category, mandi_price, views,
        units_sold, stock_quantity, season, quality_grade,
        demand_score, farmer_location
    """
    model = load_model()
    if model is None:
        return {
            "error": "Model not trained yet. Click 'Train Model' first.",
            "predicted_price": None,
        }

    # ML base prediction
    X = preprocess_for_prediction(input_data)
    base_price = float(model.predict(X)[0])
    base_price = max(base_price, 1.0)   # floor ₹1

    # Dynamic pricing adjustments
    dynamic = apply_dynamic_pricing_rules(
        base_price     = base_price,
        views          = input_data.get("views", 100),
        stock_quantity = input_data.get("stock_quantity", 200),
        demand_score   = input_data.get("demand_score", 5.0),
        units_sold     = input_data.get("units_sold", 50),
        mandi_price    = input_data.get("mandi_price", 20.0),
    )

    final_price = dynamic["adjusted_price"]
    mandi_price = input_data.get("mandi_price", 20.0)
    diff_pct    = ((final_price - mandi_price) / mandi_price) * 100

    # Fair price status
    if diff_pct > 25:
        status, emoji = "Premium",  "💎"
    elif diff_pct > 5:
        status, emoji = "Good",     "📈"
    elif diff_pct >= -5:
        status, emoji = "Fair",     "✅"
    else:
        status, emoji = "Discount", "💰"

    return {
        "base_price":    round(base_price, 2),
        "final_price":   final_price,
        "price_range": {
            "min": round(final_price * 0.90, 2),
            "max": round(final_price * 1.10, 2),
        },
        "diff_from_mandi_pct": round(diff_pct, 1),
        "status":        status,
        "status_emoji":  emoji,
        "adjustments":   dynamic["adjustments"],
        "recommendation": _get_recommendation(diff_pct, input_data),
    }


def _get_recommendation(diff_pct: float, inp: dict) -> str:
    """Human-readable recommendation for the farmer."""
    if inp.get("stock_quantity", 200) < 50 and inp.get("views", 0) > 300:
        return "🔥 High demand, low stock — sell now at this premium price!"
    if inp.get("demand_score", 5) < 4.0:
        return "⚠️ Low demand area. Consider reducing price or targeting a different market."
    if diff_pct > 20:
        return "💎 Premium pricing justified — excellent quality and high local demand."
    if diff_pct < -10:
        return "💡 Price below mandi — consider increasing slightly to improve margin."
    return "✅ Fair market price — competitive and likely to sell well."
