# train_model.py
# Train RandomForestRegressor on mandi dataset
# Run: python train_model.py
# Outputs: model.pkl, encoders.pkl, scaler.pkl, feature_cols.pkl

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from preprocessing import preprocess_for_training

DATASET_PATH = "dataset.csv"
MODEL_PATH   = "model.pkl"


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the mandi dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    return df


def train(df: pd.DataFrame):
    """Full training pipeline — preprocess, train, evaluate, save."""

    print("\n⚙️  Preprocessing...")
    X, y, encoders, scaler, feat_cols = preprocess_for_training(df)
    print(f"   Features used: {feat_cols}")
    print(f"   Training samples: {len(X)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🧠 Training models...")
    models = {
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200,
                                 max_depth=10,
                                 min_samples_split=3,
                                 random_state=42,
                                 n_jobs=-1
                             ),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=150,
                                 max_depth=5,
                                 learning_rate=0.08,
                                 random_state=42
                             ),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    print(f"\n{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 50)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2, "preds": preds}
        marker = " ← best" if r2 == max(v["r2"] for v in results.values()) else ""
        print(f"  {name:<20} {mae:>8.2f} {rmse:>8.2f} {r2:>8.3f}{marker}")

    # Pick best model
    best_name  = max(results, key=lambda k: results[k]["r2"])
    best       = results[best_name]
    best_model = best["model"]

    print(f"\n🏆 Best: {best_name}")
    print(f"   MAE  = ₹{best['mae']:.2f} avg error per prediction")
    print(f"   R²   = {best['r2']:.3f} ({best['r2']*100:.1f}% variance explained)")

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        print("\n📊 Top feature importances:")
        imp = sorted(
            zip(feat_cols, best_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for feat, val in imp[:6]:
            bar = "█" * int(val * 50)
            print(f"   {feat:<28} {val*100:5.1f}%  {bar}")

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    # Save metrics for dashboard
    metrics = {
        "best_model":  best_name,
        "mae":         best["mae"],
        "rmse":        best["rmse"],
        "r2":          best["r2"],
        "n_samples":   len(df),
        "n_features":  len(feat_cols),
        "all_results": {k: {"mae":v["mae"],"rmse":v["rmse"],"r2":v["r2"]} for k,v in results.items()},
        "y_test":      y_test.tolist(),
        "y_pred":      best["preds"].tolist(),
        "feat_cols":   feat_cols,
        "feat_imp":    dict(imp) if hasattr(best_model, "feature_importances_") else {},
    }
    joblib.dump(metrics, "metrics.pkl")
    print("✅ Metrics saved → metrics.pkl")

    return best_model, metrics


if __name__ == "__main__":
    print("=" * 55)
    print("  KisanConnect — Dynamic Price Model Training")
    print("=" * 55)
    df = load_dataset(DATASET_PATH)
    model, metrics = train(df)
    print("\n🎉 Training complete! Run: streamlit run app.py")
