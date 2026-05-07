# app.py
# KisanConnect — Dynamic Price Prediction Dashboard
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, io

from train_model import load_dataset, train
from prediction  import predict_price, load_model
from utils       import (
    get_current_season, save_prediction_to_history,
    load_prediction_history, load_metrics, is_model_ready,
    get_dataset_stats, PRODUCTS, CATEGORIES, SEASONS,
    QUALITY_GRADES, LOCATIONS, CATEGORY_MAP
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "KisanConnect · Price AI",
    page_icon   = "🌾",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark green background for main area */
  .stApp { background: #0a1a0f; color: #e0f0e3; }
  .block-container { padding-top: 1.5rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f2518 !important;
    border-right: 1px solid #1e4a28;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #122b1a;
    border: 1px solid #2e7d32;
    border-radius: 12px;
    padding: 12px !important;
  }
  [data-testid="metric-container"] label { color: #7db882 !important; }
  [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #6abf69 !important; font-size: 1.6rem !important;
  }

  /* Section headers */
  .section-header {
    background: linear-gradient(135deg, #0f2518, #1a3d27);
    border-left: 4px solid #6abf69;
    border-radius: 0 10px 10px 0;
    padding: 10px 18px;
    margin: 18px 0 12px;
  }
  .section-header h3 { color: #6abf69; margin: 0; font-size: 1.1rem; }

  /* Price result card */
  .price-card {
    background: linear-gradient(135deg, #0f2518, #1a3d27);
    border: 2px solid #6abf69;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 12px 0;
  }
  .price-big { font-size: 3.2rem; font-weight: 800; color: #6abf69; }
  .price-sub { font-size: 0.95rem; color: #7db882; margin-top: 4px; }

  /* Adjustment items */
  .adj-box {
    background: #122b1a;
    border: 1px solid #1e4a28;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.88rem;
    color: #a5c9a8;
  }

  /* Status badges */
  .badge-premium  { background:#7c3aed22; color:#c084fc; border:1px solid #7c3aed; border-radius:20px; padding:3px 12px; }
  .badge-good     { background:#16a34a22; color:#6abf69; border:1px solid #16a34a; border-radius:20px; padding:3px 12px; }
  .badge-fair     { background:#0891b222; color:#67e8f9; border:1px solid #0891b2; border-radius:20px; padding:3px 12px; }
  .badge-discount { background:#d9770622; color:#fb923c; border:1px solid #d97706; border-radius:20px; padding:3px 12px; }

  /* Input styling */
  .stSelectbox > div > div, .stNumberInput > div > div { background: #122b1a !important; }
  .stButton > button {
    background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important; font-size: 1rem !important;
  }
  .stButton > button:hover { background: #1b5e20 !important; }

  /* Table */
  .dataframe { background: #122b1a !important; color: #e0f0e3 !important; }
  thead tr th { background: #1e4a28 !important; color: #6abf69 !important; }
  tbody tr:hover td { background: #1a3d27 !important; }

  /* Plotly chart bg */
  .js-plotly-plot { border-radius: 12px; overflow: hidden; }

  /* Hide streamlit default elements */
  footer { display: none; }
  #MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2518,#1a3d27,#0f2518);
            padding:24px 32px;border-radius:16px;
            border:1px solid #2e7d32;margin-bottom:8px'>
  <div style='display:flex;align-items:center;gap:14px'>
    <span style='font-size:2.8rem'>🌾</span>
    <div>
      <h1 style='color:#6abf69;margin:0;font-size:1.9rem;font-weight:800'>
        KisanConnect · Dynamic Price AI
      </h1>
      <p style='color:#7db882;margin:4px 0 0;font-size:0.95rem'>
        ML-powered price prediction for farmer-consumer marketplace · Madhya Pradesh
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px'>
      <span style='font-size:2rem'>🌾</span>
      <p style='color:#6abf69;font-weight:700;margin:4px 0 0'>KisanConnect</p>
      <p style='color:#7db882;font-size:0.78rem;margin:0'>Price Prediction AI</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔮 Predict Price", "🧠 Train Model", "📊 Analytics", "📋 History"],
        label_visibility="collapsed"
    )
    st.divider()

    # Model status indicator
    if is_model_ready():
        st.success("✅ Model ready")
        metrics = load_metrics()
        if metrics:
            st.markdown(f"""
            <div style='font-size:0.8rem;color:#7db882;line-height:1.8'>
              <b style='color:#6abf69'>Model Stats</b><br>
              Algorithm: {metrics['best_model']}<br>
              R² Score: {metrics['r2']:.3f}<br>
              Avg Error: ₹{metrics['mae']:.2f}/kg<br>
              Samples: {metrics['n_samples']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model not trained")
        st.caption("Go to Train Model tab first")

    st.divider()
    st.markdown("<p style='color:#4a7a52;font-size:0.75rem;text-align:center'>Powered by scikit-learn + Streamlit</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1: PREDICT PRICE
# ══════════════════════════════════════════════════════════════
if page == "🔮 Predict Price":

    st.markdown("<div class='section-header'><h3>🔮 Dynamic Price Prediction</h3></div>", unsafe_allow_html=True)

    if not is_model_ready():
        st.warning("⚠️ Model not trained yet. Go to **🧠 Train Model** first.")
        st.stop()

    # ── Input form ─────────────────────────────────────────────
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🌾 Product Info**")
            product_name = st.selectbox("Product", PRODUCTS)
            category     = st.selectbox("Category", CATEGORIES,
                                        index=CATEGORIES.index(CATEGORY_MAP.get(product_name, "Vegetable")))
            quality_grade = st.selectbox("Quality Grade", QUALITY_GRADES,
                                         help="A=Premium · B=Good · C=Standard")
            farmer_location = st.selectbox("Farmer Location", LOCATIONS)

        with col2:
            st.markdown("**📈 Market Data**")
            mandi_price  = st.number_input("Mandi Price (₹/kg)", min_value=1.0, value=20.0, step=0.5)
            season       = st.selectbox("Season", SEASONS, index=SEASONS.index(get_current_season()))
            demand_score = st.slider("Demand Score", 1.0, 10.0, 7.0, 0.1,
                                     help="1=Very low demand · 10=Extreme demand")

        with col3:
            st.markdown("**📦 Stock & Sales**")
            views          = st.number_input("Product Views",   min_value=0,   value=250,  step=10)
            units_sold     = st.number_input("Units Sold (kg)", min_value=0,   value=80,   step=5)
            stock_quantity = st.number_input("Stock Left (kg)", min_value=1,   value=150,  step=10)

        submitted = st.form_submit_button("🔮 Predict Optimal Price", use_container_width=True)

    # ── Show result ────────────────────────────────────────────
    if submitted:
        input_data = {
            "product_name":    product_name,
            "category":        category,
            "mandi_price":     mandi_price,
            "views":           views,
            "units_sold":      units_sold,
            "stock_quantity":  stock_quantity,
            "season":          season,
            "quality_grade":   quality_grade,
            "demand_score":    demand_score,
            "farmer_location": farmer_location,
        }

        with st.spinner("🧠 Calculating optimal price..."):
            result = predict_price(input_data)

        if result.get("error"):
            st.error(result["error"])
            st.stop()

        # ── Main price card ───────────────────────────────────
        rc1, rc2, rc3 = st.columns([1, 1.5, 1])
        with rc2:
            badge_class = f"badge-{result['status'].lower()}"
            st.markdown(f"""
            <div class='price-card'>
              <p style='color:#7db882;margin:0 0 4px;font-size:0.9rem'>Optimal Selling Price</p>
              <div class='price-big'>₹{result['final_price']}</div>
              <div class='price-sub'>per kilogram</div>
              <div style='margin-top:10px'>
                <span class='{badge_class}'>{result['status_emoji']} {result['status']}</span>
              </div>
              <p style='color:#7db882;font-size:0.82rem;margin:10px 0 0'>
                Range: ₹{result['price_range']['min']} – ₹{result['price_range']['max']}/kg
              </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Metrics row ────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ML Base Price",   f"₹{result['base_price']}/kg")
        m2.metric("Final Price",     f"₹{result['final_price']}/kg")
        m3.metric("vs Mandi Price",  f"{result['diff_from_mandi_pct']:+.1f}%")
        m4.metric("Mandi Reference", f"₹{mandi_price}/kg")

        # ── Adjustments + Gauge ─────────────────────────────────
        ga1, ga2 = st.columns(2)

        with ga1:
            st.markdown("**⚙️ Dynamic Pricing Adjustments**")
            if result["adjustments"]:
                for adj in result["adjustments"]:
                    st.markdown(f"<div class='adj-box'>{adj}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='adj-box'>✅ No adjustments — base ML price is optimal</div>",
                            unsafe_allow_html=True)

            # Recommendation
            st.markdown(f"""
            <div style='background:#1a3d27;border:1px solid #2e7d32;
                        border-radius:10px;padding:12px;margin-top:10px'>
              <p style='color:#6abf69;margin:0;font-size:0.9rem'>{result['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

        with ga2:
            # Price gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = result["final_price"],
                delta = {"reference": mandi_price, "valueformat": ".1f",
                         "prefix": "vs mandi "},
                title = {"text": "Predicted Price (₹/kg)",
                         "font": {"color": "#6abf69", "size": 14}},
                gauge = {
                    "axis": {"range": [0, mandi_price * 3.5],
                             "tickcolor": "#7db882", "tickfont": {"color":"#7db882"}},
                    "bar":  {"color": "#6abf69"},
                    "bgcolor": "#122b1a",
                    "bordercolor": "#2e7d32",
                    "steps": [
                        {"range": [0, mandi_price * 0.9],      "color": "#1a3d27"},
                        {"range": [mandi_price * 0.9, mandi_price * 1.2], "color": "#1e4a28"},
                        {"range": [mandi_price * 1.2, mandi_price * 3.5], "color": "#163320"},
                    ],
                    "threshold": {"line": {"color": "#f9a825", "width": 3},
                                  "thickness": 0.85, "value": mandi_price},
                },
                number = {"font": {"color": "#6abf69", "size": 42}, "suffix": "₹"},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0f2518", plot_bgcolor="#0f2518",
                font_color="#7db882", height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Price comparison bar ────────────────────────────────
        comp_df = pd.DataFrame({
            "Price Type":  ["Mandi Price", "ML Base Price", "Final AI Price",
                            "Min Estimate", "Max Estimate"],
            "Price (₹/kg)":[mandi_price, result["base_price"], result["final_price"],
                            result["price_range"]["min"], result["price_range"]["max"]],
            "Color":       ["#f9a825", "#60a5fa", "#6abf69", "#7db882", "#4ade80"],
        })
        fig_bar = px.bar(
            comp_df, x="Price Type", y="Price (₹/kg)",
            color="Price Type",
            color_discrete_sequence=["#f9a825","#60a5fa","#6abf69","#7db882","#4ade80"],
            title=f"Price Breakdown — {product_name}",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
            font_color="#7db882", showlegend=False,
            title_font_color="#6abf69",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Save to history
        save_prediction_to_history(input_data, result)
        st.success("✅ Prediction saved to history!")


# ══════════════════════════════════════════════════════════════
# PAGE 2: TRAIN MODEL
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Train Model":

    st.markdown("<div class='section-header'><h3>🧠 Train Price Prediction Model</h3></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📂 Use Built-in Dataset", "⬆️ Upload Custom CSV"])

    with tab1:
        st.info("Using the built-in Madhya Pradesh mandi dataset (120 product records)")
        if os.path.exists("dataset.csv"):
            df_preview = pd.read_csv("dataset.csv")
            stats = get_dataset_stats(df_preview)
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Total Records",  stats["total_rows"])
            s2.metric("Products",       stats["products"])
            s3.metric("Categories",     stats["categories"])
            s4.metric("Locations",      stats["locations"])
            st.dataframe(df_preview.head(8), use_container_width=True, hide_index=True)

        if st.button("🚀 Train Model on Built-in Data", use_container_width=True, type="primary"):
            _run_training("dataset.csv")

    with tab2:
        uploaded = st.file_uploader(
            "Upload mandi CSV", type=["csv"],
            help="Required columns: product_name, category, mandi_price, views, units_sold, "
                 "stock_quantity, season, quality_grade, demand_score, farmer_location, final_price"
        )
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df_up):,} rows")
            st.dataframe(df_up.head(5), use_container_width=True, hide_index=True)

            # Save uploaded file
            df_up.to_csv("dataset.csv", index=False)

            if st.button("🚀 Train on Uploaded Data", use_container_width=True, type="primary"):
                _run_training("dataset.csv")


def _run_training(dataset_path: str):
    """Run training and display results."""
    with st.spinner("🧠 Training ML models... (takes ~30 seconds)"):
        progress = st.progress(0, "Loading dataset...")
        df = load_dataset(dataset_path)
        progress.progress(20, "Preprocessing data...")
        progress.progress(50, "Training Random Forest...")
        model, metrics = train(df)
        progress.progress(100, "✅ Done!")

    # Results
    st.success(f"🏆 Best model: **{metrics['best_model']}** | R² = {metrics['r2']:.3f} | MAE = ₹{metrics['mae']:.2f}/kg")

    # Model comparison table
    st.subheader("📊 Model Comparison")
    comp_rows = []
    for name, res in metrics["all_results"].items():
        comp_rows.append({
            "Model":       name,
            "MAE (₹/kg)":  round(res["mae"], 2),
            "RMSE":        round(res["rmse"], 2),
            "R² Score":    round(res["r2"], 3),
            "Accuracy":    f"{res['r2']*100:.1f}%",
            "Winner":      "✅ Best" if name == metrics["best_model"] else ""
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Feature importance chart
    if metrics.get("feat_imp"):
        feat_df = pd.DataFrame(
            list(metrics["feat_imp"].items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            feat_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Greens",
            title="Feature Importance — What drives price prediction?",
        )
        fig.update_layout(
            paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
            font_color="#7db882", showlegend=False,
            coloraxis_showscale=False, title_font_color="#6abf69",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted scatter
    y_test = metrics.get("y_test", [])
    y_pred = metrics.get("y_pred", [])
    if y_test and y_pred:
        scatter_df = pd.DataFrame({"Actual (₹/kg)": y_test, "Predicted (₹/kg)": y_pred})
        fig2 = px.scatter(
            scatter_df, x="Actual (₹/kg)", y="Predicted (₹/kg)",
            trendline="ols",
            title="Actual vs Predicted Price — Test Set",
            color_discrete_sequence=["#6abf69"],
        )
        fig2.update_layout(
            paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
            font_color="#7db882", title_font_color="#6abf69",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.balloons()
    st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 3: ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown("<div class='section-header'><h3>📊 Market Analytics Dashboard</h3></div>", unsafe_allow_html=True)

    if not os.path.exists("dataset.csv"):
        st.warning("No dataset found. Go to Train Model tab first.")
        st.stop()

    df = pd.read_csv("dataset.csv")

    # Top metrics
    stats = get_dataset_stats(df)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Records",  stats["total_rows"])
    m2.metric("Products",       stats["products"])
    m3.metric("Avg Price",      f"₹{stats['avg_price']:.1f}")
    m4.metric("Max Price",      f"₹{stats['max_price']:.1f}")
    m5.metric("Markets",        stats["locations"])

    # ── Charts row 1 ────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        # Price by category
        cat_df = df.groupby("category")["final_price"].mean().reset_index()
        fig = px.bar(
            cat_df, x="category", y="final_price",
            color="category", title="Avg Price by Category",
            color_discrete_sequence=px.colors.sequential.Greens_r,
        )
        fig.update_layout(paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
                          font_color="#7db882", showlegend=False,
                          title_font_color="#6abf69")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Price by season
        sea_df = df.groupby("season")["final_price"].mean().reset_index()
        fig2 = px.pie(
            sea_df, names="season", values="final_price",
            title="Price Distribution by Season",
            color_discrete_sequence=["#2e7d32","#6abf69","#f9a825","#60a5fa"],
        )
        fig2.update_layout(paper_bgcolor="#0f2518", font_color="#7db882",
                           title_font_color="#6abf69")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        # Mandi price vs final price scatter
        fig3 = px.scatter(
            df, x="mandi_price", y="final_price",
            color="quality_grade", size="demand_score",
            hover_data=["product_name","farmer_location"],
            title="Mandi Price vs Final Price",
            color_discrete_map={"A":"#6abf69","B":"#f9a825","C":"#ef4444"},
        )
        fig3.update_layout(paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
                           font_color="#7db882", title_font_color="#6abf69")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Top products by avg price
        top_prod = df.groupby("product_name")["final_price"].mean().nlargest(10).reset_index()
        fig4 = px.bar(
            top_prod, x="final_price", y="product_name", orientation="h",
            color="final_price", color_continuous_scale="Greens",
            title="Top 10 Products by Avg Price",
        )
        fig4.update_layout(paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
                           font_color="#7db882", coloraxis_showscale=False,
                           title_font_color="#6abf69")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Views vs Price ────────────────────────────────────────
    fig5 = px.scatter(
        df, x="views", y="final_price",
        color="category", size="demand_score",
        hover_data=["product_name"],
        title="Views vs Final Price (bubble size = demand score)",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig5.update_layout(paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
                       font_color="#7db882", title_font_color="#6abf69")
    st.plotly_chart(fig5, use_container_width=True)

    # ── Price heatmap by location and season ──────────────────
    heat_df = df.groupby(["farmer_location","season"])["final_price"].mean().unstack()
    fig6 = px.imshow(
        heat_df, text_auto=".0f",
        color_continuous_scale="Greens",
        title="Avg Price Heatmap: Location × Season",
        aspect="auto",
    )
    fig6.update_layout(paper_bgcolor="#0f2518", font_color="#7db882",
                       title_font_color="#6abf69")
    st.plotly_chart(fig6, use_container_width=True)

    # ── Full dataset table ────────────────────────────────────
    with st.expander("📄 View Full Dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Dataset CSV", csv_bytes,
                           "dataset.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4: HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "📋 History":

    st.markdown("<div class='section-header'><h3>📋 Prediction History</h3></div>", unsafe_allow_html=True)

    history = load_prediction_history()

    if history.empty:
        st.info("No predictions yet. Go to **🔮 Predict Price** to make your first prediction.")
        st.stop()

    # Summary metrics
    h1,h2,h3,h4 = st.columns(4)
    h1.metric("Total Predictions",  len(history))
    h2.metric("Avg Predicted Price", f"₹{history['predicted_price'].mean():.2f}" if "predicted_price" in history else "N/A")
    h3.metric("Products Predicted",  history["product_name"].nunique() if "product_name" in history else 0)
    h4.metric("Last Prediction",     history["timestamp"].iloc[-1][:10] if "timestamp" in history else "N/A")

    # History table
    st.subheader("All Predictions")
    st.dataframe(
        history.sort_values("timestamp", ascending=False),
        use_container_width=True, hide_index=True
    )

    # Price trend over predictions
    if len(history) > 1 and "predicted_price" in history.columns:
        history["index"] = range(1, len(history)+1)
        fig = px.line(
            history, x="index", y="predicted_price",
            color="product_name", markers=True,
            title="Price Prediction Trend",
            labels={"index": "Prediction #", "predicted_price": "Predicted Price (₹/kg)"},
        )
        fig.update_layout(paper_bgcolor="#0f2518", plot_bgcolor="#122b1a",
                          font_color="#7db882", title_font_color="#6abf69")
        st.plotly_chart(fig, use_container_width=True)

    # Download history
    csv_bytes = history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction History CSV",
        csv_bytes, "prediction_history.csv",
        "text/csv", use_container_width=True
    )
