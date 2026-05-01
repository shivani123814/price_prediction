import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Price Prediction App",
    page_icon="💰",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stApp {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}

.title {
    font-size: 50px;
    font-weight: bold;
    color: #1f3b73;
    text-align: center;
    padding-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}

.prediction-box {
    padding: 30px;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}

.stButton>button {
    width: 100%;
    background-color: #1f77ff;
    color: white;
    font-size: 20px;
    border-radius: 10px;
    height: 3em;
    border: none;
}

.stButton>button:hover {
    background-color: #125dcc;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="title">💰 Price Prediction System</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Predict prices using Machine Learning</p>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Enter Input Features")

feature1 = st.sidebar.number_input("Feature 1", min_value=0.0)
feature2 = st.sidebar.number_input("Feature 2", min_value=0.0)
feature3 = st.sidebar.number_input("Feature 3", min_value=0.0)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1,1])

with col1:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=300
    )

with col2:

    st.markdown("### Enter details and click predict")

    if st.button("Predict Price"):

        features = np.array([[feature1, feature2, feature3]])

        prediction = model.predict(features)

        st.markdown(
            f"""
            <div class="prediction-box">
                <h2>Predicted Price</h2>
                <h1 style='color:#1f77ff;'>₹ {prediction[0]:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)