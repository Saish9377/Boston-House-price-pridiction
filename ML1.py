import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -----------------------
# Page Config & Title
# -----------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction Dashboard")

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
X = df.drop('medv', axis=1)
y = df['medv']
feature_names = X.columns.tolist()

# -----------------------
# Sidebar Options
# -----------------------
st.sidebar.header("⚙️ Settings")

split_ratio = st.sidebar.slider("Train-Test Split (%)", 10, 90, 80, 5)
test_size = 1 - (split_ratio / 100)

model_options = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBoost Regressor": XGBRegressor()
}

algo_choice = st.sidebar.selectbox("Select Model", ["All Models"] + list(model_options.keys()))
run_button = st.sidebar.button("🚀 Run Model(s)")

# -----------------------
# Data Preview & Download
# -----------------------
with st.expander("📂 View Dataset"):
    st.dataframe(df)

with st.expander("📊 Summary Statistics"):
    st.write(df.describe())

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Dataset", csv_data, file_name="boston_housing.csv", mime="text/csv")

# -----------------------
# Split Data
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# -----------------------
# Evaluation Function
# -----------------------
def evaluate_model(model, name, color):
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader(f"📘 {name}")
    st.write(f"**MSE:** {mse:.2f} | **R² Score:** {r2:.2f}")
    st.write(f"⏱ Training Time: {end - start:.2f} seconds")

    # Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, color=color)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title(f"{name}: Actual vs Predicted")
    ax.grid(True)
    st.pyplot(fig)

    # Feature Importance (for tree models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)

        st.write("📌 Feature Importance:")
        st.bar_chart(feat_df.set_index("Feature"))

    return mse, r2

# -----------------------
# Run & Show Results
# -----------------------
if run_button:
    mse_scores = {}
    r2_scores = {}

    if algo_choice == "All Models":
        colors = ['blue', 'green', 'orange', 'purple']
        for (name, model), color in zip(model_options.items(), colors):
            st.divider()
            mse, r2 = evaluate_model(model, name, color)
            mse_scores[name] = mse
            r2_scores[name] = r2

        # Comparison Bar Charts
        st.subheader("📊 Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**MSE Scores**")
            st.bar_chart(pd.DataFrame(mse_scores, index=["MSE"]).T)

        with col2:
            st.write("**R² Scores**")
            st.bar_chart(pd.DataFrame(r2_scores, index=["R2"]).T)

    else:
        selected_model = model_options[algo_choice]
        evaluate_model(selected_model, algo_choice, "blue")
