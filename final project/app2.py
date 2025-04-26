
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import os
import io

# Title
st.title("📊 AI-Powered Data Visualization & ML App")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Read data
    data = pd.read_csv(uploaded_file)

    # Data Cleaning
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.replace(',', '', regex=True)
        data[col] = pd.to_numeric(data[col], errors='ignore')

    # Fill missing values
    if data.isnull().values.any():
        st.warning("⚠️ Dataset contains missing values. Filling them with nearest valid values.")
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

    # Show data preview
    st.write("### 🔍 Data Preview:")
    st.dataframe(data.head())



    # 📌 **1. Automated Data Insights**
    st.subheader("📊 AI-Generated Data Insights")
    if st.button("🔍 Generate Data Report"):
        profile = ProfileReport(data, explorative=True)
        st_profile_report(profile)

    # 📌 **2. Interactive Data Filtering**
    st.sidebar.header("🔎 Data Filters")
    col_to_filter = st.sidebar.selectbox("📌 Select Column to Filter:", data.columns)
    unique_values = data[col_to_filter].unique()
    selected_value = st.sidebar.selectbox(f"🎯 Filter {col_to_filter} by:", unique_values)
    filtered_data = data[data[col_to_filter] == selected_value]
    st.write("### 🔍 Filtered Data Preview:")
    st.dataframe(filtered_data)

    # 📌 **1. Interactive Visualizations**
    st.subheader("📊 Interactive Visualizations")
    
    graph_count = st.session_state.get("graph_count", 1)
    
    for i in range(graph_count):
        st.write(f"### 📈 Graph {i + 1}")
        viz_type = st.selectbox(f"📌 Choose Visualization Type {i + 1}:", ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Heatmap"], key=f"viz_type_{i}")
        x_col = st.selectbox(f"📌 Select X-axis column {i + 1}:", data.columns, key=f"x_col_{i}")
        y_col = st.selectbox(f"📌 Select Y-axis column {i + 1}:", data.columns, key=f"y_col_{i}")

        if viz_type == "Line Chart":
            fig = px.line(data, x=x_col, y=y_col, title=f"📊 Line Chart: {y_col} vs {x_col}")
        elif viz_type == "Bar Chart":
            fig = px.bar(data, x=x_col, y=y_col, title=f"📊 Bar Chart: {y_col} vs {x_col}")
        elif viz_type == "Scatter Plot":
            fig = px.scatter(data, x=x_col, y=y_col, title=f"🔵 Scatter Plot: {y_col} vs {x_col}")
        elif viz_type == "Pie Chart":
            category_col = st.selectbox(f"📌 Select Category Column for Pie Chart {i + 1}:", data.columns, key=f"category_col_{i}")
            fig = px.pie(data, names=category_col, title=f"🥧 Pie Chart of {category_col}")
        elif viz_type == "Heatmap":
            fig = px.imshow(data.corr(), text_auto=True, title="🔥 Heatmap of Feature Correlations", color_continuous_scale='coolwarm')

        st.plotly_chart(fig, use_container_width=True)

    if st.button("➕ Add Another Graph"):
        st.session_state.graph_count = graph_count + 1
        st.rerun()

    # 📌 **2. Train an ML Model**
    st.subheader("🧠 Train an ML Model")
    target_col = st.selectbox("🎯 Select Target Variable:", data.columns)
    features = data.drop(columns=[target_col])

    categorical_columns = features.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        st.warning(f"ℹ️ Encoding categorical columns: {', '.join(categorical_columns)}")
        features = pd.get_dummies(features, columns=categorical_columns, drop_first=True)

    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    model_type = st.selectbox("⚙️ Choose ML Model:", ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"])

    if model_type == "Decision Tree":
        max_depth = st.slider("🌳 Max Depth for Decision Tree:", 1, 20, 5)
    elif model_type in ["Random Forest", "XGBoost"]:
        n_estimators = st.slider("🌲 Number of Trees:", 50, 500, 100)

    if st.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(features, data[target_col], test_size=0.2, random_state=42)

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree":
            model = DecisionTreeRegressor(max_depth=max_depth)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=n_estimators)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=n_estimators)

        model.fit(X_train, y_train)

        # ✅ **Calculate Accuracy Metrics**
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        # ✅ **Display Accuracy Metrics**
        st.write("### 📊 Model Accuracy Metrics:")
        st.write(f"📌 **Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"📌 **Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"📌 **Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"📌 **R² Score:** {r2:.4f}")


        # ✅ **Download Model**
        with open("trained_model.pkl", "rb") as f:
            st.download_button("📥 Download Trained Model", f, file_name="trained_model.pkl")
        

    # 📌 **Forecasting**
if st.checkbox("🔮 Do you want to forecast?"):
    if os.path.exists("trained_model.pkl") and os.path.exists("model_features.pkl"):
        model = joblib.load("trained_model.pkl")
        expected_features = joblib.load("model_features.pkl")

        forecast_col = st.selectbox("📌 What to forecast?", data.columns)
        based_on = st.multiselect("📌 Forecast based on:", [col for col in data.columns if col != forecast_col])
        time_horizon = st.slider("📅 Forecast how many steps ahead?", 1, 12, 6)

        if st.button("📈 Generate Forecast"):
            # Ensure selected columns exist in the dataset
            if not set(based_on).issubset(set(data.columns)):
                st.error("⚠️ Some selected columns are not in the dataset!")
            else:
                input_data = data[based_on].tail(1)

                # Handle categorical features (One-Hot Encoding)
                categorical_columns = input_data.select_dtypes(include=['object']).columns
                if not categorical_columns.empty:
                    input_data = pd.get_dummies(input_data, columns=categorical_columns)

                # Ensure feature consistency: add missing columns
                missing_cols = set(expected_features) - set(input_data.columns)
                for col in missing_cols:
                    input_data[col] = 0  # Fill missing columns with 0

                # Ensure column order matches training
                input_data = input_data.reindex(columns=expected_features, fill_value=0)

                # Forecast future values
                forecast = []
                for _ in range(time_horizon):
                    pred = model.predict(input_data)  # Predict the next value

                    # Ensure prediction shape is correct
                    pred_value = pred[0] if isinstance(pred, (np.ndarray, list)) else pred

                    forecast.append(pred_value)

                    # Shift input data for the next step
                    input_data = input_data.shift(-1).fillna(0)
                    input_data.iloc[-1, :] = pred_value  # Fixed assignment

                st.write(f"📉 **Forecasted Values:** {forecast}")
                st.line_chart(forecast)

    else:
        st.error("⚠️ Train the model first to enable forecasting.")

# 📌 **About Section**
st.sidebar.header("👤 About Me")

st.sidebar.markdown("""
### Digvijay Hande  
📞 **Phone:** [9767966370](tel:9767966370)  
📧 **Email:** [digvijayhande07@gmail.com](mailto:digvijayhande07@gmail.com)  
🔗 **LinkedIn:** [Click Here](https://www.linkedin.com/in/digvijay-hande-1bb538264/)  
💻 **GitHub:** [Click Here](https://github.com/Digvijaycode)
""")
