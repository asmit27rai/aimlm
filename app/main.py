import streamlit as st
import pandas as pd
from joblib import load
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI/ML Model Monitoring Dashboard", layout="wide")

def load_data():
    reference_data = pd.read_csv("data/reference.csv")
    current_data = pd.read_csv("data/current.csv")
    return reference_data, current_data

@st.cache_resource
def load_model():
    model = load("models/model.pkl")
    return model

def preprocess_data(data):
    data = data.drop(columns=["id", "name", "host_id", "host_name", "last_review"])
    return data

def generate_report(reference_data, current_data):
    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=reference_data, current_data=current_data)
    return report

def generate_test_suite(reference_data, current_data):
    test_suite = TestSuite(tests=[DataStabilityTestPreset()])
    test_suite.run(reference_data=reference_data, current_data=current_data)
    return test_suite

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

def plot_feature_distribution(reference_data, current_data, feature):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=reference_data[feature], name="Reference Data", opacity=0.75))
    fig.add_trace(go.Histogram(x=current_data[feature], name="Current Data", opacity=0.75))
    fig.update_layout(
        title=f"Distribution of {feature}",
        xaxis_title=feature,
        yaxis_title="Count",
        barmode="overlay"
    )
    return fig

def plot_target_vs_prediction(current_data):
    fig = px.scatter(current_data, x="target", y="prediction", trendline="ols")
    fig.update_layout(
        title="Target vs Prediction",
        xaxis_title="Target",
        yaxis_title="Prediction"
    )
    return fig

def main():
    st.title("🚀 AI/ML Model Monitoring Dashboard")
    st.markdown("Monitor your model's performance, data drift, and target drift over time.")

    reference_data, current_data = load_data()
    model = load_model()

    reference_data_processed = preprocess_data(reference_data)
    current_data_processed = preprocess_data(current_data)

    reference_data['prediction'] = model.predict(reference_data_processed)

    st.sidebar.header("Upload New Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        current_data = pd.read_csv(uploaded_file)
        current_data_processed = preprocess_data(current_data)

    current_data['prediction'] = model.predict(current_data_processed)

    if 'target' not in current_data.columns:
        st.error("Error: The 'target' column is missing in the current dataset.")
        return

    mae, mse, r2 = calculate_metrics(current_data['target'], current_data['prediction'])

    st.sidebar.header("Model Performance")
    st.sidebar.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.sidebar.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.sidebar.metric("R² Score", f"{r2:.2f}")

    tab1, tab2, tab3 = st.tabs(["Data Drift", "Model Performance", "Target Drift"])

    with tab1:
        st.header("Data Drift Analysis")
        
        st.subheader("Feature Distributions")
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Select Feature 1", reference_data.columns, key="feature1")
            fig1 = plot_feature_distribution(reference_data, current_data, feature1)
            st.plotly_chart(fig1, use_container_width=True, key="feature1_chart")
        with col2:
            feature2 = st.selectbox("Select Feature 2", reference_data.columns, key="feature2")
            fig2 = plot_feature_distribution(reference_data, current_data, feature2)
            st.plotly_chart(fig2, use_container_width=True, key="feature2_chart")

        st.subheader("Evidently Data Drift Report")
        report = generate_report(reference_data, current_data)
        st.json(report.as_dict())

    with tab2:
        st.header("Model Performance Analysis")
        
        st.subheader("Target vs Prediction")
        fig = plot_target_vs_prediction(current_data)
        st.plotly_chart(fig, use_container_width=True, key="target_vs_prediction_chart")

        st.subheader("Evidently Model Performance Report")
        regression_report = Report(metrics=[RegressionPreset()])
        regression_report.run(reference_data=reference_data, current_data=current_data)
        st.json(regression_report.as_dict())

    with tab3:
        st.header("Target Drift Analysis")
        
        st.subheader("Target Distribution")
        fig = plot_feature_distribution(reference_data, current_data, "target")
        st.plotly_chart(fig, use_container_width=True, key="target_distribution_chart")

        st.subheader("Evidently Target Drift Report")
        target_drift_report = Report(metrics=[TargetDriftPreset()])
        target_drift_report.run(reference_data=reference_data, current_data=current_data)
        st.json(target_drift_report.as_dict())

    st.header("Data Stability Tests")
    test_suite = generate_test_suite(reference_data, current_data)
    st.json(test_suite.as_dict())

main()