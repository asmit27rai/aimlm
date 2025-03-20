# 🚀 AI/ML Model Monitoring Dashboard

Welcome to the **AI/ML Model Monitoring Dashboard**! This project provides a user-friendly interface to monitor the performance, data drift, and target drift of your machine learning models in real-time. Built with **Streamlit** and powered by **Evidently AI**, this dashboard is designed to help you keep track of your model's health and make data-driven decisions.

## 🌟 Features

- **Data Drift Analysis**: Monitor changes in feature distributions over time.
- **Model Performance Metrics**: Track key regression metrics like MAE, MSE, and R².
- **Target Drift Analysis**: Visualize changes in the target variable distribution.
- **Interactive Visualizations**: Use Plotly to create interactive and insightful charts.
- **Upload New Data**: Easily upload new datasets for monitoring.
- **Evidently AI Reports**: Generate detailed reports for data drift, model performance, and target drift.

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Data Visualization**: Plotly
- **Model Monitoring**: Evidently AI
- **Machine Learning**: Scikit-learn

---

## 🚀 Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/asmit27rai/aimlm)
   cd aiml

2. **Create a Virtual Environment**;
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**;
   ```bash
   pip install -r requirements.txt
5. **Train the Model**;
   ```bash
   python train_model.py
7. **Run the Streamlit App**;
   ```bash
   streamlit run app/main.py

---

## 🖥 Usage

1. Upload Data:
   - Use the sidebar to upload a new dataset (current.csv) for monitoring.
3. Explore Tabs:
   - Analyze feature distributions and view data drift reports.
   - View regression metrics and target vs prediction plots.
   - Monitor changes in the target variable distribution.
5. View Reports:
   - Detailed Evidently AI reports are displayed in JSON format for each analysis.

---
