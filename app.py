
import os
import requests
import numpy as np
import pandas as pd
import json
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset


def monitor_model(reference_data, current_data):
    """
    Monitor the model using Evidently AI.

    Parameters:
        reference_data (pd.DataFrame): Historical data to use as a baseline.
        current_data (pd.DataFrame): New data with predictions to analyze.
        output_path (str): Path to save the monitoring reports.

    Returns:
        None
    """
    # Generate Data Drift and Regression Performance Reports
    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset()
    ])
    report.run(reference_data=reference_data, current_data=current_data)

    # Save report as HTML
    # report.save_html(os.path.join(output_path, "monitoring_report.html"))

    # Optionally, create a dashboard for live visualization
    # dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab()])
    # dashboard.calculate(reference_data=reference_data, current_data=current_data)
    # dashboard.save(os.path.join(output_path, "monitoring_dashboard.html"))
    # report.save_html(os.path.join(output_path, "monitoring_dashboard.html"))
    return report

def create_tf_serving_json(data):
    # Format data for TensorFlow Serving
    return {"inputs": data.tolist() if not isinstance(data, dict) else {key: value.tolist() for key, value in data.items()}}

def score_model_with_labels(dataset, label_column=None):
    """
    Scores the model and returns a table containing predictions along with the original data and labels.

    Parameters:
        dataset (pd.DataFrame): The input dataset (including features and optionally the label column).
        label_column (str, optional): The name of the label column in the dataset. If provided, it will be included in the output table.

    Returns:
        pd.DataFrame: A DataFrame with predictions and the original labels (if provided).
    """
    url = 'https://dbc-a3af8ba1-8a92.cloud.databricks.com/model/XGBoost_Regressor_1/1/invocations'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Separate features and label if the label column is provided
    features = dataset.drop(columns=[label_column]) if label_column else dataset
    labels = dataset[label_column] if label_column else None

    # Prepare the input in the `dataframe_split` format
    ds_dict = {"dataframe_split": features.to_dict(orient="split")}
    data_json = json.dumps(ds_dict)

    # Send the scoring request
    response = requests.post(url, headers=headers, data=data_json)

    # Check response status
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    # Parse predictions
    predictions = response.json()["predictions"]

    # Combine predictions with the original data and label (if present)
    output_data = dataset.copy()
    output_data["prediction"] = predictions

    return output_data


# Load your data
table_test = pd.read_csv('./table_test.csv')
# table_whole = pd.read_csv('./table_whole.csv')
table_train = pd.read_csv('./table_train.csv')

token = st.text_input("Enter your token:", key="Token", type="password")

option = st.radio(
    "Choose an option:",
    ("Original", "Changed")
)

# 根据用户选择执行不同操作
if option == "Original":
    st.write("Original Dataset")
    # 在这里添加处理 Original 的逻辑
else:
    st.write("Change minimum_nights and number_of_reviews to random number")
    # 在这里添加处理 Changed 的逻辑

if token:

    if option == "Changed":
        table_test['minimum_nights'] = np.random.randint(table_test['minimum_nights'].min(), table_test['minimum_nights'].max() + 1, size=len(table_test))
        table_test['number_of_reviews'] = np.random.randint(table_test['number_of_reviews'].min(), table_test['number_of_reviews'].max() + 1, size=len(table_test))
    
    # Score the models
    test_predictions = score_model_with_labels(table_test, label_column='price')
    # whole_predictions = score_model_with_labels(table_whole, label_column='price')
    train_predictions = score_model_with_labels(table_train, label_column='price')

    # Display the predictions
    st.dataframe(test_predictions)
    
    test_predictions.rename(columns={'price': 'target'}, inplace=True)
    # whole_predictions.rename(columns={'price': 'target'}, inplace=True)
    train_predictions.rename(columns={'price': 'target'}, inplace=True)
    report = monitor_model(reference_data=train_predictions, current_data=test_predictions)

    report.save_html('DD.html')
    # Load the HTML file
    html_file_path = "DD.html"  # Replace with the path to your HTML file
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Display the HTML content in Streamlit
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.warning("Please enter your token to proceed.")

