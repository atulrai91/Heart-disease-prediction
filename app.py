import streamlit as st
import pandas as pd
import pickle
import os
import glob

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)


st.set_page_config(
    page_title="Cardiovascular disease prediction",
    layout="wide"
)

st.title("Heart Disease Prediction Model")

MODEL_FOLDER = "model"


label_encoder = pickle.load(
    open(f"{MODEL_FOLDER}/label_encoder.pkl", "rb")
)


X_train, X_test, y_train, y_test = pickle.load(
    open(f"{MODEL_FOLDER}/train_test_data.pkl", "rb")
)


test_dataset = X_test.copy()
test_dataset["Target"] = y_test


model_files = glob.glob(os.path.join(MODEL_FOLDER, "*.pkl"))

models = {}

for file in model_files:

    name = os.path.basename(file)

    if name in ["label_encoder.pkl", "train_test_data.pkl"]:
        continue

    data = pickle.load(open(file, "rb"))
    models[name.replace(".pkl", "")] = data["model"]


st.sidebar.header("Dataset Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

user_df = None


st.sidebar.subheader("Download Evaluation Dataset")

test_csv = test_dataset.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    "⬇ Download Test Dataset Used by Models",
    test_csv,
    "evaluation_test_dataset.csv",
    "text/csv"
)


if uploaded_file is not None:

    user_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(user_df.head())

    
    uploaded_csv = user_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Uploaded Dataset",
        uploaded_csv,
        "uploaded_dataset.csv",
        "text/csv"
    )


st.header("Model Prediction")

selected_model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[selected_model_name]

st.success(f"Currently Evaluating: {selected_model_name}")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)


try:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = "Not Available"
except:
    auc = "Not Available"


st.subheader("Performance Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")

col4, col5, col6 = st.columns(3)

col4.metric("F1 Score", f"{f1:.4f}")
col5.metric(
    "AUC Score",
    auc if isinstance(auc, str) else f"{auc:.4f}"
)
col6.metric("Matthews Corrcoef", f"{mcc:.4f}")


st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig = plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {selected_model_name}")

st.pyplot(fig)


if user_df is not None:

    st.header("🤖 Predict Using Uploaded Dataset")

    try:
        predictions = model.predict(user_df)

        user_df["Prediction"] = label_encoder.inverse_transform(predictions)

        st.dataframe(user_df.head())

        prediction_csv = user_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Predictions",
            prediction_csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Dataset format does not match training features.")
        st.write(e)
