import pandas as pd
import streamlit as st
import pickle
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

st.title("ML Model Analysis")
st.write("Upload a CSV file to analyze the dataset and build ML models.")
file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")


def clean_data(file):
    if file is None:
        return None

    file.seek(0)
    df = pd.read_csv(file)

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
            else:
                mode_values = df[column].mode()
                if not mode_values.empty:
                    df[column] = df[column].fillna(mode_values[0])
                else:
                    df[column] = df[column].fillna("unknown")

    df = df.drop_duplicates()

    for column in list(df.columns):
        if df[column].nunique() <= 1:
            df = df.drop(columns=[column])
        elif df[column].nunique() == 2 and df[column].dtype == 'object':
            unique_values = df[column].dropna().unique()
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            df[column] = df[column].map(mapping)

    return df


def build_demo_row(df, target_column):
    if df is None or df.empty:
        return pd.DataFrame()

    demo_row = {}
    for col in df.columns:
        if col == target_column:
            continue

        values = df[col].dropna()
        if values.empty:
            demo_row[col] = 0
        elif pd.api.types.is_numeric_dtype(values):
            demo_row[col] = float(values.median())
        else:
            mode_value = values.mode()
            demo_row[col] = mode_value.iloc[0] if not mode_value.empty else "unknown"

    return pd.DataFrame([demo_row])


def predict_demo_sample(model_result):
    if model_result is None or "model" not in model_result:
        return None

    demo_df = build_demo_row(model_result["source_df"], model_result["target_column"])
    if demo_df.empty:
        return None

    demo_features = pd.get_dummies(demo_df, drop_first=True)
    demo_features = demo_features.reindex(columns=model_result["feature_columns"], fill_value=0)
    prediction = model_result["model"].predict(demo_features)
    prediction_value = prediction[0]

    if model_result.get("target_classes") is not None:
        return model_result["target_classes"][int(prediction_value)]

    return prediction_value


df = clean_data(file)

st.write("The dataset has been analyzed. You can now proceed to build ML models.")
st.write("Choose the type of problem(Regression or Classification):")
problem_type = st.selectbox("Select a problem type", ["Regression", "Classification"], key="problem_type_selector")
if problem_type == "Regression":
    st.write("You have selected Regression. You can now choose a model to build.")
    model_type = st.radio("Select a model", options=["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor"], key="model_selector")
else:
    st.write("You have selected Classification. You can now choose a model to build.")
    model_type = st.radio("Select a model", options=["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"], key="model_selector")

target_column = None
if df is not None and not df.empty:
    target_column = st.selectbox("Select target column", options=list(df.columns), key="target_column_selector")


def build_model(model_type, df, target_column):
    if df is None or df.empty:
        st.error("Please upload a valid CSV file before building a model.")
        return None

    if target_column is None or target_column not in df.columns:
        st.error("Please select a valid target column.")
        return None

    df_model = df.copy()
    X = df_model.drop(target_column, axis=1)
    y = df_model[target_column]

    X = pd.get_dummies(X, drop_first=True)
    target_classes = None

    if not pd.api.types.is_numeric_dtype(y):
        y, target_classes = pd.factorize(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": math.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100,
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "source_df": df_model,
        }
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": math.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100,
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "source_df": df_model,
        }
    elif model_type == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": math.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100,
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "source_df": df_model,
        }
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "source_df": df_model,
            "target_classes": list(target_classes) if target_classes is not None else None,
        }
    elif model_type == "Random Forest Classifier":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "source_df": df_model,
            "target_classes": list(target_classes) if target_classes is not None else None,
        }
    elif model_type == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "feature_columns": X.columns.tolist(),
            "target_column": target_column,
            "source_df": df_model,
            "target_classes": list(target_classes) if target_classes is not None else None,
        }

    return None


if st.button("Build Model"):
    result = build_model(model_type, df, target_column)
    if result is not None:
        st.session_state["trained_model"] = result["model"]
        st.session_state["metrics"] = {key: value for key, value in result.items() if key not in {"model", "feature_columns", "target_column", "source_df", "target_classes"}}
        st.session_state["demo_row"] = build_demo_row(result["source_df"], result["target_column"])
        st.session_state["demo_prediction"] = predict_demo_sample(result)
        st.session_state["demo_model_info"] = {
            "model_type": model_type,
            "problem_type": problem_type,
            "target_column": target_column,
        }
        st.write(st.session_state["metrics"])

        if st.session_state["demo_row"].empty is False and st.session_state["demo_prediction"] is not None:
            st.subheader("Demo Prediction")
            st.dataframe(st.session_state["demo_row"], use_container_width=True)
            if isinstance(st.session_state["demo_prediction"], (int, float)):
                st.metric("Predicted value", round(float(st.session_state["demo_prediction"]), 4))
            else:
                st.write(f"Predicted class: {st.session_state['demo_prediction']}")

st.write("You can now download the model or make predictions on new data.")
st.write("To download the model, click the button below.")
if "trained_model" in st.session_state:
    model_bytes = pickle.dumps(st.session_state["trained_model"])
    st.download_button(
        "Download Model",
        data=model_bytes,
        file_name="model.pkl",
        mime="application/octet-stream",
    )
