import pandas as pd
import streamlit as st
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
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

st.set_page_config(page_title="ML Model Analysis", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 30%, #1e293b 100%);
            color: #e2e8f0;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stFileUploader"] > section {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 18px;
            padding: 0.5rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
        }
        .block-container h1 {
            color: #f8fafc;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .hero-box {
            background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.16));
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 22px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
            margin-bottom: 1.2rem;
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.2);
        }
        .metric-label {
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #cbd5e1;
        }
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            font-weight: 700;
            padding: 0.7rem 1rem;
        }
        .stDownloadButton > button {
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.9);
            color: #e2e8f0;
            border: 1px solid rgba(148, 163, 184, 0.4);
            font-weight: 600;
        }
        .stDataFrame {
            background: rgba(15, 23, 42, 0.4);
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <h1>ML Model Analysis</h1>
        <p style="color:#cbd5e1; margin:0; font-size:1.05rem;">
            Upload a dataset, train a machine learning model, and preview a demo prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
st.session_state["original_df"] = df.copy() if df is not None else None
st.session_state["working_df"] = df.copy() if df is not None else None

with st.container():
    col_left, col_right = st.columns([1.6, 1])

    current_df = st.session_state.get("working_df", df)

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("Choose the type of problem and model you want to evaluate.")
        problem_type = st.selectbox("Select a problem type", ["Regression", "Classification"], key="problem_type_selector")
        if problem_type == "Regression":
            st.write("Regression mode is active.")
            model_type = st.radio("Select a model", options=["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor"], key="model_selector")
        else:
            st.write("Classification mode is active.")
            model_type = st.radio("Select a model", options=["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"], key="model_selector")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if current_df is not None and not current_df.empty:
            target_column = st.selectbox("Select target column", options=list(current_df.columns), key="target_column_selector")
            st.caption(f"Rows: {len(current_df)} | Columns: {len(current_df.columns)}")
            if target_column:
                st.caption(f"Target selected: {target_column}")
        else:
            target_column = None
            st.caption("Upload a valid CSV to unlock target selection.")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if df is not None and not df.empty:
    st.success("Dataset loaded successfully. You can proceed to train the selected model.")
else:
    st.info("Upload a CSV file to begin analysis.")


def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return q1, q3, iqr, lower_bound, upper_bound, outliers


if "working_df" not in st.session_state:
    st.session_state["working_df"] = df


if df is not None and not df.empty:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Outlier Visualization")

        outlier_col = st.selectbox("Select a numeric column", numeric_columns, key="outlier_col")
        chart_type = st.radio(
            "Choose graph type",
            ["Box Plot", "Histogram", "Scatter Plot", "Violin Plot"],
            horizontal=True,
            key="outlier_chart_type",
        )

        values = df[outlier_col].dropna()
        q1, q3, iqr, lower_bound, upper_bound, outliers = detect_outliers(values)

        if chart_type == "Box Plot":
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.boxplot(
                values,
                patch_artist=True,
                boxprops=dict(facecolor="#60a5fa", alpha=0.8),
                medianprops=dict(color="#facc15", linewidth=2),
                whiskerprops=dict(color="#cbd5e1"),
                capprops=dict(color="#cbd5e1"),
                flierprops=dict(marker='o', markerfacecolor="#f87171", markeredgecolor="#f87171", markersize=7),
            )
            ax.set_title(f"{outlier_col} - Box Plot")
            ax.set_ylabel(outlier_col)
            plt.tight_layout()
            st.pyplot(fig)

        elif chart_type == "Histogram":
            fig, ax = plt.subplots(figsize=(9, 4.5))
            bins = min(20, max(5, len(values) // 5))
            ax.hist(values, bins=bins, color="#7dd3fc", edgecolor="white", alpha=0.9)
            ax.axvline(lower_bound, color="#f87171", linestyle="--", linewidth=2, label=f"Lower: {lower_bound:.2f}")
            ax.axvline(upper_bound, color="#f87171", linestyle="--", linewidth=2, label=f"Upper: {upper_bound:.2f}")
            ax.axvline(values.median(), color="#facc15", linestyle="-", linewidth=2, label=f"Median: {values.median():.2f}")
            ax.set_title(f"{outlier_col} - Histogram with IQR Boundaries")
            ax.set_xlabel(outlier_col)
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        elif chart_type == "Scatter Plot":
            fig, ax = plt.subplots(figsize=(9, 4.5))
            x = np.arange(len(values))
            colors = np.where((values < lower_bound) | (values > upper_bound), "#f87171", "#60a5fa")
            ax.scatter(x, values, c=colors, s=35, alpha=0.8)
            ax.axhline(lower_bound, color="#f87171", linestyle="--", linewidth=2, label=f"Lower: {lower_bound:.2f}")
            ax.axhline(upper_bound, color="#f87171", linestyle="--", linewidth=2, label=f"Upper: {upper_bound:.2f}")
            ax.axhline(values.median(), color="#facc15", linestyle="-", linewidth=2, label=f"Median: {values.median():.2f}")
            ax.set_title(f"{outlier_col} - Value Scatter")
            ax.set_xlabel("Observation index")
            ax.set_ylabel(outlier_col)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        else:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            parts = ax.violinplot(values, positions=[1], widths=0.7, showmeans=True)
            for body in parts['bodies']:
                body.set_facecolor('#60a5fa')
                body.set_alpha(0.8)
                body.set_edgecolor('#1d4ed8')
            ax.set_title(f"{outlier_col} - Violin Plot")
            ax.set_xticks([1])
            ax.set_xticklabels([outlier_col])
            ax.set_ylabel(outlier_col)
            plt.tight_layout()
            st.pyplot(fig)

        if st.checkbox("Compare original and cleaned distribution", key="compare_cleaned_distribution"):
            cleaned_values = values[(values >= lower_bound) & (values <= upper_bound)]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.hist(values, bins=min(20, max(5, len(values) // 5)), alpha=0.6, color="#60a5fa", label="Original")
            ax.hist(cleaned_values, bins=min(20, max(5, len(cleaned_values) // 5)), alpha=0.6, color="#34d399", label="Without outliers")
            ax.axvline(lower_bound, color="#f87171", linestyle="--", linewidth=2, label=f"Lower: {lower_bound:.2f}")
            ax.axvline(upper_bound, color="#f87171", linestyle="--", linewidth=2, label=f"Upper: {upper_bound:.2f}")
            ax.set_title(f"{outlier_col} - Original vs Cleaned Distribution")
            ax.set_xlabel(outlier_col)
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        if outliers.empty:
            st.success(f"No outliers detected in {outlier_col} using the IQR rule.")
            if st.button("Reset to original dataset", key="reset_outliers_button"):
                st.session_state["working_df"] = st.session_state.get("original_df", df).copy()
                st.rerun()
        else:
            st.warning(f"Detected {len(outliers)} outliers in {outlier_col} outside [{lower_bound:.2f}, {upper_bound:.2f}].")
            st.write(outliers.head(10))

            remove_col, reset_col = st.columns(2)
            with remove_col:
                if st.button("Remove outliers from dataset", key="remove_outliers_button"):
                    original_df = st.session_state.get("original_df", df)
                    cleaned_df = original_df[(original_df[outlier_col] >= lower_bound) & (original_df[outlier_col] <= upper_bound)].copy()
                    st.session_state["working_df"] = cleaned_df
                    st.success(f"Outliers removed from {outlier_col}. The cleaned dataset is now ready for model training.")
                    st.write(f"Rows before: {len(original_df)} | Rows after: {len(cleaned_df)}")
            with reset_col:
                if st.button("Reset to original dataset", key="reset_after_remove_button"):
                    st.session_state["working_df"] = st.session_state.get("original_df", df).copy()
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


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


st.markdown("<br>", unsafe_allow_html=True)

build_col, info_col = st.columns([1.3, 0.7])
with build_col:
    if st.button("Build Model", use_container_width=True):
        training_df = st.session_state.get("working_df", df)
        if training_df is None:
            training_df = df
        result = build_model(model_type, training_df, target_column)
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

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Model Metrics")
            st.write(st.session_state["metrics"])
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state["demo_row"].empty is False and st.session_state["demo_prediction"] is not None:
                st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
                st.subheader("Demo Prediction")
                st.dataframe(st.session_state["demo_row"], use_container_width=True)
                if isinstance(st.session_state["demo_prediction"], (int, float)):
                    st.metric("Predicted value", round(float(st.session_state["demo_prediction"]), 4))
                else:
                    st.write(f"Predicted class: {st.session_state['demo_prediction']}")
                st.markdown('</div>', unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Status")
    st.write("Ready to train the selected model.")
    st.write("After training, a demo prediction will be shown here.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if "trained_model" in st.session_state:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("You can now download the model or make predictions on new data.")
    model_bytes = pickle.dumps(st.session_state["trained_model"])
    st.download_button(
        "Download Model",
        data=model_bytes,
        file_name="model.pkl",
        mime="application/octet-stream",
    )
    st.markdown('</div>', unsafe_allow_html=True)
