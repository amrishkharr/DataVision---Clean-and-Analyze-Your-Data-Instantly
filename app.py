import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="DataVision", layout="wide", page_icon="📊")

uploaded_file = st.sidebar.file_uploader("📂  Upload Your Dataset", type=["csv"])

# Only show title/description if file is not uploaded
if uploaded_file is None:
    st.title("🔍 DataVision - 📊 Clean and Analyze Your Data Instantly")
    st.markdown("""
    This Web app allows you to:
    - Upload your own dataset (CSV)
    - Automatically clean missing data
    - Remove unwanted columns
    - Explore and visualize data
    """)

if uploaded_file is not None:
    progress = st.progress(0, text="📥 Loading CSV file...")
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully")
    progress.progress(20, text="📊 Checking data size...")

    if len(df) > 1000:
        st.warning(f"Large dataset detected ({len(df)} rows). Visualizations may be limited for performance.")

    progress.progress(40, text="🔎 Checking missing values...")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    cleaned_rows = df.dropna().shape[0]

    progress.progress(60, text="🧹 Dropping rows with missing data...")
    df.dropna(inplace=True)

    progress.progress(80, text="🧼 Removing selected columns...")

    # Column removal feature
    with st.sidebar.expander("🧯 Remove Columns"):
        drop_cols = st.multiselect("Select columns to remove", options=df.columns)
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            st.sidebar.info(f"Removed columns: {', '.join(drop_cols)}")

    progress.progress(100, text="✅ Data cleaned and ready!")
    st.success("✅ Data successfully loaded and cleaned!")

    with st.sidebar.expander("✂️ Trim Data Rows"):
        start_idx = st.number_input("Start Row Index", min_value=0, max_value=len(df)-1, value=0)
        end_idx = st.number_input("End Row Index", min_value=start_idx+1, max_value=len(df), value=len(df))
        if start_idx < end_idx:
            df = df.iloc[start_idx:end_idx]
            st.sidebar.info(f"Trimmed data from row {start_idx} to {end_idx}")

    target_col = "species" if "species" in df.columns else None
    if target_col:
        df["species_encoded"] = LabelEncoder().fit_transform(df[target_col])
        y = df["species_encoded"]
        X = df.drop([target_col, "species_encoded"], axis=1)
    else:
        y = None
        X = df.select_dtypes(include=['float64', 'int'])

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        @st.cache_resource
        def train_model():
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            return model

        with st.spinner("🧠 Training the model..."):
            model = train_model()
        st.success("✅ Model trained successfully!")

    with st.sidebar:
        st.header("📂 Menu")
        section = st.radio("Navigation", ["EDA", "Model Report", "Predict"], index=0)
        st.markdown("---\n**🛠️ Developed by**  \nㅤㅤㅤ-**Amrishkhar R** [[GitHub](https://github.com/amrishkharr)]")

    if section == "EDA":
        # Top section: Overview and Missing Values side-by-side
        top1, top2 = st.columns(2)
        with top1:
            st.header("📋 Dataset Overview")
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
            st.metric("Rows after Cleaning", cleaned_rows)

        with top2:
            st.header("🧹 Missing Values")
            st.dataframe(pd.DataFrame({
                "Missing Count": missing_values,
                "Missing %": missing_percent.round(2)
            }), use_container_width=True)

        # Main section: Categorical Frequencies and EDA Visualizations
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.header("📊 Categorical Frequencies")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                with st.expander(f"Value Counts for '{col}'"):
                    st.dataframe(df[col].value_counts(), use_container_width=True)

        with right_col:
            st.header("📊 Exploratory Data Analysis")
            numeric_cols = df.select_dtypes(include=['float64', 'int']).columns

            if st.checkbox("📋 Show Data Table"):
                st.dataframe(df, use_container_width=True)

            with st.expander("📈 Statistical Summary"):
                st.dataframe(df.describe(), use_container_width=True)

            with st.expander("📊 Correlation Heatmap"):
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="YlGnBu", ax=ax)
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate heatmap: {e}")

            with st.expander("📉 Line Chart"):
                st.line_chart(df.select_dtypes(include=['float64', 'int']), use_container_width=True)

            with st.expander("📊 Histograms of Numeric Features"):
                for col in numeric_cols[:6]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[col], kde=True, ax=ax)
                    ax.set_title(f'Histogram and KDE of {col}')
                    st.pyplot(fig, use_container_width=True)

            with st.expander("📊 Scatter Plots for Selected Feature Pairs"):
                for i in range(0, min(len(numeric_cols) - 1, 5), 2):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x_col, y_col = numeric_cols[i], numeric_cols[i+1]
                    sns.scatterplot(data=df, x=x_col, y=y_col, hue=target_col if target_col else None, ax=ax)
                    ax.set_title(f'Scatter plot of {x_col} vs {y_col}')
                    st.pyplot(fig, use_container_width=True)

            with st.expander("📊 Count Plots for Categorical Features"):
                for col in categorical_cols:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=df, x=col, ax=ax)
                    ax.set_title(f'Count plot of {col}')
                    st.pyplot(fig, use_container_width=True)

            if target_col:
                with st.expander("📦 Boxplots Grouped by Target Category"):
                    for col in numeric_cols[:6]:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(data=df, x=target_col, y=col, ax=ax)
                        ax.set_title(f'Boxplot of {col} grouped by {target_col}')
                        st.pyplot(fig, use_container_width=True)

            with st.expander("📊 Density (KDE) Plots of Numeric Features"):
                for col in numeric_cols[:6]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.kdeplot(df[col], fill=True, ax=ax)
                    ax.set_title(f'Density plot of {col}')
                    st.pyplot(fig, use_container_width=True)

            with st.expander("🟡 Pairplot (Sampled to 500 Rows)"):
                try:
                    st.pyplot(sns.pairplot(df.sample(min(len(df), 500)), hue=target_col if target_col else None))
                except Exception as e:
                    st.warning(f"Could not generate pairplot: {e}")

    elif section == "Model Report":
        if y is not None:
            st.subheader("🧠 Model Performance")
            y_pred = model.predict(X_test)
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.write("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig, use_container_width=True)
        else:
            st.error("No target column ('species') available in uploaded data to train a model.")

    elif section == "Predict":
        st.subheader("🔮 Make a Prediction")
        if y is not None:
            col1, col2 = st.columns(2)
            input_data = []
            for i, col in enumerate(X.columns):
                with col1 if i % 2 == 0 else col2:
                    val = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                    input_data.append(val)

            input_df = pd.DataFrame([input_data], columns=X.columns)

            with st.spinner("🔍 Making prediction..."):
                prediction = model.predict(input_df)[0]
            st.success("✅ Prediction complete!")

            try:
                label_decoded = LabelEncoder().fit(df[target_col]).inverse_transform([prediction])[0]
                st.success(f"🌼 The predicted class is: **{label_decoded}**")
            except:
                st.success(f"Predicted class: {prediction}")
        else:
            st.error("Upload data with a target column named 'species' to use the prediction feature.")
else:
    st.info("📤 Please upload a Dataset to begin analysis.")
