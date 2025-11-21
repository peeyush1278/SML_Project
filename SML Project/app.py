import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, mean_absolute_error)
import io

# Page configuration
st.set_page_config(page_title="ML Operations Dashboard", layout="wide", page_icon="🤖")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Machine Learning Operations Dashboard")
st.markdown("Upload your CSV file and perform comprehensive ML operations")

# Sidebar
st.sidebar.header("📊 Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Data Overview", "🔍 EDA", "🤖 Model Training", "📊 Results", "💾 Export"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
        with col4:
            st.metric("Categorical Columns", df.select_dtypes(include=['object']).shape[1])
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), width='stretch')
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, width='stretch')
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), width='stretch')
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Missing values
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data[missing_data > 0].sort_values(ascending=False).plot(kind='bar', ax=ax)
            ax.set_title("Missing Values by Column")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.success("No missing values found!")
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title("Feature Correlation Matrix")
            st.pyplot(fig)
        
        # Distribution plots
        st.subheader("Feature Distributions")
        selected_col = st.selectbox("Select column for distribution", numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            df[selected_col].hist(bins=30, ax=ax, edgecolor='black')
            ax.set_title(f"Histogram of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            df.boxplot(column=selected_col, ax=ax)
            ax.set_title(f"Box Plot of {selected_col}")
            ax.set_ylabel(selected_col)
            st.pyplot(fig)
    
    with tab3:
        st.header("Model Training")
        
        # Problem type selection
        problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])
        
        # Feature and target selection
        st.subheader("Feature Selection")
        all_columns = df.columns.tolist()
        target_column = st.selectbox("Select Target Variable", all_columns)
        
        feature_columns = st.multiselect(
            "Select Feature Variables",
            [col for col in all_columns if col != target_column],
            default=[col for col in all_columns if col != target_column][:5]
        )
        
        if len(feature_columns) > 0 and target_column:
            # Prepare data
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.info(f"Encoding categorical columns: {', '.join(categorical_cols)}")
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target if classification and categorical
            if problem_type == "Classification" and y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
            
            # Train-test split
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Feature scaling
            scale_features = st.checkbox("Apply Feature Scaling", value=True)
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Model selection
            st.subheader("Model Selection")
            
            if problem_type == "Classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Naive Bayes": GaussianNB()
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "SVR": SVR(),
                    "K-Nearest Neighbors": KNeighborsRegressor()
                }
            
            selected_model = st.selectbox("Select Model", list(models.keys()))
            
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    model = models[selected_model]
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Store in session state
                    st.session_state['model'] = model
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred_train'] = y_pred_train
                    st.session_state['y_pred_test'] = y_pred_test
                    st.session_state['problem_type'] = problem_type
                    st.session_state['feature_columns'] = feature_columns
                    st.session_state['target_column'] = target_column
                    
                    st.success("✅ Model trained successfully!")
    
    with tab4:
        st.header("Model Results")
        
        if 'model' in st.session_state:
            problem_type = st.session_state['problem_type']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            y_pred_train = st.session_state['y_pred_train']
            y_pred_test = st.session_state['y_pred_test']
            
            if problem_type == "Classification":
                # Metrics
                st.subheader("Classification Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Training Set**")
                    train_acc = accuracy_score(y_train, y_pred_train)
                    st.metric("Accuracy", f"{train_acc:.4f}")
                    st.metric("Precision", f"{precision_score(y_train, y_pred_train, average='weighted'):.4f}")
                    st.metric("Recall", f"{recall_score(y_train, y_pred_train, average='weighted'):.4f}")
                    st.metric("F1-Score", f"{f1_score(y_train, y_pred_train, average='weighted'):.4f}")
                
                with col2:
                    st.markdown("**Test Set**")
                    test_acc = accuracy_score(y_test, y_pred_test)
                    st.metric("Accuracy", f"{test_acc:.4f}")
                    st.metric("Precision", f"{precision_score(y_test, y_pred_test, average='weighted'):.4f}")
                    st.metric("Recall", f"{recall_score(y_test, y_pred_test, average='weighted'):.4f}")
                    st.metric("F1-Score", f"{f1_score(y_test, y_pred_test, average='weighted'):.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix (Test Set)")
                cm = confusion_matrix(y_test, y_pred_test)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_ylabel("Actual")
                ax.set_xlabel("Predicted")
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
            
            else:  # Regression
                st.subheader("Regression Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Training Set**")
                    st.metric("MSE", f"{mean_squared_error(y_train, y_pred_train):.4f}")
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
                    st.metric("MAE", f"{mean_absolute_error(y_train, y_pred_train):.4f}")
                    st.metric("R² Score", f"{r2_score(y_train, y_pred_train):.4f}")
                
                with col2:
                    st.markdown("**Test Set**")
                    st.metric("MSE", f"{mean_squared_error(y_test, y_pred_test):.4f}")
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
                    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred_test):.4f}")
                    st.metric("R² Score", f"{r2_score(y_test, y_pred_test):.4f}")
                
                # Prediction vs Actual plot
                st.subheader("Predictions vs Actual Values (Test Set)")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred_test, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
                
                # Residuals plot
                st.subheader("Residuals Plot")
                residuals = y_test - y_pred_test
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_pred_test, residuals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Residuals")
                ax.set_title("Residual Plot")
                st.pyplot(fig)
        else:
            st.info("Please train a model first in the 'Model Training' tab.")
    
    with tab5:
        st.header("Export Results")
        
        if 'model' in st.session_state:
            st.subheader("Download Predictions")
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'Actual': st.session_state['y_test'],
                'Predicted': st.session_state['y_pred_test']
            })
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
            st.subheader("Model Summary")
            summary = f"""
            Model Type: {selected_model if 'selected_model' in locals() else 'N/A'}
            Problem Type: {st.session_state['problem_type']}
            Features Used: {', '.join(st.session_state['feature_columns'])}
            Target Variable: {st.session_state['target_column']}
            Training Samples: {len(st.session_state['y_train'])}
            Test Samples: {len(st.session_state['y_test'])}
            """
            st.text_area("Model Information", summary, height=200)
        else:
            st.info("Please train a model first to export results.")

else:
    st.info("👆 Please upload a CSV file to get started")
    st.markdown("""
    ### Features:
    - 📊 **Data Overview**: View dataset statistics and information
    - 🔍 **EDA**: Exploratory data analysis with visualizations
    - 🤖 **Model Training**: Train multiple ML models
    - 📈 **Results**: View detailed metrics and visualizations
    - 💾 **Export**: Download predictions and model summary
    
    ### Supported Models:
    - **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes
    - **Regression**: Linear Regression, Decision Tree, Random Forest, SVR, KNN
    """)