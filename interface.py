import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import plotly.express as px
import joblib

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the uploaded data."""
    data = pd.read_csv(uploaded_file)
    
    # Drop unnecessary columns
    columns_to_drop = ['transaction_status', 'user_segment', 'customer_ip_location']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)
    
    return data

def train_model(train_data):
    """Train the fraud detection model."""
    # Define feature groups
    numeric_features = ['cart_size', 'transaction_amount', 'session_duration', 
                       'time_spent_on_payment_page', 'card_expiration_delay', 
                       'time_since_account_creation', 'ip_address_previous_transactions', 
                       'product_views_during_session']

    categorical_features = ['transaction_type', 'customer_age_group', 'payment_method', 
                           'login_status', 'visit_origin', 'device_type']
    
    # Prepare features and target
    X = train_data.drop('flag', axis=1)
    y = train_data['flag']
    
    # Impute missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    
    # Encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    
    # Convert encoded arrays to DataFrames
    encoded_columns = encoder.get_feature_names_out(categorical_features)
    X_encoded = pd.DataFrame(X_encoded, columns=encoded_columns, index=X.index)
    
    # Combine numeric and encoded categorical features
    X_final = pd.concat([X[numeric_features], X_encoded], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Train model
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_resampled, y_resampled)
    
    return model, numeric_imputer, categorical_imputer, encoder, scaler

def make_predictions(model, test_data, numeric_imputer, categorical_imputer, encoder, scaler):
    """Make predictions on test data."""
    numeric_features = ['cart_size', 'transaction_amount', 'session_duration', 
                       'time_spent_on_payment_page', 'card_expiration_delay', 
                       'time_since_account_creation', 'ip_address_previous_transactions', 
                       'product_views_during_session']

    categorical_features = ['transaction_type', 'customer_age_group', 'payment_method', 
                           'login_status', 'visit_origin', 'device_type']
    
    # Preprocess test data
    test_data[numeric_features] = numeric_imputer.transform(test_data[numeric_features])
    test_data[categorical_features] = categorical_imputer.transform(test_data[categorical_features])
    
    # Encode categorical variables
    X_test_encoded = encoder.transform(test_data[categorical_features])
    encoded_columns = encoder.get_feature_names_out(categorical_features)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=test_data.index)
    
    # Combine features
    X_test_final = pd.concat([test_data[numeric_features], X_test_encoded], axis=1)
    
    # Scale features
    X_test_scaled = scaler.transform(X_test_final)
    
    # Make predictions
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions, probabilities

def main():
    st.title("Fraud Detection System")
    st.write("Upload your data to detect potentially fraudulent transactions")
    
    # File upload
    train_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'], key='train')
    test_file = st.file_uploader("Upload Test Data (CSV)", type=['csv'], key='test')
    
    if train_file and test_file:
        # Load data
        with st.spinner('Loading and preprocessing data...'):
            train_data = load_and_preprocess_data(train_file)
            test_data = load_and_preprocess_data(test_file)
            
            # Train model
            st.write("Training model...")
            model, numeric_imputer, categorical_imputer, encoder, scaler = train_model(train_data)
            
            # Make predictions
            predictions, probabilities = make_predictions(
                model, test_data, numeric_imputer, categorical_imputer, encoder, scaler
            )
            
            # Create results DataFrame
            results = pd.DataFrame({
                'transaction_id': test_data['transaction_id'],
                'probability': probabilities,
                'prediction': predictions
            })
            
            # Display results
            st.subheader("Results")
            st.write(f"Total transactions analyzed: {len(results)}")
            st.write(f"Flagged transactions: {predictions.sum()}")
            st.write(f"Percentage flagged: {(predictions.sum() / len(predictions)) * 100:.2f}%")
            
            # Plot distribution of fraud probabilities
            fig = px.histogram(
                results, 
                x='probability', 
                title='Distribution of Fraud Probabilities',
                nbins=50
            )
            st.plotly_chart(fig)
            
            # Allow downloading results
            st.download_button(
                label="Download predictions",
                data=results.to_csv(index=False).encode('utf-8'),
                file_name='fraud_predictions.csv',
                mime='text/csv'
            )
            
            # Display sample of predictions
            st.subheader("Sample Predictions")
            st.write(results.head(10))
            
    else:
        st.info("Please upload both training and test data files to begin analysis")

if __name__ == "__main__":
    main()