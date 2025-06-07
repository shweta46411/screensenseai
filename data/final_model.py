import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
# from datetime import timedelta # Already imported previously

# --- Configuration ---
# Paths to your newly saved FINAL model components
FINAL_MODEL_PATH = "dataset/final_hybrid_cnn_lstm_sleep_model.keras"
FINAL_SCALER_PATH = "dataset/final_sleep_scaler.joblib"
FINAL_METADATA_PATH = "dataset/final_sleep_model_metadata.json"
DATA_PATH = "dataset/master_daily_data_cleaned_for_sleep_model.csv" # The CLEANED data

# --- Load Model, Scaler, and Metadata ---
@st.cache_resource # Cache Keras model and scaler
def load_final_model_artifacts():
    try:
        model = tf.keras.models.load_model(FINAL_MODEL_PATH)
        scaler = joblib.load(FINAL_SCALER_PATH)
        with open(FINAL_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        st.success("Final model, scaler, and metadata loaded successfully!")
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading final model/scaler/metadata: {e}")
        return None, None, None

@st.cache_data # Cache data loading
def load_main_cleaned_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        if 'uid' in df.columns: # Ensure uid is string for consistency
            df['uid'] = df['uid'].astype(str)
        return df.sort_values(by=['uid', 'date'])
    except FileNotFoundError:
        st.error(f"Error: Cleaned data file '{DATA_PATH}' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading cleaned data: {e}")
        return pd.DataFrame()

# Load everything
final_model, final_scaler, final_metadata = load_final_model_artifacts()
master_cleaned_df = load_main_cleaned_data()

if final_model is None or final_scaler is None or final_metadata is None or master_cleaned_df.empty:
    st.error("App initialization failed. Check file paths and ensure all components are available.")
    st.stop()

# Extract metadata
FEATURE_COLUMNS = final_metadata.get("feature_columns", [])
SEQUENCE_LENGTH = final_metadata.get("sequence_length", 7)
OPTIMAL_THRESHOLD = final_metadata.get("optimal_threshold", 0.5) # This will be your ~0.5065
TARGET_NAMES = final_metadata.get("target_names", ['Poor Sleep (0)', 'Good Sleep (1)'])

if not FEATURE_COLUMNS:
    st.error("Feature columns not found in loaded metadata. Cannot proceed.")
    st.stop()

st.title("Individual Sleep Prediction Dashboard 🛌 (Hybrid CNN-LSTM)")
st.write(f"Using model: {final_metadata.get('model_type', 'Hybrid CNN-LSTM')}")
st.write(f"Prediction threshold for '{TARGET_NAMES[1]}': {OPTIMAL_THRESHOLD:.4f}")


# --- User Selection ---
st.sidebar.header("Select User")
available_uids_in_cleaned_data = sorted(master_cleaned_df['uid'].unique())
if not available_uids_in_cleaned_data:
    st.error("No users found in the cleaned data file.")
    st.stop()
    
selected_uid = st.sidebar.selectbox("User ID (uid):", available_uids_in_cleaned_data)

user_df = master_cleaned_df[master_cleaned_df['uid'] == selected_uid].copy()

if user_df.empty:
    st.warning(f"No data found for user: {selected_uid}")
    st.stop()

st.header(f"Sleep Prediction for User: {selected_uid}")

# --- Prepare Data for Prediction ---
latest_day_available = user_df['date'].max()
st.write(f"Latest available data for this user is from: {latest_day_available.strftime('%Y-%m-%d')}")
st.write(f"Predicting sleep category for the night *after* {latest_day_available.strftime('%Y-%m-%d')}.")

if len(user_df) >= SEQUENCE_LENGTH:
    sequence_data_df = user_df.tail(SEQUENCE_LENGTH)
    
    # Ensure all feature columns are present, fill with 0 if any are missing (should not happen with cleaned data)
    current_features_df = pd.DataFrame(columns=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if col in sequence_data_df.columns:
            current_features_df[col] = sequence_data_df[col].values
        else:
            st.error(f"CRITICAL ERROR: Feature column '{col}' (expected by model) not found in data for user {selected_uid}. Predictions will be incorrect.")
            current_features_df[col] = 0 # Fallback, but this indicates a data issue
            
    sequence_features_ordered = current_features_df[FEATURE_COLUMNS] # Ensures correct order

    # Scale the features
    try:
        sequence_scaled = final_scaler.transform(sequence_features_ordered)
        input_sequence = np.expand_dims(sequence_scaled, axis=0)

        st.subheader("Input Features for Prediction (Last 7 Days Scaled - from Cleaned Data):")
        st.dataframe(pd.DataFrame(sequence_scaled, columns=FEATURE_COLUMNS, index=sequence_data_df['date'].dt.strftime('%Y-%m-%d')))
        
        # --- Make Prediction ---
        prediction_proba = final_model.predict(input_sequence, verbose=0)[0][0]
        predicted_class_raw = 1 if prediction_proba >= OPTIMAL_THRESHOLD else 0
        predicted_label = TARGET_NAMES[predicted_class_raw]

        st.subheader("Prediction for Next Night's Sleep")
        st.metric(label=f"Predicted Sleep Category", value=predicted_label)
        st.write(f"Probability of '{TARGET_NAMES[1]}': {prediction_proba:.4f}")

        # ... (Conceptual Feedback as before) ...

    except ValueError as ve:
        st.error(f"ValueError during scaling or prediction: {ve}")
        st.error("This might be due to a mismatch in feature numbers between training and prediction, or unexpected data.")
        st.dataframe(sequence_features_ordered) # Show what was being scaled
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.warning(f"User {selected_uid} does not have enough historical data ({len(user_df)} days) to form a sequence of {SEQUENCE_LENGTH} days for prediction.")

# ... (rest of your Streamlit app, e.g., displaying user's historical sleep proxy) ...