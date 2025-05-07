import streamlit as st
import parselmouth
from parselmouth.praat import call
import pandas as pd
import numpy as np
import os
import joblib
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="ALS Voice Analysis", layout="wide")

# Define constants
vowels = ['A', 'E', 'I', 'O', 'U']
syllables = ['PA', 'TA', 'KA']
all_sounds = vowels + syllables
metrics = ['meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localShimmer']

# Function to extract acoustic features
def measurePitch(sound, f0min=75, f0max=500, unit="Hertz"):
    """Extract acoustic features from a sound object"""
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0, 0, unit)
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return meanF0, stdevF0, hnr, localJitter, localShimmer

# Load model and scaler if they exist
@st.cache_resource
def load_model():
    try:
        model = joblib.load('als_detection_model.pkl')
        scaler = joblib.load('als_detection_scaler.pkl')
        # Load training data for feature means
        df = pd.read_excel("VOC-ALS.xlsx", sheet_name="VOC-ALS_Data", header=1)
        return model, scaler, df
    except:
        st.error("Model files not found. Please make sure als_detection_model.pkl and als_detection_scaler.pkl exist.")
        return None, None, None

model, scaler, df = load_model()

# Create feature matrix for reference
if df is not None:
    acoustic_features = []
    for sound in all_sounds:
        for metric in metrics:
            feature_name = f"{metric}_{sound}"
            acoustic_features.append(feature_name)
    
    X = df[acoustic_features].copy()
    X['Age'] = df['Age (years)']
    X['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    X = X.fillna(X.mean())
    
    # Add engineered features
    for sound in all_sounds:
        X[f'jitter_shimmer_ratio_{sound}'] = X[f'localJitter_{sound}'] / X[f'localShimmer_{sound}']
        X[f'f0_variability_{sound}'] = X[f'stdevF0Hz_{sound}'] / X[f'meanF0Hz_{sound}']

# Function to predict ALS
def predict_als(features):
    """Predict ALS from acoustic features"""
    # Create DataFrame with features
    new_df = pd.DataFrame([features])
    
    # Add engineered features
    for sound in all_sounds:
        if f'localJitter_{sound}' in new_df and f'localShimmer_{sound}' in new_df:
            new_df[f'jitter_shimmer_ratio_{sound}'] = new_df[f'localJitter_{sound}'] / new_df[f'localShimmer_{sound}']
            new_df[f'f0_variability_{sound}'] = new_df[f'stdevF0Hz_{sound}'] / new_df[f'meanF0Hz_{sound}']
    
    # Handle missing columns
    missing_cols = set(X.columns) - set(new_df.columns)
    for col in missing_cols:
        new_df[col] = X[col].mean()
    
    # Ensure columns match training data
    new_df = new_df[X.columns]
    
    # Scale features
    new_scaled = scaler.transform(new_df)
    
    # Predict
    prediction = model.predict(new_scaled)[0]
    probability = model.predict_proba(new_scaled)[0][1]
    
    return prediction, probability

# App title and description
st.title("ALS Detection from Voice Analysis")
st.write("Upload 8 audio files (vowels A, E, I, O, U and syllables PA, TA, KA) to analyze for ALS indicators")

# Demographics
st.sidebar.header("Patient Demographics")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=60)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
sex_code = 1 if sex == "Male" else 0

# File uploaders for each sound
st.subheader("Upload Voice Recordings")

# Create two columns for better layout
col1, col2 = st.columns(2)

# Dictionary to store uploaded files
uploaded_files = {}

# Create file uploaders for vowels in first column
with col1:
    st.write("Vowel Sounds:")
    for vowel in vowels:
        uploaded_files[vowel] = st.file_uploader(f"Upload vowel /{vowel}/", type=["wav"], key=f"vowel_{vowel}")

# Create file uploaders for syllables in second column
with col2:
    st.write("Syllable Sounds:")
    for syllable in syllables:
        uploaded_files[syllable] = st.file_uploader(f"Upload syllable /{syllable}/", type=["wav"], key=f"syllable_{syllable}")

# Process button
if st.button("Process Audio Files"):
    # Check if all files are uploaded
    if None in uploaded_files.values():
        st.error("Please upload all 8 audio files")
    else:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Dictionary to store features
        features = {}
        
        # Process each file
        for i, (sound_type, file) in enumerate(uploaded_files.items()):
            status_text.text(f"Processing {sound_type}...")
            
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # Extract features
            try:
                sound = parselmouth.Sound(tmp_path)
                meanF0, stdevF0, hnr, localJitter, localShimmer = measurePitch(sound)
                
                # Store features
                features[f'meanF0Hz_{sound_type}'] = meanF0
                features[f'stdevF0Hz_{sound_type}'] = stdevF0
                features[f'HNR_{sound_type}'] = hnr
                features[f'localJitter_{sound_type}'] = localJitter
                features[f'localShimmer_{sound_type}'] = localShimmer
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            except Exception as e:
                st.error(f"Error processing {sound_type}: {e}")
                break
        
        # Add demographic information
        features['Age'] = age
        features['Sex'] = sex_code
        
        # If all features were extracted successfully
        if len(features) == 42:  # 5 features × 8 sounds + 2 demographic features
            status_text.text("Analyzing voice patterns...")
            
            # Make prediction
            prediction, probability = predict_als(features)
            
            # Display results
            st.success("Analysis complete!")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.subheader("Results")
                
                # Create a visual indicator
                if prediction == 1:
                    st.error("ALS Detected")
                    st.write(f"Probability of ALS: {probability:.2f}")
                else:
                    st.success("No ALS Detected")
                    st.write(f"Probability of ALS: {probability:.2f}")
            
            with result_col2:
                # Display a gauge chart for probability
                fig, ax = plt.subplots(figsize=(4, 4))
                
                # Create gauge chart
                pos = ax.pie([probability, 1-probability], 
                       colors=['#ff9999', '#99ff99'] if prediction == 1 else ['#ffcc99', '#99ccff'],
                       startangle=90, 
                       counterclock=False,
                       wedgeprops=dict(width=0.3))
                
                plt.text(0, 0, f"{probability:.2f}", ha='center', va='center', fontsize=24)
                plt.title("ALS Probability", pad=20)
                
                # Display the chart
                st.pyplot(fig)
            
            # Display extracted features
            with st.expander("View Extracted Features"):
                feature_df = pd.DataFrame([features])
                st.dataframe(feature_df)
