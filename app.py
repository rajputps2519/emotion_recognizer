# app.py - Your Streamlit Web Application

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import io # To handle in-memory audio files
import pandas as pd # For displaying probabilities

# --- 1. Configuration and Model Loading ---

# Define the path to your saved model and label encoder
# These paths are relative to where app.py is located
MODEL_PATH = "emotion_classifier_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Parameters for MFCC extraction (MUST MATCH TRAINING EXACTLY!)
N_MFCC = 40
TARGET_DURATION = 3 # seconds
SR = 22050
FIXED_MFCC_FRAMES = 130 # Must match the value used during training MFCC padding

# Load the trained Keras model
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_emotion_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.") # This will print to your terminal running streamlit
        return model
    except Exception as e:
        st.error(f"Error loading the model. Make sure '{MODEL_PATH}' is in the correct directory. Error: {e}")
        return None

# Load the LabelEncoder
@st.cache_resource # Cache the label encoder
def load_label_encoder():
    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        print("LabelEncoder loaded successfully.")
        return le
    except Exception as e:
        st.error(f"Error loading the LabelEncoder. Make sure '{LABEL_ENCODER_PATH}' is in the correct directory. Error: {e}")
        return None

model = load_emotion_model()
le = load_label_encoder()

# --- 2. MFCC Extraction Function (MUST BE IDENTICAL TO TRAINING) ---

def extract_mfccs(audio_data, sr_input, n_mfcc=N_MFCC, target_duration=TARGET_DURATION, fixed_mfcc_frames=FIXED_MFCC_FRAMES):
    """
    Extracts MFCCs from audio data (numpy array), resampling and padding/truncating
    to ensure consistent output shape.

    Args:
        audio_data (np.array): Audio time series.
        sr_input (int): Original sampling rate of the input audio.
        n_mfcc (int): Number of MFCCs to extract.
        target_duration (int): Target duration in seconds for padding/truncation.
        fixed_mfcc_frames (int): Fixed number of MFCC frames to pad/truncate to.

    Returns:
        np.array: Padded/truncated MFCCs, or None if an error occurs.
    """
    try:
        # Resample to the target sampling rate for consistency (SR global constant)
        if sr_input != SR:
            y = librosa.resample(y=audio_data, orig_sr=sr_input, target_sr=SR)
        else:
            y = audio_data

        # Calculate target number of samples for padding/truncation
        target_length_samples = int(SR * target_duration)

        # Pad or truncate audio to the target length
        if len(y) > target_length_samples:
            y = y[:target_length_samples]
        else:
            y = np.pad(y, (0, max(0, target_length_samples - len(y))), "constant")

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=n_mfcc)

        # Pad or truncate MFCCs to fixed frames
        if mfccs.shape[1] > fixed_mfcc_frames:
            mfccs = mfccs[:, :fixed_mfcc_frames]
        else:
            pad_width = fixed_mfcc_frames - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Reshape for model input: (time_frames, n_mfcc) -> (1, time_frames, n_mfcc, 1)
        X_processed = mfccs.transpose(1, 0) # From (n_mfcc, fixed_mfcc_frames) to (fixed_mfcc_frames, n_mfcc)
        X_processed = np.expand_dims(X_processed, axis=0) # Add batch dimension: (1, fixed_mfcc_frames, n_mfcc)
        X_processed = np.expand_dims(X_processed, axis=-1) # Add channel dimension: (1, fixed_mfcc_frames, n_mfcc, 1)

        return X_processed

    except Exception as e:
        st.error(f"Error processing audio for MFCC extraction: {e}")
        return None

# --- 3. Streamlit UI and Logic ---

st.set_page_config(page_title="Speech Emotion Classifier", layout="centered")

st.title("🗣️ Speech Emotion Classifier")
st.markdown("""
Upload an audio file (WAV format recommended) to get its emotion classified.
The model recognizes 8 emotions: Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised.
""")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    if model is None or le is None:
        st.warning("Model or LabelEncoder could not be loaded. Please ensure all necessary files are present and try again.")
    else:
        st.audio(uploaded_file, format='audio/wav') # Play the uploaded audio

        # Create a spinner while processing
        with st.spinner("Analyzing emotion..."):
            try:
                # Load audio data from the uploaded file
                audio_bytes = uploaded_file.read()
                audio_io = io.BytesIO(audio_bytes)
                y, sr = librosa.load(audio_io, sr=None) # Load with original SR

                # Extract MFCCs and prepare input for the model
                X_pred_prepared = extract_mfccs(y, sr)

                if X_pred_prepared is not None:
                    # Make prediction
                    predictions = model.predict(X_pred_prepared)
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    predicted_emotion = le.inverse_transform([predicted_class_index])[0]
                    confidence = predictions[0][predicted_class_index] * 100

                    st.success(f"**Predicted Emotion:** **{predicted_emotion.upper()}**")
                    st.info(f"Confidence: {confidence:.2f}%")

                    st.subheader("Prediction Probabilities:")
                    prob_df = pd.DataFrame({
                        'Emotion': le.classes_,
                        'Probability': predictions[0]
                    }).sort_values(by='Probability', ascending=False)
                    st.dataframe(prob_df, hide_index=True)

                else:
                    st.error("Could not extract features from the audio file. Please try a different file.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e) # Display full traceback for debugging

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Librosa, and TensorFlow.")