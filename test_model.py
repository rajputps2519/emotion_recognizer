# test_model.py - Script to test the trained model locally

import librosa
import numpy as np
import tensorflow as tf
import pickle
import os

# --- Configuration (MUST MATCH TRAINING and APP.PY) ---
MODEL_PATH = "emotion_classifier_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

N_MFCC = 40
TARGET_DURATION = 3 # seconds
SR = 22050
FIXED_MFCC_FRAMES = 130

# --- MFCC Extraction Function (Identical to app.py and training) ---
def extract_mfccs(audio_data, sr_input, n_mfcc=N_MFCC, target_duration=TARGET_DURATION, fixed_mfcc_frames=FIXED_MFCC_FRAMES):
    try:
        if sr_input != SR:
            y = librosa.resample(y=audio_data, orig_sr=sr_input, target_sr=SR)
        else:
            y = audio_data

        target_length_samples = int(SR * target_duration)

        if len(y) > target_length_samples:
            y = y[:target_length_samples]
        else:
            y = np.pad(y, (0, max(0, target_length_samples - len(y))), "constant")

        mfccs = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=n_mfcc)

        if mfccs.shape[1] > fixed_mfcc_frames:
            mfccs = mfccs[:, :fixed_mfcc_frames]
        else:
            pad_width = fixed_mfcc_frames - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Reshape for model input: (time_frames, n_mfcc) -> (1, time_frames, n_mfcc, 1)
        X_processed = mfccs.transpose(1, 0)
        X_processed = np.expand_dims(X_processed, axis=0)
        X_processed = np.expand_dims(X_processed, axis=-1)

        return X_processed

    except Exception as e:
        print(f"Error processing audio for MFCC extraction: {e}")
        return None

# --- Main Test Logic ---
if __name__ == "__main__":
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model. Make sure '{MODEL_PATH}' exists. {e}")
        exit()

    print(f"Loading LabelEncoder from: {LABEL_ENCODER_PATH}")
    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        print("LabelEncoder loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load LabelEncoder. Make sure '{LABEL_ENCODER_PATH}' exists. {e}")
        exit()

    # --- IMPORTANT: Replace with a path to one of your WAV test files ---
    # Make sure this file is present in the same directory or provide its full path
    sample_audio_file = "path/to/your/sample_audio.wav"
    if not os.path.exists(sample_audio_file):
        print(f"ERROR: Sample audio file not found at '{sample_audio_file}'. Please update the path.")
        exit()

    print(f"\n--- Testing prediction with '{sample_audio_file}' ---")
    try:
        y_audio, sr_audio = librosa.load(sample_audio_file, sr=None)
        X_pred_prepared = extract_mfccs(y_audio, sr_audio)

        if X_pred_prepared is not None:
            predictions = model.predict(X_pred_prepared)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_emotion = le.inverse_transform([predicted_class_index])[0]
            confidence = predictions[0][predicted_class_index] * 100

            print(f"\nPredicted Emotion: {predicted_emotion.upper()}")
            print(f"Confidence: {confidence:.2f}%")
            print("\nAll Probabilities:")
            for i, emotion in enumerate(le.classes_):
                print(f"  {emotion}: {predictions[0][i]*100:.2f}%")
        else:
            print("Failed to extract features from the sample audio file.")

    except Exception as e:
        print(f"An error occurred during test prediction: {e}")