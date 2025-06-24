Emotion Recognition from Speech and Song Audio
Project Overview
This project implements an end-to-end pipeline for emotion classification using speech and song audio data. The system processes audio files, extracts Mel-Frequency Cepstral Coefficients (MFCCs) as features, and uses a deep learning model to classify emotions into one of eight categories: angry, calm, disgust, fearful, happy, neutral, sad, or surprised.

Key Features
Processes both speech and song audio data
Extracts 40 MFCC features from audio files
Uses a 1D Convolutional Neural Network (CNN) for classification
Achieves weighted F1 score of 85% and overall accuracy of 85%
Includes class balancing through weighted loss function
Provides detailed evaluation metrics including confusion matrix and per-class accuracy

Dataset
The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains:
24 professional actors (12 male, 12 female)
8 emotional states (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
Two modalities: speech and song
Audio files in WAV format at 48kHz sampling rate

Project Structure
EMOTION_RECOGNIZER/
├── venv/                     # Virtual environment
├── .gitignore               # Git ignore file
├── app.py                    # Streamlit web application
├── emotion_classifier_model.h5  # Trained model weights
├── finalproject.ipynb        # Jupyter notebook with full implementation
├── label_encoder.pkl         # Label encoder for emotion classes
├── requirements.txt          # Python dependencies
└── test_model.py             # Script for testing the model
Installation
Clone the repository
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies
pip install -r requirements.txt
Usage
Running the Web Application
To launch the Streamlit web app:
streamlit run app.py
The web application allows you to:

Upload audio files (WAV format)

View the waveform and spectrogram

Get emotion predictions in real-time
Testing the Model
To test the model with your own audio files
python test_model.py --file path/to/your/audio.wav

Methodology
Data Preprocessing
Audio Loading: Audio files are loaded and resampled to 22.05kHz
Padding/Truncation: All audio clips are standardized to 3 seconds duration
Feature Extraction: 40 MFCCs are extracted from each audio file
Normalization: MFCCs are normalized across the dataset

Model Architecture
The model uses a 1D CNN architecture with the following layers:
Three 1D convolutional layers with increasing filters (128, 256, 512)
Batch normalization and max pooling after each convolutional layer
Dropout layers for regularization (30% dropout)
A dense layer with 512 units and 60% dropout
Softmax output layer with 8 units (one per emotion class)

Training
Optimizer: Adam with learning rate 0.0005
Loss: Categorical cross-entropy with class weights
Callbacks: Early stopping and learning rate reduction on plateau
Batch size: 32
Epochs: 100 (with early stopping)

Performance Metrics
The model achieves the following performance on the validation set:

Metric	Score
Overall Accuracy	85%
Weighted F1 Score	85%
Per-Class Accuracy
Emotion	Accuracy
angry	96%
calm	91%
disgust	79%
fearful	81%
happy	79%
neutral	87%
sad	80%
surprised	82%
Evaluation Criteria Check
F1 score > 80%: Achieved (85%)
Each class accuracy > 75%: All classes meet this requirement
Overall accuracy > 80%: Achieved (85%)

Acknowledgments
Ryerson University for the RAVDESS dataset
The open-source community for libraries like Librosa, TensorFlow, and scikit-learn

