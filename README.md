#  Live WEb App
[![Open in Streamlit](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?logo=streamlit)](https://emotionrecognizer.streamlit.app)

  


# Emotion Recognition from Speech and Song Audio 🎤

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A deep learning pipeline that classifies emotions from speech and song audio using MFCC features and a 1D CNN model, achieving **85% accuracy** across 8 emotion classes.

## ✨ Features
- 🎙️ Processes both speech and song audio (WAV format)
- 🔢 Extracts 40 MFCC features per audio sample
- 🧠 1D CNN architecture with Batch Normalization
- ⚖️ Handles class imbalance with weighted loss
- 📈 Early Stopping and LR Reduction callbacks
- 🌍 Streamlit web interface for easy testing
- 📊 Comprehensive evaluation metrics

## 🛠️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```

2. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage
### Web Interface
```bash
streamlit run app.py
```
Access the interface at `http://localhost:8501`

### Command Line
```bash
python test_model.py --file path/to/audio.wav
```

## 📂 Project Structure
```
Emotion_Recognizer/
├── app.py                    # Streamlit web application
├── emotion_classifier_model.h5  # Trained Keras model
├── finalproject.ipynb        # Complete Jupyter notebook
├── label_encoder.pkl         # Emotion label encoder
├── requirements.txt          # Python dependencies
└── test_model.py             # CLI testing script
```

## 📊 Performance
### Overall Metrics
| Metric            | Score |
|-------------------|-------|
| Accuracy          | 85%   |
| Weighted F1 Score | 85%   |
| Loss              | 0.42  |

### Class-wise Accuracy
| Emotion    | Accuracy | Samples |
|------------|----------|---------|
| Angry      | 96%      | 75      |
| Calm       | 91%      | 75      |
| Disgust    | 79%      | 39      |
| Fearful    | 81%      | 75      |
| Happy      | 79%      | 75      |
| Neutral    | 87%      | 38      |
| Sad        | 80%      | 75      |
| Surprised  | 82%      | 39      |

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ using <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="20"> Python, <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" width="20"> TensorFlow, and Streamlit
</div>
```
