# Speech and Emotion Recognition

🎵 A machine learning project that classifies audio files as human or animal sounds and predicts the emotion conveyed in the speech.

---

## Features

- Classifies audio type: **Human** or **Animal**
- Predicts emotion from speech (e.g., happy, sad, angry, neutral)
- Uses audio features: MFCC, Chroma, Spectral Contrast, Tonnetz
- Backend built with **Flask** and **scikit-learn**
- Frontend built with **Streamlit** for easy interaction

---

## Demo

Upload a `.wav` audio file to the app, and it will analyze and display:

- Audio type (human or animal)
- Predicted emotion
- Audio waveform visualization

---

## Project Structure

```
SpeechEmotionRecognition/
├── backend/
│   ├── ser/
│   │   ├── __init__.py
│   │   ├── main.py                      # Flask app entrypoint
│   │   ├── utils.py                     # Feature extraction & prediction functions
│   │   ├── models/
│   │   │   ├── audio_type_classifier.joblib
│   │   │   └── emotion_classifier.joblib
│   │   └── requirements.txt             # Backend dependencies
│   └── serenv/                         # Virtual environment folder (optional, usually excluded from git)
│
├── frontend/
│   ├── streamlit_app.py                # Streamlit frontend app
│   ├── requirements.txt                # Frontend dependencies (can be combined with backend if preferred)
│
├── .gitignore                         # To exclude venv, __pycache__, etc.
├── README.md                         # Project overview and instructions
├── LICENSE                           # License file (e.g., MIT)
└── Dockerfile                        # (Optional) Dockerfile for containerizing backend or frontend

```

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/karthikkemidi/SpeechEmotionRecognition.git
   cd SpeechEmotionRecognition/backend
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python3 -m venv serenv
   source serenv/bin/activate
   ```

3. Install backend dependencies:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the Flask backend server:
   ```
   python -m ser.main
   ```
   The backend will run on `http://localhost:5050`.

5. In a new terminal, activate the same environment and start the frontend:
   ```
   cd ../frontend
   streamlit run streamlit_app.py
   ```

6. Open your browser at `http://localhost:8501` to use the app.

---

## Usage

- Upload a `.wav` file using the Streamlit frontend.
- Click **Analyze Audio**.
- View the predicted audio type, emotion, and waveform visualization.

---

## Dependencies

- Python 3.10+
- Flask
- scikit-learn (version 1.2.2 recommended)
- librosa
- noisereduce
- numpy
- matplotlib
- streamlit
- requests

See `requirements.txt` for exact versions.

---

## Deployment

You can deploy this app using platforms like:

- **Render.com** (for backend)
- **Streamlit Cloud** (for frontend)
- Or use Docker containers for both backend and frontend.

Deployed Link: [SER](https://speechemotionrecognition-kk.streamlit.app/)
---

## Author

Developed by [Karthik Kemidi](https://www.linkedin.com/in/karthik-kemidi-b4924a25a/)

---

