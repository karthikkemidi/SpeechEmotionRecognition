import numpy as np
import librosa
import joblib
import os
import noisereduce as nr

# Paths to models (relative to this file)
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def extract_features_type(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1).reshape(1, -1)

def extract_features_human(file_path):
    y, sr = librosa.load(file_path, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=reduced_noise, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=reduced_noise, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=reduced_noise, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(reduced_noise), sr=sr)
    return np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(tonnetz, axis=1)
    ]).reshape(1, -1)

def extract_features_animal(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1)
    ]).reshape(1, -1)

def classify_and_predict_emotion(file_path):
    # Load models
    clf_type = joblib.load(os.path.join(MODELS_DIR, 'audio_type_classifier.joblib'))
    rf_model_human = joblib.load(os.path.join(MODELS_DIR, 'rf_model_human.joblib'))
    lda_human = joblib.load(os.path.join(MODELS_DIR, 'human_lda.joblib'))
    le_human = joblib.load(os.path.join(MODELS_DIR, 'human_label_encoder.joblib'))
    rf_model_animal = joblib.load(os.path.join(MODELS_DIR, 'rf_model_animal_aug.joblib'))
    scaler_animal = joblib.load(os.path.join(MODELS_DIR, 'animal_scaler_aug.joblib'))
    le_animal = joblib.load(os.path.join(MODELS_DIR, 'animal_label_encoder_aug.joblib'))

    # Step 1: Type classification
    type_feat = extract_features_type(file_path)
    audio_type = clf_type.predict(type_feat)[0]

    # Step 2: Emotion classification
    if audio_type == 'human':
        feat = extract_features_human(file_path)
        feat_lda = lda_human.transform(feat)
        pred_enc = rf_model_human.predict(feat_lda)[0]
        emotion = le_human.inverse_transform([pred_enc])[0]
    else:
        feat = extract_features_animal(file_path)
        feat_scaled = scaler_animal.transform(feat)
        pred_enc = rf_model_animal.predict(feat_scaled)[0]
        emotion = le_animal.inverse_transform([pred_enc])[0]
    return audio_type, emotion
