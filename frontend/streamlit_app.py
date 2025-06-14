import streamlit as st
import requests
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="Speech and Emotion Recognition", layout="centered")
backend_url = st.secrets["BACKEND_API_URL"]
# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>🎵 Speech and Emotion Recognition of Animals and Humans 🎵</h1>
    """,
    unsafe_allow_html=True
)

# Project description below title
st.markdown(
    """
    <p style='text-align: center; font-size: 18px;'>
    Upload a <b>.wav</b> audio file to classify whether the sound is from a human or animal, 
    and to predict the emotion conveyed in the speech.
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose a .wav file", type=['wav'])

def plot_waveform(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="blue")  # <-- add color here
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("🔍 Analyze Audio"):
        with st.spinner("Predicting..."):
            # Visualize waveform
            plot_waveform(uploaded_file.getvalue())

            try:
                response = requests.post(
                    f"{backend_url}/predict",
                    files={'file': uploaded_file}
                )

                if response.ok:
                    result = response.json()
                    st.success("Prediction Complete!")
                    st.markdown(f"**Type:** :blue[{result['audio_type'].capitalize()}]")
                    st.markdown(f"**Emotion:** :orange[{result['emotion'].capitalize()}]")
                    with st.expander("Show Raw Response"):
                        st.json(result)
                else:
                    st.error(f"Prediction failed: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
else:
    st.info("Please upload a .wav file to get started.")

with st.sidebar:
    st.header("About")
    st.write(
        """
        This project uses machine learning to classify audio type (human/animal) and predict the emotion in speech.
        - **Backend:** Flask + scikit-learn
        - **Frontend:** Streamlit
        - **Features:** MFCC, Chroma, Spectral Contrast, Tonnetz
        """
    )
    st.markdown(
        "👨‍💻 Developed by [Karthik Kemidi](https://www.linkedin.com/in/karthik-kemidi-b4924a25a/)",
        unsafe_allow_html=True
    )
