from flask import Flask, request, jsonify
import os
from ser.utils import classify_and_predict_emotion

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

@app.route('/')
def index():
    return "Speech Emotion Recognition API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['file']
    file_path = os.path.join('/tmp', audio_file.filename)
    audio_file.save(file_path)
    
    try:
        audio_type, emotion = classify_and_predict_emotion(file_path)
        return jsonify({'audio_type': audio_type, 'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
