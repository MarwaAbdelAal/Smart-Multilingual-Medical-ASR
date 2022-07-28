from huggingsound import SpeechRecognitionModel
from flask import Flask, request
import json
from NER_models import NER_finder

english_model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
arabic_model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.data
    try:
        audio_path = json.loads(data.decode('utf-8'))['path']
    except Exception as err:
        return f'invalid request with error: {err}'

    english_transcription = english_model.transcribe([audio_path])
    arabic_transcription = arabic_model.transcribe([audio_path])

    responseDict = {
        'english': english_transcription[0]['transcription'],
        'arabic': arabic_transcription[0]['transcription']
    }
    return responseDict


@app.route('/ner', methods=['POST'])
def ner():
    data = request.data
    try:
        text = json.loads(data.decode('utf-8'))['text']
        mode = json.loads(data.decode('utf-8'))['mode']
    except Exception as err:
        return f'invalid request with error: {err}'

    updatedText, medicalEntities, personalEntities = NER_finder(text, mode)

    responseDict = {
        'text': updatedText,
        'medicalEntities': medicalEntities,
        'personalEntities': personalEntities
    }
    return responseDict


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
