from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from io import StringIO
import json
import logging
import spacy
import re

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the language dictionary
with open("languages.json", "r", encoding="utf-8") as f:
    languages = json.load(f)

# Load the crop recommendation model and label encoder
with open("model/crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the disease detection model and recompile it
disease_model = tf.keras.models.load_model("model/disease_model.h5", compile=False)
# Optionally, compile the model with a new optimizer if needed for training
disease_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
logger.info(f"Disease model input shape: {disease_model.input_shape}")
with open("model/disease_labels.txt", "r") as f:
    disease_labels = f.read().splitlines()
logger.info(f"Disease labels: {disease_labels}")

# Load spacy NLP model
nlp = spacy.load("en_core_web_sm")

# State management for chatbot conversation, chat history, and language preference
user_states = {}
chat_histories = {}

# Helper function to get translated message
def get_message(key, lang, **kwargs):
    message = languages.get(lang, languages["en"]).get(key, languages["en"][key])
    return message.format(**kwargs)

# NLP intent recognition
def detect_intent(message):
    doc = nlp(message.lower())
    if any(token.text in ["crop", "recommend", "suggest", "plant"] for token in doc):
        return "crop_recommendation"
    if any(token.text in ["disease", "sick", "ill", "problem", "leaf"] for token in doc):
        return "disease_prediction"
    if any(token.text in ["rainfall", "rain"] for token in doc):
        return "rainfall_query"
    if any(token.text in ["weather", "climate"] for token in doc):
        return "weather_query"
    if any(token.text in ["soil", "land"] for token in doc):
        return "soil_query"
    if any(token.text in ["pest", "insect", "bug"] for token in doc):
        return "pest_query"
    return "default"

# Extract numerical value from message using NLP and regex
def extract_number(message):
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in ["QUANTITY", "CARDINAL"]:
            try:
                # Extract numbers (including decimals) from the entity text
                numbers = re.findall(r'\d+\.?\d*', ent.text)
                if numbers:
                    return float(numbers[0])
            except ValueError:
                pass
    # Fallback to regex for any number in the message
    numbers = re.findall(r'\d+\.?\d*', message)
    return float(numbers[0]) if numbers else None

@app.route('/')
def index():
    user_id = request.remote_addr
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    if user_id not in user_states:
        user_states[user_id] = {
            'step': 'start',
            'crop_inputs': {},
            'awaiting_image': False,
            'language': 'en'
        }
    lang = user_states[user_id]['language']
    welcome_message = get_message("welcome_message", lang)
    chat_histories[user_id] = [{'sender': 'bot', 'message': welcome_message, 'image': None}]
    return render_template('index.html')

@app.route('/set_language', methods=['POST'])
def set_language():
    user_id = request.remote_addr
    lang = request.form.get('language', 'en')
    if user_id not in user_states:
        user_states[user_id] = {
            'step': 'start',
            'crop_inputs': {},
            'awaiting_image': False,
            'language': lang
        }
    else:
        user_states[user_id]['language'] = lang
    welcome_message = get_message("welcome_message", lang)
    chat_histories[user_id] = [{'sender': 'bot', 'message': welcome_message, 'image': None}]
    return jsonify({'response': welcome_message})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.remote_addr
    message = request.form.get('message', '').strip()

    if user_id not in user_states:
        user_states[user_id] = {
            'step': 'start',
            'crop_inputs': {},
            'awaiting_image': False,
            'language': 'en'
        }
    if user_id not in chat_histories:
        lang = user_states[user_id]['language']
        welcome_message = get_message("welcome_message", lang)
        chat_histories[user_id] = [{'sender': 'bot', 'message': welcome_message, 'image': None}]

    state = user_states[user_id]
    lang = state['language']

    if message:
        chat_histories[user_id].append({'sender': 'user', 'message': message, 'image': None})

    # Handle intents in start state using NLP
    if state['step'] == 'start':
        intent = detect_intent(message)
        if intent == "crop_recommendation":
            state['step'] = 'crop_N'
            state['crop_inputs'] = {}
            response = get_message("crop_N_prompt", lang)
            chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': None})
            return jsonify({'response': response})
        if intent == "disease_prediction":
            state['step'] = 'disease_image'
            state['awaiting_image'] = True
            response = get_message("disease_prediction_prompt", lang)
            chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': None})
            return jsonify({'response': response, 'request_image': True})
        if intent in ["rainfall_query", "weather_query", "soil_query", "pest_query"]:
            response = get_message(f"{intent.split('_')[0]}_response", lang)
            chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': None})
            return jsonify({'response': response})

    # Handle image upload for disease prediction
    if state['step'] == 'disease_image' and 'image' in request.files:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))  # Ensure 224x224 as per model input
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        logger.info(f"Image shape after preprocessing: {img_array.shape}")

        prediction = disease_model.predict(img_array)
        logger.info(f"Raw prediction output: {prediction}")
        predicted_class = np.argmax(prediction, axis=1)[0]
        logger.info(f"Predicted class index: {predicted_class}")

        if 0 <= predicted_class < len(disease_labels):
            disease = disease_labels[predicted_class]
        else:
            disease = "Unknown (prediction out of range)"
            logger.warning(f"Predicted class {predicted_class} out of range for {len(disease_labels)} labels")

        state['step'] = 'start'
        state['awaiting_image'] = False
        response = get_message("disease_prediction_result", lang, disease=disease)
        chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': None})
        return jsonify({'response': response})

    # Handle crop recommendation input collection with NLP
    if state['step'].startswith('crop_'):
        value = extract_number(message)
        if value is not None:
            if state['step'] == 'crop_N':
                state['crop_inputs']['N'] = value
                state['step'] = 'crop_P'
                response = get_message("crop_P_prompt", lang)
            elif state['step'] == 'crop_P':
                state['crop_inputs']['P'] = value
                state['step'] = 'crop_K'
                response = get_message("crop_K_prompt", lang)
            elif state['step'] == 'crop_K':
                state['crop_inputs']['K'] = value
                state['step'] = 'crop_temperature'
                response = get_message("crop_temperature_prompt", lang)
            elif state['step'] == 'crop_temperature':
                state['crop_inputs']['temperature'] = value
                state['step'] = 'crop_humidity'
                response = get_message("crop_humidity_prompt", lang)
            elif state['step'] == 'crop_humidity':
                state['crop_inputs']['humidity'] = value
                state['step'] = 'crop_ph'
                response = get_message("crop_ph_prompt", lang)
            elif state['step'] == 'crop_ph':
                state['crop_inputs']['ph'] = value
                state['step'] = 'crop_rainfall'
                response = get_message("crop_rainfall_prompt", lang)
            elif state['step'] == 'crop_rainfall':
                state['crop_inputs']['rainfall'] = value
                features = np.array([[state['crop_inputs']['N'], state['crop_inputs']['P'], state['crop_inputs']['K'],
                                   state['crop_inputs']['temperature'], state['crop_inputs']['humidity'],
                                   state['crop_inputs']['ph'], state['crop_inputs']['rainfall']]])
                prediction = crop_model.predict(features)
                crop = label_encoder.inverse_transform(prediction)[0]
                crop_image_path = f"static/images/crops/{crop}.jpg"
                crop_image_url = f"/static/images/crops/{crop}.jpg" if os.path.exists(crop_image_path) else None
                state['step'] = 'start'
                state['crop_inputs'] = {}
                response = get_message("crop_recommendation_result", lang, crop=crop)
                chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': crop_image_url})
                return jsonify({'response': response, 'crop_image': crop_image_url})
            chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': None})
            return jsonify({'response': response})
        else:
            response = get_message("invalid_number", lang)
            chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': None})
            return jsonify({'response': response})

    # Default response for unrecognized inputs
    response = get_message("default_response", lang)
    chat_histories[user_id].append({'sender': 'bot', 'message': response, 'image': None})
    return jsonify({'response': response})

@app.route('/download_chat', methods=['GET'])
def download_chat():
    user_id = request.remote_addr
    if user_id not in chat_histories:
        return "No chat history found.", 404

    html_content = StringIO()
    html_content.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
    html_content.write('    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
    html_content.write('    <title>AgriBot Chat History</title>\n    <style>\n')
    html_content.write('        body { font-family: Arial, sans personally; margin: 20px; }\n')
    html_content.write('        .chat-box { max-width: 600px; margin: auto; }\n')
    html_content.write('        .message { margin: 10px 0; padding: 10px; border-radius: 10px; }\n')
    html_content.write('        .bot-message { background-color: #f1f1f1; margin-right: auto; }\n')
    html_content.write('        .user-message { background-color: #6a1b9a; color: #fff; margin-left: auto; text-align: right; }\n')
    html_content.write('        .crop-image { max-width: 100%; margin-top: 10px; border-radius: 10px; }\n')
    html_content.write('    </style>\n</head>\n<body>\n    <h1>AgriBot Chat History</h1>\n    <div class="chat-box">\n')
    for entry in chat_histories[user_id]:
        message_class = 'bot-message' if entry['sender'] == 'bot' else 'user-message'
        html_content.write(f'        <div class="message {message_class}">\n            <p>{entry["message"]}</p>\n')
        if entry['image']:
            html_content.write(f'            <img src="{entry["image"]}" class="crop-image" alt="Crop Image">\n')
        html_content.write('        </div>\n')
    html_content.write('    </div>\n</body>\n</html>\n')

    with open('agribot_chat.html', 'w', encoding='utf-8') as f:
        f.write(html_content.getvalue())
    return send_file('agribot_chat.html', as_attachment=True, download_name='agribot_chat.html')

if __name__ == '__main__':
    app.run(debug=True)
