import numpy as np
from flask import request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({'error': 'Missing text in request'}), 400

    text_input = data['text']
    processed_text = preprocess_text(text_input)

    prediction = loaded_model.predict(processed_text)
    # Assuming binary classification, convert probability to a class label (0 or 1)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0
    
    result = {'prediction': predicted_class, 'probability': float(prediction[0][0])}
    return jsonify(result)

print("Flask app and '/predict' endpoint defined.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
