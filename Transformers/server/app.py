from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome Extension

def hindi_tokenizer(text):
    pattern = r'[\u0900-\u097F]+|\d+|[^\u0900-\u097F\s]'
    tokens = re.findall(pattern, text)
    return [token.strip() for token in tokens if token.strip()]

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.get_json()
    tokens = hindi_tokenizer(data['text'])
    return jsonify(tokens)

if __name__ == '__main__':
    app.run(port=5000)