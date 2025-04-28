from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JSON_AS_ASCII'] = False 

try:
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

@app.route('/')
def home():
    return "Hindi Tokenizer API is running! Use POST /tokenize"

@app.route('/tokenize', methods=['POST'])
def tokenize():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        
        logger.info(f"Received text for tokenization: {text}")
        
        
        tokens = hf_hindi_tokenizer(text)
        
        logger.info(f"Tokenized result: {tokens}")
        
        response = jsonify(tokens)
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
        return response
    
    except Exception as e:
        logger.error(f"Error in tokenization: {e}")
        return jsonify({"error": str(e)}), 500

def hf_hindi_tokenizer(text: str) -> list[str]:
    """
    Tokenizes Hindi text using Hugging Face's tokenizer.
    
    Args:
        text (str): Input Hindi text to tokenize.
        
    Returns:
        list: List of word tokens.
    """
 
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    

    tokens = []
    offset_mapping = encoding.offset_mapping
    
    for start, end in offset_mapping:
        if end > start:  
            token = text[start:end]
            if token.strip(): 
                tokens.append(token)
    
    return tokens

if __name__ == '__main__':
    logger.info("Starting Hindi Tokenizer API")
    app.run(host='0.0.0.0', port=8000, debug=True)