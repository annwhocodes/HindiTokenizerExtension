import re

def hindi_tokenizer(text):
    """
    Tokenizes Hindi text into words, numbers, and punctuation.
    
    Args:
        text (str): Input text to tokenize.
    
    Returns:
        list: List of tokens.
    """
    # Regex pattern to match:
    # 1. Hindi words (Devanagari Unicode range)
    # 2. Numbers (digits)
    # 3. Punctuation/symbols (non-Devanagari and non-whitespace)
    pattern = r'[\u0900-\u097F]+|\d+|[^\u0900-\u097F\s]'
    
    tokens = re.findall(pattern, text)

    tokens = [token.strip() for token in tokens if token.strip()]
    return tokens

if __name__ == "__main__":
    sample_text = "मैं हिंदी लिख सकता हूँ। क्या आप जानते हैं? 123"
    tokens = hindi_tokenizer(sample_text)
    print(tokens)
    