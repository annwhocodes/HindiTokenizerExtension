import json
import re
from collections import Counter

class SimpleHindiTokenizer:
    """
    A lightweight Hindi tokenizer specifically designed for use with the Gemini Flash model
    """
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.word_freq = Counter()
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        }
    
    def simple_word_tokenize(self, text):
        """
        Simple word tokenizer for Hindi that splits on spaces and punctuation
        """
        # First replace punctuation with spaces
        text = re.sub(r'[।,.?!]', ' ', text)
        # Split on whitespace
        words = text.split()
        return words
    
    def load_tokenizer(self, filepath="hindi_tokenizer.json"):
        """
        Load a previously saved tokenizer
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data["vocab"]
        self.special_tokens = data["special_tokens"]
        self.inverse_vocab = {int(idx): token for token, idx in self.vocab.items()}
        
        print(f"Loaded tokenizer with {len(self.vocab)} tokens")
    
    def encode(self, text, add_special_tokens=False):
        """
        Encode text into token IDs
        """
        words = self.simple_word_tokenize(text)
        token_ids = []
        tokens = []
        
        # Add [CLS] token at the beginning if requested
        if add_special_tokens:
            token_ids.append(self.special_tokens["[CLS]"])
            tokens.append("[CLS]")
        
        # Encode each word
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
                tokens.append(word)
            else:
                token_ids.append(self.special_tokens["[UNK]"])
                tokens.append("[UNK]")
        
        # Add [SEP] token at the end if requested
        if add_special_tokens:
            token_ids.append(self.special_tokens["[SEP]"])
            tokens.append("[SEP]")
        
        return {"ids": token_ids, "tokens": tokens}
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to text
        """
        words = []
        for idx in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and idx in [
                self.special_tokens["[PAD]"], 
                self.special_tokens["[CLS]"], 
                self.special_tokens["[SEP]"], 
                self.special_tokens["[MASK]"]
            ]:
                continue
                
            if idx in self.inverse_vocab:
                words.append(self.inverse_vocab[idx])
            else:
                words.append("[UNK]")
        
        return " ".join(words)
    
    def encode_batch(self, texts, max_length=None, padding=False, truncation=False):
        """
        Encode a batch of texts
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences to max_length
            
        Returns:
            Dictionary with encoded IDs and attention masks
        """
        encoded_batch = {"input_ids": [], "attention_mask": []}
        
        for text in texts:
            encoded = self.encode(text)
            ids = encoded["ids"]
            
            # Truncate if needed
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(ids)
            
            # Pad if needed
            if padding and max_length:
                padding_length = max_length - len(ids)
                ids = ids + [self.special_tokens["[PAD]"]] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            encoded_batch["input_ids"].append(ids)
            encoded_batch["attention_mask"].append(attention_mask)
        
        return encoded_batch

def load_pretrained_tokenizer(tokenizer_path="hindi_tokenizer.json"):
    """
    Helper function to load a pretrained tokenizer
    """
    tokenizer = SimpleHindiTokenizer()
    try:
        tokenizer.load_tokenizer(tokenizer_path)
        return tokenizer
    except FileNotFoundError:
        print(f"Tokenizer file not found at {tokenizer_path}. Please train a tokenizer first.")
        return None