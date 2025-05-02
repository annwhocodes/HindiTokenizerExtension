import os
import re
import json
import torch
import numpy as np
import random
import urllib.request
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SimpleHindiTokenizer:
    """
    A lightweight Hindi tokenizer that uses the same interface as the original
    but has been adapted for the Gemini Flash training.
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
    
    def download_hindi_wikipedia_page(self, page_title=None):
        """
        Download a specific Hindi Wikipedia page or a random one if no title provided
        """
        print("Downloading Hindi Wikipedia page...")
        
        # If no specific page title is provided, use a random page
        if page_title:
            # Replace spaces with underscores for the URL
            page_title = page_title.replace(' ', '_')
            # Properly encode non-ASCII characters
            import urllib.parse
            page_title = urllib.parse.quote(page_title)
            hindi_wiki_url = f"https://hi.wikipedia.org/wiki/{page_title}"
        else:
            # For random page, directly use the special page URL
            hindi_wiki_url = "https://hi.wikipedia.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%B6%E0%A5%87%E0%A4%B7:%E0%A4%AF%E0%A4%BE%E0%A4%A6%E0%A5%83%E0%A4%9A%E0%A5%8D%E0%A4%9B%E0%A4%BF%E0%A4%95_%E0%A4%AA%E0%A5%83%E0%A4%B7%E0%A5%8D%E0%A4%A0"
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            req = urllib.request.Request(hindi_wiki_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
            
            # Extract the title for logging purposes
            title_match = re.search(r'<title>(.*?) - विकिपीडिया</title>', html)
            if title_match:
                actual_title = title_match.group(1)
                print(f"Downloaded Wikipedia page: {actual_title}")
            
            # Extract the main content
            content_match = re.search(r'<div id="mw-content-text".*?>(.*?)<div class="printfooter">', html, re.DOTALL)
            if content_match:
                content = content_match.group(1)
                
                # Extract paragraphs
                paragraphs = re.findall(r'<p>(.*?)</p>', content, re.DOTALL)
                text = ' '.join(paragraphs)
                
                # Remove HTML tags
                text = re.sub(r'<.*?>', ' ', text)
                
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > 100:
                    return text
                else:
                    print("Content too short, using sample Hindi text instead...")
            else:
                print("Could not extract content, using sample Hindi text instead...")
        except Exception as e:
            print(f"Error downloading Wikipedia page: {e}")
        
        # Fallback sample text if Wikipedia download fails
        return """हिन्दी विश्व की एक प्रमुख भाषा है और भारत की राजभाषा है। यह हिंदुस्तानी भाषा की एक मानकीकृत रूप है जिसमें संस्कृत के तत्सम तथा तद्भव शब्दों का प्रयोग अधिक है और अरबी-फ़ारसी शब्द कम हैं।
        भारत एक विशाल देश है जिसमें अनेक धर्म, संस्कृति और भाषाएँ पाई जाती हैं। भारत की विविधता में ही इसकी एकता है।
        प्रौद्योगिकी का विकास आज के समय में तेज़ी से हो रहा है। कंप्यूटर, इंटरनेट, मोबाइल फोन आदि ने हमारे जीवन को बहुत सरल बना दिया है।"""
    
    def preprocess_hindi_text(self, text):
        """
        Clean and preprocess Hindi text data
        """
        print("Preprocessing Hindi text...")
        
        # Replace HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        # Keep only Hindi characters, spaces, and punctuation
        text = re.sub(r'[^\u0900-\u097F\s।,.?!]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def simple_word_tokenize(self, text):
        """
        Simple word tokenizer for Hindi that splits on spaces and punctuation
        """
        text = re.sub(r'[।,.?!]', ' ', text)
        words = text.split()
        return words
    
    def build_vocabulary(self, text, max_vocab_size=5000):
        """
        Build a vocabulary from the processed text
        """
        print("Building vocabulary...")
        
        vocab = {token: idx for token, idx in self.special_tokens.items()}
        current_idx = len(vocab)
        
        words = self.simple_word_tokenize(text)
        self.word_freq.update(words)
        
        # Print some stats about the vocabulary
        total_words = len(words)
        unique_words = len(set(words))
        print(f"Total words in corpus: {total_words}")
        print(f"Unique words in corpus: {unique_words}")
        
        # Print most common words
        print("Most common words:")
        for word, count in self.word_freq.most_common(10):
            print(f"  {word}: {count}")
        
        for word, _ in self.word_freq.most_common(max_vocab_size - len(vocab)):
            vocab[word] = current_idx
            current_idx += 1
        
        inverse_vocab = {idx: token for token, idx in vocab.items()}
        
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        
        print(f"Vocabulary built with {len(vocab)} tokens")
        return vocab, inverse_vocab
    
    def save_tokenizer(self, filepath="hindi_tokenizer.json"):
        """
        Save the tokenizer vocabulary to a JSON file
        """
        data = {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {filepath}")
    
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
    
    def encode(self, text):
        """
        Encode text into token IDs
        """
        words = self.simple_word_tokenize(text)
        token_ids = []
        tokens = []
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
                tokens.append(word)
            else:
                token_ids.append(self.special_tokens["[UNK]"])
                tokens.append("[UNK]")
        
        return {"ids": token_ids, "tokens": tokens}
    
    def decode(self, token_ids):
        """
        Decode token IDs back to text
        """
        words = []
        for idx in token_ids:
            if idx in self.inverse_vocab:
                words.append(self.inverse_vocab[idx])
            else:
                words.append("[UNK]")
        
        return " ".join(words)
    
    def save_corpus(self, text, filepath="hindi_corpus.txt"):
        """
        Save the processed text to a text file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Corpus saved to {filepath}")
        return filepath


class GeminiFlashAttention(nn.Module):
    """
    Flash Attention implementation for the Gemini Flash model
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Ensure the head dimension is compatible
        assert self.head_dim * heads == dim, "Dimension must be divisible by number of heads"
        
        # QKV projections in a single matrix
        self.qkv = nn.Linear(dim, dim * 3)
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for the attention scores
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        
        # QKV projections: [batch, seq, dim*3] -> [batch, seq, 3, heads, head_dim]
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.heads, self.head_dim)
        # Separate Q, K, V: [batch, heads, seq, head_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        
        # Reshape and project back
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        
        return out


class GeminiFlashMLP(nn.Module):
    """
    MLP block with SwiGLU activation for Gemini Flash
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU activation function
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(hidden))


class GeminiFlashBlock(nn.Module):
    """
    Transformer block for Gemini Flash
    """
    def __init__(self, dim, mlp_dim, heads=8, dropout=0.1):
        super().__init__()
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention and MLP blocks
        self.attn = GeminiFlashAttention(dim, heads, dropout)
        self.mlp = GeminiFlashMLP(dim, mlp_dim, dropout)
    
    def forward(self, x, mask=None):
        # Attention with residual connection and pre-norm
        x = x + self.attn(self.norm1(x), mask)
        # MLP with residual connection and pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


class GeminiFlashModel(nn.Module):
    """
    Gemini Flash 1.5 model - Lightweight version for demonstration
    """
    def __init__(self, 
                 vocab_size, 
                 dim=256, 
                 depth=6, 
                 heads=8, 
                 mlp_dim=1024, 
                 max_seq_len=512, 
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GeminiFlashBlock(dim, mlp_dim, heads, dropout)
            for _ in range(depth)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(dim)
        
        # Language model head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight sharing between token embedding and LM head (as in many LLMs)
        self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Ensure sequence length doesn't exceed max_seq_len
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # Get token embeddings and add position embeddings
        token_emb = self.token_embedding(x)
        position_emb = self.position_embedding[:, :seq_len, :]
        
        # Combine embeddings and apply dropout
        x = token_emb + position_emb
        x = self.dropout(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=0.8, top_k=40):
        """
        Generate text using the model
        """
        self.eval()
        
        # Make sure input_ids is a tensor on the right device
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        input_ids = input_ids.to(next(self.parameters()).device)
        
        # Generate tokens auto-regressively
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self(input_ids)
                
                # Get the next token logits (last position)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the probability distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the new token
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we've generated enough tokens or hit a special token
                if next_token.item() == 3:  # [SEP] token
                    break
                
                # Handle max_seq_len limit - truncate from beginning if needed
                if input_ids.size(1) >= self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len + 10:]  # Keep some context
        
        return input_ids.squeeze().tolist()


class WikiTextDataset(Dataset):
    """
    Dataset for Hindi Wikipedia text
    """
    def __init__(self, text, tokenizer, seq_len=128, stride=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        
        # Encode the entire text
        encoded = tokenizer.encode(text)
        self.tokens = encoded["ids"]
        
        # Create training examples with a sliding window approach
        self.examples = []
        for i in range(0, len(self.tokens) - seq_len, stride):
            self.examples.append(self.tokens[i:i + seq_len])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Get input sequence
        input_ids = torch.tensor(self.examples[idx][:-1], dtype=torch.long)
        # Get target (shifted by one position)
        labels = torch.tensor(self.examples[idx][1:], dtype=torch.long)
        
        return {"input_ids": input_ids, "labels": labels}


def get_wiki_page_and_train_tokenizer(page_title=None, max_vocab_size=5000):
    """
    Download a Wikipedia page and train a tokenizer on it
    """
    # Initialize tokenizer
    tokenizer = SimpleHindiTokenizer()
    
    # Download and preprocess Wikipedia page
    wiki_text = tokenizer.download_hindi_wikipedia_page(page_title)
    processed_text = tokenizer.preprocess_hindi_text(wiki_text)
    
    # Save the corpus
    corpus_path = tokenizer.save_corpus(processed_text)
    
    # Build vocabulary
    tokenizer.build_vocabulary(processed_text, max_vocab_size)
    
    # Save tokenizer
    tokenizer.save_tokenizer()
    
    return tokenizer, corpus_path, processed_text


def train_gemini_flash_model(tokenizer, text, 
                             dim=256, 
                             depth=6, 
                             heads=8, 
                             mlp_dim=1024, 
                             max_seq_len=128, 
                             batch_size=8, 
                             lr=3e-4, 
                             epochs=10, 
                             save_path="gemini_flash_hindi.pt"):
    """
    Train the Gemini Flash model on Hindi text
    """
    # Create dataset
    dataset = WikiTextDataset(text, tokenizer, seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    vocab_size = len(tokenizer.vocab)
    model = GeminiFlashModel(vocab_size, dim, depth, heads, mlp_dim, max_seq_len)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nStarting training Gemini Flash model for {epochs} epochs")
    print(f"Model architecture: dim={dim}, depth={depth}, heads={heads}, mlp_dim={mlp_dim}")
    
    total_steps = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            # Reshape logits to [batch_size * seq_len, vocab_size]
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_steps += 1
            
            # Print progress every 10 steps
            if total_steps % 10 == 0:
                print(f"Step {total_steps} - Loss: {loss.item():.4f}")
        
        # Print epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    # Save the model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "mlp_dim": mlp_dim,
            "max_seq_len": max_seq_len
        }
    }, save_path)
    
    print(f"\nTraining completed after {total_steps} steps")
    print(f"Model saved to {save_path}")
    
    return model, total_steps


def test_model_generation(model, tokenizer, prompt="हिन्दी", max_length=50):
    """
    Test the model by generating text from a prompt
    """
    print(f"\nGenerating text from prompt: '{prompt}'")
    
    # Encode the prompt
    encoded = tokenizer.encode(prompt)
    input_ids = encoded["ids"]
    
    # Generate text
    output_ids = model.generate(input_ids, max_length=max_length)
    
    # Decode the generated text
    generated_text = tokenizer.decode(output_ids)
    
    print(f"Generated text: {generated_text}")
    return generated_text


def main():
    print("===== Gemini Flash Hindi LLM Training =====")
    
    # Step 1: Download Hindi Wikipedia page and train tokenizer
    wiki_page_title = "भारत"  # India
    tokenizer, corpus_path, processed_text = get_wiki_page_and_train_tokenizer(wiki_page_title)
    
    # Print some stats about the downloaded text
    print(f"\nDownloaded text statistics:")
    print(f"Character count: {len(processed_text)}")
    print(f"Word count (approx): {len(processed_text.split())}")
    print(f"First 100 characters: {processed_text[:100]}...")
    
    # Step 2: Train the Gemini Flash model
    # Using smaller dimensions for faster training
    model, total_steps = train_gemini_flash_model(
        tokenizer, 
        processed_text,
        dim=128,          # Embedding dimension (smaller for demo)
        depth=4,          # Number of transformer blocks
        heads=4,          # Number of attention heads
        mlp_dim=512,      # Hidden dimension in MLP
        max_seq_len=64,   # Maximum sequence length
        batch_size=4,     # Batch size
        lr=5e-4,          # Learning rate
        epochs=5          # Number of epochs
    )
    
    # Step 3: Test the model with generation
    test_model_generation(model, tokenizer, "भारत एक")
    
    # Print summary information
    print("\n===== Training Summary =====")
    print(f"Trained on Wikipedia page: {wiki_page_title}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Total training steps: {total_steps}")
    print(f"Model architecture: Gemini Flash 1.5 (lightweight version)")
    print(f"Parameters: 4 layers, 4 heads, 128 dimensions")
    print("===============================")


if __name__ == "__main__":
    main()