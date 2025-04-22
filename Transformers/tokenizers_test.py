from transformers import AutoTokenizer

sentence = "Hello World!"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
token_ids = tokenizer(sentence).input_ids
print(token_ids)

for token_id in token_ids:
    print(tokenizer.decode(token_id))
