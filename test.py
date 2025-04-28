import requests

response = requests.post(
    "http://localhost:5000/tokenize",
    json={"text": "मेरा नाम जॉन है"}
)
print(response.json())  # Should output: ['मेरा', 'नाम', 'जॉन', 'है']