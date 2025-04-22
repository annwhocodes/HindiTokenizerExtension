import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCasualLM, AutoTokenizer, pipeline


tokenizer = AutoTokenizer.from_pretrained("../models/microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "../models/microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50, 
    do_sample=False,
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. "

output = generator(prompt)

print(output[0]['generated_text'])