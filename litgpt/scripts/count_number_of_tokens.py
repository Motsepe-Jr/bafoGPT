import json
from transformers import AutoTokenizer
from tokenizer import Tokenizer

tokenizer = Tokenizer("checkpoints/google/gemma-2-2b")

input_data = "data/train.json"

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text))

with open(input_data, 'r', encoding='utf-8') as f:
    data = json.load(f)

total_tokens = 0

for item in data:
    instruction = item.get('instruction', "")
    output = item.get('output', "")
    
    instruction_tokens = count_tokens(tokenizer, instruction)
    output_tokens = count_tokens(tokenizer, output)
    
    total_tokens += instruction_tokens + output_tokens
    
print(f"Total number of tokens: {total_tokens}")
