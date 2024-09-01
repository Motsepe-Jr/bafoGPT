import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

def convert_to_jsonl(input_text, tokenizer, max_tokens=8192):
    data_points = []
    combined_text = ""
    token_count = 0
    total_token_count = 0

    for index, line in enumerate(input_text):
        encoded_line = tokenizer.encode(line.strip())
        line_token_count = len(encoded_line)

        if token_count + line_token_count > max_tokens:
            data_point = {
                "source": "",
                "target": combined_text.strip(),
                "category": "unknown"
            }
            data_points.append(data_point)

            combined_text = line
            token_count = line_token_count
        else:
            combined_text += " " + line.strip()
            token_count += line_token_count

        total_token_count += line_token_count  
        
        print("so far --> %", index / len(input_text))

    if combined_text:
        data_point = {
            "source": "",
            "target": combined_text.strip(),
            "category": "unknown"
        }
        data_points.append(data_point)

    with open('pretrain_dataset/jsons/pretraining_dataset.jsonl', 'w', encoding='utf-8') as f:
        for data in data_points:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    token_count_data = {"total_token_count": total_token_count}
    with open('pretrain_dataset/jsons/token_count.json', 'w', encoding='utf-8') as f:
        json.dump(token_count_data, f, ensure_ascii=False, indent=4)

    print(f"Total token count: {total_token_count}")



tokenizer = AutoTokenizer.from_pretrained("tokenization/expanded_tokenizer/")
with open('pretrain_dataset/files/pretrain_dataset.txt', 'r', encoding='utf-8') as file:
    raw_data = file.readlines()

convert_to_jsonl(raw_data, tokenizer)
