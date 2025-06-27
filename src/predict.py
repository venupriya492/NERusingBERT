from transformers import BertTokenizerFast
from src.model_definition import get_model
import torch
import json
import numpy as np

def get_label_list(model_dir="model"):
    with open(f"{model_dir}/id2tag.json") as f:
        id2tag = json.load(f)
    # Sort by int key because JSON keys are strings
    return [id2tag[str(i)] for i in range(len(id2tag))]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("model", device=device)
    tokenizer = BertTokenizerFast.from_pretrained("model")
    label_list = get_label_list("model")

    model.eval()

    while True:
        text = input("\nEnter sentence (or type 'exit'): ")
        if text.lower() == "exit":
            break

        words = text.strip().split()
        encoding = tokenizer(words, is_split_into_words=True,
                             return_tensors="pt", truncation=True,
                             padding="max_length", max_length=128)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None

        print("\nNamed Entity Predictions:")
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                word = words[word_idx]
                label = label_list[predictions[idx]]
                print(f"{word} -> {label}")
            previous_word_idx = word_idx

if __name__ == "__main__":
    print("Enter text to extract named entities (type 'exit' to quit):")
    main()
