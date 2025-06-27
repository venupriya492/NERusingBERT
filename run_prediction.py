import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import json
import numpy as np

def get_label_list(model_dir="model"):
    try:
        with open(f"{model_dir}/id2tag.json") as f:
            id2tag = json.load(f)
        label_list = [id2tag[str(i)] for i in range(len(id2tag))]
    except Exception:
        label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-NAT", "I-NAT"]
    return label_list

def merge_tokens(tokens, labels, scores):
    entities = []
    entity_tokens = []
    entity_scores = []
    entity_label = None

    for token, label, score in zip(tokens, labels, scores):
        if label.startswith("B-"):
            if entity_tokens:
                entity_text = "".join(entity_tokens)
                avg_score = np.mean(entity_scores)
                entities.append({"entity": entity_text, "label": entity_label, "score": avg_score})
            clean_token = token.replace("##", "")
            entity_tokens = [clean_token]
            entity_scores = [score]
            entity_label = label[2:]
        elif label.startswith("I-") and entity_label == label[2:]:
            clean_token = token.replace("##", "")
            if token.startswith("##"):
                entity_tokens[-1] += clean_token
            else:
                entity_tokens.append(" " + clean_token)
            entity_scores.append(score)
        else:
            if entity_tokens:
                entity_text = "".join(entity_tokens)
                avg_score = np.mean(entity_scores)
                entities.append({"entity": entity_text, "label": entity_label, "score": avg_score})
                entity_tokens = []
                entity_scores = []
                entity_label = None

    if entity_tokens:
        entity_text = "".join(entity_tokens)
        avg_score = np.mean(entity_scores)
        entities.append({"entity": entity_text, "label": entity_label, "score": avg_score})

    return entities


def main():
    model_dir = "model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertForTokenClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    label_list = get_label_list(model_dir)

    print("\nEnter text to extract named entities (type 'exit' to quit):")

    while True:
        text = input("\nInput text: ")
        if text.lower() == "exit":
            break

        encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits 
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        scores = torch.softmax(logits, dim=2)[0].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [label_list[pred] if pred < len(label_list) else "O" for pred in predictions]

        conf_scores = [scores[i, pred] for i, pred in enumerate(predictions)]

        entities = merge_tokens(tokens, labels, conf_scores)

        if entities:
            print("\nNamed Entities:")
            for ent in entities:
                entity_text = ent["entity"].replace("##", "")
                print(f"{entity_text} -> {ent['label']} ({ent['score']:.2f})")
        else:
            print("No named entities found.")

if __name__ == "__main__":
    main()
