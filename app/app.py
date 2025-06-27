import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizerFast
from src.model_definition import get_model
from src.config import MODEL_NAME
import torch
import json

app = Flask(__name__)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# Load label mapping
with open("model/id2tag.json") as f:
    ID2TAG = json.load(f)

# Load model
model = get_model()

model_path = "model/pytorch_model.bin"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.config.id2label = {int(k): v for k, v in ID2TAG.items()}
    model.config.label2id = {v: int(k) for k, v in ID2TAG.items()}
    model.eval()
else:
    raise FileNotFoundError("Trained model not found at model/pytorch_model.bin")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sentence = request.form["sentence"].strip()
        words = sentence.split()

        inputs = tokenizer(words, is_split_into_words=True,
                           return_tensors="pt", truncation=True,
                           padding="max_length", max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0]

        word_ids = inputs.word_ids(batch_index=0)
        result = []
        seen = set()

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen:
                continue
            word = words[word_idx]
            label_id = predictions[idx].item()
            label = ID2TAG.get(str(label_id), "O")
            result.append({"word": word, "entity": label})
            seen.add(word_idx)

        return jsonify(result=result)

    except Exception as e:
        print(f"\nError during prediction: {e}\n")
        return render_template("index.html", input_text=sentence, predictions=None, error=True)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
