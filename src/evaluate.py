import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.metrics import classification_report
from src.model_definition import get_model
from src.data_preprocessing import prepare_data
from src.config import MODEL_NAME, ID2TAG, BATCH_SIZE

def evaluate_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = get_model()
    model.load_state_dict(torch.load("model/pytorch_model.bin", map_location=torch.device("cpu")))
    model.eval()

    dataset = prepare_data("data/ner_dataset.csv")[0]
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    predictions = []
    true_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=2)

        for i in range(len(labels)):
            true_seq = labels[i].tolist()
            pred_seq = pred_ids[i].tolist()

            for true_id, pred_id in zip(true_seq, pred_seq):
                if true_id != -100:
                    true_labels.append(ID2TAG[true_id])
                    predictions.append(ID2TAG[pred_id])

    print("\nClassification Report:\n")
    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    evaluate_model()
