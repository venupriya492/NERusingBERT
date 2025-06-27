from transformers import BertTokenizerFast, Trainer, TrainingArguments
from src.data_preprocessing import prepare_data
from src.model_definition import get_model
from src.config import BATCH_SIZE, EPOCHS, LR, MODEL_NAME
from src.metrics import compute_metrics
import torch
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    print("Cleared MPS cache memory before training.")

def train_model():
    print("Starting BERT NER model training\n")

    train_dataset, val_dataset, TAG2ID, ID2TAG = prepare_data("data/ner_dataset_tokenized.csv")

    model = get_model()
    model.config.id2label = {i: tag for tag, i in TAG2ID.items()}
    model.config.label2id = TAG2ID

    training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_dir='./logs',
    logging_steps=50,
    save_total_limit=1
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nSaving model and tokenizer...")
    trainer.save_model("model/")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained("model/")
    torch.save(model.state_dict(), "model/pytorch_model.bin")

    with open("model/tag2id.json", "w") as f:
        json.dump(TAG2ID, f, indent=2)
    with open("model/id2tag.json", "w") as f:
        json.dump(ID2TAG, f, indent=2)

    print("\n✅ Training complete. Model, tokenizer, and mappings saved to 'model/' directory.")

if __name__ == "__main__":
    train_model()
