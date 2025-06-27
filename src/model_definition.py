from transformers import BertForTokenClassification
from src.config import TAG2ID
import torch

def get_model(model_dir="bert-base-cased", device="cpu"):
    model = BertForTokenClassification.from_pretrained(model_dir, num_labels=len(TAG2ID))
    return model.to(device)
