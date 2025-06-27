MODEL_NAME = "bert-base-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5
TAG2ID = {
  "0": "O",
  "1": "B-GEO",
  "2": "B-GPE",
  "3": "B-ORG",
  "4": "I-PER",
  "5": "B-TIM",
  "6": "B-PER",
  "7": "I-ORG",
  "8": "I-GEO",
  "9": "I-TIM",
  "10": "B-ART",
  "11": "I-GPE",
  "12": "I-ART",
  "13": "B-EVE",
  "14": "I-EVE"
}
ID2TAG = {v: k for k, v in TAG2ID.items()}

