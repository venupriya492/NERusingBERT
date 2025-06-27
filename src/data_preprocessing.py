import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from transformers import BertTokenizerFast
from src.config import MODEL_NAME, MAX_LEN
import json
import os
from sklearn.utils import shuffle


tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


def read_data(file_path, limit=None):
    df = pd.read_csv(file_path)
    df = df.ffill()
    df['Tag'] = df['Tag'].str.upper()

    df['Tag'] = df['Tag'].replace({
        'B-MISC': 'O', 'I-MISC': 'O',
        'B-NAT': 'O', 'I-NAT': 'O'
    })

    if limit:
        df = df[df["Sentence #"].isin(df["Sentence #"].unique()[:limit])]

    print("\nTag distribution:\n", df['Tag'].value_counts())

    TAGS = sorted(df['Tag'].unique().tolist())
    TAG2ID = {tag: i for i, tag in enumerate(TAGS)}
    ID2TAG = {i: tag for tag, i in TAG2ID.items()}

    os.makedirs("model", exist_ok=True)
    with open("model/tag2id.json", "w") as f:
        json.dump(TAG2ID, f)
    with open("model/id2tag.json", "w") as f:
        json.dump(ID2TAG, f)

    grouped = df.groupby("Sentence #")[["Word", "Tag"]].agg(list)
    return grouped, TAG2ID, ID2TAG


class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def encode_tags(tags, encodings, TAG2ID):
    labels = []
    for i, label in enumerate(tags):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(TAG2ID[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    return labels


def prepare_data(file_path, limit=1000):
    grouped, TAG2ID, ID2TAG = read_data(file_path, limit=limit)
    texts = grouped["Word"].tolist()
    tags = grouped["Tag"].tolist()

    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

    labels = encode_tags(tags, encodings, TAG2ID)
    encodings.pop("offset_mapping") 

    dataset = NERDataset(encodings, labels)

    indices = list(range(len(dataset)))
    indices = shuffle(indices, random_state=42)
    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset, TAG2ID, ID2TAG
