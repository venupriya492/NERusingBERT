import csv

input_file = "data/ner_dataset.csv"
output_file = "data/ner_dataset_tokenized.csv"

tokenized_data = []

with open(input_file, "r") as infile:
    reader = csv.reader(infile)
    header = next(reader) 

    tokenized_data.append(["Sentence #", "Word", "POS", "Tag"])

    for row in reader:
        if len(row) != 4:
            continue  

        sentence_id, word, pos, tag = row

        sentence_id = sentence_id.strip()
        word = word.strip()
        pos = pos.strip()
        tag = tag.strip()

        words = word.split()

        for i, w in enumerate(words):
            tokenized_data.append([sentence_id, w, pos, tag])

with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(tokenized_data)

print(f"Cleaned and tokenized data saved to: {output_file}")
