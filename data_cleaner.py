import csv

COUNTRIES = {
    "America", "India", "China", "Germany", "France", "Japan", "Canada",
    "Russia", "Brazil", "Italy", "Australia", "England", "Pakistan",
    "Bangladesh", "Mexico", "Nepal", "Afghanistan", "Sri Lanka", "Norway",
    "Spain", "Korea", "USA", "UK", "UAE", "South Africa"
}

input_file = "data/ner_dataset_corrected.csv"
output_file = "data/ner_dataset_newcorrected.csv"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline='', encoding="utf-8") as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)


    header = next(reader)
    writer.writerow(header)

    for row in reader:
        if len(row) == 4:
            sentence_id, word, pos, tag = row
            clean_word = word.strip()

            if clean_word in COUNTRIES:
                if "ORG" in tag or "PER" in tag or "MISC" in tag:
                    print(f"Fixing tag: {clean_word} -> {tag} to B-LOC/I-LOC")
                    if tag.startswith("B-"):
                        tag = "B-LOC"
                    elif tag.startswith("I-"):
                        tag = "I-LOC"
                    else:
                        tag = "B-LOC"  
                    row[3] = tag

        writer.writerow(row)

print(f"Correction complete. Cleaned file saved as: {output_file}")
