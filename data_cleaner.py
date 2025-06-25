import csv

countries = [
    "America", "India", "China", "Germany", "France", "Japan", "Canada",
    "Russia", "Brazil", "Italy", "Australia", "England", "Pakistan",
    "Bangladesh", "Mexico", "Nepal", "Afghanistan", "Sri Lanka", "Norway",
    "Spain", "Korea", "USA", "UK", "UAE", "South Africa"
]

input_file = "data/ner_dataset_corrected.csv"
output_file = "data/ner_dataset_newcorrected.csv"

infile = open(input_file, "r")
outfile = open(output_file, "w", newline='')

reader = csv.reader(infile)
writer = csv.writer(outfile)

header = next(reader)
writer.writerow(header)

for row in reader:
    if len(row) == 4:
        word = row[1].strip() 
        tag = row[3]

        if word in countries and ("ORG" in tag or "PER" in tag or "MISC" in tag):
            print("Fixing tag for:", word)
            if tag.startswith("I-"):
                row[3] = "I-LOC"
            else:
                row[3] = "B-LOC"

    writer.writerow(row)

infile.close()
outfile.close()

print("New file saved as:", output_file)
