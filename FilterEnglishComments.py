import csv
from langdetect import detect

input_file = "AllCommentsBalanced.csv"
output_file = "CommentsEnglish2.csv"

filtered_data = []

with open(input_file, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        text = row["review"]
        try:
            if detect(text) == "en":
                filtered_data.append(row)
        except:
            continue  

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["film", "sort_type", "review"])
    writer.writeheader()
    writer.writerows(filtered_data)

print(f"Filtering completed. Saved only English comments to {output_file}.")
