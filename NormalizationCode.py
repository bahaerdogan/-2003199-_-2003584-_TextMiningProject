import re
import csv

input_file = "CommentsEnglish.csv"
output_file = "CommentsEnglish_Normalized.csv"

normalized_data = []

def normalize_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Remove digits
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

with open(input_file, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        normalized_review = normalize_text(row["review"])
        normalized_data.append({
            "film": row["film"],
            "sort_type": row["sort_type"],
            "review": normalized_review
        })

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["film", "sort_type", "review"])
    writer.writeheader()
    writer.writerows(normalized_data)

print(f"Normalization completed. Saved to {output_file}.")
