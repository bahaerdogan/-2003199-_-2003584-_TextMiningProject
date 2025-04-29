import re
import csv


stopwords = {
    'the', 'is', 'in', 'at', 'to', 'a', 'an', 'and', 'or', 'of', 'for', 'on', 'with', 'this', 'that',
    'it', 'as', 'was', 'but', 'are', 'by', 'be', 'has', 'had', 'have', 'from', 'not', 'so', 'if', 'its',
    'they', 'i', 'you', 'we', 'he', 'she', 'them', 'his', 'her', 'my', 'your', 'our', 'their', 'me', 'do'
}

input_file = "DatasetBeforeNormalization.csv"
output_file = "DatasetAfternormalization.csv"

normalized_data = []

def normalize_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Remove digits
    text = re.sub(r'\d+', '', text)
    
    # 4. Reduce repeated letters (e.g., "fannntastic" → "fantastic")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    text = ' '.join(filtered_words)
    
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
