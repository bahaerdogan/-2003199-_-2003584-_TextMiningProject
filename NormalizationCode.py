import pandas as pd
import re


custom_stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
]


def normalize_text(text):
    if pd.isna(text):  
        return ""
    text = text.lower()                          
    text = re.sub(r'[^\w\s]', '', text)         
    text = re.sub(r'\d+', '', text)               
    tokens = text.split()                         
    filtered_tokens = [w for w in tokens if w not in custom_stopwords]  
    return ' '.join(filtered_tokens)              

def main():
    input_file = 'DatasetBeforeNormalization.csv'        
    output_file = 'normalized_output.csv'  
    text_column = 'review'  

    try:
        df = pd.read_csv(input_file, delimiter=';', on_bad_lines='warn')
    except Exception as e:
        print(f"cantreadfile- {e}")
        return

    if text_column not in df.columns:
        print(f"eerr '{text_column}' cant file columns.")
        return

    df['clean_text'] = df[text_column].apply(normalize_text)

    df.to_csv(output_file, index=False)
    print(f"outpt {output_file}")


if __name__ == "__main__":
    main()
