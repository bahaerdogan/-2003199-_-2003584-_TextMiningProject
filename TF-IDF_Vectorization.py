import pandas as pd

df = pd.read_csv("normalized_outputxx.csv")  
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

print("TF-IDF vector size: ", X_tfidf.shape)
