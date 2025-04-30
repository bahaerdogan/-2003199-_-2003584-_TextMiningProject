import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# CSV dosyasını oku
df = pd.read_csv("normalized_outputxx.csv")
# NaN içeren satırları temizle
df.dropna(subset=['clean_text', 'sentiment'], inplace=True)


# Eğer veri boşsa kontrol et
if df.empty:
    print("Hata: CSV dosyası boş.")
    exit()

# clean_text sütununu kontrol et
if 'clean_text' not in df.columns or 'sentiment' not in df.columns:
    print("Hata: 'clean_text' ve/veya 'sentiment' sütunu eksik.")
    exit()

# TF-IDF vektörleştirme
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

# Etiketleri al
y = df['sentiment']

# Eğitim-test bölme
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Naive Bayes modelini eğit
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Tahmin yap
y_pred = nb_model.predict(X_test)

# Sonuçları yazdır
print("TF-IDF vektör boyutu:", X_tfidf.shape)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

