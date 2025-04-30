import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import string
import warnings
import os
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = '/Users/baha/MyWork/{2003199}_{2003584}_TextMiningProject/NaiveBayesOutputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a custom print function that writes to both console and file
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Initialize logger with timestamp in filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(OUTPUT_DIR, f'metrics_{timestamp}.txt')
logger = TeeLogger(log_file)

USE_EMOJI = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=True)
except ImportError:
    print("Installing NLTK package...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'nltk'])
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

def save_plot(plt, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.close()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'[^a-zA-Z\s.,!?;:()\'\"-]', ' ', text)
    
    text = ' '.join(text.split())
    
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    text = re.sub(r'!+', ' ! ', text)
    text = re.sub(r'\?+', ' ? ', text)
    
    text = ' '.join(text.split())
    
    return text

def extract_text_features(text):
    features = {}
    
    words = text.split()
    features['word_count'] = len(words)
    features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
    
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['period_count'] = text.count('.')
    features['comma_count'] = text.count(',')
    
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = re.split(r'[.!?]+', text)
    features['sentence_count'] = len(sentences)
    features['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    features['emphasis_ratio'] = (features['exclamation_count'] + features['question_count']) / features['word_count'] if features['word_count'] > 0 else 0
    
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
    features['long_word_ratio'] = sum(1 for word in words if len(word) > 6) / len(words) if words else 0
    features['short_word_ratio'] = sum(1 for word in words if len(word) < 4) / len(words) if words else 0
    
    features['punct_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
    features['upper_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    return features

print("Reading data...")
df = pd.read_csv('büyütülmüşdata.csv', sep=';', skiprows=1)

df['sentiment'] = df['sentiment'].str.lower().str.strip()
df['sentiment'] = df['sentiment'].replace({
    'neut,': 'neutral',
    'neut': 'neutral',
    'pos': 'positive',
    'neg': 'negative',
    'positive,': 'positive',
    'negative,': 'negative',
    'pos,': 'positive',
    'neg,': 'negative'
})

valid_sentiments = ['positive', 'negative', 'neutral']
df = df[df['sentiment'].isin(valid_sentiments)]

print("Balancing data...")
sentiment_counts = df['sentiment'].value_counts()
min_count = sentiment_counts.min()
balanced_df = pd.DataFrame()
for sentiment in valid_sentiments:
    sentiment_df = df[df['sentiment'] == sentiment]
    if len(sentiment_df) > min_count:
        sentiment_df = sentiment_df.sample(n=min_count, random_state=42)
    balanced_df = pd.concat([balanced_df, sentiment_df])

df = balanced_df

print("Preprocessing text data...")
df['clean_text'] = df['clean_text'].apply(preprocess_text)

print("Extracting additional features...")
text_features = df['clean_text'].apply(extract_text_features)
for feature in text_features.iloc[0].keys():
    df[feature] = text_features.apply(lambda x: x[feature])

plt.figure(figsize=(10, 6))
df['sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
save_plot(plt, 'sentiment_distribution.png')

df.dropna(subset=['clean_text', 'sentiment'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['sentiment'],
    test_size=0.2, 
    random_state=42,
    stratify=df['sentiment']
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=100000,  
        ngram_range=(1, 6),  
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        min_df=1, 
        max_df=0.99, 
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b'  
    )),
    ('clf', MultinomialNB())
])

param_grid = {
    'tfidf__max_features': [80000, 100000, 120000],
    'tfidf__ngram_range': [(1, 5), (1, 6), (2, 6)],
    'tfidf__min_df': [1, 2],
    'tfidf__max_df': [0.95, 0.99],
    'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]  
}

print("Performing grid search...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='f1_weighted'
)

grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)

y_pred = grid_search.predict(X_test)

print("\nDetailed Evaluation Metrics:")
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("F1 Score (Weighted):", f1_score(y_test, y_pred, average='weighted'))
print("Precision Score (Weighted):", precision_score(y_test, y_pred, average='weighted'))
print("Recall Score (Weighted):", recall_score(y_test, y_pred, average='weighted'))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
save_plot(plt, 'confusion_matrix.png')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_metrics_comparison(y_test, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted')
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    save_plot(plt, 'metrics_comparison.png')

def plot_feature_importance_heatmap(model, n_features=20):
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    
    feature_importance = np.abs(clf.feature_log_prob_)
    avg_importance = np.mean(feature_importance, axis=0)
    top_indices = np.argsort(avg_importance)[-n_features:]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_importance[:, top_indices], 
                xticklabels=[feature_names[i] for i in top_indices],
                yticklabels=clf.classes_,
                cmap='YlOrRd')
    plt.title('Feature Importance Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(plt, 'feature_importance_heatmap.png')

def plot_class_distribution(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(y=y)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    save_plot(plt, 'class_distribution.png')

def analyze_feature_importance(model, X, y, n_features=100):  
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    
   
    feature_importance = np.abs(clf.feature_log_prob_)
    
    avg_importance = np.mean(feature_importance, axis=0)
    
    top_indices = np.argsort(avg_importance)[-n_features:]
    
    print("\nTop {} Most Important Features:".format(n_features))
    for idx in top_indices:
        print(f"{feature_names[idx]}: {avg_importance[idx]:.4f}")

analyze_feature_importance(grid_search.best_estimator_, X_train, y_train)

cv_scores = cross_val_score(grid_search.best_estimator_, df['clean_text'], df['sentiment'], cv=5, scoring='f1_weighted')
print("\nCross-validation scores:", cv_scores)
print("Average CV score: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

plot_metrics_comparison(y_test, y_pred)
save_plot(plt, f'metrics_comparison_{timestamp}.png')

plot_feature_importance_heatmap(grid_search.best_estimator_)
save_plot(plt, f'feature_importance_heatmap_{timestamp}.png')

plot_class_distribution(y_train)
save_plot(plt, f'class_distribution_{timestamp}.png')

plot_learning_curve(grid_search.best_estimator_, 
                   "Learning Curves", X_train, y_train, cv=5)
save_plot(plt, f'learning_curves_{timestamp}.png')

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=grid_search.best_estimator_.named_steps['clf'].classes_,
            yticklabels=grid_search.best_estimator_.named_steps['clf'].classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
save_plot(plt, f'confusion_matrix_{timestamp}.png')

plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores)
plt.title('Cross-validation Scores Distribution')
plt.ylabel('F1 Score')
save_plot(plt, f'cv_scores_distribution_{timestamp}.png')

sys.stdout = logger.terminal
logger.log.close()

print(f"\nAll outputs have been saved to: {OUTPUT_DIR}")
print(f"Metrics log file: {log_file}")

def show_top_features(model, n=100):  
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    
    print("\nMost important features (for each class):")
    for i, label in enumerate(clf.classes_):
        top_features = np.argsort(clf.feature_log_prob_[i])[-n:]
        print(f"\n{label.upper()}:")
        for idx in top_features:
            print(f"{feature_names[idx]}: {np.exp(clf.feature_log_prob_[i][idx]):.4f}")

show_top_features(grid_search.best_estimator_)

