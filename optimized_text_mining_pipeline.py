import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os
import sys
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"./NaiveBayesOutputs_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(plt, filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = ' '.join(text.split())
    return text

df = pd.read_csv("büyütülmüşdata.csv", sep=";", skiprows=1)
df['sentiment'] = df['sentiment'].str.lower().str.strip()
df['sentiment'] = df['sentiment'].replace({
    'neut,': 'neutral', 'neut': 'neutral',
    'pos': 'positive', 'neg': 'negative',
    'positive,': 'positive', 'negative,': 'negative',
    'pos,': 'positive', 'neg,': 'negative'
})
valid_sentiments = ['positive', 'negative', 'neutral']
df = df[df['sentiment'].isin(valid_sentiments)]
df['clean_text'] = df['clean_text'].astype(str).apply(preprocess_text)
df = df[df['clean_text'].str.split().apply(len) > 5]

min_count = df['sentiment'].value_counts().min()
df = df.groupby('sentiment').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), sublinear_tf=True)),
    ('clf', MultinomialNB())
])
param_grid = {
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    'clf__alpha': [0.001, 0.01, 0.1]
}
grid = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred, average='weighted'),
    'Precision': precision_score(y_test, y_pred, average='weighted'),
    'Recall': recall_score(y_test, y_pred, average='weighted')
}
print("Best Parameters:", grid.best_params_)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=grid.best_estimator_.named_steps['clf'].classes_,
            yticklabels=grid.best_estimator_.named_steps['clf'].classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
save_plot(plt, 'confusion_matrix.png')

plt.figure(figsize=(8,5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.ylim(0,1)
plt.title("Model Evaluation Metrics")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
save_plot(plt, 'metrics_comparison.png')

feature_names = grid.best_estimator_.named_steps['tfidf'].get_feature_names_out()
class_labels = grid.best_estimator_.named_steps['clf'].classes_
log_probs = grid.best_estimator_.named_steps['clf'].feature_log_prob_
topn = 20

plt.figure(figsize=(14,6))
for i, label in enumerate(class_labels):
    top_indices = np.argsort(log_probs[i])[-topn:]
    plt.subplot(1, 3, i+1)
    plt.barh([feature_names[j] for j in top_indices], [log_probs[i][j] for j in top_indices])
    plt.title(f"Top Words for {label.capitalize()}")
    plt.tight_layout()
save_plot(plt, 'top_features.png')

def plot_metric_curve(estimator, X, y, metric_name, scorer):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, scoring=scorer,
                                                            train_sizes=np.linspace(0.1, 1.0, 5))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
    plt.title(f"{metric_name} Curve")
    plt.xlabel("Training examples")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    save_plot(plt, f"{metric_name.lower().replace(' ', '_')}_curve.png")

plot_metric_curve(grid.best_estimator_, df['clean_text'], df['sentiment'], 'Accuracy', 'accuracy')
plot_metric_curve(grid.best_estimator_, df['clean_text'], df['sentiment'], 'F1 Score', 'f1_weighted')
plot_metric_curve(grid.best_estimator_, df['clean_text'], df['sentiment'], 'Precision', 'precision_weighted')
plot_metric_curve(grid.best_estimator_, df['clean_text'], df['sentiment'], 'Recall', 'recall_weighted')

def plot_all_metrics_together(estimator, X, y):
    metrics = {
        'Accuracy': 'accuracy',
        'F1 Score': 'f1_weighted',
        'Precision': 'precision_weighted',
        'Recall': 'recall_weighted'
    }
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 5)
    plt.figure(figsize=(10,6))

    for metric_name, scorer in metrics.items():
        _, _, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, scoring=scorer, train_sizes=train_sizes)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes * len(X), test_mean, label=metric_name)

    plt.title("Cross-Validation Scores vs Training Size")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    save_plot(plt, "all_metrics_curve.png")

plot_all_metrics_together(grid.best_estimator_, df['clean_text'], df['sentiment'])

print(f"All visualizations and outputs are saved to {OUTPUT_DIR}")
