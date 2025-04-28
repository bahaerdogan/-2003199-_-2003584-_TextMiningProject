# Install necessary libraries
# pip install transformers datasets scikit-learn torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TrainerCallback
import json

# 1. Load your dataset
data = pd.read_csv('reviews.csv')  # assumes columns: 'review', 'rating'
print(data.head())

# 2. Preprocess the labels (optional: if ratings are not from 0)
# Here we map ratings (1-5) to (0-4) because Hugging Face expects labels starting from 0
data['rating'] = data['rating'] - 1

# 3. Split into train/test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['review'], data['rating'], test_size=0.2, random_state=42, stratify=data['rating']
)

# 4. Load tokenizer and tokenize data
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

# 5. Prepare Hugging Face Dataset objects
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': list(train_labels)
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': list(test_labels)
})

# 6. Load DistilBERT model for classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

# Create a custom callback to store training metrics
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.training_loss = []
        self.eval_loss = []
        self.eval_accuracy = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_loss.append(logs['eval_loss'])
            if 'eval_accuracy' in logs:
                self.eval_accuracy.append(logs['eval_accuracy'])

# Function to plot training metrics
def plot_training_metrics(callback):
    plt.figure(figsize=(12, 4))
    
    # Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(callback.training_loss, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    # Evaluation Metrics
    plt.subplot(1, 2, 2)
    steps = range(len(callback.eval_accuracy))
    plt.plot(steps, callback.eval_accuracy, label='Accuracy')
    plt.plot(steps, callback.eval_loss, label='Loss')
    plt.title('Evaluation Metrics Over Time')
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Function to compute all metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate all metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Create callback instance
metrics_callback = MetricsCallback()

# Update training arguments to include evaluation metrics
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# Update Trainer with compute_metrics and callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback]
)

# Train the model
trainer.train()

# Evaluate and get predictions
results = trainer.evaluate()
predictions = trainer.predict(test_dataset)

# Get predicted labels
predicted_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

# Plot training metrics
plot_training_metrics(metrics_callback)

# Plot confusion matrix
plot_confusion_matrix(true_labels, predicted_labels)

# Save evaluation metrics to a JSON file
evaluation_results = {
    'accuracy': float(results['eval_accuracy']),
    'precision': float(results['eval_precision']),
    'recall': float(results['eval_recall']),
    'f1': float(results['eval_f1']),
    'loss': float(results['eval_loss'])
}

with open('evaluation_metrics.json', file='w') as f:
    json.dump(evaluation_results, f, indent=4)

print("\nEvaluation Results:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")

# 11. Example prediction
sample_text = ["The movie was an absolute masterpiece, highly recommended!"]
sample_tokens = tokenizer(sample_text, truncation=True, padding=True, return_tensors="pt")
with torch.no_grad():
    output = model(**sample_tokens)
predicted_class = torch.argmax(output.logits, dim=1)
print(f"\nPredicted Rating (0=1-star, 4=5-star): {predicted_class.item() + 1}")
