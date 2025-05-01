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
import os
from sklearn.utils.class_weight import compute_class_weight

# Define all functions and classes first
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

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

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

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights to loss calculation
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Main execution code
if __name__ == '__main__':
    # 1. Load your dataset
    data = pd.read_csv('büyütülmüşdata.csv', sep=';')
    
    print("\nDataset columns:", data.columns.tolist())
    print("\nFirst few rows of the dataset:")
    print(data.head())

    # 2. Convert sentiment to lowercase and map to numerical values
    print("\nChecking sentiment values...")
    print("Unique sentiment values before cleaning:", data['sentiment'].unique())

    # Define label mapping
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Convert to string and lowercase
    data['sentiment'] = data['sentiment'].astype(str).str.lower().str.strip()

    # Handle common variations
    data['sentiment'] = data['sentiment'].replace({
        'neut,': 'neutral',
        'neut': 'neutral',
        'pos': 'positive',
        'neg': 'negative'
    })

    # Check for invalid sentiment values
    invalid_sentiments = data[~data['sentiment'].isin(label_map.keys())]['sentiment'].unique()
    if len(invalid_sentiments) > 0:
        print(f"\nWarning: Found invalid sentiment values: {invalid_sentiments}")
        print("Dropping rows with invalid sentiment values...")
        print(f"Number of rows before cleaning: {len(data)}")
        data = data[data['sentiment'].isin(label_map.keys())]
        print(f"Number of rows after cleaning: {len(data)}")

    if len(data) == 0:
        raise ValueError("No valid data remaining after cleaning. Please check your sentiment values.")

    print("\nFinal sentiment distribution:")
    print(data['sentiment'].value_counts())

    data['label'] = data['sentiment'].map(label_map)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data['label']),
        y=data['label']
    )
    class_weight_dict = dict(zip(np.unique(data['label']), class_weights))
    print("\nClass weights:", class_weight_dict)

    # Convert class weights to tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Split and prepare data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['clean_text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
    )

    # Initialize tokenizer and tokenize data
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=256)

    # Create datasets
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

    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3,
        problem_type="single_label_classification",
        attention_dropout=0.3,
        dropout=0.3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id=label_map
    ).to(device)

    # Create callback instance
    metrics_callback = MetricsCallback()

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_num_workers=0,
        optim='adamw_torch'
    )

    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )

    # Train and evaluate
    print("\nStarting training...")
    trainer.train()

    results = trainer.evaluate()
    predictions = trainer.predict(test_dataset)

    # Generate evaluation artifacts
    predicted_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids

    plot_training_metrics(metrics_callback)
    plot_confusion_matrix(true_labels, predicted_labels)

    # Save results
    evaluation_results = {
        'accuracy': float(results['eval_accuracy']),
        'precision': float(results['eval_precision']),
        'recall': float(results['eval_recall']),
        'f1': float(results['eval_f1']),
        'loss': float(results['eval_loss'])
    }

    with open('evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print("\nEvaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

    # Save model artifacts
    print("\nSaving the model and tokenizer...")
    model_save_path = './sentiment_model'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save label mapping
    label_mapping = {
        'id2label': {str(i): label for label, i in label_map.items()},
        'label2id': label_map
    }
    with open(f"{model_save_path}/label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=4)

    # Test prediction
    sample_text = ["The movie was an absolute masterpiece, highly recommended!"]
    sample_tokens = tokenizer(sample_text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**sample_tokens)
    predicted_class = torch.argmax(output.logits, dim=1)
    sentiment = ['negative', 'neutral', 'positive'][predicted_class.item()]
    print(f"\nPredicted Sentiment: {sentiment}")
