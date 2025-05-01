import matplotlib.pyplot as plt
import numpy as np
import json
import os
import glob

def load_metrics(file_path):
    """Load metrics from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_training_history():
    """Extract training history from checkpoint files."""
    # Initialize lists to store the data
    epochs = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    checkpoint_files = glob.glob('results/checkpoint-*/trainer_state.json')
    
    # Collect all evaluation data
    for file_path in checkpoint_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for log in data['log_history']:
                if 'eval_accuracy' in log:
                    epochs.append(log['epoch'])
                    accuracies.append(log['eval_accuracy'])
                    precisions.append(log['eval_precision'])
                    recalls.append(log['eval_recall'])
                    f1_scores.append(log['eval_f1'])
    
    # Sort all lists based on epochs
    sorted_data = sorted(zip(epochs, accuracies, precisions, recalls, f1_scores))
    epochs, accuracies, precisions, recalls, f1_scores = zip(*sorted_data)
    
    # Return as dictionary
    return {
        'epochs': list(epochs),
        'eval_accuracy': list(accuracies),
        'eval_precision': list(precisions),
        'eval_recall': list(recalls),
        'eval_f1': list(f1_scores)
    }

def plot_metric_graph(epochs, values, metric_name, save_path):
    """Plot individual metric graph."""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker='o', linewidth=2, markersize=8)
    plt.title(f'{metric_name} Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on points
    for i, v in enumerate(values):
        plt.text(epochs[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def export_individual_graphs(output_dir='graphs'):
    """Export individual graphs for each metric."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get training history
    history = extract_training_history()
    
    # Define metrics to plot
    metrics = {
        'eval_accuracy': 'Accuracy',
        'eval_precision': 'Precision',
        'eval_recall': 'Recall',
        'eval_f1': 'F1 Score'
    }
    
    # Create individual graphs
    for metric_key, metric_name in metrics.items():
        save_path = os.path.join(output_dir, f'{metric_key.replace("eval_", "")}_graph.png')
        plot_metric_graph(history['epochs'], history[metric_key], metric_name, save_path)
        print(f"Exported {metric_name} graph to {save_path}")

if __name__ == '__main__':
    export_individual_graphs()
