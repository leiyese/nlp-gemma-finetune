#!/usr/bin/env python3
"""
Visualize results from the best BERT checkpoint (1755).
Generates confusion matrix and performance graphs.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model results")
    parser.add_argument(
        "--model-name",
        type=str,
        default="models/bert-small-finance/checkpoint-1755",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/test.csv",
        help="Path to the test data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    return parser.parse_args()

def load_and_tokenize_data(tokenizer, test_file):
    """Load and tokenize the test dataset."""
    print(f"[visualize_results] Loading data from {test_file}")
    
    # Load dataset
    test_dataset = load_dataset('csv', data_files={'test': test_file})['test']
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    
    tokenized_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Ensure we have the right columns
    tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")
    
    return tokenized_dataset, test_dataset

def get_predictions(model, dataset, device):
    """Get model predictions on the dataset."""
    model.eval()
    predictions = []
    true_labels = []
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        collate_fn=default_data_collator
    )
    
    print(f"[visualize_results] Getting predictions for {len(dataset)} examples")
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'label_name'}
            
            # Get predictions
            outputs = model(**batch)
            logits = outputs.logits
            
            # Get predicted class
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            true = batch['labels'].cpu().numpy()
            
            predictions.extend(pred)
            true_labels.extend(true)
    
    return np.array(predictions), np.array(true_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Create and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    # Save figure
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[visualize_results] Saved confusion matrix to {output_path}")
    
    return cm

def plot_class_distribution(dataset, output_dir):
    """Plot class distribution in the dataset."""
    # Convert to pandas DataFrame first
    df = pd.DataFrame(dataset)
    # Count classes
    class_counts = df['label_name'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # Add percentage labels on top of bars
    total = sum(class_counts.values)
    for i, p in enumerate(ax.patches):
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=12
        )
    
    plt.title('Class Distribution in Test Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Save figure
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[visualize_results] Saved class distribution to {output_path}")

def plot_performance_metrics(y_true, y_pred, class_names, output_dir):
    """Plot precision, recall, and F1 score for each class."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract metrics for each class
    metrics = pd.DataFrame({
        'Precision': [report[cls]['precision'] for cls in class_names],
        'Recall': [report[cls]['recall'] for cls in class_names],
        'F1-Score': [report[cls]['f1-score'] for cls in class_names]
    }, index=class_names)
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    metrics.plot(kind='bar', figsize=(12, 6))
    plt.title('Performance Metrics by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    output_path = os.path.join(output_dir, 'performance_metrics.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[visualize_results] Saved performance metrics to {output_path}")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[visualize_results] Using device: {device}")
    
    # Load model and tokenizer
    print(f"[visualize_results] Loading model from {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.to(device)
    
    # Load and tokenize data
    tokenized_dataset, raw_dataset = load_and_tokenize_data(tokenizer, args.test_file)
    
    # Get predictions
    y_pred, y_true = get_predictions(model, tokenized_dataset, device)
    
    # Get class names
    class_names = ["Negative", "Neutral", "Positive"]
    
    # Create visualizations
    plot_class_distribution(raw_dataset, args.output_dir)
    cm = plot_confusion_matrix(y_true, y_pred, class_names, args.output_dir)
    plot_performance_metrics(y_true, y_pred, class_names, args.output_dir)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save classification report to file
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print(f"[visualize_results] All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
