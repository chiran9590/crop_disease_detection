"""
Evaluation script for crop disease detection CNN.
Computes accuracy, precision, recall, F1, confusion matrix, and classification report.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from data_pipeline import build_data_pipeline

# Evaluation configuration
EVAL_CONFIG = {
    'model_path': 'best_model.keras',
    'confusion_matrix_path': 'confusion_matrix.png',
    'report_path': 'classification_report.txt',
}


def load_model(model_path):
    """Load trained model from disk."""
    model = tf.keras.models.load_model(model_path)
    return model


def evaluate_model(model, test_ds, class_names):
    """Evaluate model on test dataset and compute metrics."""
    print("Evaluating model on test set...")
    
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1: {f1[i]:.4f}")
        print(f"  Support: {support[i]}")
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print(f"\nMacro-average metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1: {macro_f1:.4f}")
    
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_classification_report(y_true, y_pred, class_names, save_path):
    """Generate and save classification report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
    
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    
    print(f"Classification report saved to {save_path}")
    print("\nClassification Report:")
    print(report)


def main():
    """Main evaluation function."""
    pipeline = build_data_pipeline()
    test_ds = pipeline['test_dataset']
    class_names = pipeline['class_names']
    
    model = load_model(EVAL_CONFIG['model_path'])
    
    y_true, y_pred = evaluate_model(model, test_ds, class_names)
    
    plot_confusion_matrix(y_true, y_pred, class_names, EVAL_CONFIG['confusion_matrix_path'])
    
    save_classification_report(y_true, y_pred, class_names, EVAL_CONFIG['report_path'])


if __name__ == '__main__':
    main()
