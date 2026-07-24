"""
Data pipeline for PlantVillage crop disease detection dataset.
Loads images, splits train/val/test (70/15/15) stratified by class,
applies preprocessing and augmentation, and builds tf.data.Dataset pipelines.
"""

import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import kagglehub

# Configuration dictionary
CONFIG = {
    'dataset_path': os.path.join(os.getcwd(), 'plantvillage_dataset'),
    'kaggle_dataset': 'emmarex/plantdisease',
    'image_size': (224, 224),
    'batch_size': 32,
    'seed': 42,
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'cache_dir': os.path.join(os.getcwd(), 'cache'),
    'num_classes': None,  # Will be determined from dataset
}


def download_dataset():
    """Download PlantVillage dataset from Kaggle if not present locally."""
    if os.path.exists(CONFIG['dataset_path']) and os.listdir(CONFIG['dataset_path']):
        print(f"Dataset found at {CONFIG['dataset_path']}")
        return CONFIG['dataset_path']
    
    print("Downloading PlantVillage dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download(CONFIG['kaggle_dataset'])
        print(f"Dataset downloaded to: {dataset_path}")
        CONFIG['dataset_path'] = dataset_path
        return dataset_path
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")


def load_image_paths_and_labels(dataset_path):
    """
    Load all image paths and their corresponding labels from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory.
    
    Returns:
        Tuple of (image_paths, labels, class_names).
    """
    image_paths = []
    labels = []
    class_names = []
    
    dataset_path = Path(dataset_path)
    
    # Find all class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = dataset_path / class_name
        image_files = list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.jpg')) + \
                      list(class_dir.glob('*.PNG')) + list(class_dir.glob('*.png')) + \
                      list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.JPEG'))
        
        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"Total images found: {len(image_paths)}")
    CONFIG['num_classes'] = len(class_names)
    
    return np.array(image_paths), np.array(labels), class_names


def stratified_split(image_paths, labels):
    """
    Split data into train/val/test sets with stratified sampling.
    
    Args:
        image_paths: Array of image file paths.
        labels: Array of corresponding labels.
    
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, 
                  test_paths, test_labels).
    """
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=CONFIG['test_split'],
        random_state=CONFIG['seed'],
        stratify=labels
    )
    
    # Second split: separate train and val from remaining
    val_ratio = CONFIG['val_split'] / (CONFIG['train_split'] + CONFIG['val_split'])
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_ratio,
        random_state=CONFIG['seed'],
        stratify=train_val_labels
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def compute_class_weights(labels):
    """
    Compute class weights to handle class imbalance.
    
    Args:
        labels: Array of labels.
    
    Returns:
        Dictionary mapping class indices to weights.
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx in range(num_classes):
        class_weights[class_idx] = total_samples / (num_classes * class_counts[class_idx])
    
    print("Class weights:", class_weights)
    return class_weights


# Data augmentation layer using tf.keras.Sequential preprocessing layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])


def load_and_preprocess_image(image_path, label, augment=False):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file.
        label: Corresponding label.
        augment: Whether to apply augmentation.
    
    Returns:
        Tuple of (preprocessed_image, label).
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, CONFIG['image_size'])
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment:
        img = data_augmentation(img)
    
    return img, label


def create_tf_dataset(image_paths, labels, augment=False, shuffle=False):
    """
    Create a tf.data.Dataset from image paths and labels.
    
    Args:
        image_paths: Array of image file paths.
        labels: Array of corresponding labels.
        augment: Whether to apply augmentation.
        shuffle: Whether to shuffle the dataset.
    
    Returns:
        tf.data.Dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=CONFIG['seed'])
    
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(CONFIG['batch_size'])
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def print_class_distribution(labels, class_names, split_name):
    """
    Print the distribution of classes in a dataset split.
    
    Args:
        labels: Array of labels.
        class_names: List of class names.
        split_name: Name of the split (train/val/test).
    """
    print(f"\n{split_name} class distribution:")
    class_counts = Counter(labels)
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_idx, 0)
        percentage = (count / len(labels)) * 100 if len(labels) > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.2f}%)")


def visualize_sample_grid(dataset, class_names, num_samples=9):
    """
    Display a grid of sample images with their labels.
    
    Args:
        dataset: tf.data.Dataset to sample from.
        class_names: List of class names.
        num_samples: Number of samples to display.
    """
    plt.figure(figsize=(12, 12))
    
    for images, labels in dataset.take(1):
        for i in range(min(num_samples, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(class_names[labels[i].numpy()])
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'sample_grid.png'))
    print(f"Sample grid saved to {os.path.join(os.getcwd(), 'sample_grid.png')}")
    plt.close()


def save_class_names(class_names, save_path='class_names.json'):
    """Save class names to JSON file for consistent labeling."""
    with open(save_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {save_path}")


def build_data_pipeline():
    """
    Build the complete data pipeline for crop disease detection.
    
    Returns:
        Dictionary containing train_dataset, val_dataset, test_dataset,
        class_names, and class_weights.
    """
    # Download/load dataset
    dataset_path = download_dataset()
    
    # Load image paths and labels
    image_paths, labels, class_names = load_image_paths_and_labels(dataset_path)
    
    # Save class names for consistent labeling in app
    save_class_names(class_names)
    
    # Print overall class distribution
    print_class_distribution(labels, class_names, "Overall")
    
    # Stratified split
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        stratified_split(image_paths, labels)
    
    # Print class distributions for each split
    print_class_distribution(train_labels, class_names, "Train")
    print_class_distribution(val_labels, class_names, "Validation")
    print_class_distribution(test_labels, class_names, "Test")
    
    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(train_labels)
    
    # Create tf.data.Dataset pipelines
    train_dataset = create_tf_dataset(train_paths, train_labels, augment=True, shuffle=True)
    val_dataset = create_tf_dataset(val_paths, val_labels, augment=False, shuffle=False)
    test_dataset = create_tf_dataset(test_paths, test_labels, augment=False, shuffle=False)
    
    # Visualize sample grid
    visualize_sample_grid(train_dataset, class_names, num_samples=9)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'class_names': class_names,
        'class_weights': class_weights,
        'num_classes': CONFIG['num_classes']
    }


if __name__ == '__main__':
    # Set random seeds for reproducibility
    tf.random.set_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Build and return data pipeline
    pipeline = build_data_pipeline()
    
    print("\nData pipeline built successfully!")
    print(f"Number of classes: {pipeline['num_classes']}")
    print(f"Class names: {pipeline['class_names']}")
