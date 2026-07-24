"""
Training script for crop disease detection CNN.
Supports MobileNetV2 transfer-learning and from-scratch CNN options.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from data_pipeline import build_data_pipeline, CONFIG

# Training configuration
TRAIN_CONFIG = {
    'use_transfer_learning': True,
    'fine_tune_layers': 20,
    'initial_epochs': 10,
    'fine_tune_epochs': 10,
    'initial_lr': 0.001,
    'fine_tune_lr': 1e-5,
    'dropout_rate': 0.5,
    'l2_reg': 0.01,
    'model_path': 'best_model.keras',
    'history_path': 'training_history.png',
}


def build_transfer_learning_model(num_classes):
    """Build MobileNetV2 transfer-learning model with custom head."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*CONFIG['image_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(*CONFIG['image_size'], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(TRAIN_CONFIG['dropout_rate'])(x)
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(TRAIN_CONFIG['l2_reg'])
    )(x)
    x = layers.Dropout(TRAIN_CONFIG['dropout_rate'])(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model


def build_scratch_cnn(num_classes):
    """Build CNN from scratch."""
    model = models.Sequential([
        layers.Input(shape=(*CONFIG['image_size'], 3)),
        layers.Conv2D(32, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(TRAIN_CONFIG['l2_reg'])),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(TRAIN_CONFIG['l2_reg'])),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(TRAIN_CONFIG['l2_reg'])),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(TRAIN_CONFIG['l2_reg'])),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(TRAIN_CONFIG['dropout_rate']),
        layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(TRAIN_CONFIG['l2_reg'])),
        layers.Dropout(TRAIN_CONFIG['dropout_rate']),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model, None


def compile_model(model, learning_rate):
    """Compile model with Adam optimizer and categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    """Create training callbacks."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            TRAIN_CONFIG['model_path'],
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    return callbacks


def plot_training_history(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def train_model():
    """Train the model with frozen base then fine-tune."""
    pipeline = build_data_pipeline()
    train_ds = pipeline['train_dataset']
    val_ds = pipeline['val_dataset']
    num_classes = pipeline['num_classes']
    class_weights = pipeline['class_weights']
    
    if TRAIN_CONFIG['use_transfer_learning']:
        model, base_model = build_transfer_learning_model(num_classes)
    else:
        model, base_model = build_scratch_cnn(num_classes)
    
    model = compile_model(model, TRAIN_CONFIG['initial_lr'])
    model.summary()
    
    callbacks = get_callbacks()
    
    print("\nPhase 1: Training with frozen base...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAIN_CONFIG['initial_epochs'],
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    if TRAIN_CONFIG['use_transfer_learning'] and base_model is not None:
        print("\nPhase 2: Fine-tuning top layers...")
        base_model.trainable = True
        
        for layer in base_model.layers[:-TRAIN_CONFIG['fine_tune_layers']]:
            layer.trainable = False
        
        model = compile_model(model, TRAIN_CONFIG['fine_tune_lr'])
        
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=TRAIN_CONFIG['fine_tune_epochs'],
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        history.history['loss'].extend(fine_tune_history.history['loss'])
        history.history['val_loss'].extend(fine_tune_history.history['val_loss'])
        history.history['accuracy'].extend(fine_tune_history.history['accuracy'])
        history.history['val_accuracy'].extend(fine_tune_history.history['val_accuracy'])
    
    plot_training_history(history, TRAIN_CONFIG['history_path'])
    
    return model, history


if __name__ == '__main__':
    tf.random.set_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    model, history = train_model()
    print("\nTraining completed!")
    print(f"Best model saved to {TRAIN_CONFIG['model_path']}")
