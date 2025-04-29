#!/usr/bin/env python3
"""
Food Classification Model Training
---------------------------------
This script trains a deep learning model to classify food images 
using the Food-101 dataset from TensorFlow Datasets.
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds #Pretty neat that Food101 can be pulled directly from TF's dataset module.
from tensorflow.keras.applications import MobileNetV2 #Our main CNN.
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam #Optimizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 101  # Food-101 has 101 classes
MODEL_SAVE_PATH = 'models/food_classifier_best.h5'
CLASS_INDICES_PATH = 'models/class_indices.json'
VISUALIZE_EXAMPLES = True
TRAIN_MODEL = True
EVALUATE_MODEL = True

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def preprocess_image(image, label):
    """Preprocess images for the model"""
    # Resize the image
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # One-hot encode the label
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def augment_image(image, label):
    """Apply data augmentation to training images"""
    # Random flip
    image = tf.image.random_flip_left_right(image)
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random rotation (using TensorFlow's image ops)
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, label

def setup_datasets():
    """Load Food-101 dataset from TensorFlow Datasets and prepare for training"""
    print("Loading Food-101 dataset from TensorFlow Datasets...")
    
    # Load the Food-101 dataset
    (train_ds, validation_ds), ds_info = tfds.load(
        'food101',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True
    )
    
    # Get the test dataset separately
    test_ds = tfds.load('food101', split='validation', as_supervised=True)
    
    # Get class names from dataset info
    class_names = ds_info.features['label'].names
    print(f"Loaded Food-101 dataset with {len(class_names)} classes")
    print(f"First 10 classes: {class_names[:10]}")
    
    # Create a mapping for class indices
    class_indices = {i: name for i, name in enumerate(class_names)}
    
    # Preprocess training data with augmentation
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Preprocess validation data (no augmentation)
    validation_ds = validation_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Preprocess test data (no augmentation)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, validation_ds, test_ds, class_indices

def create_model():
    """Create and return a MobileNetV2-based model fine-tuned for food classification"""
    print("Creating MobileNetV2-based food classification model...")
    
    # Use MobileNetV2 as base model (lightweight and efficient for deployment)
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_examples(dataset, class_indices, num_examples=10):
    """Visualize examples from the dataset"""
    if not VISUALIZE_EXAMPLES:
        return
    
    print("Visualizing example images...")
    class_indices_inverted = {v: k for k, v in class_indices.items()}
    
    plt.figure(figsize=(15, 8))
    
    example_images = []
    example_labels = []
    
    # Collect some example images
    for images, labels in dataset.unbatch().take(num_examples):
        example_images.append(images.numpy())
        example_labels.append(tf.argmax(labels).numpy())
    
    # Plot them
    for i, (image, label) in enumerate(zip(example_images, example_labels)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(class_indices[label])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('example_food_images.png')
    plt.close()
    print(f"Example visualization saved to 'example_food_images.png'")

def train_model():
    """Train the model and save the best weights"""
    if not TRAIN_MODEL:
        return None, None
    
    print("\nTraining the food classification model...")
    model = create_model()
    
    # Set up datasets and get class indices
    train_ds, validation_ds, test_ds, class_indices = setup_datasets()
    
    # Visualize some examples
    visualize_examples(train_ds, class_indices)
    
    # Save class indices for later use
    class_indices_flipped = {v: k for k, v in class_indices.items()}
    os.makedirs(os.path.dirname(CLASS_INDICES_PATH), exist_ok=True)
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_indices_flipped, f)
    print(f"Class indices saved to {CLASS_INDICES_PATH}")
    
    # Set up callbacks for training
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=0.00001
    )
    
    # Train the model
    print("\nPhase 1: Training with frozen base model...")
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=10,  # Initial training with frozen base
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # After initial training, unfreeze some of the top layers of the base model for fine-tuning
    print("\nPhase 2: Fine-tuning with unfrozen top layers...")
    
    # Get the actual base model - not just the input layer
    # In the create_model function, the base model is wrapped in a Functional model
    # We need to get the actual MobileNetV2 model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):  # Find the MobileNetV2 model
            base_model = layer
            break
    
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    history_fine_tune = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=20,  # Additional fine-tuning
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save the final model
    final_model_path = 'models/food_classifier_final.h5'
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, class_indices

def evaluate_model(model=None, class_indices=None):
    """Evaluate the trained model on the test set"""
    if not EVALUATE_MODEL:
        return
    
    print("\nEvaluating the food classification model...")
    
    # Load model if not provided
    if model is None:
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"Loading model from {MODEL_SAVE_PATH}")
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        else:
            print(f"Model file {MODEL_SAVE_PATH} not found. Train the model first.")
            return
    
    # Load class indices if not provided
    if class_indices is None:
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices_flipped = json.load(f)
                class_indices = {int(v): k for k, v in class_indices_flipped.items()}
        else:
            print(f"Class indices file {CLASS_INDICES_PATH} not found.")
            _, _, _, class_indices = setup_datasets()
    
    # Get the test dataset
    _, _, test_ds, _ = setup_datasets()
    
    # Evaluate the model
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds)
    print(f"Test loss: {results[0]:.4f}")
    print(f"Test accuracy: {results[1]:.4f}")
    
    # Get predictions for a sample of the test set
    print("\nGenerating classification report for a sample of the test set...")
    all_predictions = []
    all_labels = []
    
    # Take a subset of the test set for the classification report
    test_sample = test_ds.take(20)  # Adjust this number based on your computational resources
    
    for images, labels in test_sample:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        
        all_predictions.extend(predicted_classes)
        all_labels.extend(true_classes)
    
    # Get class names for the report
    class_names = [class_indices[i] for i in range(len(class_indices))]
    
    # Generate and print classification report
    report = classification_report(
        all_labels, 
        all_predictions, 
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
    print(report)
    
    # Save the report to a file
    with open('models/classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved to 'models/classification_report.txt'")

def main():
    """Main function to run the training and evaluation pipeline"""
    print("Food-101 Classification with TensorFlow")
    print("=" * 40)
    
    # Train the model
    model, class_indices = train_model()
    
    # Evaluate the model
    evaluate_model(model, class_indices)
    
    print("\nDone!")

if __name__ == "__main__":
    main()