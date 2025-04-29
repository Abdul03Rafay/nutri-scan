#!/usr/bin/env python3
"""
Simple Model Converter for Food Classification
----------------------------------------------
This script creates a new model with explicit input shapes and
converts it to TensorFlow.js format.
"""

import os
import json
import shutil
import tensorflow as tf
import tensorflowjs as tfjs

# Paths based on local directories.
MODEL_PATH = '/Users/...'
TFJS_OUTPUT_DIR = '/Users/.../food-classifier-frontend/public/models/food_classifier_tfjs'
CLASS_INDICES_PATH = '/Users/.../models/class_indices.json'

def create_simple_model():
    """Create a simple model for testing"""
    print("Creating a simple model for testing...")
    
    # Create a simple model with explicit input shape
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(101, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(TFJS_OUTPUT_DIR, exist_ok=True)
    
    # Convert the model to TensorFlow.js format
    print("Converting model to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, TFJS_OUTPUT_DIR)
    print(f"Model converted and saved to {TFJS_OUTPUT_DIR}")
    
    # Create a simple dummy class indices file for testing
    if not os.path.exists(CLASS_INDICES_PATH):
        print("Creating dummy class indices file...")
        dummy_classes = {str(i): f"class_{i}" for i in range(101)}
        
        # Save to the models directory
        dest_path = os.path.join(os.path.dirname(TFJS_OUTPUT_DIR), 'class_indices.json')
        with open(dest_path, 'w') as f:
            json.dump(dummy_classes, f)
        print(f"Dummy class indices saved to {dest_path}")
    else:
        # Copy the class indices file
        dest_path = os.path.join(os.path.dirname(TFJS_OUTPUT_DIR), 'class_indices.json')
        shutil.copy2(CLASS_INDICES_PATH, dest_path)
        print(f"Class indices copied to {dest_path}")
    
    return True

def main():
    """Main function to convert the model and prepare it for deployment"""
    print("Simple Food Classification Model Converter")
    print("=" * 45)
    
    # Create and save the model
    success = create_simple_model()
    
    if success:
        print("\nModel conversion completed successfully!")
    else:
        print("\nModel conversion failed.")

if __name__ == "__main__":
    main()
