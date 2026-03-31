"""
Digit Recognition Model Trainer (CNN Upgrade)
This script builds, trains, and saves a Convolutional Neural Network (CNN)
using the full MNIST dataset. This model is significantly more powerful and
accurate than the previous scikit-learn model.

Requirements: pip install tensorflow
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_and_train_cnn_model():
    print("🚀 Starting CNN model training with TensorFlow/Keras...")

    # 1. Load the full MNIST dataset (70,000 images)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Dataset loaded: {len(x_train)} training samples, {len(x_test)} testing samples.")

    # 2. Preprocess the data for the CNN
    # Reshape data to fit the model input (28x28 pixels, 1 color channel)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Normalize pixel values from 0-255 to 0-1. This helps the model learn faster.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    print("Data preprocessed for CNN.")

    # 3. Build the CNN Model Architecture
    # A CNN is like a specialist for images. It learns features like edges and corners.
    model = Sequential([
        # First convolutional layer: 32 filters to find basic features
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Pooling layer: Shrinks the image to focus on the most important features
        MaxPooling2D(pool_size=(2, 2)),
        # Second convolutional layer: 64 filters to find more complex features
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Flatten the 2D image data into a 1D array
        Flatten(),
        # A standard 'Dense' layer for classification
        Dense(128, activation='relu'),
        # Dropout layer: Prevents the model from "memorizing" the training data
        Dropout(0.5),
        # Output layer: 10 neurons (one for each digit), with softmax to give probabilities
        Dense(10, activation='softmax')
    ])

    # 4. Compile the model
    # We define the optimizer, how to measure error (loss), and what to track (accuracy)
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("CNN Model built and compiled.")
    model.summary()

    # 5. Train the model
    print("\n🔥 Training the model... (This may take a few minutes)")
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10, # We'll go through the whole dataset 10 times
              verbose=1,
              validation_data=(x_test, y_test))

    # 6. Evaluate the final model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Training Complete!")
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")

    # 7. Save the trained model
    model_filename = 'digit_model_cnn.h5'
    model.save(model_filename)
    print(f"💾 Saved trained CNN model to {model_filename}")

if __name__ == '__main__':
    create_and_train_cnn_model()
