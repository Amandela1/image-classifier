"""
Image Classifier using TensorFlow/Keras — by Amanuel
======================================================
Trains a CNN to classify images from CIFAR-10 into 10 categories.

Categories: airplane, automobile, bird, cat, deer,
            dog, frog, horse, ship, truck

Requirements:
    pip install tensorflow numpy matplotlib

Run: python image_classifier.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_and_preprocess_data():
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples:     {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Saved: training_history.png")


def show_predictions(model, x_test, y_test, num_images=10):
    predictions = model.predict(x_test[:num_images])
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    for i in range(num_images):
        pred_label = CLASS_NAMES[np.argmax(predictions[i])]
        true_label = CLASS_NAMES[y_test[i][0]]
        color = "green" if pred_label == true_label else "red"
        axes[i].imshow(x_test[i])
        axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    print("Saved: sample_predictions.png")


def main():
    print("=" * 55)
    print("  Image Classifier (CIFAR-10) — by Amanuel")
    print("=" * 55)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_model()
    model.summary()

    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    model.save("image_classifier_model.h5")
    print("Model saved!")

    plot_training_history(history)
    show_predictions(model, x_test, y_test)


if __name__ == "__main__":
    main()
