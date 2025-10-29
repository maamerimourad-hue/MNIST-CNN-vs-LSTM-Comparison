from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import LSTM
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def load_and_preprocess_data(for_lstm=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisation

    if for_lstm:
        x_train = x_train.reshape(x_train.shape[0], 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_and_preprocess_data(for_lstm=False)
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()
cnn.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)
cnn_time = time.time() - start
cnn_acc = cnn.evaluate(x_test, y_test, verbose=0)[1]

(x_train, y_train), (x_test, y_test) = load_and_preprocess_data(for_lstm=True)
lstm = Sequential([
    LSTM(128, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()
lstm.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)
lstm_time = time.time() - start
lstm_acc = lstm.evaluate(x_test, y_test, verbose=0)[1]
results = pd.DataFrame([
    {"Model": "CNN", "Accuracy": cnn_acc, "Training Time (s)": cnn_time},
    {"Model": "LSTM", "Accuracy": lstm_acc, "Training Time (s)": lstm_time}
])

results.to_csv("results/training_times.csv", index=False)
print("Résultats enregistrés dans results/training_times.csv")
print(results)
plt.figure(figsize=(6, 4))
plt.bar(results["Model"], results["Accuracy"], color=['skyblue', 'lightgreen'])
plt.title("Comparaison de la précision entre CNN et LSTM (MNIST)")
plt.ylabel("Accuracy")
plt.ylim(0.95, 1.0)  # pour mieux voir la différence
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Bar plot : temps d'entraînement ---
plt.figure(figsize=(6, 4))
plt.bar(results["Model"], results["Training Time (s)"], color=['orange', 'salmon'])
plt.title("Comparaison du temps d'entraînement entre CNN et LSTM (MNIST)")
plt.ylabel("Temps (secondes)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.savefig("results/accuracy_comparison.png")
plt.savefig("results/training_time_comparison.png")

