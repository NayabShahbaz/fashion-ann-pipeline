"""Stage 3: build the ANN, train it, save models/model.h5 and models/history.csv."""
import csv
import os

import numpy as np
import tensorflow as tf
import yaml

DATA_DIR = "data/processed"
MODEL_DIR = "models"


def build_model(p):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(p["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(p["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]
    tf.keras.utils.set_random_seed(p["seed"])

    train = np.load(os.path.join(DATA_DIR, "train.npz"))
    val = np.load(os.path.join(DATA_DIR, "val.npz"))

    model = build_model(p)
    model.summary()
    history = model.fit(
        train["x"], train["y"],
        validation_data=(val["x"], val["y"]),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "model.h5"))

    hist = history.history
    with open(os.path.join(MODEL_DIR, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(hist.keys()))
        for i in range(len(hist["loss"])):
            writer.writerow([i + 1] + [round(hist[k][i], 4) for k in hist])
    print(f"Saved model and history to {MODEL_DIR}/")


if __name__ == "__main__":
    main()