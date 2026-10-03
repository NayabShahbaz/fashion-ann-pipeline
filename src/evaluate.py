import json
import os

import matplotlib
matplotlib.use("Agg")  # save plots without needing a display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    model = tf.keras.models.load_model("models/model.h5")
    test = np.load(os.path.join("data/processed", "test.npz"))

    loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
    y_pred = np.argmax(model.predict(test["x"], verbose=0), axis=1)

    cm = confusion_matrix(test["y"], y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Fashion-MNIST confusion matrix (test acc = {acc:.4f})")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=120)

    metrics = {"test_loss": round(float(loss), 4), "test_accuracy": round(float(acc), 4)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()