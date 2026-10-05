"""Stage 2: normalize pixels to [0, 1] and split a validation set off the training data."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

def normalize(x):
    """Scale pixels to [-1, 1]."""
    return x.astype("float32") / 127.5 - 1.0


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train = np.load(os.path.join(RAW_DIR, "train.npz"))
    test = np.load(os.path.join(RAW_DIR, "test.npz"))

    x_train, x_val, y_train, y_val = train_test_split(
        normalize(train["x"]), train["y"],
        test_size=params["val_size"],
        random_state=params["seed"],
        stratify=train["y"],
    )
    x_test, y_test = normalize(test["x"]), test["y"]

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(OUT_DIR, "val.npz"), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(OUT_DIR, "test.npz"), x=x_test, y=y_test)
    print(f"train={x_train.shape}  val={x_val.shape}  test={x_test.shape}")


if __name__ == "__main__":
    main()