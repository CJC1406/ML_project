from pathlib import Path
from pathsnap.data.preprocess import preprocess_training


if __name__ == "__main__":
    root = Path(__import__("os").environ.get("PATHSNAP_TRAIN_ROOT", "/Users/cjc/Downloads/Student folder/train"))
    manifest = preprocess_training(root, Path("data/processed/train"))
    print(f"preprocessed {manifest['trip_count']} valid trips")
