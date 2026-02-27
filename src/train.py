import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

import features as F

DATASET_FILE = "asl_dataset.csv"
MODEL_FILE   = "asl_model.pkl"
LABELS_FILE  = "asl_labels.pkl"


def main():
    # load dataset
    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: {DATASET_FILE} not found. Run 1_collect_data.py first.")
        return

    df = pd.read_csv(DATASET_FILE)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} letters.")

    counts = Counter(df["label"])
    print("\nSample counts per letter:")
    for letter, cnt in sorted(counts.items()):
        bar  = "#" * (cnt // 2)
        warn = " << LOW" if cnt < 15 else ""
        print(f"  {letter}: {cnt:3d}  {bar}{warn}")

    low = [l for l, c in counts.items() if c < 10]
    if low:
        print(f"\n[!] Warning: {low} have fewer than 10 samples.")

    # Engineer features
    print("\nEngineering features...")
    raw_X  = df.drop(columns=["label"]).values
    y      = df["label"].values
    labels = sorted(df["label"].unique())

    X = np.array([F.from_csv_row(row) for row in raw_X])
    print(f"Feature vector size: {X.shape[1]}")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train rdmfrst #1
    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"RandomForest accuracy: {rf_acc*100:.1f}%")

    if rf_acc < 0.90:
        print("Trying GradientBoosting for better accuracy (slower)...")
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        gb_acc = accuracy_score(y_test, gb.predict(X_test))
        print(f"GradientBoosting accuracy: {gb_acc*100:.1f}%")
        model    = gb if gb_acc > rf_acc else rf
        best_acc = max(rf_acc, gb_acc)
    else:
        model    = rf
        best_acc = rf_acc

    # evaluate
    y_pred = model.predict(X_test)
    print(f"\nBest model accuracy: {best_acc*100:.1f}%")
    print("\nPer-letter report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if best_acc < 0.70:
        print("[!] Low accuracy. Tips:")
        print("    - Collect more samples (50+ per letter)")
        print("    - Make sure lighting is good when recording")
        print("    - Keep hand fully visible in frame")
    elif best_acc >= 0.90:
        print("[+] Great accuracy! Model is ready.")
    else:
        print("[~] Decent. More samples for confused letters will help.")

    # confused pairs
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nMost confused letter pairs:")
    confused = []
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i != j and cm[i][j] > 0:
                confused.append((cm[i][j], la, lb))
    confused.sort(reverse=True)
    for count, la, lb in confused[:8]:
        print(f"  {la} mistaken as {lb}: {count}x")

    # save
    with open(MODEL_FILE,  "wb") as f: pickle.dump(model,  f)
    with open(LABELS_FILE, "wb") as f: pickle.dump(labels, f)

    print(f"\nSaved: {MODEL_FILE}, {LABELS_FILE}")
    print("-> Now run: python 3_hand_recognition.py")


if __name__ == "__main__":
    main()
