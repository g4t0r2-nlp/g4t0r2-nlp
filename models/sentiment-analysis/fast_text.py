import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import tqdm
import random
import argparse

import numpy as np
import pandas as pd
tqdm.tqdm.pandas()

import matplotlib.pyplot as plt

import fasttext

import gensim
from gensim import corpora, models, similarities

from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="FastText for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="/mnt/d/work2/teknofest-tddi/data/processed/cleaned.csv", help="Dataset path")
    parser.add_argument("--embedding-dim", type=int, default=300, help="Embeddings dim")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--split-size", type=float, default=0.9, help="Train-Test split size")
    parser.add_argument("--output", type=str, default=".", help="Model save path")
    return parser.parse_args()

def train_validate_test_split(df, split_size):
    perm = np.random.permutation(df.index)
    train_end = int(split_size * len(df.index))
    validate_end = int(((1 - split_size) / 2) * len(df.index)) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def main():
    args = parse_arguments()

    set_seed(args.seed)

    df = pd.read_csv(args.dataset)
    df = df.dropna()
    df = df.reset_index(drop=True)

    df["text"] = df["aspect"] + " " + df["cleaned"]

    train_df, valid_df, test_df = train_validate_test_split(df, args.split_size)

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df["label_format"] = 0
    for i in range(len(train_df)):
        train_df.label_format[i] = "__label__" + str(train_df["aspect_polarity"][i]) + " " + str(train_df["text"][i])

    valid_df["label_format"] = 0
    for i in range(len(valid_df)):
        valid_df.label_format[i] = "__label__" + str(valid_df["aspect_polarity"][i]) + " " + str(valid_df["text"][i])

    test_df["label_format"] = 0
    for i in range(len(test_df)):
        test_df.label_format[i] = "__label__" + str(test_df["aspect_polarity"][i]) + " " + str(test_df["text"][i])

    train_df.label_format.to_csv("fasttext_train.txt", index=None, header=None)
    valid_df.label_format.to_csv("fasttext_valid.txt", index=None, header=None)
    test_df.label_format.to_csv("fasttext_test.txt", index=None, header=None)

    model5_train_start = time.time()
    fasttext_model = fasttext.train_supervised("fasttext_train.txt", epoch=args.epochs, lr=args.learning_rate, label_prefix="__label__", dim=args.embedding_dim)
    model5_train_time = time.time() - model5_train_start
    print(f"FastText Train Time = {model5_train_time:.4f}")

    def predict_fasttext(row):
        pred = fasttext_model.predict(row)[0][0].replace("__label__", "")
        return int(pred)
    
    model5_test_start = time.time()
    fasttext_pred_test = [predict_fasttext(test) for test in test_df.label_format]
    model5_test_time = time.time() - model5_test_start
    print(f"FastText Test Time = {model5_test_time:.4f}")

    fasttext_pred_train = [predict_fasttext(train) for train in train_df.label_format]
    fasttext_train_score = accuracy_score(fasttext_pred_train, train_df.aspect_polarity)
    fasttext_test_score = accuracy_score(fasttext_pred_test, test_df.aspect_polarity)
    print(f"FastText Train Score = {fasttext_train_score * 100:.4f}%")
    print(f"FastText Test Score = {fasttext_test_score * 100:.4f}%")

    fasttext_precision_score = precision_score(test_df.aspect_polarity, fasttext_pred_test, average="macro")
    fasttext_f1_score = f1_score(test_df.aspect_polarity, fasttext_pred_test, average="macro")
    fasttext_recall_score = recall_score(test_df.aspect_polarity, fasttext_pred_test, average="macro")
    fasttext_accuracy_score = accuracy_score(test_df.aspect_polarity, fasttext_pred_test)

    print(f"FastText Precision Score = {fasttext_precision_score * 100:.4f}%")
    print(f"FastText F1 Score = {fasttext_f1_score * 100:.4f}%")
    print(f"FastText Recall Score = {fasttext_recall_score * 100:.4f}%")
    print(f"FastText Accuracy Score = {fasttext_accuracy_score * 100:.4f}%")

    print(classification_report(test_df.aspect_polarity, fasttext_pred_test, target_names=["Negative", "Neutral", "Positive"]))

    fasttext_cm = confusion_matrix(test_df.aspect_polarity, fasttext_pred_test)
    fig, ax = plot_confusion_matrix(conf_mat=fasttext_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["Negative", "Neutral", "Positive"], figsize=(10, 10))
    plt.title("FastText - Sentiment Analysis")
    plt.savefig("./output/fasttext.png")
    plt.show()

if __name__ == "__main__":
    main()