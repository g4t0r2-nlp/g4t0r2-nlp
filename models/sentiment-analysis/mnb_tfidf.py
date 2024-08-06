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

import gensim
from gensim import corpora, models, similarities

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.naive_bayes import MultinomialNB

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
    parser = argparse.ArgumentParser(description="MultinomialNB for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="/mnt/d/work2/teknofest-tddi/data/processed/cleaned.csv", help="Dataset path")
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

    X = df["text"]
    y = df["aspect_polarity"]

    df_train, df_validation, df_test = train_validate_test_split(df, args.split_size)
    
    X_train = df_train["text"]
    y_train = df_train["aspect_polarity"]

    X_valid = df_validation["text"]
    y_valid = df_validation["aspect_polarity"]

    X_test = df_test["text"]
    y_test = df_test["aspect_polarity"]

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_valid_tfidf = tfidf.transform(X_valid)
    X_test_tfidf = tfidf.transform(X_test)

    mnb = MultinomialNB()
    mnb_tfidf_train_start = time.time()
    mnb.fit(X_train_tfidf, y_train)
    mnb_tfidf_train_time = time.time() - mnb_tfidf_train_start
    print(f"MultnomialNB + TFIDF Train Time = {mnb_tfidf_train_time:.4f}")

    mnb_tfidf_pred_train = mnb.predict(X_train_tfidf)
    mnb_tfidf_test_start = time.time()
    mnb_tfidf_pred_test = mnb.predict(X_test_tfidf)
    mnb_tfidf_test_time = time.time() - mnb_tfidf_test_start

    mnb_tfidf_train_score = accuracy_score(mnb_tfidf_pred_train, y_train)
    mnb_tfidf_test_score = accuracy_score(mnb_tfidf_pred_test, y_test)
    print(f"MultnomialNB + TFIDF Train Score = {mnb_tfidf_train_score * 100:.4f}%")
    print(f"MultnomialNB + TFIDF Test Score = {mnb_tfidf_test_score * 100:.4f}%")
    print(f"MultnomialNB + TFIDF Test Time = {mnb_tfidf_test_time:.4f}")

    mnb_tfidf_precision_score = precision_score(y_test, mnb_tfidf_pred_test, average='macro')
    mnb_tfidf_f1_score = f1_score(y_test, mnb_tfidf_pred_test, average='macro')
    mnb_tfidf_recall_score = recall_score(y_test, mnb_tfidf_pred_test, average='macro')
    mnb_tfidf_accuracy_score = accuracy_score(y_test, mnb_tfidf_pred_test)

    print(f"MultnomialNB + TFIDF Precision Score = {mnb_tfidf_precision_score * 100:.4f}%")
    print(f"MultnomialNB + TFIDF F1 Score = {mnb_tfidf_f1_score * 100:.4f}%")
    print(f"MultnomialNB + TFIDF Recall Score = {mnb_tfidf_recall_score * 100:.4f}%")
    print(f"MultnomialNB + TFIDF Accuracy Score = {mnb_tfidf_accuracy_score * 100:.4f}%")

    print(classification_report(y_test, mnb_tfidf_pred_test, target_names=["Negative", "Neutral", "Positive"]))

    model_cm = confusion_matrix(y_test, mnb_tfidf_pred_test)
    fig, ax = plot_confusion_matrix(conf_mat=model_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["Negative", "Neutral", "Positive"], figsize=(10, 10))
    plt.title("MultinomialNB + TFIDF - Sentiment Analysis")
    plt.savefig("./output/mnb_tfidf.png")

if __name__ == "__main__":
    main()