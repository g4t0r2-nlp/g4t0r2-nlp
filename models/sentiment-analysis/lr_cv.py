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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression

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
    parser = argparse.ArgumentParser(description="LogisticRegression for Entity Extraction")
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

    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    X_valid_cv = cv.transform(X_valid)
    X_test_cv = cv.transform(X_test)

    rf = LogisticRegression()
    rf_cv_train_start = time.time()
    rf.fit(X_train_cv, y_train)
    rf_cv_train_time = time.time() - rf_cv_train_start
    print(f"Logistic Regression + CV Train Time = {rf_cv_train_time:.4f}")

    rf_cv_pred_train = rf.predict(X_train_cv)
    rf_cv_test_start = time.time()
    rf_cv_pred_test = rf.predict(X_test_cv)
    rf_cv_test_time = time.time() - rf_cv_test_start

    rf_cv_train_score = accuracy_score(rf_cv_pred_train, y_train)
    rf_cv_test_score = accuracy_score(rf_cv_pred_test, y_test)
    print(f"Logistic Regression + CV Train Score = {rf_cv_train_score * 100:.4f}%")
    print(f"Logistic Regression + CV Test Score = {rf_cv_test_score * 100:.4f}%")
    print(f"Logistic Regression + CV Test Time = {rf_cv_test_time:.4f}")

    rf_cv_precision_score = precision_score(y_test, rf_cv_pred_test, average='macro')
    rf_cv_f1_score = f1_score(y_test, rf_cv_pred_test, average='macro')
    rf_cv_recall_score = recall_score(y_test, rf_cv_pred_test, average='macro')
    rf_cv_accuracy_score = accuracy_score(y_test, rf_cv_pred_test)

    print(f"Logistic Regression + CV Precision Score = {rf_cv_precision_score * 100:.4f}%")
    print(f"Logistic Regression + CV F1 Score = {rf_cv_f1_score * 100:.4f}%")
    print(f"Logistic Regression + CV Recall Score = {rf_cv_recall_score * 100:.4f}%")
    print(f"Logistic Regression + CV Accuracy Score = {rf_cv_accuracy_score * 100:.4f}%")

    print(classification_report(y_test, rf_cv_pred_test, target_names=["Negative", "Neutral", "Positive"]))

    model_cm = confusion_matrix(y_test, rf_cv_pred_test)
    fig, ax = plot_confusion_matrix(conf_mat=model_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["Negative", "Neutral", "Positive"], figsize=(10, 10))
    plt.title("LogisticRegression + CV - Sentiment Analysis")
    plt.savefig("./output/logisticregression_cv.png")

if __name__ == "__main__":
    main()