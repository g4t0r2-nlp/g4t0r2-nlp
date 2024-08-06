import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from mlxtend.plotting import plot_confusion_matrix

def set_seed(seed):
    np.random.seed(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="CRF for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="../data/all_tagged_aspects_just_aspect_sentences.csv", help="Dataset path")
    parser.add_argument("--algorithm", type=str, default="lbfgs", help="Algorithm")
    parser.add_argument("--max_iterations", type=int, default=100, help="Max iterations")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--split-size", type=float, default=0.9, help="Train-Test split size")
    parser.add_argument("--output", type=str, default=".", help="Model save path")
    return parser.parse_args()

def create_sentences_and_labels(df):
    sentences = df['sentence'].apply(lambda x: x.split())
    labels = df['word_labels'].apply(lambda x: x.split(','))
    return [(list(zip(sent, lab))) for sent, lab in zip(sentences, labels)]

def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def evaluate_model(y_test, y_pred, class_names):
    recall_val = metrics.flat_recall_score(y_true=y_test, y_pred=y_pred, average='micro')
    print(f'Recall Score: {recall_val}')

    precision_val = metrics.flat_precision_score(y_true=y_test, y_pred=y_pred, average='micro')
    print(f'Precision Score: {precision_val}')

    f1_val = metrics.flat_f1_score(y_true=y_test, y_pred=y_pred, average='micro')
    print(f'F1 Score: {f1_val}')

    acc_val = metrics.flat_accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Accuracy Score: {acc_val}")

    cm = confusion_matrix(y_true=metrics.flatten(y=y_test), y_pred=metrics.flatten(y=y_pred))
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(metrics.flat_classification_report(y_test, y_pred, labels=class_names, digits=3))

    plt.figure()
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True, class_names=class_names)
    plt.title("CRF - Aspect Extraction")
    plt.savefig("./output/crf_confusion_matrix.png")

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

    sentences_and_labels = create_sentences_and_labels(df)

    train_sents, valid_sents, test_sents = train_validate_test_split(df, args.split_size)

    print("\n====================")
    print(f"Full dataset: {df.shape[0]}")
    print(f"Train dataset: {train_sents.shape[0]}")
    print(f"Validation dataset: {valid_sents.shape[0]}")
    print(f"Test dataset: {test_sents.shape[0]}")
    print("====================")

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_valid = [sent2features(s) for s in valid_sents]
    y_valid = [sent2labels(s) for s in valid_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    
    crf = sklearn_crfsuite.CRF(
        algorithm=args.algorithm,
        c1=0.1,
        c2=0.1,
        max_iterations=args.max_iterations,
        all_possible_transitions=False
    )

    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    class_names = list(crf.classes_)

    evaluate_model(y_test, y_pred, class_names)

if __name__ == '__main__':
    main()
