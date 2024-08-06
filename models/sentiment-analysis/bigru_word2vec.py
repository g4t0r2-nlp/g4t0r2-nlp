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

from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, preprocessing
from tensorflow.keras.utils import to_categorical

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bi-GRU for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="/mnt/d/work2/teknofest-tddi/data/processed/cleaned.csv", help="Dataset path")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embeddings dim")
    parser.add_argument("--maxlen", type=int, default=64, help="Max length")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
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

    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)

    tokenizer = preprocessing.text.Tokenizer(
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower = False,
        split = " "
    )
    tokenizer.fit_on_texts(X)

    X_train_tokenizer = tokenizer.texts_to_sequences(X_train)
    X_valid_tokenizer = tokenizer.texts_to_sequences(X_valid)
    X_test_tokenizer = tokenizer.texts_to_sequences(X_test)

    X_train_tokenizer = preprocessing.sequence.pad_sequences(X_train_tokenizer, maxlen=args.maxlen)
    X_valid_tokenizer = preprocessing.sequence.pad_sequences(X_valid_tokenizer, maxlen=args.maxlen)
    X_test_tokenizer = preprocessing.sequence.pad_sequences(X_test_tokenizer, maxlen=args.maxlen)

    input_dim = len(tokenizer.word_index) + 1

    documents = [_text.split() for _text in df["text"]]

    w2v_model = gensim.models.word2vec.Word2Vec(
        vector_size = 100,
        window = 2,
        min_count = 10,
        workers = 10
    )

    w2v_model.build_vocab(documents)

    vocab_len = len(w2v_model.wv)

    w2v_model.train(documents, total_examples=len(documents), epochs=16)

    wv_embedding_matrix = np.zeros((input_dim, args.embedding_dim))
    for word, i in tqdm.tqdm(tokenizer.word_index.items()):
        if word in w2v_model.wv:
            wv_embedding_matrix[i] = w2v_model.wv[word]

    model = models.Sequential([
        layers.Input(shape=X_train_tokenizer.shape[1]),
        layers.Embedding(len(tokenizer.word_index)+1, args.embedding_dim, weights=[wv_embedding_matrix], input_length=args.maxlen, trainable=False),
        layers.Bidirectional(layers.GRU(100, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate), 
                  metrics=['accuracy'])
    
    model1_train_start = time.time()
    model_history = model.fit(X_train_tokenizer, 
                                    y_train, 
                                    epochs=args.epochs, 
                                    batch_size=args.batch_size, 
                                    validation_data=[X_valid_tokenizer, y_valid], 
                                    callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=3)])
    model1_train_time = time.time() - model1_train_start
    print(f"Bi-GRU Train Time = {model1_train_time:.4f}")

    model1_test_start = time.time()
    model_pred_test = model.predict(X_test_tokenizer, verbose=0)
    model1_test_time = time.time() - model1_test_start
    print(f"Bi-GRU Test Time = {model1_test_time:.4f}")

    true_labels_train = np.argmax(y_train, axis=1)
    true_labels_test = np.argmax(y_test, axis=1)

    model_pred_train = model.predict(X_train_tokenizer, verbose=0)
    model_pred_train = np.argmax(model_pred_train, axis=1)
    model_pred_test = np.argmax(model_pred_test, axis=1)
    model_train_score = accuracy_score(model_pred_train, true_labels_train)
    model_test_score = accuracy_score(model_pred_test, true_labels_test)
    print(f"Bi-GRU Train Score = {model_train_score * 100:.4f}%")
    print(f"Bi-GRU Test Score = {model_test_score * 100:.4f}%")

    model_precision_score = precision_score(true_labels_test, model_pred_test, average="macro")
    model_f1_score = f1_score(true_labels_test, model_pred_test, average="macro")
    model_recall_score = recall_score(true_labels_test, model_pred_test, average="macro")
    model_accuracy_score = accuracy_score(true_labels_test, model_pred_test)

    print(f"Bi-GRU Precision Score = {model_precision_score * 100:.4f}%")
    print(f"Bi-GRU F1 Score = {model_f1_score * 100:.4f}%")
    print(f"Bi-GRU Recall Score = {model_recall_score * 100:.4f}%")
    print(f"Bi-GRU Accuracy Score = {model_accuracy_score * 100:.4f}%")

    print(classification_report(true_labels_test, model_pred_test, target_names=["Negative", "Neutral", "Positive"]))

    model_cm = confusion_matrix(true_labels_test, model_pred_test)
    fig, ax = plot_confusion_matrix(conf_mat=model_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["Negative", "Neutral", "Positive"], figsize=(10, 10))
    plt.title("Bi-GRU + Word2Vec - Sentiment Analysis")
    plt.savefig("./output/bigru_word2vec.png")

if __name__ == "__main__":
    main()