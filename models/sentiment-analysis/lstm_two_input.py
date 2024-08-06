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
    parser = argparse.ArgumentParser(description="LSTM (Two Input) for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="/mnt/d/work2/teknofest-tddi/data/processed/cleaned.csv", help="Dataset path")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embeddings dim")
    parser.add_argument("--max-review-len", type=int, default=64, help="Max review length")
    parser.add_argument("--max-aspect-len", type=int, default=16, help="Max aspect length")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
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

    label_map = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }

    df["sentiment_polarity"] = df["sentiment"].map(label_map)

    df_train, df_validation, df_test = train_validate_test_split(df, args.split_size)

    X_train_review = df_train["cleaned"]
    X_train_aspect = df_train["aspect"]
    X_train_sentiment = df_train["sentiment_polarity"]

    X_valid_review = df_validation["cleaned"]
    X_valid_aspect = df_validation["aspect"]
    X_valid_sentiment = df_validation["sentiment_polarity"]

    X_test_review = df_test["cleaned"]
    X_test_aspect = df_test["aspect"]
    X_test_sentiment = df_test["sentiment_polarity"]

    tokenizer_review = preprocessing.text.Tokenizer()
    tokenizer_review.fit_on_texts(df["cleaned"])

    tokenizer_aspect = preprocessing.text.Tokenizer()
    tokenizer_aspect.fit_on_texts(df["aspect"])

    X_train_review = tokenizer_review.texts_to_sequences(X_train_review)
    X_train_review = preprocessing.sequence.pad_sequences(X_train_review, maxlen=args.max_review_len)

    X_train_aspect = tokenizer_aspect.texts_to_sequences(X_train_aspect)
    X_train_aspect = preprocessing.sequence.pad_sequences(X_train_aspect, maxlen=args.max_aspect_len)

    X_valid_review = tokenizer_review.texts_to_sequences(X_valid_review)
    X_valid_review = preprocessing.sequence.pad_sequences(X_valid_review, maxlen=args.max_review_len)

    X_valid_aspect = tokenizer_aspect.texts_to_sequences(X_valid_aspect)
    X_valid_aspect = preprocessing.sequence.pad_sequences(X_valid_aspect, maxlen=args.max_aspect_len)

    X_test_review = tokenizer_review.texts_to_sequences(X_test_review)
    X_test_review = preprocessing.sequence.pad_sequences(X_test_review, maxlen=args.max_review_len)

    X_test_aspect = tokenizer_aspect.texts_to_sequences(X_test_aspect)
    X_test_aspect = preprocessing.sequence.pad_sequences(X_test_aspect, maxlen=args.max_aspect_len)

    train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_review, X_train_aspect), X_train_sentiment))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train_sentiment)).batch(args.batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices(((X_valid_review, X_valid_aspect), X_valid_sentiment))
    valid_dataset = valid_dataset.shuffle(buffer_size=len(X_valid_sentiment)).batch(args.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(((X_test_review, X_test_aspect), X_test_sentiment))
    test_dataset = test_dataset.shuffle(buffer_size=len(X_test_sentiment)).batch(args.batch_size)



    input1 = layers.Input(shape=(args.max_review_len,))
    input2 = layers.Input(shape=(args.max_aspect_len,))
    embedding_layer = layers.Embedding(input_dim=len(tokenizer_review.word_index) + 1, output_dim=64)

    embedded_sequences1 = embedding_layer(input1)
    embedded_sequences2 = embedding_layer(input2)

    lstm1 = layers.LSTM(128)(embedded_sequences1)
    lstm2 = layers.LSTM(128)(embedded_sequences2)

    merged = layers.Concatenate()([lstm1, lstm2])
    dense = layers.Dense(64, activation='relu')(merged)
    output = layers.Dense(3, activation='softmax')(dense)

    model = models.Model(inputs=[input1, input2], outputs=output)

    model.compile(optimizer=optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model1_train_start = time.time()
    model_history = model.fit(train_dataset, 
                                  validation_data=valid_dataset,
                                  epochs=args.epochs, 
                                  batch_size=args.batch_size, 
                                  validation_split=0.1, 
                                  callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=3)])
    model1_train_time = time.time() - model1_train_start
    print(f"LSTM Train Time = {model1_train_time:.4f}")

    model1_test_start = time.time()
    model_pred_test = model.predict([X_test_review, X_test_aspect], verbose=0)
    model1_test_time = time.time() - model1_test_start
    print(f"LSTM Test Time = {model1_test_time:.4f}")

    model_pred_train = model.predict([X_train_review, X_train_aspect], verbose=0)
    model_pred_train = np.argmax(model_pred_train, axis=1)
    model_pred_test = np.argmax(model_pred_test, axis=1)
    model_train_score = accuracy_score(model_pred_train, X_train_sentiment)
    model_test_score = accuracy_score(model_pred_test, X_test_sentiment)
    print(f"LSTM Train Score = {model_train_score * 100:.4f}%")
    print(f"LSTM Test Score = {model_test_score * 100:.4f}%")

    model_precision_score = precision_score(X_test_sentiment, model_pred_test, average="macro")
    model_f1_score = f1_score(X_test_sentiment, model_pred_test, average="macro")
    model_recall_score = recall_score(X_test_sentiment, model_pred_test, average="macro")
    model_accuracy_score = accuracy_score(X_test_sentiment, model_pred_test)

    print(f"LSTM Precision Score = {model_precision_score * 100:.4f}%")
    print(f"LSTM F1 Score = {model_f1_score * 100:.4f}%")
    print(f"LSTM Recall Score = {model_recall_score * 100:.4f}%")
    print(f"LSTM Accuracy Score = {model_accuracy_score * 100:.4f}%")

    print(classification_report(X_test_sentiment, model_pred_test, target_names=["Negative", "Neutral", "Positive"]))

    model_cm = confusion_matrix(X_test_sentiment, model_pred_test)
    fig, ax = plot_confusion_matrix(conf_mat=model_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["Negative", "Neutral", "Positive"], figsize=(10, 10))
    plt.title("LSTM (Two Input) - Sentiment Analysis")
    plt.savefig("./output/lstm_two_input.png")

if __name__ == "__main__":
    main()