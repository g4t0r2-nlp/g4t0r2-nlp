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
import seaborn as sns

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
    parser = argparse.ArgumentParser(description="Attention for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="/mnt/d/work2/teknofest-tddi/data/processed/cleaned.csv", help="Dataset path")
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

    tokenizer = preprocessing.text.Tokenizer(
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower = False,
        split = " "
    )
    tokenizer.fit_on_texts(X)

    X_train_tokenizer = tokenizer.texts_to_sequences(X_train)
    X_test_tokenizer = tokenizer.texts_to_sequences(X_test)

    num_tokens = [len(tokens) for tokens in X_train_tokenizer + X_test_tokenizer]
    num_tokens = np.array(num_tokens)
    maxlen = int(np.mean(num_tokens) + (2 * np.std(num_tokens)))

    X_train_tokenizer = preprocessing.sequence.pad_sequences(X_train_tokenizer, maxlen=args.maxlen)
    X_test_tokenizer = preprocessing.sequence.pad_sequences(X_test_tokenizer, maxlen=args.maxlen)

    input_dim = len(tokenizer.word_index) + 1

    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.dense_dim = dense_dim
            self.num_heads = num_heads
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.dense_proj = models.Sequential([
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim)
            ])
            self.layernorm1 = layers.LayerNormalization()
            self.layernorm2 = layers.LayerNormalization()

        def call(self, inputs, mask=None):
            if mask is not None:
                mask = mask[:, tf.newaxis, :]
            attention_output = self.attention(inputs, inputs, attention_mask=mask)
            proj_input = self.layernorm1(inputs + attention_output)
            proj_output = self.dense_proj(proj_input)
            return self.layernorm2(proj_input + proj_output)
        
    class PositionalEmbedding(layers.Layer):
        def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.token_embeddings = layers.Embedding(input_dim, output_dim)
            self.position_embeddings = layers.Embedding(sequence_length, output_dim)
            self.sequence_length = sequence_length
            self.input_dim = input_dim
            self.output_dim = output_dim
            
        def call(self, inputs):
            length = tf.shape(inputs)[-1]
            positions = tf.range(start=0, limit=length, delta=1)
            embedded_tokens = self.token_embeddings(inputs)
            embedded_positions = self.position_embeddings(positions)
            return embedded_tokens + embedded_positions
        
    n_class = 3
    input_layer = layers.Input(shape=(None,), dtype="int64")
    posemb_layer = PositionalEmbedding(args.maxlen, len(tokenizer.word_index) + 1, 256)(input_layer)
    transformer_block = TransformerBlock(256, 64, 2)(posemb_layer)
    gmaxpool_layer = layers.GlobalMaxPooling1D()(transformer_block)
    dropout_layer = layers.Dropout(0.5)(gmaxpool_layer)
    output_layer = layers.Dense(n_class, activation="softmax")(dropout_layer)

    model = models.Model(input_layer, output_layer)

    model.compile(loss='categorical_crossentropy', 
                    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate), 
                    metrics=['accuracy'])    

    model1_train_start = time.time()
    model_history = model.fit(
        X_train_tokenizer, 
        y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        validation_split=0.1, 
        callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=3)]
    )
    model1_train_time = time.time() - model1_train_start
    print(f"Attention Train Time = {model1_train_time:.4f}")

    model1_test_start = time.time()
    model_pred_test = model.predict(X_test_tokenizer, verbose=0)
    model1_test_time = time.time() - model1_test_start
    print(f"Attention Test Time = {model1_test_time:.4f}")

    true_labels_train = np.argmax(y_train, axis=1)
    true_labels_test = np.argmax(y_test, axis=1)

    model_pred_train = model.predict(X_train_tokenizer, verbose=0)
    model_pred_train = np.argmax(model_pred_train, axis=1)
    model_pred_test = np.argmax(model_pred_test, axis=1)
    model_train_score = accuracy_score(model_pred_train, true_labels_train)
    model_test_score = accuracy_score(model_pred_test, true_labels_test)
    print(f"Attention Train Score = {model_train_score * 100:.4f}%")
    print(f"Attention Test Score = {model_test_score * 100:.4f}%")

    model_precision_score = precision_score(true_labels_test, model_pred_test, average="macro")
    model_f1_score = f1_score(true_labels_test, model_pred_test, average="macro")
    model_recall_score = recall_score(true_labels_test, model_pred_test, average="macro")
    model_accuracy_score = accuracy_score(true_labels_test, model_pred_test)

    print(f"Attention Precision Score = {model_precision_score * 100:.4f}%")
    print(f"Attention F1 Score = {model_f1_score * 100:.4f}%")
    print(f"Attention Recall Score = {model_recall_score * 100:.4f}%")
    print(f"Attention Accuracy Score = {model_accuracy_score * 100:.4f}%")

    print(classification_report(true_labels_test, model_pred_test, target_names=["Negative", "Neutral", "Positive"]))

    model_cm = confusion_matrix(true_labels_test, model_pred_test)
    fig, ax = plot_confusion_matrix(conf_mat=model_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["Negative", "Neutral", "Positive"], figsize=(10, 10))
    plt.title("Attention - Sentiment Analysis")
    plt.savefig("./output/attention.png")

if __name__ == "__main__":
    main()