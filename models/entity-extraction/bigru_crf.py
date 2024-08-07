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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix
from sklearn_crfsuite import metrics

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, preprocessing
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from tensorflow_addons.layers import CRF
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow_addons.optimizers import AdamW

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
    parser = argparse.ArgumentParser(description="Bi-GRU + CRF for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="../data/processed/all_tagged_aspects_just_aspect_cleaned.csv", help="Dataset path")
    parser.add_argument("--maxlen", type=str, default=128, help="Max Length")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--split-size", type=float, default=0.9, help="Train-Test split size")
    parser.add_argument("--output", type=str, default=".", help="Model save path")
    return parser.parse_args()

def build_model(input_dim, embedding_dim, maxlen):
    input_layer = Input(shape=(maxlen,))

    embeddings = Embedding(input_dim, embedding_dim, input_length=maxlen, mask_zero=True, trainable=True)(input_layer)

    output_sequences = Bidirectional(GRU(units=32, return_sequences=True))(embeddings)
    output_sequences = Bidirectional(GRU(units=64, return_sequences=True))(output_sequences)
    output_sequences = Bidirectional(GRU(units=32, return_sequences=True))(output_sequences)

    dense_out = TimeDistributed(Dense(16, activation="relu"))(output_sequences)

    mask = Input(shape=(maxlen,), dtype=tf.bool)
    crf = CRF(4, name='crf')
    predicted_sequence, potentials, sequence_length, crf_kernel = crf(dense_out)

    model = Model(input_layer, potentials)
    model.compile(optimizer=AdamW(weight_decay=0.001), loss= SigmoidFocalCrossEntropy())
    return model

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

    words = list(set(df['Word'].values))
    tags = list(set(df["Tag"].values))

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    idx2word = {i: w for w, i in word2idx.items()}

    tag2idx = {t: i+1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0

    idx2tag = {i: w for w, i in tag2idx.items()}

    sentences = [(list(zip(group['Word'], group['Tag']))) for _, group in df.groupby('rid')]

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = preprocessing.sequence.pad_sequences(maxlen=args.maxlen, sequences=X, padding="post", value=word2idx["PAD"])

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = preprocessing.sequence.pad_sequences(maxlen=args.maxlen, sequences=y, padding="post", value=tag2idx["PAD"])
    y = [to_categorical(i, num_classes=len(tags)+1) for i in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.05, random_state=1)

    print("X_train shape:", X_train.shape)
    print("X_valid shape:", X_valid.shape)
    print("X_test shape:", X_test.shape)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)

    print("\n====================")
    print(f"Full dataset: {len(sentences)}")
    print(f"Train dataset: {X_train.shape[0]}")
    print(f"Test dataset: {X_test.shape[0]}")
    print("====================")

    input_dim = len(word2idx) + 1
    embedding_dim = 300

    model = build_model(input_dim, embedding_dim, args.maxlen)

    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file="bilstm_crf_model.png")

    history = model.fit(X_train, np.array(y_train), validation_split=0.1, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

    model1_train_start = time.time()
    model_history = model.fit(
        X_train, 
        y_train, 
        epochs=2, 
        batch_size=128, 
        validation_data=[X_valid, y_valid], 
        callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=3)]
    )
    model1_train_time = time.time() - model1_train_start
    print(f"Bi-GRU + CRF Train Time = {model1_train_time:.4f}")

    model1_test_start = time.time()
    model_pred_test = model.predict(X_test, verbose=0)
    model1_test_time = time.time() - model1_test_start
    print(f"Bi-GRU + CRF Test Time = {model1_test_time:.4f}")

    true_labels_train = np.argmax(y_train, axis=-1)
    true_labels_test = np.argmax(y_test, axis=-1)

    model_pred_train = model.predict(X_train, verbose=0)
    model_pred_train = np.argmax(model_pred_train, axis=-1)
    model_pred_test = np.argmax(model_pred_test, axis=-1)
    model_train_score = metrics.flat_accuracy_score(model_pred_train, true_labels_train)
    model_test_score = metrics.flat_accuracy_score(model_pred_test, true_labels_test)
    print(f"Bi-GRU + CRF Train Score = {model_train_score * 100:.4f}%")
    print(f"Bi-GRU + CRF Test Score = {model_test_score * 100:.4f}%")

    model_precision_score = metrics.flat_precision_score(true_labels_test, model_pred_test, average="macro")
    model_f1_score = metrics.flat_f1_score(true_labels_test, model_pred_test, average="macro")
    model_recall_score = metrics.flat_recall_score(true_labels_test, model_pred_test, average="macro")
    model_accuracy_score = metrics.flat_accuracy_score(true_labels_test, model_pred_test)

    print(f"Bi-GRU + CRF Precision Score = {model_precision_score * 100:.4f}%")
    print(f"Bi-GRU + CRF F1 Score = {model_f1_score * 100:.4f}%")
    print(f"Bi-GRU + CRF Recall Score = {model_recall_score * 100:.4f}%")
    print(f"Bi-GRU + CRF Accuracy Score = {model_accuracy_score * 100:.4f}%")

    print(metrics.flat_classification_report(true_labels_test, model_pred_test, target_names=["PAD", "O", "I-A", "B-A"]))

    model_cm = confusion_matrix(metrics.flatten(true_labels_test), metrics.flatten(model_pred_test))
    fig, ax = plot_confusion_matrix(conf_mat=model_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["PAD", "O", "I-A", "B-A"], figsize=(10, 10))
    plt.title("Bi-GRU + CRF - Entity Extraction")
    plt.savefig("./output/bigru_crf.png")
    plt.show()

if __name__ == '__main__':
    main()
