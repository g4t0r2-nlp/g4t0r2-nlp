import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics

from mlxtend.plotting import plot_confusion_matrix

from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from tensorflow_addons.layers import CRF
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow_addons.optimizers import AdamW

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="CRF for Entity Extraction")
    parser.add_argument("--dataset", type=str, default="../data/all_tagged_aspects_just_aspect.csv", help="Dataset path")
    parser.add_argument("--maxlen", type=str, default=128, help="Max Length")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--split-size", type=float, default=0.9, help="Train-Test split size")
    parser.add_argument("--output", type=str, default=".", help="Model save path")
    return parser.parse_args()

def build_model(input_dim, embedding_dim, maxlen, num_classes):
    input_ = layers.Input(shape=(maxlen,))

    embeddings = layers.Embedding(input_dim, embedding_dim, input_length=maxlen, mask_zero=True, trainable=True,
    name = 'embedding_layer'
    )(input_)

    output_sequences = layers.Bidirectional(layers.LSTM(units=50, return_sequences=True))(embeddings)
    output_sequences = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True))(output_sequences)
    output_sequences = layers.Bidirectional(layers.LSTM(units=50, return_sequences=True))(output_sequences)

    dense_out = layers.TimeDistributed(layers.Dense(num_classes, activation="softmax"))(output_sequences)

    model = models.Model(input_, dense_out)
    model.compile(optimizer = AdamW(weight_decay=0.001), loss = SigmoidFocalCrossEntropy(), metrics=["accuracy"])
    return model

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
    plt.title("Bi-LSTM - Aspect Extraction")
    plt.savefig("./output/bilstm_confusion_matrix.png")

def main():
    args = parse_arguments()

    set_seed(args.seed)

    df = pd.read_csv(args.dataset)

    sentences = df.groupby('rid').apply(lambda x: (x['Word'].tolist(), x['Tag'].tolist()))
    sentences = sentences.tolist()

    words = [str(s[0]) for s in sentences]
    tags = [s[1] for s in sentences]

    word_tokenizer = Tokenizer(lower=False, oov_token="UNK")
    tag_tokenizer = Tokenizer(lower=False)

    word_tokenizer.fit_on_texts(words)

    word_tokenizer.word_index["PAD"] = 0

    word_tokenizer.index_word = {i: w for w, i in word_tokenizer.word_index.items()}

    tag_tokenizer.fit_on_texts(tags)

    X = word_tokenizer.texts_to_sequences(words)
    y = tag_tokenizer.texts_to_sequences(tags)

    X = pad_sequences(X, maxlen=args.maxlen, padding='post')
    y = pad_sequences(y, maxlen=args.maxlen, padding='post')

    num_classes = len(tag_tokenizer.word_index) + 1

    y = [to_categorical(i, num_classes=num_classes) for i in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.split_size, random_state=42)

    print("\n====================")
    print(f"Full dataset: {len(sentences)}")
    print(f"Train dataset: {X_train.shape[0]}")
    print(f"Test dataset: {X_test.shape[0]}")
    print("====================")

    input_dim = len(word_tokenizer.word_index) + 1
    embedding_dim = 300

    model = build_model(input_dim, embedding_dim, args.maxlen, num_classes)

    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file="bilstm_model.png")

    history = model.fit(X_train, np.array(y_train), validation_split=0.1, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("history.csv", index=False)

    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "valid"])
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "valid"])
    plt.savefig("accuracy_curve.png")

    model_predictions = model.predict(np.array(X_test), verbose=0)
    pred_labels = np.argmax(model_predictions, axis=-1)
    test_labels = np.argmax(y_test, axis=-1)
    tags = tag_tokenizer.index_word | {0: "PAD"}
    class_names = list((tag_tokenizer.index_word | {0: "PAD"}).values())
    evaluate_model(test_labels, pred_labels, class_names)

if __name__ == '__main__':
    main()
