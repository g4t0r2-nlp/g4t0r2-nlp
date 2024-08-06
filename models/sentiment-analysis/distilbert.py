import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import torch
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, Dataset
from transformers import AdamW
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import argparse
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="DistilBERT for Entity Extraction")
    parser.add_argument("--bert-repository", type=str, default="dbmdz/distilbert-base-turkish-cased", help="DistilBERT Repository")
    parser.add_argument("--dataset", type=str, default="/mnt/d/work2/teknofest-tddi/data/processed/cleaned.csv", help="Dataset path")
    parser.add_argument("--maxlen", type=int, default=128, help="Max length")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=int, default=10, help="Max grad norm")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--split-size", type=float, default=0.9, help="Train-Test split size")
    parser.add_argument("--output", type=str, default=".", help="Model save path")
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_validate_test_split(df, split_size):
    perm = np.random.permutation(df.index)
    train_end = int(split_size * len(df.index))
    validate_end = int(((1 - split_size) / 2) * len(df.index)) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm.tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def validate_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm.tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def evaluate_model(labels, predictions):
    labels = [ids_to_labels[id.item()] for id in labels]
    predictions = [ids_to_labels[id.item()] for id in predictions]
    
    recall_val = recall_score(y_true=labels, y_pred=predictions, average='micro')
    print(f'Recall Score: {recall_val}')

    precision_val = precision_score(y_true=labels, y_pred=predictions, average='micro')
    print(f'Precision Score: {precision_val}')

    f1_val = f1_score(y_true=labels, y_pred=predictions, average='micro')
    print(f'F1 Score: {f1_val}')

    acc_val = accuracy_score(y_true=labels, y_pred=predictions)
    print(f"Accuracy Score: {acc_val}")

    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(labels, predictions))

    plt.figure()
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["Negative", "Neutral", "Positive"])
    plt.title("DistilBERT - Aspect Extraction")
    plt.show()

def main():
    args = parse_arguments()

    set_seed(args.seed)

    global labels_to_ids
    global ids_to_labels
    labels_to_ids = {'O': 0, 'B-A': 1, 'I-A': 2}
    ids_to_labels = {0: 'O', 1: 'B-A', 2: 'I-A'}

    tokenizer = AutoTokenizer.from_pretrained(args.bert_repository)
    
    df = pd.read_csv(args.dataset)
    df_train, df_validation, df_test = train_validate_test_split(df, args.split_size)
    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("\n====================")
    print(f"Full dataset: {df.shape[0]}")
    print(f"Train dataset: {df_train.shape[0]}")
    print(f"Valid dataset: {df_validation.shape[0]}")
    print(f"Test dataset: {df_test.shape[0]}")
    print("====================")

    train_params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0}
    valid_params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0}
    test_params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0}

    training_set = CustomDataset(df_train["text"].to_numpy(), df_train["aspect_polarity"].to_numpy(), tokenizer, args.maxlen)
    validation_set = CustomDataset(df_validation["text"].to_numpy(), df_validation["aspect_polarity"].to_numpy(), tokenizer, args.maxlen)
    testing_set = CustomDataset(df_test["text"].to_numpy(), df_test["aspect_polarity"].to_numpy(), tokenizer, args.maxlen)

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **valid_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = AutoModelForTokenClassification.from_pretrained(args.bert_repository, num_labels=len(labels_to_ids)).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    total_steps = len(training_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, training_loader, optimizer, scheduler, device)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = validate_model(model, validation_loader, device)
        print(f'Val loss {val_loss} accuracy {val_acc}')
    
    print("##### Validation #####")
    labels, predictions = validate_model(model, validation_loader)
    evaluate_model(labels, predictions)

    print("##### Testing #####")
    labels, predictions = validate_model(model, testing_loader)
    evaluate_model(labels, predictions)

if __name__ == "__main__":
    main()