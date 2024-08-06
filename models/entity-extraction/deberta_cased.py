import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import torch
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, Dataset
from transformers import AdamW
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="DeBERTa for Entity Extraction")
    parser.add_argument("--bert-repository", type=str, default="microsoft/mdeberta-v3-base", help="DeBERTa Repository")
    parser.add_argument("--dataset", type=str, default="/mnt/d/work2/teknofest-tddi/data/processed/all_tagged_aspects_just_aspect_cleaned_sentences.csv", help="Dataset path")
    parser.add_argument("--maxlen", type=int, default=128, help="Max length")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=int, default=10, help="Max grad norm")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--split-size", type=float, default=0.9, help="Train-Test split size")
    parser.add_argument("--output", type=str, default=".", help="Model save path")
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index].split(",") 

        encoding = self.tokenizer(sentence, return_offsets_mapping=True, padding='max_length', truncation=True,  max_length=self.max_len)
        
        labels = [labels_to_ids[label] for label in word_labels] 
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = labels[i]
                i += 1
      
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels) 
        return item

    def __len__(self):
        return self.len

def train_validate_test_split(df, split_size):
    perm = np.random.permutation(df.index)
    train_end = int(split_size * len(df.index))
    validate_end = int(((1 - split_size) / 2) * len(df.index)) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def train_model(model, training_loader, optimizer, max_grad_norm, epochs):
    for epoch in range(epochs):
        print(f"Training epoch: {epoch + 1}")
        model.train()
        tr_loss, tr_accuracy = 0, 0
        nb_tr_steps = 0
        tr_preds, tr_labels = [], []

        for idx, batch in enumerate(training_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = output.loss
            tr_logits = output.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            if idx % 1000 == 0:
                print(f"Training loss per 1000 training steps: {tr_loss / nb_tr_steps}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            flattened_targets = labels.view(-1)
            active_logits = tr_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            active_accuracy = labels.view(-1) != -100
            active_targets = torch.masked_select(flattened_targets, active_accuracy)
            active_predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_labels.extend(active_targets.cpu().numpy())
            tr_preds.extend(active_predictions.cpu().numpy())

        tr_loss /= nb_tr_steps
        tr_accuracy = np.sum(np.array(tr_preds) == np.array(tr_labels)) / len(tr_labels)

        print(f"Training loss epoch: {tr_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

def validate_model(model, testing_loader):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids=ids, attention_mask=mask, labels=labels)
            eval_loss += output.loss.item()
            eval_logits = output.logits

            nb_eval_steps += 1
            if idx % 1000 == 0:
                print(f"Validation loss per 1000 evaluation steps: {eval_loss / nb_eval_steps}")

            flattened_targets = labels.view(-1)
            active_logits = eval_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            active_accuracy = labels.view(-1) != -100
            active_targets = torch.masked_select(flattened_targets, active_accuracy)
            active_predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(active_targets.cpu().numpy())
            eval_preds.extend(active_predictions.cpu().numpy())

    eval_loss /= nb_eval_steps
    eval_accuracy = np.sum(np.array(eval_preds) == np.array(eval_labels)) / len(eval_labels)

    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    return eval_labels, eval_preds

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
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True, class_names=np.unique(labels))
    plt.title("DeBERTa - Aspect Extraction")
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

    training_set = CustomDataset(df_train, tokenizer, args.maxlen)
    validation_set = CustomDataset(df_validation, tokenizer, args.maxlen)
    testing_set = CustomDataset(df_test, tokenizer, args.maxlen)

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **valid_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = AutoModelForTokenClassification.from_pretrained(args.bert_repository, num_labels=len(labels_to_ids)).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    train_model(model, training_loader, optimizer, args.max_grad_norm, args.epochs)
    
    print("##### Validation #####")
    labels, predictions = validate_model(model, validation_loader)
    evaluate_model(labels, predictions)

    print("##### Testing #####")
    labels, predictions = validate_model(model, testing_loader)
    evaluate_model(labels, predictions)

if __name__ == "__main__":
    main()