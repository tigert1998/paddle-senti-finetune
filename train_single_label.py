import os.path as osp
import random
import argparse

import paddle
import paddlenlp
from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack
import pandas as pd
from sklearn.metrics import accuracy_score


def get_dataset(dataset_path, train_ratio):
    df0 = pd.read_csv(
        osp.join(
            dataset_path,
            "NLPcc2013-2014微博文本情感分类数据集/Nlpcc2013/Nlpcc2013Train_NoNone.tsv",
        )
    )
    df1 = pd.read_csv(
        osp.join(
            dataset_path,
            "NLPcc2013-2014微博文本情感分类数据集/Nlpcc2014/Nlpcc2014Train_NoNone.tsv",
        )
    )
    df = pd.concat([df0, df1])
    labels = sorted(df["标签"].drop_duplicates())
    labels = {k: v for v, k in enumerate(labels)}

    indices = list(range(len(df)))
    random.seed(10086)
    random.shuffle(indices)
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    def read(indices):
        for i in indices:
            label_id = labels[df.iloc[i]["标签"]]
            text = df.iloc[i]["文本"]
            yield {"text": text, "label": label_id}

    train_ds = load_dataset(lambda: read(train_indices), lazy=False)
    val_ds = load_dataset(lambda: read(val_indices), lazy=False)
    return labels, train_ds, val_ds


def preprocess_dataset(examples, tokenizer, max_length):
    result = tokenizer(
        examples["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    result["labels"] = examples["label"]
    return result


def collate(batch):
    input_ids = Stack(dtype="int64")([example["input_ids"] for example in batch])
    token_type_ids = Stack(dtype="int64")(
        [example["token_type_ids"] for example in batch]
    )
    labels = Stack(dtype="int64")([example["labels"] for example in batch])
    return input_ids, token_type_ids, labels


def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with paddle.no_grad():
        for batch in data_loader:
            input_ids, token_type_ids, labels = batch

            logits = model(input_ids, token_type_ids=token_type_ids)
            preds = paddle.argmax(logits, axis=1).numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return accuracy_score(all_labels, all_preds)


def train(
    dataset_path,
    train_ratio,
    max_length,
    batch_size,
    num_epochs,
    lr,
    warmup_ratio,
    weight_decay,
    log_steps,
):
    labels, train_ds, val_ds = get_dataset(dataset_path, train_ratio)
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    trans_func = lambda t: preprocess_dataset(t, tokenizer, max_length)
    train_ds = train_ds.map(trans_func)
    val_ds = val_ds.map(trans_func)
    train_data_loader = paddle.io.DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=collate,
        batch_size=batch_size,
        drop_last=False,
    )
    val_data_loader = paddle.io.DataLoader(
        val_ds,
        shuffle=False,
        collate_fn=collate,
        batch_size=batch_size,
        drop_last=False,
    )
    num_training_steps = len(train_data_loader) * num_epochs
    lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(
        lr, num_training_steps, warmup_ratio
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "ernie-3.0-medium-zh", num_labels=len(labels), dtype="float32"
    )
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x
        in [
            p.name
            for n, p in model.named_parameters()
            if not any([nd in n for nd in ["bias", "norm"]])
        ],
    )
    criterion = paddle.nn.loss.CrossEntropyLoss()

    for epoch_id in range(num_epochs):
        model.train()
        total_loss = 0.0

        for step_id, batch in enumerate(train_data_loader):
            input_ids, token_type_ids, labels = batch

            logits = model(input_ids, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if (step_id + 1) % log_steps == 0:
                print(
                    f"Epoch [{epoch_id}/{num_epochs}], Step [{step_id}/{len(train_data_loader)}], "
                    f"Loss: {loss.item():.4f}, LR: {optimizer.get_lr():.6f}"
                )

        avg_loss = total_loss / len(train_data_loader)

        val_accuracy = evaluate(model, val_data_loader)

        print(
            f"Epoch [{epoch_id}/{num_epochs}] Train Loss: {avg_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training ERNIE model for single label")
    parser.add_argument("--dataset-path")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        train_ratio=0.9,
        max_length=128,
        batch_size=32,
        num_epochs=5,
        lr=2e-5,
        warmup_ratio=0.1,
        weight_decay=1e-2,
        log_steps=128,
    )
