import os.path as osp

import xml.etree.ElementTree as ET
import numpy as np

import paddle
import paddle.nn as nn
import paddlenlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack
from paddlenlp.transformers import AutoModel, AutoTokenizer, AutoConfig


def get_blurb_genre_collection_dataset(dataset_path):
    # https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html
    labels = set()
    with open(osp.join(dataset_path, "hierarchy.txt"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            labels.update(map(lambda s: s.strip(), line.split("\t")))
    labels = list(labels)

    def read(split):
        f = open(
            osp.join(dataset_path, f"BlurbGenreCollection_EN_{split}.txt"),
            "r",
            encoding="utf-8",
        )
        xmls = [c + "</book>" for c in f.read().split("</book>") if c.strip() != ""]
        f.close()

        for xml in xmls:
            content = xml.strip().replace("&", "&amp;")
            tree = ET.fromstring(content)
            title = tree.find("title").text
            body = tree.find("body").text
            text = f"Title: {title}\nBody: {body}"
            topics = tree.find("metadata").find("topics")
            label_ids = [labels.index(d.text) for d in topics]
            yield {"text": text, "labels": label_ids}

    train_ds = load_dataset(lambda: read("train"), lazy=False)
    val_ds = load_dataset(lambda: read("dev"), lazy=False)
    return labels, train_ds, val_ds


def tokenize(samples, tokenizer, max_length):
    output = tokenizer(
        samples["text"],
        max_seq_len=max_length,
        pad_to_max_seq_len="max_length",
    )
    output["labels"] = samples["labels"]
    return output


def collate(samples, num_labels):
    input_ids = Stack(dtype="int64")([s["input_ids"] for s in samples])
    labels = Stack(dtype="float32")(
        [np.eye(num_labels)[s["labels"]].sum(axis=0) for s in samples]
    )
    return input_ids, labels


class ModelForMultiLabelClassification(nn.Layer):
    def __init__(self, num_labels):
        super().__init__(None, "float32")
        self.model = AutoModel.from_pretrained("distilbert-base-cased", dtype="float32")
        config = AutoConfig.from_pretrained("distilbert-base-cased")
        self.linear = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        pooled = paddle.mean(outputs, axis=1)
        return self.linear(pooled)


def train(
    dataset_path,
    max_length,
    batch_size,
    lr,
    num_epochs,
    weight_decay,
    log_steps,
    warmup_ratio,
):
    labels_array, train_ds, val_ds = get_blurb_genre_collection_dataset(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    model = ModelForMultiLabelClassification(len(labels_array))
    trans_func = lambda samples: tokenize(samples, tokenizer, max_length)
    train_ds = train_ds.map(trans_func)
    val_ds = val_ds.map(trans_func)
    collate_fn = lambda samples: collate(samples, len(labels_array))
    train_data_loader = paddle.io.DataLoader(
        train_ds,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=False,
        batch_size=batch_size,
    )
    val_data_loader = paddle.io.DataLoader(
        val_ds,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,
    )

    lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(
        learning_rate=lr,
        total_steps=num_epochs * len(train_data_loader),
        warmup=warmup_ratio,
    )
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: "bias" not in x and "norm" not in x,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch_id in range(num_epochs):
        model.train()

        for step_id, batch in enumerate(train_data_loader):
            input_ids, labels = batch
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if (step_id + 1) % log_steps == 0:
                print(
                    f"Epoch [{epoch_id}/{num_epochs}], Step [{step_id}/{len(train_data_loader)}], "
                    f"Loss: {loss.item():.4f}, LR: {optimizer.get_lr():.6f}"
                )


if __name__ == "__main__":
    train(
        dataset_path="C:/Users/tiger/Projects/datasets/blurbgenrecollectionen",
        max_length=256,
        batch_size=16,
        lr=2e-5,
        num_epochs=5,
        weight_decay=1e-2,
        log_steps=256,
        warmup_ratio=0.1,
    )
