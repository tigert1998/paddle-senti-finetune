import argparse
import os.path as osp
import random

import pandas as pd


def preprocess_csri_content_page_dataset(dataset_path, train_ratio):
    # https://csri.scu.edu.cn/info/1012/2827.htm

    df = pd.read_excel(osp.join(dataset_path, "hierarchical_data.xlsx"))
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["keywords"] = df["keywords"].fillna("")
    labels = list(
        set(
            df["first_level"].astype("string")
            + "##"
            + df["second_level"].astype("string")
        )
    )
    with open(osp.join(dataset_path, "label.txt"), "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")

    random.seed(10086)
    indices = list(range(len(df)))
    random.shuffle(indices)
    train_size = int(len(df) * train_ratio)
    train_indices = indices[:train_size]
    dev_indices = indices[train_size:]
    for split, split_indices in {"train": train_indices, "dev": dev_indices}.items():
        f = open(osp.join(dataset_path, f"{split}.txt"), "w", encoding="utf-8")
        for i in split_indices:
            text = (
                "Title: "
                + str(df.iloc[i]["title"])
                + "; Description: "
                + str(df.iloc[i]["description"])
                + "; Keywords: "
                + str(df.iloc[i]["keywords"])
            ).replace("\n", "<br>")
            label = df.iloc[i]["first_level"] + "##" + df.iloc[i]["second_level"]
            f.write(f"{text}\t{label}\n")
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path")
    args = parser.parse_args()

    preprocess_csri_content_page_dataset(args.dataset_path, 0.9)
