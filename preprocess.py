import argparse
import os.path as osp

import xml.etree.ElementTree as ET


def label_escape(label):
    return label.replace(",", "<comma>")


def preprocess_blurb_genre_collection_dataset(dataset_path):
    # https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html
    labels = set()
    with open(osp.join(dataset_path, "hierarchy.txt"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            labels.update(map(lambda s: s.strip(), line.split("\t")))
    labels = list(labels)
    with open(osp.join(dataset_path, "label.txt"), "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label_escape(label)}\n")

    for split in ["train", "dev"]:
        with open(
            osp.join(dataset_path, f"BlurbGenreCollection_EN_{split}.txt"),
            "r",
            encoding="utf-8",
        ) as f:
            xmls = [c + "</book>" for c in f.read().split("</book>") if c.strip() != ""]

        f = open(osp.join(dataset_path, f"{split}.txt"), "w", encoding="utf-8")
        for xml in xmls:
            content = xml.strip().replace("&", "&amp;")
            tree = ET.fromstring(content)
            title = tree.find("title").text
            body = tree.find("body").text
            text = f"Title: {title}\nBody: {body}".replace("\n", "<br>")
            topics = tree.find("metadata").find("topics")
            labels = [label_escape(d.text) for d in topics]
            labels_str = ",".join(labels)
            f.write(f"{text}\t{labels_str}\n")
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path")
    args = parser.parse_args()

    preprocess_blurb_genre_collection_dataset(args.dataset_path)
