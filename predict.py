import argparse

import paddle
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dev")
    parser.add_argument("--label")
    parser.add_argument("--max-length", default=128, type=int)
    args = parser.parse_args()

    with open(args.label, "r", encoding="utf-8") as f:
        labels = f.readlines()

    with open(args.dev, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_preds = []
    all_gts = []

    tokenizer = AutoTokenizer.from_pretrained("checkpoint")
    model = AutoModelForSequenceClassification.from_pretrained("checkpoint")
    model.eval()
    with paddle.no_grad():
        for line in tqdm(lines):
            sentence, label = line.split("\t")
            label_id = labels.index(label)
            tokenize_result = tokenizer(
                sentence,
                max_length=args.max_length,
                padding="max_length",
                truncation=True,
            )
            logits = model(
                input_ids=paddle.tensor(tokenize_result["input_ids"])[None, :],
                token_type_ids=paddle.tensor(tokenize_result["token_type_ids"])[
                    None, :
                ],
            )
            pred = np.argmax(logits.numpy(), axis=1)[0]
            all_preds.append(pred)
            all_gts.append(label_id)

    print("accuracy score: ", accuracy_score(all_preds, all_gts))
