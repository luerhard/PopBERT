import numpy as np
import torch
from sklearn.metrics import f1_score

import src.bert.utils as bert_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, dataloader, optimizer, lr_scheduler, clip):
    train_loss = 0.0
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        encodings = batch["encodings"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(**encodings, labels=labels)

        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        lr_scheduler.step()

        train_loss += out.loss.item()

    return train_loss


def eval_epoch(model, dataloader):
    eval_loss = 0.0
    y_true = []
    y_pred = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            encodings = batch["encodings"]
            encodings = encodings.to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            out = model(**encodings, labels=labels)
            if model.config.problem_type == "multi_label_classification":
                preds = torch.nn.functional.sigmoid(out.logits)
            elif model.config.problem_type == "single_label_classification":
                preds = torch.nn.functional.softmax(out.logits)

            eval_loss += out.loss.item()
            y_true.extend(batch["labels"].numpy())
            y_pred.extend(preds.to("cpu").detach().numpy())

    y_true = np.array(y_true)
    if model.config.problem_type == "multi_label_classification":
        y_pred_bin = np.where(np.array(y_pred) > 0.5, 1, 0)
        thresh_finder = bert_utils.ThresholdFinder(type="single_task")
        thresholds = thresh_finder.find_thresholds(np.array(y_true), np.array(y_pred))
    elif model.config.problem_type == "single_label_classification":
        y_pred_bin = np.argmax(np.array(y_pred), axis=1)
        thresholds = {k: 0.5 for k in range(model.config.num_labels)}

    try:
        score = f1_score(np.array(y_true), np.array(y_pred_bin), average="macro", zero_division=0)
    except Exception as e:  # noqa: E722
        print(e)
        raise

    return eval_loss, score, thresholds
