import torch
import torch.nn as nn
import tqdm
import numpy as np
from src.dataset import KGData


def loss_fn(output, labels):
    return nn.BCEWithLogitsLoss()(output, labels)


def train_fn(model: nn.Module, device, train_dataloader, optimizer, epoch, scheduler=None):
    """ a loop of train"""
    model.train()
    all_loss = 0
    tk = tqdm.tqdm(train_dataloader, desc="Train Iter")
    for idx, data in enumerate(tk):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        labels = data["label"].to(device, dtype=torch.float)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        all_loss += loss.item()
        avg_loss = all_loss / (idx + 1)

        tk.set_postfix(epoch=(epoch + 1), avg_loss=avg_loss)


def eval_fn(model, device, eval_dataloader):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    preds = []
    tk = tqdm.tqdm(eval_dataloader, desc="Eval Iter")
    with torch.no_grad():
        for idx, data in enumerate(tk):

            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            labels = data["label"].to(device, dtype=torch.float)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs, labels)
            eval_loss += loss.item()

            if len(preds) == 0:
                preds.append(outputs.detach().cpu().numpy())
            else:
                np.append(
                    preds[0], outputs.detach().cpu().numpy(), axis=0
                )
            pred = np.argmax(outputs.detach().cpu().numpy(), axis=-1)
            if pred == np.argmax(labels.detach().cpu().numpy(), axis=-1):
                eval_acc += 1
            avg_loss = eval_loss / (idx + 1)
            avg_acc = eval_acc / (idx + 1)
            tk.set_postfix(avg_loss=avg_loss, avg_acc=avg_acc)

        print("---------------------------avg Loss------------------------------")
        print(eval_loss / len(eval_dataloader))
        print("---------------------------avg Acc------------------------------")
        print(eval_acc / len(eval_dataloader))
        print("---------------------------num Acc------------------------------")
        print(eval_acc)
        print("---------------------------avg Wro------------------------------")
        print(len(eval_dataloader - eval_acc))
