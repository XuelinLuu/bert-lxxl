import os
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.file_utils import WEIGHTS_NAME

import src.config as config
from src.model import BertClassifierModel
from src.dataset import KGDataset, KGExample
from src.engine import train_fn


def run_train():

    data_dir = config.DATA_DIR

    tokenizer = BertTokenizer.from_pretrained(config.BERT_TOKENIZER_PATH)
    train_examples = KGExample(data_dir).get_train_example()
    train_dataset = KGDataset(train_examples, tokenizer, config.MAX_SEQUENCE_LEN)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        sampler=train_sampler
    )


    num_train_step = len(train_dataset) / config.TRAIN_BATCH_SIZE * config.TRAIN_EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifierModel(config.BERT_MODEL_PATH)
    optimizer = AdamW(model.parameters(), lr=1e-3)


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_step
    )

    model.to(device)
    for epoch in range(config.TRAIN_EPOCHS):
        train_fn(model, device, train_dataloader, optimizer, epoch, scheduler)

        model_to_save = model.module if hasattr(model, "module") else model
        output_file = os.path.join(f"{config.BERT_MODEL_OUTPUT}/{epoch+1}", WEIGHTS_NAME)
        if not os.path.exists(f"{config.BERT_MODEL_OUTPUT})/{epoch+1}"):
            os.mkdir(f"{config.BERT_MODEL_OUTPUT}/{epoch+1}")
        torch.save(model_to_save.state_dict(), output_file)
        tokenizer.save_vocabulary(f"{config.BERT_MODEL_OUTPUT}/{epoch+1}/vocab.txt")

    model_to_save = model.module if hasattr(model, "module") else model
    output_file = os.path.join(config.BERT_MODEL_OUTPUT, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_file)
    tokenizer.save_vocabulary(config.BERT_TOKENIZER_OUTPUT)

if __name__ == '__main__':
    run_train()