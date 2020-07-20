import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

import src.config as config
from src.model import BertClassifierModel
from src.dataset import KGDataset, KGExample
from src.engine import eval_fn


def run_eval():

    data_dir = config.DATA_DIR

    tokenizer = BertTokenizer.from_pretrained(config.BERT_TOKENIZER_OUTPUT)
    eval_examples = KGExample(data_dir).get_test_example()
    eval_dataset = KGDataset(eval_examples, tokenizer, config.MAX_SEQUENCE_LEN)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        sampler=eval_sampler
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifierModel(config.BERT_MODEL_OUTPUT)

    model.to(device)
    eval_fn(model, device, eval_dataloader)


if __name__ == '__main__':
    run_eval()