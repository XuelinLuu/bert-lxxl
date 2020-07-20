import torch.nn as nn
import transformers

class BertClassifierModel(nn.Module):
    def __init__(self, bert_path):
        super(BertClassifierModel, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.dropout(pooled_output)
        return self.classifier(output)