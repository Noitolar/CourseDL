import transformers as tfm
import torch.nn as nn


class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = tfm.BertModel.from_pretrained("bert-base-uncased", cache_dir="./cache")
        self.tokenizer = tfm.BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs


class DistilBertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.distil_bert = tfm.DistilBertModel.from_pretrained("distilbert-base-uncased", cache_dir="./cache")
        self.tokenizer = tfm.DistilBertTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="./cache")
        self.pre_fc = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distil_bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        outputs = self.pre_fc(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs


class Gpt2Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = tfm.GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem", cache_dir="./cache")
        self.tokenizer = tfm.BertTokenizer.from_pretrained("uer/gpt2-chinese-poem", cache_dir="./cache")
        self.config = self.gpt2.record_metadata
        self.generate = self.gpt2.generate

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs
