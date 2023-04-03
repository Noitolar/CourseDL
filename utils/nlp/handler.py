import torch
import torch.nn as nn
import transformers as tfm
import time
import utils.common.recorder as recorder


class ModelHandlerNlpClassifier(nn.Module):
    def __init__(self, model, criterion, device, logpath=None):
        super().__init__()
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.recorder = recorder.Recorder(logpath)
        self.extensions = dict()

    def log_metadata(self):
        self.recorder.audit(f"\n\n[+] exp starts from: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        self.recorder.audit(f"[+] arch: {self.model.__class__.__name__}")
        self.recorder.audit(f"[+] device: {self.device}")
        for key, value in self.extensions.items():
            self.recorder.audit(f"[+] {key}: {value}")
        self.recorder.audit(f"[+] criterion: {self.criterion.__class__.__name__}")

    def device_transfer(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        if isinstance(data, dict):
            data = {key: value.to(self.device) for key, value in data.items()}
        return data

    def forward(self, inputs: dict, targets=None):
        inputs = self.device_transfer(inputs)
        targets = self.device_transfer(targets)
        preds = self.model(**inputs)
        loss = self.criterion(preds, targets) if targets is not None else None
        return preds, loss


class ModelHandlerNlpGenerator(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.generator = tfm.TextGenerationPipeline(self.model.gpt2, self.model.tokenizer, device=device)
        self.extensions = dict()

    def forward(self, text_inputs, max_length, do_sample):
        text_inputs = "[CLS] " + text_inputs if not text_inputs.startswith("[CLS]") else text_inputs
        outputs = self.generator(text_inputs=text_inputs, max_length=max_length, do_sample=do_sample)[0]["generated_text"]
        outputs = outputs.replace("[CLS]", "")
        outputs = outputs.replace(" ", "")
        outputs = outputs[:-len(outputs.split("。")[-1])]
        outputs = outputs.replace("。", "。\n")

        return outputs
