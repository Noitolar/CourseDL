import torch
import torch.nn as nn
import torchvision.transforms as trans
import time
import utils.nlp.recorder as recorder


class ModelHandlerGenerator(nn.Module):
    def __init__(self, model, criterion, vocab: dict, device, logpath):
        super().__init__()
        self.metadata = dict()
        self.recorder = recorder.Recorder(logpath)
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.vocab_encode = vocab["encode"]
        self.vocab_decode = vocab["decode"]
        self.record_metadata("model", model)
        self.record_metadata("criterion", criterion)
        self.record_metadata("device", device)

    def encode(self, character: str):
        return self.vocab_encode[character]

    def decode(self, token: int):
        return self.vocab_decode[token]

    def log(self, message):
        self.recorder.audit(message)

    def record_metadata(self, config_key, config_value):
        if isinstance(config_value, str) or isinstance(config_value, int):
            self.metadata[config_key] = config_value
        elif isinstance(config_value, dict):
            for key, value in config_value.items():
                self.metadata[f"{config_key}.{key}"] = value
        elif isinstance(config_value, trans.transforms.Compose):
            self.metadata[config_key] = ""
            for index, value in enumerate(str(config_value).replace(" ", "").split("\n")[1:-1]):
                self.metadata[f"{config_key}.{index:02d}"] = value
        else:
            self.metadata[config_key] = type(config_value).__name__

    def log_metadata(self, show_model=True):
        self.log(f"\n\n[+] exp starts from: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        for key, value in self.metadata.items():
            indentation = " " * 4 * key.count(".")
            self.log(f"{indentation}[+] {key}: {value}")
            if key == "model" and show_model:
                for layer_string in str(self.model).split("\n"):
                    self.log(f"    {indentation}[+] {layer_string}")

    def device_transfer(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        if isinstance(data, dict):
            data = {key: value.to(self.device) for key, value in data.items()}
        if isinstance(data, tuple):
            data = tuple([child.to(self.device) for child in data])
        if isinstance(data, list):
            data = [child.to(self.device) for child in data]
        return data

    def forward(self, inputs: torch.Tensor, hiddens: tuple = None, targets: torch.Tensor = None):
        if hiddens is None:
            batch_size = inputs.shape[0]
            lstm_h0 = torch.zeros(size=(self.model.num_lstm_layers, batch_size, self.model.lstm_output_size), dtype=torch.float)
            lstm_c0 = torch.zeros(size=(self.model.num_lstm_layers, batch_size, self.model.lstm_output_size), dtype=torch.float)
            hiddens = (lstm_h0, lstm_c0)
        inputs = self.device_transfer(inputs)
        targets = self.device_transfer(targets)
        hiddens = self.device_transfer(hiddens)
        preds, hiddens = self.model(inputs, hiddens)

        # 相当于把batch内的多个样本拼接起来算损失函数
        if targets is not None:
            batch_size, sequence_length, vocab_size = preds.shape
            preds = preds.reshape(batch_size * sequence_length, vocab_size)
            targets = targets.reshape(batch_size * sequence_length)
            loss = self.criterion(preds, targets)
            self.recorder.update(preds, targets, loss)
        else:
            loss = None
        return preds, loss, hiddens
