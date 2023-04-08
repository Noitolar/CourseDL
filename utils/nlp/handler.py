import torch
import torch.nn as nn
import torchvision.transforms as trans
import time
import transformers as tfm
import utils.nlp as unlp


class ModelHandlerGenerator(nn.Module):
    def __init__(self, config: unlp.config.ConfigObject):
        super().__init__()
        self.config = config
        tfm.set_seed(config.seed)
        self.recorder = unlp.recorder.Recorder(config.log_path)
        self.device = config.device
        self.model = config.model_class(**config.model_params).to(config.device)
        self.criterion = config.criterion_class(**config.criterion_params)

    def log(self, message):
        self.recorder.audit(message)

    def log_config(self):
        self.log(f"\n\n[+] exp starts from: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        for config_key, config_value in self.config.params_dict.items():
            if config_key.endswith("_class"):
                self.log(f"[+] {config_key.replace('_class', '')}: {config_value.__name__ if config_value is not None else None}")
            elif config_key.endswith("_params") and isinstance(config_value, dict):
                for param_key, param_value in config_value.items():
                    self.log(f"    [-] {config_key.replace('_params', '')}.{param_key}: {param_value}")
            elif isinstance(config_value, trans.transforms.Compose):
                self.log(f"[+] {config_key}:")
                for index, value in enumerate(str(config_value).replace(" ", "").split("\n")[1:-1]):
                    self.log(f"    [-] {index:02d}: {value}")
            else:
                self.log(f"[+] {config_key}: {config_value}")

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
            lstm_h0 = torch.zeros(size=(self.model.num_lstm_layers, batch_size, self.model.lstm_output_size), dtype=torch.float, requires_grad=False)
            lstm_c0 = torch.zeros(size=(self.model.num_lstm_layers, batch_size, self.model.lstm_output_size), dtype=torch.float, requires_grad=False)
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
