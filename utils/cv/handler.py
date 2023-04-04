import torch
import torch.nn as nn
import torchvision.transforms as trans
import time
import utils.cv.recorder as recorder


class ModelHandlerCv(nn.Module):
    def __init__(self, model, criterion, device, logpath):
        super().__init__()
        self.metadata = dict()
        self.recorder = recorder.Recorder(logpath)
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.record_metadata("model", model)
        self.record_metadata("criterion", criterion)
        self.record_metadata("device", device)

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
        return data

    def forward(self, inputs: torch.Tensor, targets=None):
        inputs = self.device_transfer(inputs)
        targets = self.device_transfer(targets)
        preds = self.model(inputs)
        loss = self.criterion(preds, targets) if targets is not None else None
        return preds, loss
