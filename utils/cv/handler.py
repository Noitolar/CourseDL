import torch
import torch.nn as nn
import torchvision.transforms as trans
import time
import transformers as tfm
import utils.cv as ucv


class ModelHandlerCv(nn.Module):
    def __init__(self, config: ucv.config.ConfigObject):
        super().__init__()
        self.config = config
        tfm.set_seed(config.seed)
        self.recorder = ucv.recorder.Recorder(config.log_path)
        self.device = config.device
        self.model = config.model_class(**config.model_params).to(config.device)
        self.criterion = config.criterion_class(**config.criterion_params)

    def log(self, message):
        self.recorder.audit(message)

    def log_config(self):
        self.log(f"\n\n[+] exp starts from: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        for config_key, config_value in self.config.params_dict.items():
            if config_value is None:
                continue
            elif config_key.endswith("_class"):
                self.log(f"[+] {config_key.replace('_class', '')}: {config_value.__name__}")
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
        return data


class ModelHandlerVanilla(ModelHandlerCv):
    def __init__(self, config: ucv.config.ConfigObject):
        super().__init__(config)

    def forward(self, inputs: torch.Tensor, targets=None):
        inputs = self.device_transfer(inputs)
        targets = self.device_transfer(targets)
        preds = self.model(inputs)
        loss = self.criterion(preds, targets) if targets is not None else None
        return preds, loss


class ModelHandlerLpr(ModelHandlerCv):
    def __init__(self, config: ucv.config.ConfigObject):
        super().__init__(config)

    def forward(self, inputs: torch.Tensor, targets=None):
        inputs = self.device_transfer(inputs)
        targets = self.device_transfer(targets)
        preds = self.model(inputs)
        loss = self.criterion(preds, targets) if targets is not None else None
        return preds, loss
