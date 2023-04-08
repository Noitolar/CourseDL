import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import torchvision.transforms as trans
import utils.cv as ucv

if __name__ == "__main__":
    config = ucv.config.ConfigObject()

    config.model_class = ucv.nnmodels.SimpleConvClassifier
    config.model_params = {"num_classes": 10, "num_channels": 1}
    config.device = "cuda:0"
    config.criterion_class = nn.CrossEntropyLoss
    config.criterion_params = {}
    config.log_path = "./logs/mnist.conv.log"

    config.dataset = "mnist"
    config.seed = 0
    config.trn_preprocess = trans.Compose([trans.ToTensor(), trans.Normalize(mean=[0.485], std=[0.229])])
    config.val_preprocess = config.trn_preprocess
    config.batch_size = 32
    config.num_epochs = 8
    config.optimizer_class = optim.SGD
    config.optimizer_params = {"lr": 0.001, "momentum": 0.9, "nesterov": True}
    config.scheduler_class = optim.lr_scheduler.MultiStepLR
    config.scheduler_params = {"milestones": [4, 6], "gamma": 0.1}
    config.checkpoint_path = "./checkpoints/mnist.conv.pt"

    handler = ucv.handler.ModelHandlerCv(config)
    handler.log_config()

    trn_set, val_set = ucv.dataset.get_dataset(config)
    trn_loader = tdata.DataLoader(trn_set, batch_size=config.batch_size, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=config.batch_size * 8)
    trainer = ucv.trainer.Trainer(handler, config)
    trainer.train_and_validate(trn_loader, val_loader)
