import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import torchvision.transforms as trans
import utils.cv as ucv

if __name__ == "__main__":
    config = ucv.config.ConfigObject()

    config.model_class = ucv.nnmodels.Resnet18Classifier
    config.model_params = {"num_classes": 2, "num_channels": 3, "from_pretrained": "imagenet"}
    config.device = "cuda:0"
    config.criterion_class = nn.CrossEntropyLoss
    config.criterion_params = {}
    config.log_path = "./logs/dogs_vs_cats.resnet18.log"

    config.dataset = "dogs_vs_cats"
    config.seed = 0
    config.trn_preprocess = trans.Compose([
        trans.Resize((256, 256)),
        trans.RandomCrop((224, 224)),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
    ])
    config.val_preprocess = trans.Compose([
        trans.Resize((256, 256)),
        trans.ToTensor(),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
    ])
    config.batch_size = 32
    config.num_epochs = 8
    config.optimizer_class = optim.SGD
    config.optimizer_params = {"lr": 0.001, "momentum": 0.9, "nesterov": True}
    config.scheduler_class = optim.lr_scheduler.StepLR
    config.scheduler_params = {"step_size": 4, "gamma": 0.1}
    config.checkpoint_path = "./checkpoints/dogs_vs_cats.resnet18.pt"

    handler = ucv.handler.ModelHandlerCv(config)
    handler.log_config()

    trn_set, val_set = ucv.dataset.get_dataset(config)
    trn_loader = tdata.DataLoader(trn_set, batch_size=config.batch_size, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=config.batch_size * 8)
    trainer = ucv.trainer.Trainer(handler, config)
    trainer.train_and_validate(trn_loader, val_loader)
