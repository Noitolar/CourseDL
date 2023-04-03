import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import transformers as tfm
import random
import numpy as np
import torchvision.transforms as trans
import utils.cv as ucv
import utils.common as ucommon

if __name__ == "__main__":
    handler = ucv.handler.ModelHandlerCv(None, nn.CrossEntropyLoss(), torch.device("cuda:0"), logpath="./logs/dogs_vs_cats.resnet18.log")
    handler.metadata["dataset"] = "dogs_vs_cats"
    handler.metadata["seed"] = 0
    handler.metadata["batch_size"] = 32
    handler.metadata["num_pochs"] = 10
    handler.metadata["optimizer_params"] = {"lr": 0.001, "momentum": 0.9, "nesterov": True, "weight_decay": 5e-4}
    handler.metadata["scheduler_params"] = {"step_size": 3, "gamma": 0.2}
    handler.metadata["checkpoint_path"] = "./checkpoints/dogs_vs_cats.resnet18.pt"

    random.seed(handler.metadata["seed"])
    np.random.seed(handler.metadata["seed"])
    tfm.set_seed(handler.metadata["seed"])

    trn_preprocess = trans.Compose([
        trans.Resize((256, 256)),
        trans.RandomCrop((224, 224)),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
    ])
    val_preprocess = trans.Compose([
        trans.Resize((256, 256)),
        trans.ToTensor(),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
    ])
    trn_set, val_set = ucv.dataset.get_dataset(handler.metadata["dataset"], trn_preprocess, val_preprocess)
    trn_loader = tdata.DataLoader(trn_set, batch_size=handler.metadata["batch_size"], shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=handler.metadata["batch_size"] * 8)

    handler.model = ucv.nnmodels.Resnet18Classifier(num_classes=2, num_channels=3, use_pretrained=True).to(handler.device)
    handler.optimizer = optim.SGD(handler.model.parameters(), **handler.metadata["optimizer_params"])
    handler.scheduler = optim.lr_scheduler.StepLR(handler.optimizer, **handler.metadata["scheduler_params"])
    handler.log_metadata()

    trainer = ucommon.trainer.Trainer(handler)
    trainer.train_and_validate(trn_loader, val_loader, num_epochs=handler.metadata["num_pochs"], save_to=handler.metadata["checkpoint_path"])
