import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import transformers as tfm
import random
import numpy as np
import torchvision.transforms as trans
import utils.cv as ucv

if __name__ == "__main__":
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    tfm.set_seed(seed)

    model = ucv.nnmodels.SimpleConvClassifier(num_classes=10, num_channels=1)
    criterion = nn.CrossEntropyLoss()
    device = "cuda:0"
    logpath = "./logs/mnist.conv.log"

    optimizer_params = {"lr": 0.001, "momentum": 0.9, "nesterov": True}
    optimizer = optim.SGD(model.parameters(), **optimizer_params)
    scheduler_params = {}
    scheduler = None

    dataset = "mnist"
    batch_size = 32
    num_epochs = 8
    checkpoint_path = "./checkpoints/mnist.conv.pt"

    handler = ucv.handler.ModelHandlerCv(model, criterion, device, logpath)
    handler.record_metadata("dataset", dataset)
    handler.record_metadata("seed", seed)
    handler.record_metadata("batch_size", batch_size)
    handler.record_metadata("num_epochs", num_epochs)
    handler.record_metadata("checkpoint_path", checkpoint_path)
    handler.record_metadata("optimizer", optimizer)
    handler.record_metadata("optimizer_params", optimizer_params)
    handler.record_metadata("scheduler", optimizer)
    handler.record_metadata("scheduler_params", optimizer_params)

    preprocess = trans.Compose([trans.ToTensor(), trans.Normalize(mean=[0.485], std=[0.229])])
    trn_set, val_set = ucv.dataset.get_dataset(dataset, preprocess)
    trn_loader = tdata.DataLoader(trn_set, batch_size=batch_size, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=batch_size * 8)
    trainer = ucv.trainer.Trainer(optimizer, scheduler)
    trainer.train_and_validate(handler, trn_loader, val_loader, num_epochs, save_to=checkpoint_path)
