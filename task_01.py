import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import transformers as tfm
import torchvision.transforms as trans
import utils.cv as ucv

if __name__ == "__main__":
    seed = 0
    tfm.set_seed(seed)

    model = ucv.nnmodels.SimpleConvClassifier(num_classes=10, num_channels=1)
    criterion = nn.CrossEntropyLoss()
    device = "cuda:0"
    logpath = "./logs/mnist.conv.log"

    dataset = "mnist"
    trn_preprocess = trans.Compose([trans.ToTensor(), trans.Normalize(mean=[0.485], std=[0.229])])
    val_preprocess = trn_preprocess

    batch_size = 32
    num_epochs = 8
    optimizer_params = {"lr": 0.001, "momentum": 0.9, "nesterov": True}
    optimizer = optim.SGD(model.parameters(), **optimizer_params)
    scheduler_params = {"milestones": [4, 6], "gamma": 0.1}
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
    checkpoint_path = "./checkpoints/mnist.conv.pt"

    handler = ucv.handler.ModelHandlerCv(model, criterion, device, logpath)
    handler.record_metadata("dataset", dataset)
    handler.record_metadata("seed", seed)
    handler.record_metadata("trn_preprocess", trn_preprocess)
    handler.record_metadata("val_preprocess", val_preprocess)
    handler.record_metadata("batch_size", batch_size)
    handler.record_metadata("num_epochs", num_epochs)
    handler.record_metadata("optimizer", optimizer)
    handler.record_metadata("optimizer_params", optimizer_params)
    handler.record_metadata("scheduler", scheduler)
    handler.record_metadata("scheduler_params", scheduler_params)
    handler.record_metadata("checkpoint_path", checkpoint_path)
    handler.log_metadata()

    trn_set, val_set = ucv.dataset.get_dataset(dataset, trn_preprocess, val_preprocess)
    trn_loader = tdata.DataLoader(trn_set, batch_size=batch_size, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=batch_size * 8)
    trainer = ucv.trainer.Trainer(optimizer, scheduler)
    trainer.train_and_validate(handler, trn_loader, val_loader, num_epochs, save_to=checkpoint_path)
