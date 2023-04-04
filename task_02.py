import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import transformers as tfm
import torchvision.transforms as trans
import utils.cv as ucv

if __name__ == "__main__":
    seed = 0
    tfm.set_seed(seed)

    model = ucv.nnmodels.Resnet18Classifier(num_classes=2, num_channels=3, from_pretrained="imagenet")
    criterion = nn.CrossEntropyLoss()
    device = "cuda:0"
    logpath = "./logs/dogs_vs_cats.resnet18.log"

    dataset = "dogs_vs_cats"
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

    batch_size = 32
    num_epochs = 8
    optimizer_params = {"lr": 0.001, "momentum": 0.9, "nesterov": True}
    optimizer = optim.SGD(model.parameters(), **optimizer_params)
    scheduler_params = {"milestones": [4, 6], "gamma": 0.2}
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
    checkpoint_path = "./checkpoints/dogs_vs_cats.resnet18.pt"

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
