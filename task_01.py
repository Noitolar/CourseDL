import torch
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
    trainer = ucv.trainer.Trainer(handler)

    best_val_accuracy = 0.0
    for epoch in range(config.num_epochs):
        handler.log("    " + "=" * 40)
        trn_report = trainer.train(trn_loader)
        handler.log(f"    [{epoch + 1:03d}] trn-loss: {trn_report['loss']:.4f} --- trn-acc: {trn_report['accuracy']:.2%}")
        val_report = trainer.validate(val_loader)
        handler.log(f"    [{epoch + 1:03d}] val-loss: {val_report['loss']:.4f} --- val-acc: {val_report['accuracy']:.2%}")
        if val_report["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_report["accuracy"]
            if config.checkpoint_path is not None:
                torch.save(handler.model.state_dict, config.checkpoint_path)
    handler.log(f"[=] best-val-acc: {best_val_accuracy:.2%}")
