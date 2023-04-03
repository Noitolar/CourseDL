import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import transformers as tfm
import random
import numpy as np
import utils.nlp as unlp
import utils.common as ucommon

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    tfm.set_seed(0)

    model = unlp.nnmodels.DistilBertClassifier(num_classes=2)
    handler = unlp.handler.ModelHandlerNlpClassifier(model, nn.CrossEntropyLoss(), torch.device("cuda:0"), logpath="./logs/run.log")

    trn_set = unlp.dataset.DatasetNlpClassifier(model.tokenizer, 64, "./datasets/sst2", "validation")
    val_set = unlp.dataset.DatasetNlpClassifier(model.tokenizer, 64, "./datasets/sst2", "validation")
    trn_loader = tdata.DataLoader(trn_set, batch_size=32, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=512)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = None
    trainer = ucommon.trainer.Trainer(optimizer, scheduler)
    trainer.train_and_validate(handler, trn_loader, val_loader, num_epochs=8)
