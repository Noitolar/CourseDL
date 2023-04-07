import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import utils.nlp as unlp

if __name__ == "__main__":
    dataset = unlp.dataset.DatasetPoemGenerator("poem", 50)
    inputs, targets = dataset[0]
    trn_loader = tdata.DataLoader(dataset, batch_size=1)
    model = unlp.nnmodels.LstmGnerator(len(dataset.vocab["encode"]), 100, 1024, 1)
    handler = unlp.handler.ModelHandlerGenerator(model, nn.CrossEntropyLoss(), dataset.vocab, "cuda:0", None)
    trainer = unlp.trainer.Trainer(optim.SGD(model.parameters(), lr=0.001))
    report = trainer.train(handler, trn_loader, 1)
    # x = trainer.validate(handler, [handler.encode(x) for x in "泰山"], 16)
    # print(x)
