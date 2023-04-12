import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import utils.nlp as unlp

if __name__ == "__main__":
    val_set = unlp.dataset.DatasetSentimentClassifier(from_file="./datasets/movie/test.txt", sequence_length=64)
    val_loader = tdata.DataLoader(val_set, batch_size=4)
    model = unlp.nnmodels.TextConvClassifier(64, 2, 0.1)

    for inputs, targets in val_loader:
        print(inputs.shape)
        preds = model(inputs)
        print(preds.shape)
        exit()





