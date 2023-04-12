import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import utils.nlp as unlp

if __name__ == "__main__":
    dataset = unlp.dataset.DatasetSentimentClassifier(from_file="./datasets/movie/test.txt", sequence_length=64)
    print(dataset[0])



