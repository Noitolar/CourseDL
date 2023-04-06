import torch
import torch.utils.data as tdata
import numpy as np


class DatasetPoemGenerator(tdata.Dataset):
    def __init__(self, dataset_name, window=124):
        assert dataset_name in ["poem"]
        npz_data = np.load(f"./datasets/{dataset_name}/tang.npz", allow_pickle=True)
        self.vocab_encode = npz_data["word2ix"].item()
        self.vocab_decode = npz_data["ix2word"].item()
        self.sentences = torch.tensor(npz_data["data"], dtype=torch.long)
        self.window = window

    def show_sentence(self, index):
        print("".join([self.vocab_decode[int(x)] for x in self.sentences[index]]))

    def __getitem__(self, index):
        sentence = self.sentences[index, 0:self.window]
        target = self.sentences[index, 1:1 + self.window]
        return sentence, target

    def __len__(self):
        return len(self.sentences)
