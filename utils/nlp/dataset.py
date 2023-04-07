import torch
import torch.utils.data as tdata
import torch.nn.functional as func
import numpy as np


class DatasetPoemGenerator(tdata.Dataset):
    def __init__(self, dataset_name, sequence_length=50):
        assert dataset_name in ["poem"]
        npz_data = np.load(f"./datasets/{dataset_name}/tang.npz", allow_pickle=True)
        self.vocab = {"encode": npz_data["word2ix"].item(), "decode": npz_data["ix2word"].item()}
        self.sentences = npz_data["data"]
        self.sequence_length = sequence_length
        self.preprocess()

    def preprocess(self):
        new_sentences = []
        for sentence in self.sentences:
            new_sentence = [token for token in sentence if token != 8292]
            if len(new_sentence) < self.sequence_length:
                new_sentence.extend([8292] * (self.sequence_length - len(new_sentence)))
            else:
                new_sentence = new_sentence[:self.sequence_length]
            new_sentences.append(new_sentence)
        self.sentences = np.array(new_sentences)
        self.sentences = torch.tensor(self.sentences, dtype=torch.long)

    def show_sentence(self, index):
        print("".join([self.vocab["decode"][int(x)] for x in self.sentences[index]]))

    def __getitem__(self, index):
        sentence = self.sentences[index, :-1]
        target = self.sentences[index, 1:]
        return sentence, target

    def __len__(self):
        return len(self.sentences)
