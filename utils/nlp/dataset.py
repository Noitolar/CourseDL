import os
import tqdm
import torch
import torch.utils.data as tdata
import gensim
import numpy as np


class DatasetPoemGenerator(tdata.Dataset):
    def __init__(self, sequence_length=50, use_samples=-1):
        npz_data = np.load(f"./datasets/poem/tang.npz", allow_pickle=True)
        self.vocab = {"encode": npz_data["word2ix"].item(), "decode": npz_data["ix2word"].item()}
        if use_samples == -1:
            self.sentences = npz_data["data"]
        else:
            self.sentences = npz_data["data"][:use_samples]
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

    def encode(self, character: str):
        return self.vocab["encode"][character]

    def decode(self, token: int):
        return self.vocab["decode"][token]

    def __getitem__(self, index):
        sentence = self.sentences[index, :-1]
        target = self.sentences[index, 1:]
        return sentence, target

    def __len__(self):
        return len(self.sentences)


class DatasetSentimentClassifier(tdata.Dataset):

    def __init__(self, from_file, sequence_length=64):
        self.vocab_embeddings = np.load("./datasets/movie/vocab.npy", allow_pickle=True).item()
        self.sequence_length = sequence_length
        self.sentences = []
        self.targets = []
        self.load_data(from_file)

    def load_data(self, from_file):
        with open(from_file, "r", encoding="utf-8") as file:
            for line in tqdm.tqdm(file.readlines(), desc=f"[+] reading \"{from_file}\"", delay=0.2, leave=False, ascii="->"):
                elements = line.strip().split()
                if len(elements) < 2:
                    continue
                self.targets.append(int(elements[0]))
                sentence = elements[1:]
                if len(sentence) > self.sequence_length:
                    sentence = sentence[:self.sequence_length]
                else:
                    sentence.extend(["_PAD_"] * (self.sequence_length - len(sentence)))
                self.sentences.append(sentence)

    def __getitem__(self, index):
        sentence = torch.tensor(np.array([self.vocab_embeddings[word] for word in self.sentences[index]]))
        target = torch.tensor(self.targets[index])
        return sentence, target

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def build_embeddings(from_dir, to_file, from_pretrained_embeddings_model):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(from_pretrained_embeddings_model, binary=True)
        vocab_embeddings = {"_PAD_": np.random.uniform(-1, 1, size=w2v_model.vector_size)}
        for file_name in [name for name in os.listdir(from_dir) if name.endswith(".txt")]:
            with open(f"{from_dir}/{file_name}", "r", encoding="utf-8") as file:
                for line in tqdm.tqdm(file.readlines(), desc=f"[+] reading \"{file_name}\"", delay=0.2, leave=False, ascii="->"):
                    for word in line.strip().split()[1:]:
                        if word not in vocab_embeddings.keys():
                            try:
                                vocab_embeddings[word] = w2v_model[word]
                            except KeyError:
                                vocab_embeddings[word] = np.random.uniform(-1, 1, size=w2v_model.vector_size)
        np.save(to_file, vocab_embeddings)
