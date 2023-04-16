import torch
import torch.nn as nn
import torch.nn.functional as func


class LstmGnerator(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_output_size, num_lstm_layers=1, lstm_dropout=0.0):
        super().__init__()
        self.lstm_output_size = lstm_output_size
        self.num_lstm_layers = num_lstm_layers
        # 在windows上多层lstm使用dropout或导致driver shutdown告警，应该是torch的问题
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_output_size, num_layers=num_lstm_layers, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 2048),
            nn.Tanh(),
            nn.Linear(2048, vocab_size))

    def forward(self, inputs, hiddens):
        outputs = self.embedding(inputs)
        # lstm_outputs.shape: (batch_size, sequence_length, vocab_size)
        # lstm_hiddens: (lstm_h0, lstm_c0)
        outputs, lstm_hiddens = self.lstm(outputs, hiddens)
        outputs = self.fc(outputs)
        return outputs, lstm_hiddens


class TextConvClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate, conv_out_channelses, kernel_sizes, pretrained_embeddings, freeze_embeddings=False):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        self.embed_size = int(pretrained_embeddings.shape[-1])
        self.parallel_conv_layers = nn.ModuleList([nn.Conv2d(1, conv_out_channels, (kernel_size, self.embed_size)) for conv_out_channels, kernel_size in zip(conv_out_channelses, kernel_sizes)])
        # self.bn = nn.BatchNorm2d(conv_out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(sum(conv_out_channelses), num_classes)

    def forward(self, inputs):
        outputs = self.embeddings(inputs).unsqueeze(dim=1)
        outputs = [conv_layer(outputs).squeeze(dim=3) for conv_layer in self.parallel_conv_layers]
        outputs = [func.relu(output) for output in outputs]
        outputs = [func.max_pool1d(output, output.size(dim=2)).squeeze(dim=2) for output in outputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc(outputs)
        return outputs
