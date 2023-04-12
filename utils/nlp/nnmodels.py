import torch.nn as nn


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
    def __init__(self, sequence_length, num_classes, dropout_rate, embed_size=50):
        super().__init__()
        self.layer1 = self.build_layer(1, 16, conv_kernel_size=(2, embed_size), pool_kernel_size=(sequence_length - 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(888, num_classes)

    @staticmethod
    def build_layer(conv_in_channels, conv_out_channels, conv_kernel_size=(5, 5), conv_stride=1, conv_padding=2, pool_kernel_size=(2, 2)):
        layer = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_out_channels, conv_kernel_size, conv_stride, conv_padding),
            nn.ReLU(), nn.BatchNorm2d(conv_out_channels), nn.MaxPool2d(pool_kernel_size))
        return layer

    def forward(self, inputs):
        outputs = self.layer1(inputs)
        return outputs
