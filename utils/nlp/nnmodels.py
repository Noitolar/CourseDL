import torch.nn as nn


class LstmGnerator(nn.Module):
    def __init__(self, vocab_size, lstm_input_size, lstm_output_size, num_lstm_layers=1, lstm_dropout=0.0):
        super().__init__()
        self.lstm_output_size = lstm_output_size
        self.num_lstm_layers = num_lstm_layers
        # 在windows上多层lstm使用dropout或导致driver shutdown告警，应该是torch的问题
        self.embedding = nn.Embedding(vocab_size, lstm_input_size)
        self.lstm = nn.LSTM(lstm_input_size, lstm_output_size, num_layers=num_lstm_layers, batch_first=True, dropout=lstm_dropout)
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
