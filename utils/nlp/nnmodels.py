import torch
import torch.nn as nn


class LstmGnerator(nn.Module):
    def __init__(self, vocab_size, lstm_input_size, lstm_output_size, num_lstm_layers):
        super().__init__()
        self.lstm_output_size = lstm_output_size
        self.num_lstm_layers = num_lstm_layers
        self.embedding = nn.Embedding(vocab_size, lstm_input_size)
        self.lstm = nn.LSTM(lstm_input_size, lstm_output_size, num_layers=num_lstm_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(lstm_output_size, vocab_size)

    def forward(self, inputs, hiddens):
        outputs = self.embedding(inputs)
        # lstm_outputs.shape: (batch_size, sequence_length, vocab_size)
        # lstm_hiddens: (lstm_h0, lstm_c0)
        outputs, lstm_hiddens = self.lstm(outputs, hiddens)
        outputs = self.fc(outputs)
        return outputs, lstm_hiddens
