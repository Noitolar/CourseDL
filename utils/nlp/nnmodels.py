import torch
import torch.nn as nn


class LstmGnerator(nn.Module):
    def __init__(self, embedding_input_dim, embedding_output_dim, lstm_output_dim, num_lstm_layers):
        super().__init__()
        self.lstm_output_dim = lstm_output_dim
        self.num_lstm_layers = num_lstm_layers

        self.embedding = nn.Embedding(embedding_input_dim, embedding_output_dim)
        self.lstm = nn.LSTM(embedding_output_dim, lstm_output_dim, num_layers=num_lstm_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(lstm_output_dim, embedding_input_dim)

    def forward(self, inputs, previous_lstm_hiddens=None):
        if previous_lstm_hiddens is None:
            batch_size, sequence_length = inputs.shape
            lstm_h0 = inputs.data.new(self.num_lstm_layers, batch_size, self.lstm_output_dim).fill_(0).float()
            lstm_c0 = inputs.data.new(self.num_lstm_layers, batch_size, self.lstm_output_dim).fill_(0).float()
            previous_lstm_hiddens = (lstm_h0, lstm_c0)
        embedded_inputs = self.embedding(inputs)
        lstm_outputs, lstm_hiddens = self.lstm(embedded_inputs, previous_lstm_hiddens)
        outputs = self.fc(lstm_outputs.flatten())
        return outputs, lstm_hiddens

