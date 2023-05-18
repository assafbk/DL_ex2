import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PTBRNN(nn.Module):
    def __init__(self, vocab_size=50000, emb_dim=200, dropout_p=0, use_gru=False):
        super(PTBRNN, self).__init__()
        if use_gru:
            self.rnn_layer_1 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
            self.rnn_layer_2 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        else:
            self.rnn_layer_1 = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
            self.rnn_layer_2 = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.token2emb = nn.Embedding(vocab_size, emb_dim)
        # self.emb2token = nn.Linear(emb_dim, vocab_size, bias=False)
        self.emb2token = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout_p)


    # h_and_c is a list of length num_of_layers. each entry holds the layer's (h,c) tuple
    # h_and_c hold the initial values for of the hidden states in the current batch, and return the final values of the hidden states of the current batch
    def forward(self, input, h_and_c):
        embs = self.token2emb(input)
        embs = self.dropout(embs)
        output1, h_and_c[0] = self.rnn_layer_1(embs, h_and_c[0])
        output1 = self.dropout(output1)
        output2, h_and_c[1] = self.rnn_layer_2(output1, h_and_c[1])
        output2 = self.dropout(output2)
        model_output = self.emb2token(output2)
        return model_output, h_and_c

    def init_weights(self):
        initrange = 0.1
        self.token2emb.weight.data.uniform_(-initrange, initrange)
        self.emb2token.bias.data.fill_(0)
        self.emb2token.weight.data.uniform_(-initrange, initrange)