import torch
import numpy as np


class DataHandler:
    def __init__(self, num_of_batches=20):
        self.init_dict()
        self.num_of_batches = num_of_batches

    def init_dict(self):
        with open("./data/ptb.train.txt") as f:
            file = f.read()
            train_data = file[1:].split(' ')

        self.dict = sorted(set(train_data))
        self.coded_dict = {c: i for i, c in enumerate(self.dict)}
        self.decode_dict = {i: c for i, c in enumerate(self.dict)}

        self.vocab_size = len(self.dict)


    def get_data(self, data_group):
        if data_group == 'train':
            with open("./data/ptb.train.txt") as f:
                file = f.read()
                return self.get_coded_data_from_file(file)

        elif data_group == 'validation':
            with open("./data/ptb.valid.txt") as f:
                file = f.read()
                return self.get_coded_data_from_file(file)

        elif data_group == 'test':
            with open("./data/ptb.test.txt") as f:
                file = f.read()
                return self.get_coded_data_from_file(file)

        else:
            raise('data_group should be "train" \ "validation" \ "test"')

    def get_coded_data_from_file(self, file):
        data = file[1:].split(' ')
        batch_size = int(np.floor(len(data) / self.num_of_batches))
        data = data[0:batch_size * self.num_of_batches]
        coded_data = torch.LongTensor([self.coded_dict[word] for word in data])
        coded_data = torch.reshape(coded_data, (self.num_of_batches, batch_size))
        return coded_data

    def decode_seq(self, seq):
        batch_size = seq.shape[0]
        seq = seq.reshape(-1)
        decoded_data = np.array([self.decode_dict[int(token)] for token in seq])
        decoded_data = np.reshape(decoded_data, (batch_size, -1))
        return decoded_data