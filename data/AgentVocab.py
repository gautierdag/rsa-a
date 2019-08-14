import os
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))


class AgentVocab(object):
    """
    Vocab object to create vocabulary and load if exists
    """

    SOS_TOKEN = "<S>"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.full_vocab_size = vocab_size+3 # add specific tokens

        self.file_path = dir_path + "/dict_{}.pckl".format(self.vocab_size)
        if self.does_vocab_exist():
            self.load_vocab()
        else:
            self.build_vocab()

    def does_vocab_exist(self):
        return os.path.exists(self.file_path)

    def load_vocab(self):
        with open(self.file_path, "rb") as f:
            d = pickle.load(f)
            self.stoi = d["stoi"]  # dictionary w->i
            self.itos = d["itos"]  # list of words

            # load specific tokens
            self.pad = self.stoi[self.PAD_TOKEN]
            self.sos = self.stoi[self.SOS_TOKEN]
            self.eos = self.stoi[self.EOS_TOKEN]

    def save_vocab(self):
        with open(self.file_path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def build_vocab(self):
        self.stoi = {}
        self.itos = []

        # 0 is reserved for padding
        self.itos.append(self.PAD_TOKEN)
        self.stoi[self.PAD_TOKEN] = 0

        # add vocab tokens to itos and stoi
        for i in range(1, self.vocab_size+1):
            self.itos.append(str(i))
            self.stoi[str(i)] = i

        # add sos and eos
        self.itos.append(self.SOS_TOKEN)
        self.stoi[self.SOS_TOKEN] = len(self.itos) - 1
        self.itos.append(self.EOS_TOKEN)
        self.stoi[self.EOS_TOKEN] = len(self.itos) - 1

        self.pad = self.stoi[self.PAD_TOKEN]
        self.sos = self.stoi[self.SOS_TOKEN]
        self.eos = self.stoi[self.EOS_TOKEN]
        
        self.save_vocab()
