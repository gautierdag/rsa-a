import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from data import AgentVocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gumbel_softmax(probs, tau: float, hard: bool):
    """ Computes sampling from the Gumbel Softmax (GS) distribution
    Args:
        probs (torch.tensor): probabilities of shape [batch_size, n_classes] 
        tau (float): temperature parameter for the GS
        hard (bool): discretize if True
    """

    rohc = RelaxedOneHotCategorical(tau, probs)
    y = rohc.rsample()

    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y

    return y


class Receiver(nn.Module):
    def __init__(
        self,
        vocab: AgentVocab,
        embedding_size: int = 64,
        hidden_size: int = 64,
        output_size: int = 64,
        cell_type: str = "lstm",
    ):
        super().__init__()

        self.vocab_size = vocab.full_vocab_size

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type

        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        else:
            raise ValueError(
                "Receiver case with cell_type '{}' is undefined".format(cell_type)
            )

        self.embedding = nn.Parameter(
            torch.empty((self.vocab_size, embedding_size), dtype=torch.float32)
        )

        self.output_module = nn.Identity()
        if self.output_size != self.hidden_size:
            self.output_module = nn.Linear(hidden_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)
        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
            )

    def forward(self, messages):
        batch_size = messages.shape[0]

        emb = (
            torch.matmul(messages, self.embedding)
            if self.training
            else self.embedding[messages]
        )

        # initialize hidden
        h = torch.zeros([batch_size, self.hidden_size], device=device)
        if self.cell_type == "lstm":
            c = torch.zeros([batch_size, self.hidden_size], device=device)
            h = (h, c)

        # make sequence_length be first dim
        seq_iterator = emb.transpose(0, 1)
        for w in seq_iterator:
            h = self.rnn(w, h)

        if self.cell_type == "lstm":
            h = h[0]  # keep only hidden state

        out = self.output_module(h)

        return out, emb


class Sender(nn.Module):
    def __init__(
        self,
        vocab: AgentVocab,
        output_len: int,
        input_size: int = 64,
        embedding_size: int = 64,
        hidden_size: int = 64,
        greedy: bool = False,
        cell_type: str = "lstm",
    ):
        super().__init__()

        # set vocab
        self.vocab_size = vocab.full_vocab_size
        self.sos_id = vocab.sos
        self.eos_id = vocab.eos
        self.pad_id = vocab.pad

        self.cell_type = cell_type
        self.output_len = output_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.input_module = nn.Identity()
        if self.input_size != self.hidden_size:
            self.input_module = nn.Linear(input_size, hidden_size)

        self.greedy = greedy

        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        else:
            raise ValueError(
                "ShapesSender case with cell_type '{}' is undefined".format(cell_type)
            )

        self.embedding = nn.Parameter(
            torch.empty((self.vocab_size, embedding_size), dtype=torch.float32)
        )
        self.linear_out = nn.Linear(
            hidden_size, self.vocab_size
        )  # from a hidden state to the vocab

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)

        self.input_module.reset_parameters()

        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            # # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
            # # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
            )

    def _init_state(self, hidden_state, rnn_type):
        """
            Handles the initialization of the first hidden state of the decoder.
            Hidden state + cell state in the case of an LSTM cell or
            only hidden state in the case of a GRU cell.
            Args:
                hidden_state (torch.tensor): The state to initialize the decoding with.
                rnn_type (type): Type of the rnn cell.
            Returns:
                state: (h, c) if LSTM cell, h if GRU cell
                batch_size: Based on the given hidden_state if not None, 1 otherwise
        """

        # h0
        if hidden_state is None:
            batch_size = 1
            h = torch.zeros([batch_size, self.hidden_size], device=device)
        else:
            batch_size = hidden_state.shape[0]
            h = hidden_state  # batch_size, hidden_size

        # c0
        if rnn_type is nn.LSTMCell:
            c = torch.zeros([batch_size, self.hidden_size], device=device)
            state = (h, c)
        else:
            state = h

        return state, batch_size

    def _calculate_seq_len(self, seq_lengths, token, initial_length, seq_pos):
        """
            Calculates the lengths of each sequence in the batch in-place.
            The length goes from the start of the sequece up until the eos_id is predicted.
            If it is not predicted, then the length is output_len + n_sos_symbols.
            Args:
                seq_lengths (torch.tensor): To keep track of the sequence lengths.
                token (torch.tensor): Batch of predicted tokens at this timestep.
                initial_length (int): The max possible sequence length (output_len + n_sos_symbols).
                seq_pos (int): The current timestep.
        """
        if self.training:
            max_predicted, vocab_index = torch.max(token, dim=1)
            mask = (vocab_index == self.eos_id) * (max_predicted == 1.0)
        else:
            mask = token == self.eos_id
        mask *= seq_lengths == initial_length
        seq_lengths[mask.nonzero()] = seq_pos + 1  # start always token appended

    def forward(self, hidden_state, tau=1.2):
        """
        Performs a forward pass. If training, use Gumbel Softmax (hard) for sampling, else use
        discrete sampling.
        Hidden state here represents the encoded image/metadata - initializes the RNN from it.
        """
        hidden_state = self.input_module(hidden_state)
        state, batch_size = self._init_state(hidden_state, type(self.rnn))

        # Init output
        if self.training:
            output = [
                torch.zeros(
                    (batch_size, self.vocab_size), dtype=torch.float32, device=device
                )
            ]
            output[0][:, self.sos_id] = 1.0
        else:
            output = [
                torch.full(
                    (batch_size,),
                    fill_value=self.sos_id,
                    dtype=torch.int64,
                    device=device,
                )
            ]

        # Keep track of sequence lengths
        initial_length = self.output_len + 1  # add the sos token
        seq_lengths = (
            torch.ones([batch_size], dtype=torch.int64, device=device) * initial_length
        )

        embeds = []  # keep track of the embedded sequence
        entropy = 0.0

        for i in range(self.output_len):
            if self.training:
                emb = torch.matmul(output[-1], self.embedding)
            else:
                emb = self.embedding[output[-1]]

            embeds.append(emb)
            state = self.rnn(emb, state)

            if type(self.rnn) is nn.LSTMCell:
                h, c = state
            else:
                h = state

            p = F.softmax(self.linear_out(h), dim=1)
            entropy += Categorical(p).entropy()

            if self.training:
                token = _gumbel_softmax(p, tau, hard=True)
            else:
                if self.greedy:
                    _, token = torch.max(p, -1)

                else:
                    token = Categorical(p).sample()

                if batch_size == 1:
                    token = token.unsqueeze(0)

            output.append(token)

            self._calculate_seq_len(seq_lengths, token, initial_length, seq_pos=i + 1)

        return (
            torch.stack(output, dim=1),
            seq_lengths,
            torch.mean(entropy) / self.output_len,
            hidden_state,
            torch.stack(embeds, dim=1),
        )

