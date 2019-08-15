import random
import numpy as np
import torch
from typing import Tuple
import os


# Training and Evaluation helper functions


class AverageMeter:
    def __init__(self):
        """
        Computes and stores the average and current value
        Taken from:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_batch(model, optimizer, targets, distractors):
    """
    Train for single batch
    """
    model.train()
    optimizer.zero_grad()
    loss, acc, _ = model(targets, distractors)
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def evaluate(model, data) -> dict:
    """
    Evaluates model on data
    """
    # metrics
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ent_meter = AverageMeter()

    # messages
    sequences = []

    # hidden states
    h_sender = []
    h_rnn_sender = []
    h_receiver = []
    h_rnn_receiver = []

    # targets and distractors
    T = []
    D = []

    model.eval()
    with torch.no_grad():
        for (targets, distractors) in data:
            T.append(targets)
            D.append(torch.cat(distractors, 0))

            loss, acc, ent, seq, h_s, h_rnn_s, h_r, h_rnn_r = model(
                targets, distractors
            )

            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            ent_meter.update(ent)

            sequences.append(seq)

            h_sender.append(h_s)
            h_rnn_sender.append(h_rnn_s)
            h_receiver.append(h_r)
            h_rnn_receiver.append(h_rnn_r)

    metrics = {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "entropy": ent_meter.avg,
        "messages": torch.cat(sequences, 0).cpu().numpy(),
        "h_sender": torch.cat(h_rnn_sender, 0).cpu().numpy(),
        "h_rnn_sender": torch.cat(h_rnn_sender, 0).cpu().numpy(),
        "h_receiver": torch.cat(h_rnn_sender, 0).cpu().numpy(),
        "h_rnn_receiver": torch.cat(h_rnn_receiver, 0).cpu().numpy(),
        "targets": torch.cat(T, 0).cpu().numpy(),
        "distractors": torch.cat(D, 0).cpu().numpy(),
    }

    return metrics


# Folder/Saving/Loading functions
def get_filename(params: dict) -> str:
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = "lstm"  # params.model_type
    name += "_h_{}".format(params.hidden_size)
    name += "_lr_{}".format(params.lr)
    name += "_max_len_{}".format(params.max_length)
    name += "_vocab_{}".format(params.vocab_size)
    return name


def seed_torch(seed: int = 42) -> None:
    """
    Seed random, numpy and torch with same seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_folder_if_not_exists(folder_name: str) -> None:
    """
    Creates folder at folder name if folder does not exist
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def save_model_state(model, model_path: str, epoch: int, iteration: int) -> None:
    checkpoint_state = {}
    checkpoint_state["sender"] = model.sender.state_dict()
    checkpoint_state["receiver"] = model.receiver.state_dict()
    checkpoint_state["epoch"] = epoch
    checkpoint_state["iteration"] = iteration
    torch.save(checkpoint_state, model_path)


def load_model_state(model, model_path: str) -> Tuple[int, int]:
    checkpoint = torch.load(model_path)
    model.sender.load_state_dict(checkpoint["sender"])
    model.receiver.load_state_dict(checkpoint["receiver"])
    epoch = checkpoint["epoch"]
    iteration = checkpoint["iteration"]
    return epoch, iteration
