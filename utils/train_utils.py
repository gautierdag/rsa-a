import random
import numpy as np
import torch
import pickle
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from functools import partial
import glob
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


def evaluate(model, data):
    """
    Evaluates model on data
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    sequences = []
    hidden_states = []

    model.eval()
    with torch.no_grad():
        for (targets, distractors) in data:
            loss, acc, seq, hid, _, _, _ = model(targets, distractors)
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            sequences.append(seq)
            hidden_states.append(hid)

    return loss_meter, acc_meter, torch.cat(sequences, 0), torch.cat(hidden_states, 0)


# Folder/Saving/Loading functions
def get_filename(params):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = "lstm"  # params.model_type
    name += "_h_{}".format(params.hidden_size)
    name += "_lr_{}".format(params.lr)
    name += "_iters_{}".format(params.iterations)
    name += "_max_len_{}".format(params.max_length)
    name += "_vocab_{}".format(params.vocab_size)
    name += "_btch_size_{}".format(params.batch_size)
    return name


def seed_torch(seed=42):
    """
    Seed random, numpy and torch with same seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_folder_if_not_exists(folder_name):
    """
    Creates folder at folder name if folder does not exist
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def save_model_state(model, model_path: str, epoch: int, iteration: int):
    checkpoint_state = {}
    if model.sender:
        checkpoint_state["sender"] = model.sender.state_dict()
    if model.receiver:
        checkpoint_state["receiver"] = model.receiver.state_dict()
    if epoch:
        checkpoint_state["epoch"] = epoch
    if iteration:
        checkpoint_state["iteration"] = iteration

    torch.save(checkpoint_state, model_path)


def load_model_state(model, model_path):
    if not os.path.isfile(model_path):
        raise Exception(f'Model not found at "{model_path}"')
    checkpoint = torch.load(model_path)
    if "sender" in checkpoint.keys() and checkpoint["sender"]:
        model.sender.load_state_dict(checkpoint["sender"])
    if "receiver" in checkpoint.keys() and checkpoint["receiver"]:
        model.receiver.load_state_dict(checkpoint["receiver"])
    if "epoch" in checkpoint.keys() and checkpoint["epoch"]:
        epoch = checkpoint["epoch"]
    if "iteration" in checkpoint.keys() and checkpoint["iteration"]:
        iteration = checkpoint["iteration"]
    return epoch, iteration
