"""
Calculates all the RSA for all run files
"""
import glob
import pickle
from itertools import combinations
from scipy import spatial
from metrics import rsa
from data import one_hot
from tqdm import tqdm

VOCAB = 28  # Vocab size + 3 special case tokens (eos, sos, pad)


def flatten_cos(a, b) -> float:
    return spatial.distance.cosine(a.flatten(), b.flatten())


def on_hot_hamming(a, b):
    return spatial.distance.hamming(
        one_hot(a, n_cols=VOCAB).flatten(), one_hot(b, n_cols=VOCAB).flatten()
    )


DIST = {
    "h_sender": spatial.distance.cosine,
    "h_rnn_sender": flatten_cos,
    "h_receiver": spatial.distance.cosine,
    "h_rnn_receiver": flatten_cos,
    "targets": spatial.distance.hamming,
    "messages": on_hot_hamming,
}

if __name__ == "__main__":

    metric_files = glob.glob(f"runs/*/*/*.pkl")

    for file in tqdm(metric_files):
        m = pickle.load(open(file, "rb"))
        for (space_x, space_y) in combinations(list(DIST.keys()), 2):
            rsa_title = f"RSA:{space_x}/{space_y}"
            if rsa_title not in m:
                r = rsa(m[space_x], m[space_y], DIST[space_x], DIST[space_y])
                m[rsa_title] = r
        pickle.dump(m, open(file, "wb"))

