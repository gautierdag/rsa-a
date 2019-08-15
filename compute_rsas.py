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

    # Calculate Cross-Seed RSA for all in baseline
    baseline_path = "runs/lstm_h_64_lr_0.001_max_len_10_vocab_25"
    seed_folders = glob.glob(f"{baseline_path}/*")

    RESULTS = {}
    for space in tqdm(DIST):
        RESULTS[space] = {}
        for s1, s2 in combinations(seed_folders, 2):

            seed1 = s1.split("/")[-1]
            seed2 = s2.split("/")[-1]

            RESULTS[space][seed1 + seed2] = {}

            files_s1 = glob.glob(f"{s1}/*.pkl")
            for f1 in files_s1:
                metric_file = f1.split("/")[-1]
                iteration = int(metric_file.split("_")[-1].split(".")[0])
                f2 = f"{s2}/{metric_file}"
                if os.path.isfile(f2):

                    m1 = pickle.load(open(f1, "rb"))
                    m2 = pickle.load(open(f2, "rb"))

                    r = rsa(m1[space], m2[space], DIST[space], DIST[space])
                    RESULTS[space][seed1 + seed2][iteration] = r

    pickle.dump(RESULTS, open(f"{baseline_path}/rsa_analysis.pkl", "wb"))

