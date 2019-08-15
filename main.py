import argparse
import sys
import torch
import pickle

from utils import *
from data import *
from model import Sender, Receiver
from trainer import ReferentialTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Trains a Sender/Receiver Agent in a referential game"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        metavar="N",
        help="number of batch iterations to train (default: 10k)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=64,
        metavar="N",
        help="embedding size for embedding layer (default: 64)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        metavar="N",
        help="hidden size for hidden layer (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10,
        metavar="N",
        help="max sentence length allowed for communication (default: 10)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=25,
        metavar="N",
        help="Size of vocabulary (default: 25)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="N",
        help="Adam learning rate (default: 1e-3)",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="number of iterations between logs (default: 200)",
    )
    parser.add_argument(
        "--resume-training",
        help="Resume the training from the saved model state",
        action="store_true",
    )

    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_arguments(args)
    seed_torch(seed=args.seed)
    model_name = get_filename(args)
    run_folder = "runs/" + model_name + "/" + str(args.seed)

    create_folder_if_not_exists("runs")
    create_folder_if_not_exists("runs/" + model_name)
    create_folder_if_not_exists("runs/" + model_name + "/" + str(args.seed))

    model_path = run_folder + "/model.p"

    vocab = AgentVocab(args.vocab_size)

    # get sender and receiver models and save them
    sender = Sender(
        vocab,
        args.max_length,
        input_size=14,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        greedy=True,
    )

    receiver = Receiver(
        vocab,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=14,
    )

    model = ReferentialTrainer(sender, receiver)

    epoch, iteration = 0, 0
    if args.resume_training:
        epoch, iteration = load_model_state(model, model_path)
        print(f"Loaded model. Resuming from - epoch: {epoch} | iteration: {iteration}")

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    # Print info
    print("----------------------------------------")
    print(f"Model name: {model_name} \n|V|: {args.vocab_size}\nL: { args.max_length}")
    print(sender)
    if receiver:
        print(receiver)
    print(f"Total number of parameters: {pytorch_total_params}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # datasets
    train_data = get_referential_dataloader("shapes", shuffle=True)
    valid_data = get_referential_dataloader("shapes")

    # Train
    while iteration < args.iterations:
        for (targets, distractors) in train_data:
            print(f"{iteration}/{args.iterations}\r", end="")

            train_one_batch(model, optimizer, targets, distractors)

            if iteration % args.log_interval == 0:
                metrics = evaluate(model, valid_data)
                save_model_state(model, model_path, epoch, iteration)

                pickle.dump(
                    metrics, open(run_folder + f"/metrics_at_{iteration}.pkl", "wb")
                )
                print(f"\t\t acc: {metrics['acc']:.3f}\r", end="")

            iteration += 1
            if iteration >= args.iterations:
                break

        epoch += 1


if __name__ == "__main__":
    main(sys.argv[1:])
