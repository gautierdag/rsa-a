import argparse
import sys
import torch

from utils import *
from data import *
from .model import Sender, Receiver
from .trainer import ReferentialTrainer

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
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
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
        help="input batch size for training (default: 1024)",
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
    model_path = run_folder + "/model.p"

    vocab = AgentVocab(args.vocab_size)

    # get sender and receiver models and save them
    sender = Sender(
        args.vocab_size,
        args.max_length,
        vocab.eos,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        greedy=True,
        gumbel_softmax=True,
        input_size=15,
    )

    receiver = Receiver(
        args.vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=15,
    )

    model = ReferentialTrainer(sender, receiver)

    epoch, iteration = 0, 0

    if args.resume_training:
        epoch, iteration = load_model_state(model, model_path)
        print(f"Loaded model. Resuming from - epoch: {epoch} | iteration: {iteration}")

    # meta = get_encoded_metadata(size=1000)
    # meaning_space = np.unique(meta, axis=0)
    # dataset = ReferentialDataset(meaning_space.astype(np.float32))
    # sampler = ReferentialSampler
    # train_data, valid_data = split_dataset_into_dataloaders(
    #     dataset, sizes=[100, 62], sampler=sampler
    # )

    train_data, valid_data, _, _, _ = get_training_data(
        device=device,
        batch_size=args.batch_size,
        k=3,
        dataset_type="meta",
        debugging=False,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    # Print info
    print("----------------------------------------")
    print(
        "Model name: {} \n|V|: {}\nL: {}".format(
            model_name, args.vocab_size, args.max_length
        )
    )
    print(sender)
    if receiver:
        print(receiver)

    print("Total number of parameters: {}".format(pytorch_total_params))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    while iteration < args.iterations:
        for train_batch in train_data:
            print(f"{iteration}/{args.iterations}\r", end="")

            train_helper.train_one_batch(model, train_batch, optimizer, None, device)

            if iteration % args.log_interval == 0:
                valid_loss_meter, valid_acc_meter, messages, = train_helper.evaluate(
                    model, valid_data, None, device, False
                )
                save_model_state(model, model_path, epoch, iteration)
                torch.save(messages, run_folder + "/messages_at_{}.p".format(iteration))

                metrics = {
                    "loss": valid_loss_meter.avg,
                    "accuracy": valid_acc_meter.avg,
                }

                logger.log_metrics(iteration, metrics)

            iteration += 1
            if iteration >= args.iterations:
                break

        epoch += 1

    return run_folder


if __name__ == "__main__":
    main(sys.argv[1:])